"""Offline ZINC latent protocols from Cls_vae a47090b.

Atom edits encode the complete input batch. Pair experiments encode endpoints
individually, without rotation alignment. Sampling uses one continuous RNG stream
per experiment. Batching is part of the protocol and is recorded in run.json.
"""

import contextlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, MACCSkeys, QED, rdFMCS, rdMolDescriptors, rdShapeHelpers
from scipy.stats import pearsonr

from .utils import engine_for, run_manifest, seed_everything, sha256, write_json

SOURCE_REVISION = "a47090b56f7f8db07b6d002640891807f8c96512"
ATOM_TO_INDEX = {6: 0, 7: 1, 8: 2, 9: 3, 15: 4, 16: 5, 17: 6, 35: 7, 53: 8}
INDEX_TO_ATOM = list(ATOM_TO_INDEX)
DESCRIPTORS = {
    "LabuteASA": Descriptors.LabuteASA,
    "TPSA": Descriptors.TPSA,
    "logP": Crippen.MolLogP,
    "MR": Crippen.MolMR,
    "sp3frac": rdMolDescriptors.CalcFractionCSP3,
    "BertzCT": Descriptors.BertzCT,
    "QED": QED.qed,
}


def _input_specs(args):
    """Keep invalid entries and stable IDs; names are never filesystem paths."""
    path = Path(args.pairs or args.molecules).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    files = {str(path): sha256(path)} if path.is_file() else {}

    def sdf_spec(file, record=0):
        file = file.resolve()
        if file.is_file():
            files[str(file)] = sha256(file)
        return {"sdf": str(file), "record": record}

    if args.pairs:
        if path.is_dir():
            starts = sorted(path.rglob("*_start.sdf"))
            pairs = [
                {"name": str(p.relative_to(path)), "a": sdf_spec(p),
                 "b": sdf_spec(p.with_name(p.name[:-10] + "_end.sdf"))}
                for p in starts
            ]
        else:
            pairs = json.loads(path.read_text())
            if not isinstance(pairs, list):
                raise ValueError("Pair JSON must be a list of {name, a, b} objects")
        inputs, pair_rows = [], []
        for i, pair in enumerate(pairs):
            if not isinstance(pair, dict):
                pair = {"a": None, "b": None}
            pair_id = f"pair-{i:06d}"
            for side in ("a", "b"):
                spec = pair.get(side)
                if isinstance(spec, str):
                    spec = {"smiles": spec}
                if isinstance(spec, dict) and "sdf" in spec:
                    sdf = Path(spec["sdf"])
                    if not sdf.is_absolute():
                        sdf = path.parent / sdf
                    spec = {**spec, **sdf_spec(sdf, spec.get("record", 0))}
                inputs.append({"id": pair_id + "-" + side, "spec": spec})
            pair_rows.append({"id": pair_id, "name": str(pair.get("name", pair_id)),
                              "a": pair_id + "-a", "b": pair_id + "-b"})
    else:
        inputs, pair_rows = [], []
        for sdf in sorted(path.rglob("*.sdf")) if path.is_dir() else [path]:
            spec = sdf_spec(sdf)
            supplier = Chem.SDMolSupplier(str(sdf), removeHs=True)
            for i in range(len(supplier)):
                inputs.append({"id": f"molecule-{len(inputs):06d}", "spec": {**spec, "record": i}})
    if not inputs:
        raise ValueError("No input molecules or pairs found")
    return inputs, pair_rows, files


def _read_molecule(spec, module, seed):
    if not isinstance(spec, dict):
        raise ValueError("Expected a molecule specification")
    kinds = [k for k in ("sdf", "smiles", "molblock") if spec.get(k)]
    if len(kinds) != 1:
        raise ValueError("Specify exactly one of sdf, smiles or molblock")
    if kinds[0] == "sdf":
        supplier = Chem.SDMolSupplier(str(spec["sdf"]), removeHs=True)
        mol = supplier[int(spec.get("record", 0))]
        source = "Original SDF coordinates; hydrogens removed"
    else:
        mol, _, source = module.prepare_molecule(seed=seed, **{kinds[0]: spec[kinds[0]]})
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("Invalid or empty molecule")
    if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
        raise ValueError("A 3D conformer is required")
    if any(a.GetAtomicNum() not in ATOM_TO_INDEX for a in mol.GetAtoms()):
        raise ValueError("Unsupported atom type")
    if not np.isfinite(mol.GetConformer().GetPositions()).all():
        raise ValueError("Non-finite input coordinates")
    return mol, source


def encode_batch(engine, key, molecules):
    """Original centering and batch-dependent UniMol normalization, before splitting."""
    model = engine.load(key)
    device = engine.device
    atoms = torch.tensor([ATOM_TO_INDEX[a.GetAtomicNum()] for mol in molecules for a in mol.GetAtoms()], device=device)
    x = torch.tensor(np.concatenate([m.GetConformer().GetPositions() for m in molecules]), dtype=torch.float32, device=device)
    batch = torch.cat([torch.full((m.GetNumAtoms(),), i, dtype=torch.long, device=device) for i, m in enumerate(molecules)])
    center_pos = importlib.import_module(key + ".models.autoencoder").center_pos
    with torch.inference_mode():
        x, _ = center_pos(x, batch, mode=True)
        zh, zx, global_batch, *_ = model.encode(F.one_hot(atoms, 9).float(), x, batch, deterministic=True)
        condition = model._attach_unimol_to_Zh(zh, global_batch, atoms, x, batch) if key == "molalign3d" else zh
    if not all(torch.isfinite(t).all() for t in (zh, zx, condition)):
        raise ValueError("Non-finite encoder output")
    return [dict(zh=zh[global_batch == i].detach().cpu(), zx=zx[global_batch == i].detach().cpu(),
                 condition=condition[global_batch == i].detach().cpu(), coordinates=x[batch == i].detach().cpu())
            for i in range(len(molecules))]


def decode_batch(engine, key, frames, steps):
    """Decode a batch without reseeding. Node counts come from encoded tensors."""
    device = engine.device
    zx = torch.cat([f["latent"]["zx"] for f in frames]).to(device)
    condition = torch.cat([f["latent"]["condition"] for f in frames]).to(device)
    batch_protein = torch.cat([torch.full((f["latent"]["zx"].shape[0],), i, dtype=torch.long, device=device)
                               for i, f in enumerate(frames)])
    batch_ligand = torch.cat([torch.full((f["atom_count"],), i, dtype=torch.long, device=device) for i, f in enumerate(frames)])
    with torch.inference_mode():
        _, chain, _ = engine.load(key).decoder.sample(
            protein_pos=zx, protein_v=condition, batch_protein=batch_protein,
            batch_ligand=batch_ligand, n_nodes=len(frames), sample_steps=steps, desc="latent experiment")
    positions = chain[-1][0].detach().cpu()
    types = chain[-1][1].argmax(-1).detach().cpu()
    batch_ligand = batch_ligand.cpu()
    return [(positions[batch_ligand == i].numpy(), [INDEX_TO_ATOM[j] for j in types[batch_ligand == i].tolist()])
            for i in range(len(frames))]


def _perceive(builder, coordinates, numbers):
    # Original scripts retained generated hydrogens, including for interpolation descriptors.
    with contextlib.redirect_stdout(io.StringIO()):
        mol = builder.build_mol(coordinates, numbers, require_H=True)
    if mol is None:
        raise ValueError("Bond perception returned no molecule")
    Chem.SanitizeMol(mol)
    return mol


def _stats(values):
    values = [float(x) for x in values if x is not None and math.isfinite(x)]
    n = len(values)
    return {"count": n, "mean": float(np.mean(values)) if n else None,
            "variance": float(np.var(values)) if n else None,
            "std": float(np.std(values)) if n else None,
            "sem": float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else None}


def similarity_metrics(reference, generated, with_mcs=False):
    """Original MACCS and shape metrics; coordinates are never aligned here."""
    result = {"maccs_tanimoto": None, "shape_tanimoto_in_situ": None}
    if with_mcs:
        result.update(mcs_iou=None, mcs_timed_out=None)
    if reference is None or generated is None:
        return result
    try:
        # Upstream metrics read exported SDFs (four-decimal coordinates).
        a, b = [Chem.MolFromMolBlock(Chem.MolToMolBlock(mol), removeHs=True) for mol in (reference, generated)]
        if a is None or b is None:
            raise ValueError("SDF round-trip failed")
    except Exception as error:
        return {**result, "error": str(error)}
    for metric, compute in [
        ("maccs_tanimoto", lambda: DataStructs.TanimotoSimilarity(MACCSkeys.GenMACCSKeys(a), MACCSkeys.GenMACCSKeys(b))),
        ("shape_tanimoto_in_situ", lambda: 1.0 - rdShapeHelpers.ShapeTanimotoDist(a, b)),
    ]:
        try:
            result[metric] = float(compute())
        except Exception as error:
            result[metric + "_error"] = str(error)
    if with_mcs:
        try:
            match = rdFMCS.FindMCS([a, b], timeout=2, bondCompare=rdFMCS.BondCompare.CompareAny)
            union = a.GetNumAtoms() + b.GetNumAtoms() - match.numAtoms
            result.update(mcs_timed_out=bool(match.canceled), mcs_iou_lower_bound=match.numAtoms / union if union else 0.0)
            if not match.canceled:
                result["mcs_iou"] = result["mcs_iou_lower_bound"]
        except Exception as error:
            result["mcs_error"] = str(error)
    return result


def interpolation_metrics(molecules):
    """Original frame indices, decoded endpoint preference and sign-corrected r."""
    n = len(molecules)
    values = {name: [None] * n for name in [*DESCRIPTORS, "Similarity"]}
    start, end = molecules[0], molecules[-1]
    fps = (MACCSkeys.GenMACCSKeys(start), MACCSkeys.GenMACCSKeys(end)) if start is not None and end is not None else None
    for i, mol in enumerate(molecules):
        if mol is None:
            continue
        for name, function in DESCRIPTORS.items():
            try:
                value = float(function(mol))
                values[name][i] = value if math.isfinite(value) else None
            except Exception:
                pass
        if fps:
            fp = MACCSkeys.GenMACCSKeys(mol)
            a, b = [DataStructs.TanimotoSimilarity(f, fp) for f in fps]
            values["Similarity"][i] = (b - a) / (b + a) if a + b else None
    result = {}
    for name, row in values.items():
        indices = [i for i, value in enumerate(row) if value is not None]
        metric = {"values": row, "valid_frames": len(indices), "r": None, "p": None, "neg_log10_p": None}
        if len(indices) < 3:
            metric["reason"] = "Fewer than three usable frames"
        elif name != "Similarity" and (row[0] is None or row[-1] is None):
            metric["reason"] = "Missing decoded endpoint for direction correction"
        elif np.ptp([row[i] for i in indices]) == 0:
            metric["reason"] = "Constant descriptor"
        else:
            r, p = pearsonr(indices, [row[i] for i in indices])
            if name != "Similarity":
                r *= np.sign(row[-1] - row[0])
            if math.isfinite(r) and math.isfinite(p):
                metric.update(r=float(r), p=float(p), neg_log10_p=-math.log10(p) if p > 0 else None,
                              p_underflow=bool(p == 0))
            else:
                metric["reason"] = "Undefined Pearson statistic"
        result[name] = metric
    return result


def _sdf_record(writer, mol, **tags):
    copy = Chem.Mol(mol)
    for key, value in tags.items():
        copy.SetProp(key, str(value))
    writer.write(copy)


def _raw_molecule(coordinates, numbers):
    editable = Chem.RWMol()
    conformer = Chem.Conformer(len(numbers))
    conformer.Set3D(True)
    for i, (number, point) in enumerate(zip(numbers, coordinates)):
        editable.AddAtom(Chem.Atom(number))
        conformer.SetAtomPosition(i, point.astype(float))
    mol = editable.GetMol()
    mol.AddConformer(conformer)
    return mol


def run(args):
    started = time.monotonic()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; use a new run directory")
    inputs, pairs, files = _input_specs(args)
    module = importlib.import_module(args.model + ".engine")
    molecules, input_rows = {}, []
    for row in inputs:
        try:
            mol, source = _read_molecule(row["spec"], module, args.seed)
            molecules[row["id"]] = mol
            row.update(completed=True, source=source, heavy_atoms=mol.GetNumHeavyAtoms())
        except Exception as error:
            row.update(completed=False, error=str(error))
        input_rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "inputs.json", {"molecules": input_rows, "pairs": pairs, "input_file_sha256": files})
    if not molecules:
        raise ValueError("No usable inputs; see inputs.json")
    # Seed before model construction, as the original scripts did. Restore this
    # post-construction state for each separately reproducible experiment.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    seed_everything(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    engine = engine_for(args.model, args.checkpoint, args.device, args.threads)
    engine.load(args.model)
    initial_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    manifest = run_manifest(**vars(args))
    manifest.update(source_revision=SOURCE_REVISION, protocol="original-zinc-latent",
                    checkpoint_sha256=sha256(args.checkpoint or module.ROOT / "model.ckpt"),
                    source_scripts=["latent_experiment/ex1/1_atomic_number_*", "latent_experiment/ex2/2_inter_*", "latent_experiment/ex3/3_switch_*"],
                    semantics={"alignment": "none; independent centered frames", "rng": "continuous per experiment",
                               "atom_edit_encoding": "all usable inputs together (UniMol normalization depends on this batch)",
                               "interpolation_encoding": "each endpoint separately", "swap_atom_count": "A for both AB and BA",
                               "swap_features": "complete conditioning, including all UniMol blocks",
                               "postprocessing": "original builder with require_H=True; metrics use sanitized outputs; shape uses SDF coordinate precision",
                               "wander": "extension: anchored Gaussian walk; fixed semantic blocks; not a paper benchmark"})
    write_json(output / "run.json", manifest)
    all_summaries = {}
    for experiment in dict.fromkeys(args.experiments):
        random.setstate(initial_rng[0])
        np.random.set_state(initial_rng[1])
        torch.set_rng_state(initial_rng[2])
        if initial_rng[3] is not None:
            torch.cuda.set_rng_state_all(initial_rng[3])
        directory = output / experiment
        directory.mkdir()
        # Atom edits use A endpoints for pair JSON, or all molecules for SDF input.
        chosen = [p["a"] for p in pairs] if pairs and experiment not in ("interpolation", "swap") else list(molecules)
        chosen = [identity for identity in chosen if identity in molecules]
        encoded = {}
        if chosen and experiment in ("atom-edit", "reconstruct"):
            encoded.update(zip(chosen, encode_batch(engine, args.model, [molecules[i] for i in chosen])))
        else:
            for identity in chosen:
                encoded[identity] = encode_batch(engine, args.model, [molecules[identity]])[0]
        torch.save(encoded, directory / "latents.pt")
        references, reference_errors = {}, {}
        with Chem.SDWriter(str(directory / "references.sdf")) as writer:
            writer.SetKekulize(False)
            for identity in chosen:
                latent = encoded[identity]
                try:
                    reference = _perceive(engine.builder, latent["coordinates"].numpy(), [a.GetAtomicNum() for a in molecules[identity].GetAtoms()])
                    references[identity] = reference
                    _sdf_record(writer, reference, input_id=identity)
                except Exception as error:
                    references[identity] = None
                    reference_errors[identity] = str(error)
        write_json(directory / "reference_errors.json", reference_errors)
        batches, planned = [], []

        def frame(identity, index, latent=None, count=None, **metadata):
            row = {"id": identity, "frame": index, "atom_count": count, **metadata, "completed": False}
            if latent is None:
                row["error"] = "Input molecule or pair could not be read"
            row["latent"] = latent
            planned.append(row)
            return row

        def chunk(rows, size):
            usable = [row for row in rows if row["latent"] is not None]
            batches.extend(usable[i:i + size] for i in range(0, len(usable), size))

        ids = [p["a"] for p in pairs] if pairs else [r["id"] for r in inputs]
        if experiment in ("atom-edit", "reconstruct"):
            for delta in range(-args.radius, args.radius + 1) if experiment == "atom-edit" else [0]:
                rows = [frame(identity, delta, encoded.get(identity), max(molecules[identity].GetNumAtoms() + delta, 1) if identity in encoded else None,
                              reference_id=identity, delta=delta) for identity in ids]
                chunk(rows, args.batch_size)
        elif experiment == "interpolation":
            for offset in range(0, len(pairs), args.group_size):
                rows = []
                for pair in pairs[offset:offset + args.group_size]:
                    a, b = encoded.get(pair["a"]), encoded.get(pair["b"])
                    for i, alpha in enumerate(torch.linspace(0, 1, args.frames)):
                        latent, count = None, None
                        if a is not None and b is not None:
                            latent = {k: a[k] + alpha * (b[k] - a[k]) for k in ("zh", "zx", "condition")}
                            count = max(round(molecules[pair["a"]].GetNumAtoms() + float(alpha) *
                                              (molecules[pair["b"]].GetNumAtoms() - molecules[pair["a"]].GetNumAtoms())), 1)
                        rows.append(frame(pair["id"], i, latent, count, alpha=float(alpha), a=pair["a"], b=pair["b"]))
                chunk(rows, max(len(rows), 1))
        elif experiment == "swap":
            # Upstream reuses its two-frame interpolation function to obtain
            # endpoints; preserve even the float32 arithmetic at alpha=1.
            endpoints = {}
            for pair in pairs:
                a, b = encoded.get(pair["a"]), encoded.get(pair["b"])
                if a is not None and b is not None:
                    endpoints[pair["id"]] = {
                        "a": {k: a[k] + 0.0 * (b[k] - a[k]) for k in ("zh", "zx", "condition")},
                        "b": {k: a[k] + 1.0 * (b[k] - a[k]) for k in ("zh", "zx", "condition")},
                    }
            for geometry, features in (("a", "b"), ("b", "a")):
                rows = []
                for pair in pairs:
                    a = endpoints.get(pair["id"], {}).get(geometry)
                    b = endpoints.get(pair["id"], {}).get(features)
                    latent = {"zx": a["zx"], "zh": b["zh"], "condition": b["condition"]} if a is not None and b is not None else None
                    count = molecules[pair["a"]].GetNumAtoms() if latent is not None else None
                    rows.append(frame(pair["id"], "AB" if geometry == "a" else "BA", latent, count,
                                      geometry_id=pair[geometry], feature_id=pair[features], reference_id=pair["a"]))
                chunk(rows, args.swap_batch_size or max(len(rows), 1))
        else:
            rng = torch.Generator(device="cpu").manual_seed(args.seed)
            for identity in ids:
                original = encoded.get(identity)
                rows = []
                if original is not None:
                    walk_x, walk_h = torch.zeros_like(original["zx"]), torch.zeros_like(original["zh"])
                    x_scale = (original["zx"] - original["zx"].mean(0)).square().mean().sqrt().clamp_min(1e-3)
                    h_scale = original["zh"].square().mean().sqrt().clamp_min(1e-3)
                for i in range(args.frames):
                    latent = None
                    if original is not None:
                        if i and args.space in ("zx", "both"):
                            noise = torch.randn(walk_x.shape, generator=rng)
                            walk_x += noise - noise.mean(0)
                        if i and args.space in ("zh", "both"):
                            walk_h += torch.randn(walk_h.shape, generator=rng)
                        latent = dict(original, zx=original["zx"] + args.step_size * x_scale * walk_x,
                                      zh=original["zh"] + args.step_size * h_scale * walk_h)
                        if i and args.step_size and args.space in ("zh", "both"):
                            latent["condition"] = engine._replace_structural({**original, "key": args.model}, latent["zh"])
                    rows.append(frame(identity, i, latent, molecules[identity].GetNumAtoms() if original is not None else None, reference_id=identity))
                chunk(rows, args.batch_size)
        generated = {}
        with Chem.SDWriter(str(directory / "predictions.sdf")) as writer:
            writer.SetKekulize(False)
            for batch in batches:
                try:
                    predictions = decode_batch(engine, args.model, batch, args.steps)
                except Exception as error:
                    for row in batch:
                        row["error"] = str(error)
                    continue
                for row, (coordinates, numbers) in zip(batch, predictions):
                    if not np.isfinite(coordinates).all():
                        row["error"] = "Non-finite decoder output"
                        continue
                    row.update(completed=True, coordinates=coordinates.tolist(), atomic_numbers=numbers)
                    try:
                        mol = _perceive(engine.builder, coordinates, numbers)
                        row.update(rdkit_valid=True, fragments=len(Chem.GetMolFrags(mol)), smiles=Chem.MolToSmiles(Chem.RemoveHs(mol)))
                        generated[(row["id"], row["frame"])] = mol
                    except Exception as error:
                        row.update(rdkit_valid=False, postprocessing_error=str(error))
                        mol = _raw_molecule(coordinates, numbers)
                    _sdf_record(writer, mol, input_id=row["id"], frame=row["frame"], experiment=experiment, rdkit_valid=row["rdkit_valid"])
        for row in planned:
            row.pop("latent")
            if experiment != "interpolation":
                row["metrics"] = similarity_metrics(references.get(row.get("reference_id")), generated.get((row["id"], row["frame"])), experiment == "atom-edit")
        trajectory_metrics = {}
        if experiment == "interpolation":
            trajectory_metrics = {pair["id"]: interpolation_metrics([generated.get((pair["id"], i)) for i in range(args.frames)]) for pair in pairs}
            write_json(directory / "trajectories.json", trajectory_metrics)
        summary = {"attempted": len(planned), "completed": sum(r["completed"] for r in planned),
                   "valid": sum(r.get("rdkit_valid", False) for r in planned),
                   "valid_connected": sum(r.get("rdkit_valid", False) and r.get("fragments") == 1 for r in planned),
                   "reference_failures": sum(m is None for m in references.values()),
                   "decode_batch_sizes": [len(b) for b in batches]}
        summary["valid_rate"] = summary["valid"] / len(planned) if planned else None
        summary["valid_connected_rate"] = summary["valid_connected"] / len(planned) if planned else None
        if trajectory_metrics:
            summary["metrics"] = {name: {field: _stats([m[name][field] for m in trajectory_metrics.values()])
                                         for field in ("r", "neg_log10_p")} for name in [*DESCRIPTORS, "Similarity"]}
            for name in summary["metrics"]:
                summary["metrics"][name]["p_underflow_count"] = sum(m[name].get("p_underflow", False) for m in trajectory_metrics.values())
        else:
            labels = list(dict.fromkeys(r["frame"] for r in planned))
            summary["metrics_by_frame"] = {str(label): {name: _stats([r.get("metrics", {}).get(name) for r in planned if r["frame"] == label])
                                          for name in ["maccs_tanimoto", "shape_tanimoto_in_situ", *(["mcs_iou"] if experiment == "atom-edit" else [])]}
                                          for label in labels}
            if experiment == "atom-edit":
                summary["mcs_timeout_count"] = sum(r["metrics"].get("mcs_timed_out") is True for r in planned)
        summary["failed_records"] = summary["attempted"] - summary["completed"]
        (directory / "records.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in planned))
        write_json(directory / "summary.json", summary)
        all_summaries[experiment] = summary
        print(f"{experiment}: {summary['completed']}/{summary['attempted']} decoded", flush=True)
    report = {"model": args.model, "steps": args.steps, "seed": args.seed, "experiments": all_summaries,
              "failed_records": sum(s["failed_records"] for s in all_summaries.values()),
              "runtime_seconds": time.monotonic() - started,
              "metric_policy": "Similarity means include sanitized pairs (including disconnected outputs), as upstream; validity uses all planned frames. Each metric has its own count. Missing frames retain their indices. Timed-out MCS values are lower bounds only. p=0 is recorded as underflow, not clipped."}
    write_json(output / "summary.json", report)
    return report
