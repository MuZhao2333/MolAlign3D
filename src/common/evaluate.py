"""Reconstruction evaluation with explicit denominators and retained failures."""

import json
from pathlib import Path
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, QED

from .utils import engine_for, run_manifest, sha256, write_json
from .data import as_molecule, read_records, validate_splits


def summarize(rows):
    n = len(rows)
    complete = [r for r in rows if r.get("completed")]
    valid = [r for r in complete if r.get("rdkit_valid") and r.get("fragments") == 1]
    exact = [r for r in valid if r.get("graph_match")]
    rmsds = [r["aligned_rmsd"] for r in complete if r.get("aligned_rmsd") is not None]
    similarity = [r["fingerprint_tanimoto"] for r in complete if r.get("fingerprint_tanimoto") is not None]
    return {
        "attempted": n,
        "completed": len(complete),
        "valid_connected": len(valid),
        "graph_matches": len(exact),
        "rmsd_matched_count": len(rmsds),
        "completion_rate": len(complete) / n if n else None,
        "valid_connected_rate": len(valid) / n if n else None,
        "exact_graph_rate": len(exact) / n if n else None,
        "mean_matched_rmsd": float(np.mean(rmsds)) if rmsds else None,
        "median_matched_rmsd": float(np.median(rmsds)) if rmsds else None,
        "mean_fingerprint_tanimoto_on_valid_pairs": float(np.mean(similarity)) if similarity else None,
        "mean_runtime_seconds_on_complete": float(np.mean([r["runtime_seconds"] for r in complete])) if complete else None,
        "denominators": {
            "completion_validity_exact_graph": "all attempted conformers, including errors",
            "rmsd": "graph-matched outputs only",
            "fingerprint_tanimoto": "sanitized connected outputs only",
        },
    }


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = read_records(args.data)
    splits = json.loads(Path(args.splits).read_text())
    validate_splits(records, splits)
    lookup = {r["id"]: r for r in records}
    chosen = [lookup[i] for i in splits[args.split]]
    if args.limit:
        chosen = chosen[: args.limit]
    engine = engine_for(args.model, args.checkpoint, args.device, args.threads)
    rows = []
    sdf = []
    for record in chosen:
        for conf in range(min(len(record["conformers"]), args.conformers)):
            row = {"id": record["id"], "conformer": conf, "input_smiles": record["smiles"], "completed": False}
            start = time.monotonic()
            try:
                mol = as_molecule(record, conf)
                result = engine.reconstruct(key=args.model, molblock=Chem.MolToMolBlock(mol), seed=args.seed, steps=args.steps)
                pred = result["decoded"]
                connected = pred["rdkit_valid"] and pred["fragments"] == 1
                generated = Chem.MolFromMolBlock(pred["molblock"], sanitize=True, removeHs=True) if connected else None
                original = Chem.MolFromSmiles(record["smiles"])
                graph_match = generated is not None and Chem.MolToSmiles(generated, isomericSmiles=True) == Chem.MolToSmiles(
                    original, isomericSmiles=True
                )
                tanimoto = (
                    DataStructs.TanimotoSimilarity(Chem.RDKFingerprint(original), Chem.RDKFingerprint(generated))
                    if generated is not None
                    else None
                )
                row.update(
                    completed=True,
                    rdkit_valid=pred["rdkit_valid"],
                    fragments=pred["fragments"],
                    generated_smiles=pred["smiles"],
                    graph_match=graph_match,
                    aligned_rmsd=pred.get("aligned_rmsd"),
                    fingerprint_tanimoto=tanimoto,
                    runtime_seconds=round(time.monotonic() - start, 5),
                    qed=float(QED.qed(generated)) if generated is not None else None,
                )
                write_json(output / "predictions" / (record["id"].replace("/", "_") + f"-{conf}.json"), result)
                sdf.append(pred["molblock"] + "\n> <record_id>\n" + record["id"] + "\n\n$$$$\n")
            except Exception as error:
                row["error"] = str(error)
            rows.append(row)
    (output / "predictions.sdf").write_text("".join(sdf))
    (output / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = {
        "model": args.model,
        "steps": args.steps,
        "seed": args.seed,
        "split": args.split,
        "dataset_sha256": sha256(args.data),
        "splits_sha256": sha256(args.splits),
        "metrics": summarize(rows),
        "provenance": engine.provenance(args.model),
        "run": run_manifest(**vars(args)),
    }
    if args.checkpoint:
        report["checkpoint_override_sha256"] = sha256(args.checkpoint)
    write_json(output / "metrics.json", report)
    return report
