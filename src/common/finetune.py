"""End-to-end single-target property fine-tuning with disjoint molecular splits."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

from .utils import engine_for, run_manifest, seed_everything, write_json
from .data import identity


def run(args):
    seed_everything(args.seed)
    engine = engine_for(args.model, args.checkpoint, args.device, args.threads)
    backbone = engine.load(args.model)
    for module in [backbone.decoder, getattr(backbone, "atom_num_predictor", None), getattr(backbone, "token_classifier", None)]:
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = False
    for name in ["Wh_log_var", "Wx_log_var"]:
        for parameter in getattr(backbone, name).parameters():
            parameter.requires_grad = False
    if hasattr(backbone, "logit_scale"):
        backbone.logit_scale.requires_grad = False
    head = torch.nn.Sequential(
        torch.nn.Linear(engine.provenance(args.model)["global_nodes"] * engine.provenance(args.model)["decoder_dim"], 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout),
        torch.nn.Linear(512, 1),
    ).to(engine.device)
    records = list(csv.DictReader(Path(args.csv).open()))
    groups = {name: [r for r in records if r["split"] == name] for name in ("train", "val", "test")}
    if any(not group for group in groups.values()):
        raise ValueError("CSV needs train, val and test rows.")
    identities = {name: {identity(r["smiles"]) for r in group} for name, group in groups.items()}
    if any(identities[a] & identities[b] for a, b in [("train", "val"), ("train", "test"), ("val", "test")]):
        raise ValueError("Molecular identity leakage")
    tensors = {}
    prepare = __import__(args.model + ".engine", fromlist=["prepare_molecule"]).prepare_molecule
    mapper = __import__(args.model + ".engine", fromlist=["ATOM_TO_INDEX"]).ATOM_TO_INDEX
    for name, group in groups.items():
        tensors[name] = []
        for row in group:
            mol, coords, _ = prepare(smiles=row["smiles"], seed=args.seed)
            atoms = torch.tensor([mapper[a.GetAtomicNum()] for a in mol.GetAtoms()], dtype=torch.long, device=engine.device)
            target = float(row[args.target])
            if not np.isfinite(target):
                raise ValueError("This single-target protocol requires finite labels.")
            if args.task == "classification" and target not in (0, 1):
                raise ValueError("Binary targets must be 0 or 1.")
            tensors[name].append((atoms, torch.tensor(coords, dtype=torch.float32, device=engine.device), target))
    train_targets = np.array([t[2] for t in tensors["train"]])
    target_mean = float(train_targets.mean()) if args.task == "regression" else 0.0
    target_scale = max(float(train_targets.std()), 1e-6) if args.task == "regression" else 1.0
    criterion = torch.nn.BCEWithLogitsLoss() if args.task == "classification" else torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        [p for p in backbone.parameters() if p.requires_grad] + list(head.parameters()), lr=args.lr, weight_decay=args.weight_decay
    )

    def prediction(batch):
        atoms = torch.cat([b[0] for b in batch])
        x = torch.cat([b[1] for b in batch])
        graph = torch.cat([torch.full((len(b[0]),), i, dtype=torch.long, device=engine.device) for i, b in enumerate(batch)])
        zh, zx, gb, *_ = backbone.encode(F.one_hot(atoms, 9).float(), x, graph, deterministic=True)
        condition = backbone._attach_unimol_to_Zh(zh, gb, atoms, x, graph) if args.model == "molalign3d" else zh
        return head(condition.reshape(len(batch), -1))

    def labels(batch):
        return torch.tensor([(b[2] - target_mean) / target_scale for b in batch], device=engine.device, dtype=torch.float32).view(-1, 1)

    history = []
    best = None
    best_val = float("inf")
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        backbone.train()
        head.train()
        if getattr(backbone, "unimol_model", None) is not None:
            backbone.unimol_model.eval()
        order = rng.permutation(len(tensors["train"]))
        train_loss = []
        for start in range(0, len(order), args.batch_size):
            batch = [tensors["train"][i] for i in order[start : start + args.batch_size]]
            optimizer.zero_grad()
            loss = criterion(prediction(batch), labels(batch))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite supervised loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in backbone.parameters() if p.requires_grad] + list(head.parameters()), 10.0, error_if_nonfinite=True
            )
            optimizer.step()
            train_loss.append(float(loss.detach()))
        backbone.eval()
        head.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, len(tensors["val"]), args.batch_size):
                batch = tensors["val"][start : start + args.batch_size]
                val_losses.append((float(criterion(prediction(batch), labels(batch))), len(batch)))
        val_loss = sum(v * n for v, n in val_losses) / sum(n for _, n in val_losses)
        history.append({"epoch": epoch, "mean_train_batch_loss": float(np.mean(train_loss)), "validation_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best = {
                "backbone": {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()},
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }
    backbone.load_state_dict(best["backbone"], strict=True)
    head.load_state_dict(best["head"], strict=True)
    backbone.eval()
    head.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(tensors["test"]), args.batch_size):
            predictions.extend(prediction(tensors["test"][start : start + args.batch_size]).flatten().cpu().tolist())
    actual = np.array([r[2] for r in tensors["test"]])
    pred = np.array(predictions) * target_scale + target_mean
    metrics = (
        {"auc": float(roc_auc_score(actual, pred)) if len(set(actual)) == 2 else None}
        if args.task == "classification"
        else {"rmse": float(np.sqrt(mean_squared_error(actual, pred))), "mae": float(mean_absolute_error(actual, pred))}
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **best,
            "model": args.model,
            "target": args.target,
            "task": args.task,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "seed": args.seed,
        },
        output / "best-property-model.pt",
    )
    write_json(
        output / "metrics.json",
        {
            "model": args.model,
            "protocol": "end-to-end encoder + 512-unit MLP; frozen UniMol; original decoder-feature flattening; single-target CSV protocol",
            "claim": "portable fine-tuning implementation, not a reproduced MoleculeNet paper score",
            "task": args.task,
            "target": args.target,
            "metrics": metrics,
            "history": history,
            "split_sizes": {k: len(v) for k, v in groups.items()},
            "run": run_manifest(**vars(args)),
        },
    )
    return metrics
