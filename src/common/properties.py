"""Frozen encoder property probe; a separate protocol from joint fine-tuning."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

from .utils import engine_for, seed_everything, write_json
from .data import identity


def run(args):
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    rows = list(csv.DictReader(Path(args.csv).open()))
    splits = {name: [r for r in rows if r["split"] == name] for name in ("train", "val", "test")}
    if any(not group for group in splits.values()):
        raise ValueError("CSV must have nonempty train, val and test splits.")
    identities = {name: {identity(r["smiles"]) for r in group} for name, group in splits.items()}
    if any(identities[a] & identities[b] for a, b in [("train", "val"), ("train", "test"), ("val", "test")]):
        raise ValueError("Molecular identity leakage in property split.")
    engine = engine_for(args.model, args.checkpoint, args.device, args.threads)
    features = {}
    targets = {}
    for name, group in splits.items():
        vectors = []
        labels = []
        for row in group:
            label = float(row[args.target])
            if not np.isfinite(label):
                raise ValueError("This single-target probe requires finite labels.")
            encoded = engine.encode(key=args.model, smiles=row["smiles"], seed=args.seed)
            vectors.append(np.asarray(encoded["latent"]["decoder_features"]).ravel())
            labels.append(label)
        features[name] = torch.tensor(np.array(vectors), dtype=torch.float32)
        targets[name] = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    mean = features["train"].mean(0)
    scale = features["train"].std(0, unbiased=False).clamp_min(1e-6)
    features = {name: (x - mean) / scale for name, x in features.items()}
    head = torch.nn.Sequential(torch.nn.Linear(len(mean), 512), torch.nn.ReLU(), torch.nn.Dropout(args.dropout), torch.nn.Linear(512, 1))
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss() if args.task == "classification" else torch.nn.MSELoss()
    if args.task == "classification" and any(not set(y.flatten().tolist()) <= {0.0, 1.0} for y in targets.values()):
        raise ValueError("Classification labels must be 0 or 1.")
    best = None
    best_val = float("inf")
    history = []
    for epoch in range(args.epochs):
        head.train()
        optimizer.zero_grad()
        loss = criterion(head(features["train"]), targets["train"])
        loss.backward()
        optimizer.step()
        head.eval()
        with torch.no_grad():
            val = float(criterion(head(features["val"]), targets["val"]))
        history.append({"epoch": epoch, "train_loss": float(loss.detach()), "val_loss": val})
        if val < best_val:
            best_val = val
            best = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best)
    head.eval()
    with torch.no_grad():
        pred = head(features["test"]).numpy().ravel()
    truth = targets["test"].numpy().ravel()
    metrics = (
        {"auc": float(roc_auc_score(truth, pred)) if len(set(truth)) == 2 else None}
        if args.task == "classification"
        else {"rmse": float(np.sqrt(mean_squared_error(truth, pred))), "mae": float(mean_absolute_error(truth, pred))}
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": best, "feature_mean": mean, "feature_scale": scale, "model": args.model, "target": args.target, "seed": args.seed},
        output / "probe.pt",
    )
    write_json(
        output / "metrics.json",
        {
            "model": args.model,
            "protocol": "frozen complete-decoder-feature MLP probe; not end-to-end paper fine-tuning",
            "target": args.target,
            "task": args.task,
            "seed": args.seed,
            "split_sizes": {k: len(v) for k, v in splits.items()},
            "metrics": metrics,
            "history": history,
        },
    )
    return metrics
