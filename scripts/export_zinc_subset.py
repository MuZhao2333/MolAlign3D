"""Export indexed original ZINC records without regenerating conformers."""

import argparse
import json
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
from rdkit import Chem

from common.utils import write_json
from common.data import validate_splits
from molflae.utils.build_mol import MoleculeBuilder

parser = argparse.ArgumentParser()
parser.add_argument("--lmdb", required=True)
parser.add_argument("--indices-dir", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--per-split", type=int, default=100)
args = parser.parse_args()
if args.per_split < 1:
    parser.error("--per-split must be positive")
env = lmdb.open(args.lmdb, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
with env.begin() as txn:
    keys = list(txn.cursor().iternext(values=False))
records = []
splits = {}
rejections = []
builder = MoleculeBuilder()
for split in ["train", "val", "test"]:
    indices = torch.load(Path(args.indices_dir) / (split + "_indices.pt"), map_location="cpu", weights_only=True)
    splits[split] = []
    for index in indices[: args.per_split]:
        key = keys[int(index)]
        try:
            with env.begin() as txn:
                raw = pickle.loads(txn.get(key))
            numbers = np.array([Chem.GetPeriodicTable().GetAtomicNumber(s) for s in raw[0]["atoms"]])
            keep = numbers != 1
            numbers = numbers[keep]
            confs = [np.asarray(c["coordinates"], dtype=float).reshape(-1, 3)[keep].tolist() for c in raw]
            reference = builder.build_mol(np.asarray(confs[0]), numbers.tolist(), require_H=False)
            Chem.SanitizeMol(reference)
            if [a.GetAtomicNum() for a in reference.GetAtoms()] != numbers.tolist():
                raise ValueError("Reference postprocessor changed atom ordering")
            name = key.hex()
            record = {
                "id": name,
                "smiles": Chem.MolToSmiles(reference),
                "molblock": Chem.MolToMolBlock(reference),
                "atomic_numbers": numbers.tolist(),
                "conformers": confs,
                "source": {
                    "type": "original-ZINC-LMDB",
                    "key_hex": key.hex(),
                    "original_index": int(index),
                    "split": split,
                    "reference_bonds": "inferred by original molecule postprocessor",
                },
            }
            records.append(record)
            splits[split].append(name)
        except Exception as error:
            rejections.append({"index": int(index), "split": split, "key_hex": key.hex(), "error": str(error)})
validate_splits(records, splits)
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
(output / "molecules.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
write_json(output / "splits.json", splits)
write_json(
    output / "export.json",
    {
        "source": "original ZINC indexed LMDB",
        "requested_per_split": args.per_split,
        "accepted": len(records),
        "rejections": rejections,
        "bond_reference": "inferred, not supplied ground-truth bonds; do not conflate with graph-labeled paper metrics",
    },
)
print(json.dumps({"accepted": len(records), "rejected": len(rejections), "split_sizes": {k: len(v) for k, v in splits.items()}}))
