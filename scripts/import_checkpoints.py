"""Copy matching checkpoint assets into a checkout with SHA256 validation."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--molflae")
parser.add_argument("--molalign3d")
parser.add_argument("--unimol")
parser.add_argument("--dictionary")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
registry = {model: json.loads((root / "src" / model / "configs/checkpoint.json").read_text()) for model in ["molalign3d", "molflae"]}
for supplied, model, name in [
    (args.molflae, "molflae", "model.ckpt"),
    (args.molalign3d, "molalign3d", "model.ckpt"),
    (args.unimol, "molalign3d", "mol_pre_no_h_220816.pt"),
    (args.dictionary, "molalign3d", "mol.dict.txt"),
]:
    if supplied is None:
        continue
    path = Path(supplied)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != registry[model]["files"][name]["sha256"]:
        raise ValueError("Wrong checkpoint asset: " + str(path))
    target = root / "weights" / model / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.resolve() != target.resolve():
        shutil.copy2(path, target)
    print("Imported", target)
