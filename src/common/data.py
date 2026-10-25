"""Validated multi-conformer records, identity-disjoint splits and LMDB support."""

import hashlib
import json
import pickle
import random
import os
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset

SUPPORTED = {6, 7, 8, 9, 15, 16, 17, 35, 53}
_LMDB_HANDLES = {}


def _lmdb_environment(path):
    import lmdb

    for inherited_key, handle in list(_LMDB_HANDLES.items()):
        if inherited_key[0] != os.getpid():
            handle.close()
            del _LMDB_HANDLES[inherited_key]
    key = (os.getpid(), str(Path(path).resolve()))
    if key not in _LMDB_HANDLES:
        _LMDB_HANDLES[key] = lmdb.open(key[1], subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    return _LMDB_HANDLES[key]


def identity(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES: " + smiles)
    return Chem.MolToSmiles(Chem.RemoveHs(molecule), canonical=True, isomericSmiles=True)


def validate_record(record):
    atoms = np.asarray(record["atomic_numbers"])
    coordinates = np.asarray(record["conformers"], dtype=np.float64)
    if not 2 <= len(atoms) <= 80 or any(int(a) not in SUPPORTED for a in atoms):
        raise ValueError("Unsupported atom count or element: " + record["id"])
    if coordinates.ndim != 3 or coordinates.shape[1:] != (len(atoms), 3) or not len(coordinates) or not np.isfinite(coordinates).all():
        raise ValueError("Invalid conformer array: " + record["id"])
    if "smiles" in record:
        molecule = Chem.RemoveHs(Chem.MolFromSmiles(record["smiles"]))
        if molecule.GetNumAtoms() != len(atoms):
            raise ValueError("SMILES and atom array disagree: " + record["id"])
    return record


def make_record(smiles, name, conformers, seed):
    molecule = Chem.AddHs(Chem.MolFromSmiles(identity(smiles)))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    params.pruneRmsThresh = -1
    ids = list(AllChem.EmbedMultipleConfs(molecule, numConfs=conformers, params=params))
    if len(ids) != conformers:
        raise ValueError("Could not embed all conformers: " + name)
    if AllChem.UFFHasAllMoleculeParams(molecule):
        for conf in ids:
            AllChem.UFFOptimizeMolecule(molecule, confId=conf, maxIters=200)
    molecule = Chem.RemoveHs(molecule)
    return validate_record(
        {
            "id": name,
            "smiles": Chem.MolToSmiles(molecule),
            "atomic_numbers": [a.GetAtomicNum() for a in molecule.GetAtoms()],
            "conformers": [np.asarray(molecule.GetConformer(c).GetPositions()).tolist() for c in ids],
            "source": {"type": "ETKDGv3-UFF", "seed": seed},
        }
    )


def prepare(input_path, output, conformers=2, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    rejected = []
    seen = set()
    for i, line in enumerate(Path(input_path).read_text().splitlines()):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        tokens = line.split(maxsplit=1)
        name = tokens[1] if len(tokens) > 1 else f"molecule-{i:06d}"
        try:
            record = make_record(tokens[0], name, conformers, seed + i)
            canonical = identity(record["smiles"])
            if canonical in seen:
                raise ValueError("Duplicate molecular identity")
            seen.add(canonical)
            records.append(record)
        except Exception as error:
            rejected.append({"line": i + 1, "id": name, "error": str(error)})
    if len(records) < 5:
        raise ValueError("At least five valid distinct molecules are required for train/val/test splits.")
    ids = list(range(len(records)))
    random.Random(seed).shuffle(ids)
    n_val = max(1, round(len(ids) * 0.1))
    n_test = max(1, round(len(ids) * 0.1))
    split = {
        "train": [records[i]["id"] for i in ids[n_val + n_test :]],
        "val": [records[i]["id"] for i in ids[:n_val]],
        "test": [records[i]["id"] for i in ids[n_val : n_val + n_test]],
    }
    (output / "molecules.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (output / "splits.json").write_text(json.dumps(split, indent=2) + "\n")
    (output / "preparation.json").write_text(
        json.dumps(
            {
                "source": str(input_path),
                "seed": seed,
                "conformers": conformers,
                "accepted": len(records),
                "rejected": rejected,
                "split_protocol": "seeded random by canonical molecular identity; fixture split, not the published ZINC split",
            },
            indent=2,
        )
        + "\n"
    )
    validate_splits(records, split)
    return {"records": len(records), "rejected": len(rejected), "split_sizes": {k: len(v) for k, v in split.items()}}


def read_records(path):
    records = [validate_record(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate record IDs")
    return records


def validate_splits(records, splits):
    lookup = {r["id"]: r for r in records}
    identities = {}
    for split in ("train", "val", "test"):
        ids = splits[split]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Empty or duplicate IDs in " + split)
        if not set(ids) <= set(lookup):
            raise ValueError("Unknown IDs in " + split)
        identities[split] = {identity(lookup[i]["smiles"]) if "smiles" in lookup[i] else i for i in ids}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        if identities[a] & identities[b]:
            raise ValueError("Molecular identity leakage between " + a + " and " + b)
    return True


def as_molecule(record, conf_index=0):
    coordinates = np.array(record["conformers"][conf_index], dtype=float)
    if record.get("molblock"):
        mol = Chem.MolFromMolBlock(record["molblock"], removeHs=True, sanitize=True)
        if mol is None:
            raise ValueError("Invalid reference molblock in " + record["id"])
    elif "smiles" in record:
        mol = Chem.RemoveHs(Chem.MolFromSmiles(record["smiles"]))
    else:
        rw = Chem.RWMol()
        for number in record["atomic_numbers"]:
            rw.AddAtom(Chem.Atom(int(number)))
        mol = rw.GetMol()
    if [a.GetAtomicNum() for a in mol.GetAtoms()] != record["atomic_numbers"]:
        raise ValueError("Atom ordering mismatch in " + record["id"])
    conformer = Chem.Conformer(len(coordinates))
    conformer.Set3D(True)
    for i, point in enumerate(coordinates):
        conformer.SetAtomPosition(i, point)
    mol.RemoveAllConformers()
    mol.AddConformer(conformer)
    return mol


class ConformerDataset(Dataset):
    def __init__(self, data, splits, split, paired=True):
        self.paired = paired
        path = Path(data)
        self.lmdb_path = str(path) if path.suffix == ".lmdb" else None
        if self.lmdb_path:
            index = Path(splits) / (split + "_indices.pt")
            self.indices = torch.load(index, map_location="cpu", weights_only=True)
            env = _lmdb_environment(path)
            with env.begin() as txn:
                self.keys = list(txn.cursor().iternext(values=False))
            if any(int(i) < 0 or int(i) >= len(self.keys) for i in self.indices):
                raise ValueError("Split index exceeds LMDB key count")
        else:
            records = read_records(path)
            split_data = json.loads(Path(splits).read_text())
            validate_splits(records, split_data)
            lookup = {r["id"]: r for r in records}
            self.records = [lookup[i] for i in split_data[split]]

    def __len__(self):
        return len(self.indices) if self.lmdb_path else len(self.records)

    def __getitem__(self, index):
        if self.lmdb_path:
            key = self.keys[int(self.indices[index])]
            with _lmdb_environment(self.lmdb_path).begin() as txn:
                confs = pickle.loads(txn.get(key))
            tensors = []
            for conf in confs:
                numbers = np.array([Chem.GetPeriodicTable().GetAtomicNumber(s) for s in conf["atoms"]])
                mask = numbers != 1
                coords = np.asarray(conf["coordinates"], dtype=np.float32).reshape(-1, 3)[mask]
                numbers = numbers[mask]
                if not 2 <= len(numbers) <= 80 or any(int(n) not in SUPPORTED for n in numbers):
                    raise ValueError("Invalid LMDB molecule " + str(key))
                if not np.isfinite(coords).all():
                    raise ValueError("Nonfinite LMDB coordinates")
                tensors.append({"h": torch.tensor(numbers, dtype=torch.long), "x": torch.tensor(coords)})
        else:
            record = self.records[index]
            tensors = [
                {"h": torch.tensor(record["atomic_numbers"], dtype=torch.long), "x": torch.tensor(c, dtype=torch.float32)}
                for c in record["conformers"]
            ]
        if self.paired:
            if len(tensors) < 2:
                raise ValueError("Paired training requires two real conformers; zero padding is not allowed.")
            return random.sample(tensors, 2)
        return random.choice(tensors)


def collate_pairs(batch):
    return batch


def collate_single(batch):
    return {
        "h": torch.cat([b["h"] for b in batch]),
        "x": torch.cat([b["x"] for b in batch]),
        "batch": torch.cat([torch.full((len(b["h"]),), i, dtype=torch.long) for i, b in enumerate(batch)]),
    }
