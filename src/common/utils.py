"""Shared configuration, model construction and run provenance."""

import copy
import hashlib
import importlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from .configuration import model_config

MODELS = ("molalign3d", "molflae")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(model, config=None, checkpoint=None, output_dir="outputs/model"):
    cfg = model_config(model, config)
    cfg["evaluation"]["save_dir"] = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if model == "molalign3d":
        engine_module = importlib.import_module(model + ".engine")
        weight_dir = engine_module.ROOT
        os.environ["UNIMOL_WEIGHT_DIR"] = str(weight_dir)
        cfg["unimol"]["ckpt_path"] = str(weight_dir / "mol_pre_no_h_220816.pt")
    cls = importlib.import_module(model + ".models.autoencoder").TrainLoop
    instance = cls(cfg)
    if checkpoint:
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        instance.load_state_dict(data.get("state_dict", data), strict=True)
    return instance, cfg


def move_decoder_constants(model, device):
    model.decoder.device = torch.device(device)
    for name in ("sigma1_coord", "sigma1_v", "beta1"):
        value = getattr(model.decoder, name, None)
        if isinstance(value, torch.Tensor):
            setattr(model.decoder, name, value.to(device))


def engine_for(model, checkpoint=None, device="cpu", threads=4):
    os.environ["MOLECULAR_DEVICE"] = device
    os.environ["MOLECULAR_CPU_THREADS"] = str(threads)
    module = importlib.import_module(model + ".engine")
    engine = module.Engine()
    if checkpoint:
        instance, _ = build_model(model, checkpoint=checkpoint, output_dir=module.RUNTIME / model)
        instance.to(engine.device).eval()
        move_decoder_constants(instance, engine.device)
        engine.models[model] = instance
        provenance = engine.provenance(model)
        provenance.update(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha256(checkpoint), checkpoint_override=True)
        engine.provenance = lambda _key: copy.deepcopy(provenance)
    return engine


def run_manifest(**settings):
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        revision = None
    return {
        "settings": settings,
        "software": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "git_revision": revision,
    }


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
