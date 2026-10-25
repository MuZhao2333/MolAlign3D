"""Source-traced model presets and explicit training overrides.

The registry preserves the original config values, checkpoint evidence and
paper table values separately. Conflicting evidence is never silently merged.
"""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent / "configs"
PROFILES = ("checkpoint", "original", "paper")


def registry():
    return json.loads((CONFIG_DIR / "hyperparameters.json").read_text())


def checkpoint_metadata(model):
    package = importlib.import_module(model)
    return json.loads((Path(package.__file__).parent / "configs/checkpoint.json").read_text())


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def set_value(config, path, value):
    parts = path.split(".")
    target = config
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = copy.deepcopy(value)


def model_config(model, config=None, profile=None):
    if config is not None and profile is not None:
        raise ValueError("Use either a custom config or a named profile.")
    package = importlib.import_module(model)
    path = Path(config) if config else Path(package.__file__).parent / "configs/model.yaml"
    resolved = copy.deepcopy(yaml.safe_load(path.read_text()))
    if config is not None:
        return resolved
    selected = profile or "checkpoint"
    if selected not in PROFILES:
        raise ValueError("Unknown configuration profile: " + selected)
    for field, value in registry()["models"][model]["profiles"][selected].items():
        set_value(resolved, field, value)
    return resolved


def resolve_training(args, cuda_available=False):
    """Resolve once; the trainer and both dataloaders consume this same result."""
    selected = "custom" if args.config else (args.profile or "checkpoint")
    cfg = model_config(args.model, args.config, args.profile)
    overrides = {}
    fields = {
        "epochs": "train.max_epochs", "batch_size": "train.batch_size",
        "workers": "train.num_workers", "lr": "train.optimizer.lr",
        "validation_batch_size": "evaluation.batch_size",
        "validation_workers": "evaluation.num_workers",
        "validation_samples": "evaluation.num_samples",
        "validation_decode_steps": "evaluation.sample_steps",
    }
    for argument, field in fields.items():
        value = getattr(args, argument, None)
        if value is not None:
            overrides[field] = value
            set_value(cfg, field, value)
    accelerator = args.accelerator
    if accelerator == "auto":
        accelerator = "gpu" if cuda_available else "cpu"
    configured_devices = cfg["train"].get("gpus", [0])
    devices = args.devices
    if devices is None:
        devices = copy.deepcopy(configured_devices) if accelerator == "gpu" else 1
    if accelerator == "cpu" and isinstance(devices, list):
        raise ValueError("CPU devices use a count; GPU ID lists require --accelerator gpu.")
    world_size = len(devices) if isinstance(devices, list) else devices
    batch = int(cfg["train"]["batch_size"])
    accumulation = int(cfg["train"].get("accumulate_grad_batches", 1))
    if min(batch, accumulation, int(cfg["train"]["max_epochs"]), world_size) < 1:
        raise ValueError("Batch size, accumulation, epochs and device count must be positive.")
    if not 10 <= cfg["evaluation"]["sample_steps"] <= 200:
        raise ValueError("Validation decoding requires 10..200 steps.")
    if min(cfg["train"]["num_workers"], cfg["evaluation"]["num_workers"]) < 0:
        raise ValueError("Worker counts must be nonnegative.")
    if min(cfg["evaluation"]["batch_size"], cfg["evaluation"]["num_samples"]) < 1:
        raise ValueError("Validation batch size and sample count must be positive.")
    output = Path(args.output).resolve()
    cfg["train"]["ckpt_path"] = str(output / "checkpoints")
    cfg["evaluation"]["save_dir"] = str(output / "validation")
    cfg["data"]["path"] = str(Path(args.data).resolve())
    cfg["data"]["indices_path"] = str(Path(args.splits).resolve())
    notes = registry()["models"][args.model]["unresolved_paper_differences"]
    return cfg, {
        "profile": selected,
        "explicit_overrides": overrides,
        "accelerator": accelerator, "devices": devices, "world_size": world_size,
        "configured_gpu_ids": configured_devices,
        "batch_per_device": batch, "gradient_accumulation": accumulation,
        "effective_molecules_per_optimizer_step": batch * accumulation * world_size,
        "conformers_per_molecule": 2 if args.model == "molalign3d" else 1,
        "validation_batch_size": cfg["evaluation"]["batch_size"],
        "validation_workers": cfg["evaluation"]["num_workers"],
        "validation_samples": cfg["evaluation"]["num_samples"],
        "validation_decode_steps": cfg["evaluation"]["sample_steps"],
        "ema_enabled": False,
        "scheduler_monitor": "val_loss_epoch",
        "scheduler_interval": "epoch" if cfg["train"]["scheduler"]["type"] == "plateau" else "step",
        "paper_table_equivalence_verified": False,
        "unresolved_paper_differences": notes,
        "preset_scope": "Numeric hyperparameters; original checkpoint-compatible objective implementation retained.",
    }


def verify_resume_config(checkpoint, config):
    """A resume restores optimizer state; conflicting presets must not disappear."""
    previous = checkpoint.get("research_config")
    if previous:
        before, after = flatten(previous), flatten(config)
        prefixes = ("encoder_config.", "optimal_layer_config.", "decoder_config.",
                    "train.kl_loss.", "train.recon_loss.", "train.contrastive_loss.",
                    "train.atom_num_loss.", "train.zh_noise.", "train.optimizer.", "train.scheduler.")
        changed = [field for field in sorted(set(before) | set(after))
                   if field.startswith(prefixes) and before.get(field) != after.get(field)]
        if changed:
            raise ValueError("Resume configuration conflicts with the saved run: " + ", ".join(changed) +
                             ". Use --init-checkpoint for a new experiment with changed settings.")
    for state in checkpoint.get("optimizer_states", []):
        for group in state["param_groups"]:
            expected = config["train"]["optimizer"]
            if list(group["betas"]) != [expected["beta1"], expected["beta2"]] or group["weight_decay"] != expected["weight_decay"]:
                raise ValueError("Resume optimizer conflicts with the selected preset; use --init-checkpoint for a new experiment.")
    return {"full_config_available": previous is not None,
            "optimizer_state_restored": bool(checkpoint.get("optimizer_states")),
            "restored_learning_rates": [group["lr"] for state in checkpoint.get("optimizer_states", []) for group in state["param_groups"]]}
