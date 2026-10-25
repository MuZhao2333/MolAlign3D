"""Lightning training with original model losses and portable run configuration."""

import copy
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from .utils import move_decoder_constants, run_manifest, write_json
from .configuration import resolve_training, verify_resume_config
from .data import ConformerDataset, collate_pairs, collate_single


class AdaptiveGradientClip(Callback):
    """The upstream Q rule, with callback-local state and checkpointed history."""

    def __init__(self, max_norm="Q"):
        self.max_norm = max_norm
        self.history = [1.0]

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        threshold = 1.5 * np.mean(self.history) + 2 * np.std(self.history) if self.max_norm == "Q" else float(self.max_norm)
        norm = torch.nn.utils.clip_grad_norm_(pl_module.parameters(), float(threshold), error_if_nonfinite=True)
        if self.max_norm == "Q":
            self.history = ([min(float(norm), float(threshold))] + self.history)[:3000]
        pl_module.log("grad_norm", norm, on_step=True, on_epoch=False, batch_size=pl_module.cfg["train"]["batch_size"])

    def state_dict(self):
        return {"history": self.history}

    def load_state_dict(self, state_dict):
        self.history = state_dict["history"]


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(args.seed, workers=True)
    torch.set_num_threads(args.threads)
    cfg, resolved = resolve_training(args, cuda_available=torch.cuda.is_available())
    if args.model == "molalign3d":
        module = importlib.import_module(args.model + ".engine")
        os.environ["UNIMOL_WEIGHT_DIR"] = str(module.ROOT)
        cfg["unimol"]["ckpt_path"] = str(module.ROOT / "mol_pre_no_h_220816.pt")
    if args.resume and not args.dry_run:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        resolved["resume"] = verify_resume_config(saved, cfg)
        del saved
    settings = vars(args).copy()
    manifest = run_manifest(**settings)
    manifest["resolved_training"] = resolved
    write_json(output / "run.json", manifest)
    write_json(output / "hyperparameters.json", resolved)
    import yaml

    (output / "resolved-config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if args.dry_run:
        return {"dry_run": True, "output": str(output), "resolved_training": resolved}
    base = importlib.import_module(args.model + ".models.autoencoder").TrainLoop

    class ResearchModel(base):
        def configure_optimizers(self):
            optimizer_cfg = self.cfg["train"]["optimizer"]
            self.optim = torch.optim.Adam(
                self.parameters(),
                lr=float(optimizer_cfg["lr"]),
                weight_decay=float(optimizer_cfg["weight_decay"]),
                betas=(float(optimizer_cfg["beta1"]), float(optimizer_cfg["beta2"])),
            )
            self.get_last_lr = lambda: self.optim.param_groups[0]["lr"]
            schedule = self.cfg["train"]["scheduler"]
            if schedule["type"] == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optim, factor=float(schedule["factor"]), patience=int(schedule["patience"]), min_lr=float(schedule["min_lr"])
                )
                return {"optimizer": self.optim, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss_epoch", "interval": "epoch"}}
            if schedule["type"] == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optim, T_max=int(schedule["max_iters"]), eta_min=float(schedule["min_lr"])
                )
                return {"optimizer": self.optim, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
            raise ValueError("Unsupported scheduler " + schedule["type"])

        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            if loss is None or not torch.isfinite(loss).all() or not loss.requires_grad:
                raise FloatingPointError("Original loss is nonfinite or disconnected; stopping without a silent update.")
            return loss

        def on_fit_start(self):
            move_decoder_constants(self, self.device)

        def on_train_epoch_end(self):
            # Lightning can finish a restored epoch before consuming another batch.
            if self.train_losses:
                super().on_train_epoch_end()

        def on_save_checkpoint(self, checkpoint):
            checkpoint["research_config"] = copy.deepcopy(self.cfg)
            checkpoint["research_provenance"] = manifest

    instance = ResearchModel(cfg)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        instance.load_state_dict(checkpoint["state_dict"], strict=True)
    paired = args.model == "molalign3d"
    train = ConformerDataset(args.data, args.splits, "train", paired)
    val = ConformerDataset(args.data, args.splits, "val", paired)
    collate = collate_pairs if paired else collate_single
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"],
        collate_fn=collate, generator=generator, pin_memory=resolved["accelerator"] == "gpu",
    )
    val_loader = DataLoader(
        val, batch_size=cfg["evaluation"]["batch_size"], shuffle=False,
        num_workers=cfg["evaluation"]["num_workers"], collate_fn=collate,
    )
    callback = ModelCheckpoint(
        dirpath=output / "checkpoints",
        monitor="val_loss_epoch",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="epoch-{epoch:03d}-val-{val_loss_epoch:.4f}",
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        accelerator=resolved["accelerator"],
        devices=resolved["devices"],
        strategy="ddp_find_unused_parameters_true" if resolved["world_size"] > 1 else "auto",
        max_epochs=cfg["train"]["max_epochs"],
        max_steps=args.max_steps,
        logger=CSVLogger(str(output), "logs"),
        callbacks=[callback, AdaptiveGradientClip(cfg["train"].get("max_grad_norm", "Q"))],
        accumulate_grad_batches=int(cfg["train"].get("accumulate_grad_batches", 1)),
        num_sanity_val_steps=0,
        val_check_interval=float(cfg["train"].get("val_freq", 1.0)),
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=1,
        enable_model_summary=False,
        default_root_dir=str(output),
    )
    # Upstream MolFLAE calls wandb.log directly. Disabled mode keeps training local.
    import wandb

    with wandb.init(mode="disabled"):
        trainer.fit(instance, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume)
    # Keep Lightning's resumable callback checkpoint. Overwriting it after fit
    # can mark both the epoch and batch complete in a max_steps-limited run,
    # causing a later resume to skip an optimizer update.
    final_checkpoint = Path(callback.last_model_path) if callback.last_model_path else output / "checkpoints/last.ckpt"
    if not final_checkpoint.exists():
        trainer.save_checkpoint(final_checkpoint)
    write_json(
        output / "training-summary.json",
        {
            "model": args.model,
            "optimizer_steps": trainer.global_step,
            "epochs_completed": trainer.current_epoch,
            "best_checkpoint": callback.best_model_path,
            "last_checkpoint": str(final_checkpoint),
            "best_validation_loss": float(callback.best_model_score) if callback.best_model_score is not None else None,
            "data_records": {"train": len(train), "val": len(val)},
            "validation_decode_steps": cfg["evaluation"]["sample_steps"],
            "resolved_training": resolved,
            "protocol": "original model training and validation losses; portable logging and scheduler compatibility",
        },
    )
