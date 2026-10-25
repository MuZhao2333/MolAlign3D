"""Portable research commands; run with python -m common COMMAND --help."""

import argparse
import json


def parse_devices(value):
    if "," in value:
        ids = [int(item) for item in value.split(",")]
        if len(set(ids)) != len(ids) or min(ids) < 0:
            raise argparse.ArgumentTypeError("GPU IDs must be distinct nonnegative integers")
        return ids
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("Device count must be positive")
    return count


def build_parser():
    parser = argparse.ArgumentParser(description="ZINC research workflows for MolAlign3D and MolFLAE")
    sub = parser.add_subparsers(dest="command", required=True)

    def model_options(p, include_device=True):
        p.add_argument("--model", choices=["molalign3d", "molflae"], default="molalign3d")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--threads", type=int, default=4)
        if include_device:
            p.add_argument("--device", choices=["cpu", "auto"], default="cpu")

    def data_options(p):
        p.add_argument("--data", required=True)
        p.add_argument("--splits", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--conformers", type=int, default=2)
    prepare.add_argument("--seed", type=int, default=42)
    train = sub.add_parser("train")
    model_options(train, include_device=False)
    data_options(train)
    selection = train.add_mutually_exclusive_group()
    selection.add_argument("--config")
    selection.add_argument("--profile", choices=["checkpoint", "original", "paper"], help="Default: checkpoint; paper selects table values where they conflict")
    train.add_argument("--output", required=True)
    initialization = train.add_mutually_exclusive_group()
    initialization.add_argument("--init-checkpoint")
    initialization.add_argument("--resume")
    train.add_argument("--epochs", type=int, help="Override the selected model configuration")
    train.add_argument("--max-steps", type=int, default=-1)
    train.add_argument("--batch-size", type=int, help="Molecules per device; inherited from the model config")
    train.add_argument("--workers", type=int)
    train.add_argument("--accelerator", choices=["cpu", "gpu", "auto"], default="auto")
    train.add_argument("--devices", type=parse_devices, help="Device count or comma-separated GPU IDs; GPU default inherits train.gpus")
    train.add_argument("--lr", type=float)
    train.add_argument("--validation-batch-size", type=int)
    train.add_argument("--validation-workers", type=int)
    train.add_argument("--validation-samples", type=int)
    train.add_argument("--validation-decode-steps", type=int)
    train.add_argument("--dry-run", action="store_true", help="Save resolved configuration and exit before loading data or models")
    train.add_argument("--limit-train-batches", type=int, default=1000000000)
    train.add_argument("--limit-val-batches", type=int, default=1000000000)
    evaluate = sub.add_parser("evaluate")
    model_options(evaluate)
    data_options(evaluate)
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--split", choices=["train", "val", "test"], default="test")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--steps", type=int, default=100)
    evaluate.add_argument("--conformers", type=int, default=1)
    evaluate.add_argument("--limit", type=int)
    experiments = sub.add_parser("experiments")
    model_options(experiments)
    experiments.add_argument("--checkpoint")
    inputs = experiments.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--pairs", help="Pair JSON, or directory of *_start.sdf / *_end.sdf files")
    inputs.add_argument("--molecules", help="SDF file or directory; use for atom-edit, reconstruct or wander")
    experiments.add_argument("--output", required=True)
    experiments.add_argument(
        "--experiments",
        nargs="+",
        choices=["interpolation", "swap", "atom-edit", "wander", "reconstruct"],
        default=["atom-edit", "interpolation", "swap"],
    )
    experiments.add_argument("--steps", type=int, default=100)
    experiments.add_argument("--frames", type=int, default=12)
    experiments.add_argument("--radius", type=int, default=2)
    experiments.add_argument("--step-size", type=float, default=0.1)
    experiments.add_argument("--batch-size", type=int, default=500, help="Atom-edit / reconstruction decode batch size")
    experiments.add_argument("--group-size", type=int, default=50, help="Interpolation decode pairs per batch")
    experiments.add_argument("--swap-batch-size", type=int, help="Default: all pairs, as in the original swap script")
    experiments.add_argument("--space", choices=["zx", "zh", "both"], default="both", help="Wander extension only")
    prop = sub.add_parser("property-probe")
    model_options(prop)
    prop.add_argument("--csv", required=True)
    prop.add_argument("--target", required=True)
    prop.add_argument("--task", choices=["classification", "regression"], default="regression")
    prop.add_argument("--checkpoint")
    prop.add_argument("--epochs", type=int, default=100)
    prop.add_argument("--lr", type=float, default=1e-3)
    prop.add_argument("--dropout", type=float, default=0.1)
    prop.add_argument("--output", required=True)
    tune = sub.add_parser("finetune")
    model_options(tune)
    tune.add_argument("--csv", required=True)
    tune.add_argument("--target", required=True)
    tune.add_argument("--task", choices=["classification", "regression"], default="regression")
    tune.add_argument("--checkpoint")
    tune.add_argument("--epochs", type=int, default=100)
    tune.add_argument("--batch-size", type=int, default=32)
    tune.add_argument("--lr", type=float, default=1e-4)
    tune.add_argument("--weight-decay", type=float, default=0.0)
    tune.add_argument("--dropout", type=float, default=0.1)
    tune.add_argument("--output", required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for field in ["steps", "validation_decode_steps"]:
        if getattr(args, field, None) is not None and not 10 <= getattr(args, field) <= 200:
            parser.error(field + " must be in 10..200")
    for field in ["threads", "conformers", "frames", "epochs", "batch_size", "group_size", "swap_batch_size", "validation_batch_size", "validation_samples"]:
        if getattr(args, field, None) is not None and getattr(args, field) < 1:
            parser.error(field + " must be positive")
    for field in ["workers", "validation_workers"]:
        if getattr(args, field, None) is not None and getattr(args, field) < 0:
            parser.error(field + " must be nonnegative")
    if args.command == "experiments":
        import math

        if args.molecules and set(args.experiments) & {"interpolation", "swap"}:
            parser.error("interpolation and swap require --pairs; select --experiments atom-edit for molecule inputs")
        if args.frames < 2 or args.radius < 0:
            parser.error("frames must be at least 2 and radius must be nonnegative")
        if not math.isfinite(args.step_size) or args.step_size < 0:
            parser.error("step-size must be finite and nonnegative")
        if not 0 <= args.seed <= 2147483647:
            parser.error("seed must be in 0..2147483647")
    if args.command == "prepare":
        from .data import prepare as run

        result = run(args.input, args.output, args.conformers, args.seed)
    elif args.command == "train":
        from .train import run

        result = run(args)
    elif args.command == "evaluate":
        from .evaluate import run

        result = run(args)
    elif args.command == "experiments":
        from .experiments import run

        result = run(args)
    elif args.command == "property-probe":
        from .properties import run

        result = run(args)
    else:
        from .finetune import run

        result = run(args)
    print(json.dumps(result, indent=2, default=str))
    if args.command == "experiments" and result["failed_records"]:
        raise SystemExit(1)
