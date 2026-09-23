"""
Command-line interface for PyTorch Mastery Hub.

Usage::

    pytorch-hub info                                # environment + device report
    pytorch-hub train --config configs/train_synthetic.yaml
    pytorch-hub train --model mlp --epochs 3 --precision auto --ema 0.999
    pytorch-hub benchmark [--sizes 1024 2048] [--dtype bf16]
    pytorch-hub test [--module MODULE]
    pytorch-hub download-data [--datasets mnist cifar10]

``train`` is a small but complete reference pipeline: YAML/flag config ->
seeded data -> model -> :class:`~pytorch_mastery_hub.neural_networks.training.Trainer`
with AMP/EMA/early stopping/checkpointing -> metrics.jsonl + summary JSON.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["build_parser", "main"]


# ------------------------------------------------------------------------ info


def cmd_info(_args: argparse.Namespace) -> int:
    """Print environment and package information."""
    import torch

    from .. import __version__
    from .device_utils import device_info

    info = device_info()
    print(f"PyTorch Mastery Hub v{__version__}")
    print("-" * 40)
    width = max(len(k) for k in info)
    for key, value in info.items():
        print(f"  {key:<{width}} : {value}")
    try:
        import torchvision

        print(f"  {'torchvision':<{width}} : {torchvision.__version__}")
    except ImportError:  # pragma: no cover
        pass
    print(
        f"  {'sdpa_flash':<{width}} : {torch.backends.cuda.flash_sdp_enabled() if torch.cuda.is_available() else 'n/a'}"
    )
    return 0


# ----------------------------------------------------------------------- train

DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "seed": 42,
    "data": {
        "name": "synthetic",
        "n_samples": 2048,
        "n_features": 32,
        "n_classes": 4,
        "batch_size": 64,
    },
    "model": {"type": "mlp", "hidden_sizes": [128, 64], "dropout": 0.1},
    "optimizer": {"type": "adamw", "lr": 1e-3, "weight_decay": 0.01},
    "scheduler": {"type": "cosine"},
    "trainer": {
        "epochs": 5,
        "precision": "no",
        "accumulation_steps": 1,
        "clip_grad_norm": 1.0,
        "ema_decay": None,
        "compile": False,
    },
    "early_stopping": {"patience": 3, "monitor": "val_loss"},
    "output_dir": "outputs/run",
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        elif v is not None:
            out[k] = v
    return out


def load_train_config(
    path: str | Path | None, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Merge defaults <- YAML file <- CLI overrides (None values are ignored)."""
    cfg = dict(DEFAULT_TRAIN_CONFIG)
    if path is not None:
        import yaml

        with Path(path).open(encoding="utf-8") as fh:
            cfg = _deep_update(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg


def _build_data(cfg: dict[str, Any], seed: int) -> tuple[Any, Any, int, int]:
    """Return (train_loader, val_loader, input_size, num_classes)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset, random_split

    from .reproducibility import make_generator

    d = cfg["data"]
    if d["name"] == "synthetic":
        g = make_generator(seed)
        x = torch.randn(d["n_samples"], d["n_features"], generator=g)
        w = torch.randn(d["n_features"], d["n_classes"], generator=g)
        y = (x @ w + 0.3 * torch.randn(d["n_samples"], d["n_classes"], generator=g)).argmax(1)
        ds = TensorDataset(x, y)
        in_size, n_classes = d["n_features"], d["n_classes"]
    elif d["name"] in {"mnist", "fashion_mnist", "cifar10"}:
        from .data_utils import load_dataset

        data = load_dataset(d["name"], data_dir=d.get("data_dir", "data"))
        ds = data["train"]
        sample = ds[0][0]
        in_size, n_classes = int(sample.numel()), int(data.get("num_classes", 10))
    else:
        raise ValueError(f"unknown dataset {d['name']!r}")

    n_val = max(1, int(0.2 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=make_generator(seed))
    bs = d["batch_size"]
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, generator=make_generator(seed)),
        DataLoader(val_ds, batch_size=bs),
        in_size,
        n_classes,
    )


def _build_model(cfg: dict[str, Any], in_size: int, n_classes: int) -> Any:
    from torch import nn

    m = cfg["model"]
    if m["type"] == "mlp":
        from ..neural_networks.models import SimpleMLP

        return nn.Sequential(
            nn.Flatten(),
            SimpleMLP(in_size, list(m["hidden_sizes"]), n_classes, dropout=m.get("dropout", 0.0)),
        )
    if m["type"] == "cnn":
        from ..computer_vision.models import SimpleCNN

        return SimpleCNN(num_classes=n_classes, in_channels=m.get("in_channels", 1))
    raise ValueError(f"unknown model type {m['type']!r}")


def cmd_train(args: argparse.Namespace) -> int:
    """Run a full training job from a YAML config and/or flags."""
    import torch

    from ..neural_networks.training import EarlyStopping, ModelCheckpoint, Trainer, TrainerConfig
    from .device_utils import get_device
    from .logging_utils import MetricsLogger, setup_logger
    from .model_utils import count_parameters, param_groups_with_weight_decay
    from .reproducibility import seed_everything

    overrides = {
        "seed": args.seed,
        "output_dir": args.output_dir,
        "model": {"type": args.model},
        "optimizer": {"lr": args.lr},
        "trainer": {
            "epochs": args.epochs,
            "precision": args.precision,
            "ema_decay": args.ema,
            "compile": True if args.compile else None,
        },
    }
    cfg = load_train_config(args.config, overrides)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger("pytorch_hub.train", log_file=out_dir / "train.log")
    seed = seed_everything(int(cfg["seed"]))
    device = get_device(args.device)
    log.info("config: %s", json.dumps(cfg))
    log.info("device: %s", device)

    train_loader, val_loader, in_size, n_classes = _build_data(cfg, seed)
    model = _build_model(cfg, in_size, n_classes)
    log.info("model: %s parameters", f"{count_parameters(model):,}")

    o = cfg["optimizer"]
    groups = param_groups_with_weight_decay(model, o.get("weight_decay", 0.0))
    opt_cls = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}[
        o["type"]
    ]
    optimizer = opt_cls(groups, lr=o["lr"], **({"momentum": 0.9} if o["type"] == "sgd" else {}))

    t = cfg["trainer"]
    scheduler = None
    if cfg["scheduler"]["type"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
    elif cfg["scheduler"]["type"] == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=o["lr"], total_steps=t["epochs"] * len(train_loader)
        )

    metrics = MetricsLogger(out_dir / "metrics.jsonl")
    trainer = Trainer(
        model,
        torch.nn.CrossEntropyLoss(),
        optimizer,
        device,
        scheduler=scheduler,
        config=TrainerConfig(
            epochs=int(t["epochs"]),
            precision=t.get("precision", "no"),
            accumulation_steps=int(t.get("accumulation_steps", 1)),
            clip_grad_norm=t.get("clip_grad_norm"),
            ema_decay=t.get("ema_decay"),
            compile=bool(t.get("compile", False)),
            seed=seed,
            verbose=not args.quiet,
        ),
        callbacks=[
            EarlyStopping(
                monitor=cfg["early_stopping"]["monitor"],
                patience=int(cfg["early_stopping"]["patience"]),
                verbose=not args.quiet,
            ),
            ModelCheckpoint(
                out_dir / "best.pt", monitor=cfg["early_stopping"]["monitor"], verbose=False
            ),
        ],
    )
    trainer.add_callback(_MetricsCallback(metrics))

    start = time.perf_counter()
    history = trainer.fit(train_loader, val_loader)
    elapsed = time.perf_counter() - start
    metrics.close()

    summary = {
        "epochs_run": len(history["loss"]),
        "best_val_loss": min(history["val_loss"]),
        "final_val_accuracy": history["val_accuracy"][-1] if "val_accuracy" in history else None,
        "seconds": round(elapsed, 2),
        "checkpoint": str(out_dir / "best.pt"),
        "device": str(device),
        "config": cfg,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(
        "done in %.1fs - best val_loss %.4f - artifacts in %s",
        elapsed,
        summary["best_val_loss"],
        out_dir,
    )
    return 0


class _MetricsCallback:
    """Bridge Trainer epoch logs into a MetricsLogger (duck-typed Callback)."""

    def __init__(self, metrics: Any) -> None:
        self.metrics = metrics

    def on_train_begin(self, trainer: Any) -> None: ...

    def on_train_end(self, trainer: Any) -> None: ...

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None: ...

    def on_batch_end(self, trainer: Any, step: int, logs: dict[str, float]) -> None: ...

    def on_validation_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None: ...

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        self.metrics.log(epoch, **logs)


# ------------------------------------------------------------------- benchmark


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Measure matmul TFLOP/s and attention throughput on the selected device."""
    import torch
    import torch.nn.functional as F

    from .device_utils import get_device, synchronize

    device = get_device(args.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    print(f"device={device} dtype={args.dtype} iters={args.iters}")
    print(f"{'op':<24}{'size':>10}{'ms/iter':>12}{'TFLOP/s':>12}")
    for n in args.sizes:
        a = torch.randn(n, n, device=device, dtype=dtype)
        b = torch.randn(n, n, device=device, dtype=dtype)
        for _ in range(3):
            a @ b
        synchronize(device)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            a @ b
        synchronize(device)
        ms = (time.perf_counter() - t0) / args.iters * 1e3
        tflops = 2 * n**3 / (ms / 1e3) / 1e12
        print(f"{'matmul':<24}{n:>10}{ms:>12.3f}{tflops:>12.2f}")

    b, h, d = 8, 8, 64
    for t in args.seq_lens:
        q = torch.randn(b, h, t, d, device=device, dtype=dtype)
        for _ in range(3):
            F.scaled_dot_product_attention(q, q, q, is_causal=True)
        synchronize(device)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            F.scaled_dot_product_attention(q, q, q, is_causal=True)
        synchronize(device)
        ms = (time.perf_counter() - t0) / args.iters * 1e3
        flops = 4 * b * h * t * t * d  # QK^T and PV, causal ~half but count full
        print(f"{'sdpa (causal)':<24}{t:>10}{ms:>12.3f}{flops / (ms / 1e3) / 1e12:>12.2f}")
    return 0


# ----------------------------------------------------------------- test / data


def cmd_test(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "-m", "pytest", "-q"]
    cmd.append(f"tests/test_{args.module}/" if args.module else "tests/")
    return subprocess.call(cmd)


def cmd_download_data(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "scripts/download_datasets.py"]
    if args.datasets:
        cmd += ["--datasets", *args.datasets]
    return subprocess.call(cmd)


# ---------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pytorch-hub", description="PyTorch Mastery Hub command-line interface"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="Show environment information").set_defaults(func=cmd_info)

    tr = sub.add_parser("train", help="Train a model from a YAML config and/or flags")
    tr.add_argument("--config", type=str, default=None, help="YAML config (see configs/)")
    tr.add_argument("--model", choices=["mlp", "cnn"], default=None)
    tr.add_argument("--epochs", type=int, default=None)
    tr.add_argument("--lr", type=float, default=None)
    tr.add_argument("--precision", choices=["no", "auto", "fp16", "bf16"], default=None)
    tr.add_argument("--ema", type=float, default=None, help="EMA decay, e.g. 0.999")
    tr.add_argument("--compile", action="store_true", help="torch.compile the model")
    tr.add_argument("--seed", type=int, default=None)
    tr.add_argument("--device", type=str, default=None, help="cpu | cuda | cuda:1 | mps")
    tr.add_argument("--output-dir", type=str, default=None)
    tr.add_argument("--quiet", action="store_true")
    tr.set_defaults(func=cmd_train)

    bm = sub.add_parser("benchmark", help="Measure matmul / attention throughput")
    bm.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048])
    bm.add_argument("--seq-lens", type=int, nargs="+", default=[256, 1024])
    bm.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    bm.add_argument("--iters", type=int, default=10)
    bm.add_argument("--device", type=str, default=None)
    bm.set_defaults(func=cmd_benchmark)

    te = sub.add_parser("test", help="Run the test suite")
    te.add_argument("--module", type=str, default=None, help="e.g. fundamentals, nlp")
    te.set_defaults(func=cmd_test)

    dl = sub.add_parser("download-data", help="Download datasets")
    dl.add_argument(
        "--datasets",
        nargs="+",
        choices=["mnist", "fashion_mnist", "cifar10", "cifar100"],
        default=["mnist", "cifar10"],
    )
    dl.set_defaults(func=cmd_download_data)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
