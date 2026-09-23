"""
Example: Train a Simple MLP on MNIST
======================================
Demonstrates end-to-end training of an MLP with the pytorch_mastery_hub ``Trainer``
(mixed precision, EMA, early stopping, checkpointing). Uses real MNIST when it can be
downloaded, otherwise (or with ``--synthetic``) a small MNIST-shaped synthetic dataset.
Run: python examples/train_mnist.py [--epochs 3] [--batch-size 64] [--synthetic]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_mastery_hub.neural_networks.models import SimpleMLP
from pytorch_mastery_hub.neural_networks.training import (
    EarlyStopping,
    ModelCheckpoint,
    Trainer,
    TrainerConfig,
)
from pytorch_mastery_hub.utils.data_utils import create_data_loaders, load_dataset
from pytorch_mastery_hub.utils.device_utils import get_device
from pytorch_mastery_hub.utils.io_utils import save_model
from pytorch_mastery_hub.utils.model_utils import count_parameters
from pytorch_mastery_hub.utils.reproducibility import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP on MNIST")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--hidden-sizes", nargs="+", type=int, default=[256, 128], help="Hidden layer sizes"
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--precision", default="auto", choices=["no", "auto", "fp16", "bf16"], help="AMP mode"
    )
    parser.add_argument("--data-dir", default="data", help="Where to download/look for MNIST")
    parser.add_argument(
        "--synthetic", action="store_true", help="Skip MNIST and use a synthetic stand-in"
    )
    parser.add_argument(
        "--save-path", type=str, default="outputs/mnist_mlp.pth", help="Path to save trained model"
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save the model")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_synthetic_mnist(n: int, seed: int, num_classes: int = 10) -> TensorDataset:
    """MNIST-shaped stand-in: noisy 1×28×28 images where each class lights up its own patch."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, num_classes, (n,), generator=g)
    x = torch.randn(n, 1, 28, 28, generator=g)
    for k in range(num_classes):
        r, c = (k // 5) * 14, (k % 5) * 5
        x[y == k, :, r : r + 14, c : c + 5] += 0.8
    return TensorDataset(x, y)


def build_loaders(args):
    """Return (train_loader, test_loader, dataset_name); never requires the network."""
    if not args.synthetic:
        try:
            data = load_dataset("mnist", data_dir=args.data_dir, download=True)
            if "train_dataset" in data:
                loaders = create_data_loaders(
                    data["train_dataset"],
                    data["test_dataset"],
                    batch_size=args.batch_size,
                    num_workers=0,
                )
                return loaders["train"], loaders["test"], "MNIST"
            print("  MNIST could not be loaded.")
        except Exception as e:  # no network, corrupt download, ...
            print(f"  MNIST download failed ({e.__class__.__name__}: {e}).")
        print("  Falling back to synthetic MNIST-like data (use --synthetic to skip the attempt).")

    train_ds = make_synthetic_mnist(4096, seed=args.seed)
    test_ds = make_synthetic_mnist(1024, seed=args.seed + 1)
    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=args.batch_size),
        "synthetic MNIST-like",
    )


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # ── Data ─────────────────────────────────────────────────────
    print("\nLoading data...")
    train_loader, test_loader, dataset_name = build_loaders(args)
    print(f"  Dataset       : {dataset_name}")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Test  batches : {len(test_loader)}")

    # ── Model ────────────────────────────────────────────────────
    # SimpleMLP expects flat vectors, so flatten the 1×28×28 images first.
    model = nn.Sequential(
        nn.Flatten(),
        SimpleMLP(784, hidden_sizes=args.hidden_sizes, output_size=10, dropout=args.dropout),
    )
    print(f"\nModel: Flatten + SimpleMLP{args.hidden_sizes}")
    print(f"  Total parameters: {count_parameters(model):,}")

    # ── Trainer: AMP + EMA + cosine LR + early stopping + best checkpoint ──
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ckpt_path = os.path.join(os.path.dirname(args.save_path) or ".", "mnist_best.pt")
    trainer = Trainer(
        model,
        nn.CrossEntropyLoss(),
        optimizer,
        device,
        scheduler=scheduler,
        config=TrainerConfig(
            epochs=args.epochs,
            precision=args.precision,
            ema_decay=0.99,  # validation uses the EMA weights
            ema_warmup_steps=100,
        ),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=2),
            ModelCheckpoint(ckpt_path, monitor="val_accuracy", mode="max"),
        ],
    )

    print(f"\nTraining for {args.epochs} epochs (precision={args.precision})...")
    history = trainer.fit(train_loader, test_loader)

    # ── Results ───────────────────────────────────────────────────
    best_epoch = max(range(len(history["val_accuracy"])), key=history["val_accuracy"].__getitem__)
    print(
        f"\n✓ Best validation accuracy: {history['val_accuracy'][best_epoch]:.2f}% "
        f"(epoch {best_epoch + 1}, {sum(history['epoch_time']):.1f}s total)"
    )

    if not args.no_save:
        save_model(
            trainer.unwrapped_model,
            args.save_path,
            metadata={"dataset": dataset_name, "val_accuracy": history["val_accuracy"][-1]},
        )
    print("Training complete!")


if __name__ == "__main__":
    main()
