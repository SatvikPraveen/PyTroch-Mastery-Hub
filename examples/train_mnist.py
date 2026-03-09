"""
Example: Train a Simple MLP on MNIST
======================================
Demonstrates end-to-end training of an MLP using src/ utilities.
Run: python examples/train_mnist.py [--epochs 5] [--batch-size 64]
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.neural_networks.models import SimpleMLP
from src.neural_networks.training import train_epoch, validate_epoch
from src.utils.data_utils import load_dataset
from src.utils.metrics import accuracy
from src.utils.io_utils import save_model
from src.utils.visualization import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP on MNIST")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 256],
                        help="Hidden layer dimensions")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--save-path", type=str, default="outputs/mnist_mlp.pth",
                        help="Path to save trained model")
    parser.add_argument("--no-save", action="store_true", help="Don't save the model")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Device ───────────────────────────────────────────────────
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # ── Data ─────────────────────────────────────────────────────
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = load_dataset("mnist", batch_size=args.batch_size)
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Test  batches : {len(test_loader)}")

    # ── Model ────────────────────────────────────────────────────
    model = SimpleMLP(
        input_dim=784,           # 28*28 flattened MNIST
        hidden_dims=args.hidden_dims,
        output_dim=10,           # 10 digit classes
        dropout=args.dropout,
    ).to(device)
    print(f"\nModel: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ── Optimizer & loss ─────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training ─────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs...")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_preds, val_targets = validate_epoch(model, test_loader, criterion, device)
        val_acc = accuracy(val_preds, val_targets)
        scheduler.step()

        elapsed = time.time() - start
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch:2d}/{args.epochs}  |  "
              f"train_loss: {train_loss:.4f}  |  "
              f"val_loss: {val_loss:.4f}  |  "
              f"val_acc: {val_acc:.2%}  |  "
              f"time: {elapsed:.1f}s")

    # ── Save ─────────────────────────────────────────────────────
    if not args.no_save:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        save_model(model, args.save_path)
        print(f"\nModel saved to: {args.save_path}")

    print(f"\n✓ Final validation accuracy: {history['val_acc'][-1]:.2%}")
    print("Training complete!")


if __name__ == "__main__":
    main()
