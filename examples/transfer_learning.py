"""
Example: Transfer Learning with a Pretrained CNN
==================================================
Demonstrates fine-tuning a pretrained ResNet-18 on a custom dataset (simulated with
synthetic data) using the pytorch_mastery_hub Trainer and model utilities.
Run: python examples/transfer_learning.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch.utils.data import DataLoader, TensorDataset

from pytorch_mastery_hub.neural_networks.training import Trainer, TrainerConfig
from pytorch_mastery_hub.utils.device_utils import get_device
from pytorch_mastery_hub.utils.model_utils import count_parameters, freeze, unfreeze
from pytorch_mastery_hub.utils.reproducibility import seed_everything


def create_synthetic_image_data(num_train=160, num_val=40, num_classes=4, img_size=64):
    """Synthetic 'photos': every class has its own colour tint, so there is something to learn."""
    tints = torch.rand(num_classes, 3) * 2 - 1

    def make(n):
        y = torch.randint(0, num_classes, (n,))
        x = tints[y].view(n, 3, 1, 1) + 0.5 * torch.randn(n, 3, img_size, img_size)
        return TensorDataset(x, y)

    return (
        DataLoader(make(num_train), batch_size=16, shuffle=True),
        DataLoader(make(num_val), batch_size=16),
        num_classes,
    )


def build_resnet18(num_classes: int) -> nn.Module:
    """Load ResNet-18 (ImageNet weights if available offline/online) and replace the head."""
    try:
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        print("  Loaded ImageNet-pretrained weights")
    except Exception as e:  # no network and no cached weights
        print(f"  Pretrained weights unavailable ({e.__class__.__name__}); using random init")
        model = tv_models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    return model


def run_phase(name, model, train_loader, val_loader, device, lr, epochs):
    """Train only the parameters that currently require grad, using the Trainer."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    trainer = Trainer(
        model, nn.CrossEntropyLoss(), optimizer, device, config=TrainerConfig(epochs=epochs)
    )
    history = trainer.fit(train_loader, val_loader)
    print(f"  → {name}: best val_accuracy {max(history['val_accuracy']):.1f}%")
    return history


def main():
    print("=" * 60)
    print("PyTorch Mastery Hub — Transfer Learning Example")
    print("=" * 60)
    seed_everything(0)
    device = get_device()
    print(f"\nDevice: {device}")

    # ── Data ─────────────────────────────────────────────────────
    print("\nCreating synthetic image dataset (160 train / 40 val, 64×64)...")
    train_loader, val_loader, num_classes = create_synthetic_image_data()
    print(f"  Classes: {num_classes}")

    # ── Model ────────────────────────────────────────────────────
    print("\nBuilding ResNet-18...")
    model = build_resnet18(num_classes)
    total = count_parameters(model)

    # ── Phase 1: train only the new classification head ──────────
    freeze(model)  # everything...
    unfreeze(model, ["fc"])  # ...except the head
    trainable = count_parameters(model, trainable_only=True)
    print(
        f"\nPhase 1 — frozen backbone: {trainable:,} / {total:,} trainable "
        f"({100 * trainable / total:.2f}%)"
    )
    run_phase("head only", model, train_loader, val_loader, device, lr=1e-3, epochs=3)

    # ── Phase 2: unfreeze everything and fine-tune with a small LR ─
    unfreeze(model)
    trainable = count_parameters(model, trainable_only=True)
    print(f"\nPhase 2 — full fine-tuning: {trainable:,} / {total:,} trainable")
    run_phase("full fine-tune", model, train_loader, val_loader, device, lr=1e-4, epochs=2)

    print("\n✓ Transfer learning example completed!")


if __name__ == "__main__":
    main()
