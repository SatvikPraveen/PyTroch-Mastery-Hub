"""
Example: Transfer Learning with a Pretrained CNN
==================================================
Demonstrates fine-tuning a pretrained ResNet on a custom dataset
(simulated with synthetic data) using src.computer_vision modules.
Run: python examples/transfer_learning.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch.utils.data import DataLoader, TensorDataset

from src.computer_vision.augmentation import MixUp
from src.neural_networks.training import train_epoch, validate_epoch
from src.utils.metrics import accuracy


def create_synthetic_image_data(num_train=200, num_val=50, num_classes=5, img_size=224):
    """Create synthetic image-like tensors to simulate a classification dataset."""
    x_train = torch.randn(num_train, 3, img_size, img_size)
    y_train = torch.randint(0, num_classes, (num_train,))
    x_val = torch.randn(num_val, 3, img_size, img_size)
    y_val = torch.randint(0, num_classes, (num_val,))
    return (
        DataLoader(TensorDataset(x_train, y_train), batch_size=16, shuffle=True),
        DataLoader(TensorDataset(x_val, y_val), batch_size=16),
        num_classes,
    )


def build_fine_tuned_resnet(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """Load pretrained ResNet-18 and replace the final layer."""
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    # Replace classification head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def main():
    print("=" * 60)
    print("PyTorch Mastery Hub — Transfer Learning Example")
    print("=" * 60)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"\nDevice: {device}")

    # ── Data ─────────────────────────────────────────────────────
    print("\nCreating synthetic image dataset (200 train / 50 val)...")
    train_loader, val_loader, num_classes = create_synthetic_image_data()
    print(f"  Classes: {num_classes}")

    # ── Model ────────────────────────────────────────────────────
    print("\nBuilding fine-tuned ResNet-18...")
    model = build_fine_tuned_resnet(num_classes=num_classes, freeze_backbone=True).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params : {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}% of total)")

    # ── Training ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()
    mixup = MixUp(alpha=0.2)

    print("\nTraining for 3 epochs (frozen backbone)...")
    for epoch in range(1, 4):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, preds, targets = validate_epoch(model, val_loader, criterion, device)
        val_acc = accuracy(preds, targets)
        print(f"  Epoch {epoch}/3  |  train_loss: {train_loss:.4f}  |  "
              f"val_loss: {val_loss:.4f}  |  val_acc: {val_acc:.2%}")

    # ── Unfreeze and fine-tune ────────────────────────────────────
    print("\nUnfreezing all layers for fine-tuning...")
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(4, 6):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, preds, targets = validate_epoch(model, val_loader, criterion, device)
        val_acc = accuracy(preds, targets)
        print(f"  Epoch {epoch}/5  |  train_loss: {train_loss:.4f}  |  "
              f"val_loss: {val_loss:.4f}  |  val_acc: {val_acc:.2%}")

    print("\n✓ Transfer learning example completed!")


if __name__ == "__main__":
    main()
