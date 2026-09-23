"""
Example: Knowledge Distillation
=================================
Demonstrates transferring knowledge from a large teacher model to a
smaller student model using pytorch_mastery_hub.advanced.optimization.KnowledgeDistillation.
Run: python examples/knowledge_distillation.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_mastery_hub.advanced.optimization import KnowledgeDistillation
from pytorch_mastery_hub.neural_networks.models import SimpleMLP
from pytorch_mastery_hub.neural_networks.training import train_epoch, validate_epoch
from pytorch_mastery_hub.utils.device_utils import get_device
from pytorch_mastery_hub.utils.model_utils import count_parameters
from pytorch_mastery_hub.utils.reproducibility import seed_everything

INPUT_SIZE, NUM_CLASSES = 64, 5


def make_synthetic_data(n_train=600, n_val=400):
    """Toy task: overlapping Gaussian blobs, one per class (accuracy ceiling well below 100%)."""
    n = n_train + n_val
    centers = 0.5 * torch.randn(NUM_CLASSES, INPUT_SIZE)
    y = torch.randint(0, NUM_CLASSES, (n,))
    x = centers[y] + torch.randn(n, INPUT_SIZE)
    train_loader = DataLoader(TensorDataset(x[:n_train], y[:n_train]), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(x[n_train:], y[n_train:]), batch_size=64)
    return train_loader, val_loader


def fit(model, loader, device, epochs, lr=3e-3):
    """Plain supervised training with the functional train_epoch API."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        train_epoch(model, loader, criterion, optimizer, device)


def evaluate(model, loader, device) -> float:
    """Validation accuracy in percent (validate_epoch returns a metrics dict)."""
    return validate_epoch(model, loader, nn.CrossEntropyLoss(), device)["accuracy"]


def main():
    print("=" * 60)
    print("PyTorch Mastery Hub — Knowledge Distillation")
    print("=" * 60)
    seed_everything(0)
    device = get_device()
    print(f"\nDevice: {device}")

    train_loader, val_loader = make_synthetic_data()

    # ── Teacher model (large) ─────────────────────────────────────
    teacher = SimpleMLP(INPUT_SIZE, hidden_sizes=[256, 256, 128], output_size=NUM_CLASSES)
    print(f"\nTeacher params : {count_parameters(teacher):,}")
    print("Training teacher (15 epochs)...")
    fit(teacher, train_loader, device, epochs=15)
    teacher_acc = evaluate(teacher, val_loader, device)
    print(f"Teacher accuracy: {teacher_acc:.1f}%")

    # ── Student models (small) ────────────────────────────────────
    student_scratch = SimpleMLP(INPUT_SIZE, hidden_sizes=[16], output_size=NUM_CLASSES)
    student_distill = SimpleMLP(INPUT_SIZE, hidden_sizes=[16], output_size=NUM_CLASSES)
    student_distill.load_state_dict(student_scratch.state_dict())  # identical starting point
    s_params = count_parameters(student_scratch)
    print(
        f"\nStudent params : {s_params:,}  ({100 * s_params / count_parameters(teacher):.1f}% of teacher)"
    )

    # ── Train student from scratch (hard labels only) ─────────────
    print("\nTraining student from scratch (15 epochs)...")
    fit(student_scratch, train_loader, device, epochs=15)
    scratch_acc = evaluate(student_scratch, val_loader, device)
    print(f"Student (scratch) accuracy : {scratch_acc:.1f}%")

    # ── Train student with knowledge distillation ─────────────────
    # loss = alpha * T² * KL(student_T || teacher_T) + (1 - alpha) * CE(student, y)
    print("\nTraining student via knowledge distillation (15 epochs)...")
    kd = KnowledgeDistillation(
        teacher_model=teacher.to(device),
        student_model=student_distill.to(device),
        temperature=4.0,
        alpha=0.7,  # weight of the soft-target (KD) term
    )
    d_optimizer = torch.optim.Adam(student_distill.parameters(), lr=3e-3)
    for epoch in range(1, 16):
        totals = {"total_loss": 0.0, "kd_loss": 0.0, "ce_loss": 0.0}
        for x_b, y_b in train_loader:
            logs = kd.train_step(x_b.to(device), y_b.to(device), d_optimizer)
            for k in totals:
                totals[k] += logs[k]
        n = len(train_loader)
        print(
            f"  Epoch {epoch:2d}/15  |  total: {totals['total_loss'] / n:.4f}  |  "
            f"kd: {totals['kd_loss'] / n:.4f}  |  ce: {totals['ce_loss'] / n:.4f}"
        )
    distill_acc = evaluate(student_distill, val_loader, device)
    print(f"Student (distilled) accuracy : {distill_acc:.1f}%")

    # ── Summary ──────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────")
    print(f"  {'Model':<30} {'Params':>8} {'Val Accuracy':>14}")
    print(f"  {'-' * 54}")
    rows = [
        ("Teacher (large)", teacher, teacher_acc),
        ("Student (trained scratch)", student_scratch, scratch_acc),
        ("Student (distilled)", student_distill, distill_acc),
    ]
    for name, model, acc in rows:
        print(f"  {name:<30} {count_parameters(model):>8,} {acc:>13.1f}%")

    print(f"\n  Distillation improvement: {distill_acc - scratch_acc:+.1f} points")
    print("  (on such a small toy task the gap can be small or even negative)")
    print("\n✓ Knowledge distillation example completed!")


if __name__ == "__main__":
    main()
