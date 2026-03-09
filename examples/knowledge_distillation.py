"""
Example: Knowledge Distillation
=================================
Demonstrates transferring knowledge from a large teacher model to a
smaller student model using src.advanced.optimization.KnowledgeDistillation.
Run: python examples/knowledge_distillation.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.neural_networks.models import SimpleMLP
from src.advanced.optimization import KnowledgeDistillation
from src.utils.metrics import accuracy


def model_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    print("=" * 60)
    print("PyTorch Mastery Hub — Knowledge Distillation")
    print("=" * 60)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"\nDevice: {device}")

    # ── Synthetic dataset ─────────────────────────────────────────
    x_data = torch.randn(500, 64)
    y_data = torch.randint(0, 5, (500,))
    split = 400
    train_loader = DataLoader(
        TensorDataset(x_data[:split], y_data[:split]), batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(x_data[split:], y_data[split:]), batch_size=32
    )

    # ── Teacher model (large) ─────────────────────────────────────
    teacher = SimpleMLP(input_dim=64, hidden_dims=[256, 256, 128], output_dim=5).to(device)
    print(f"\nTeacher params : {model_param_count(teacher):,}")

    # Quick-train the teacher
    t_optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    teacher.train()
    for _ in range(10):
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            t_optimizer.zero_grad()
            criterion(teacher(x_b), y_b).backward()
            t_optimizer.step()

    # Evaluate teacher
    teacher.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x_b, y_b in val_loader:
            preds.append(teacher(x_b.to(device)).argmax(dim=-1).cpu())
            targets.append(y_b)
    teacher_acc = accuracy(torch.cat(preds), torch.cat(targets))
    print(f"Teacher accuracy: {teacher_acc:.2%}")

    # ── Student model (small) ─────────────────────────────────────
    student_scratch = SimpleMLP(input_dim=64, hidden_dims=[64], output_dim=5).to(device)
    student_distill = SimpleMLP(input_dim=64, hidden_dims=[64], output_dim=5).to(device)
    print(f"\nStudent params : {model_param_count(student_scratch):,}  "
          f"({100*model_param_count(student_scratch)/model_param_count(teacher):.1f}% of teacher)")

    # ── Train student from scratch ────────────────────────────────
    print("\nTraining student from scratch (5 epochs)...")
    s_optimizer = torch.optim.Adam(student_scratch.parameters(), lr=1e-3)
    for epoch in range(1, 6):
        student_scratch.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            s_optimizer.zero_grad()
            criterion(student_scratch(x_b), y_b).backward()
            s_optimizer.step()

    student_scratch.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x_b, y_b in val_loader:
            preds.append(student_scratch(x_b.to(device)).argmax(dim=-1).cpu())
            targets.append(y_b)
    scratch_acc = accuracy(torch.cat(preds), torch.cat(targets))
    print(f"Student (scratch) accuracy : {scratch_acc:.2%}")

    # ── Train student with knowledge distillation ─────────────────
    print("\nTraining student via knowledge distillation (5 epochs)...")
    kd = KnowledgeDistillation(
        teacher=teacher,
        student=student_distill,
        temperature=4.0,
        alpha=0.5,         # blend factor: 0.5 CE + 0.5 KD loss
    )
    d_optimizer = torch.optim.Adam(student_distill.parameters(), lr=1e-3)

    for epoch in range(1, 6):
        student_distill.train()
        epoch_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            d_optimizer.zero_grad()
            loss = kd.distillation_loss(x_b, y_b)
            loss.backward()
            d_optimizer.step()
            epoch_loss += loss.item()

    student_distill.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x_b, y_b in val_loader:
            preds.append(student_distill(x_b.to(device)).argmax(dim=-1).cpu())
            targets.append(y_b)
    distill_acc = accuracy(torch.cat(preds), torch.cat(targets))
    print(f"Student (distilled) accuracy : {distill_acc:.2%}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────")
    print(f"  {'Model':<30} {'Params':>8} {'Val Accuracy':>14}")
    print(f"  {'-'*52}")
    print(f"  {'Teacher (large)':<30} {model_param_count(teacher):>8,} {teacher_acc:>14.2%}")
    print(f"  {'Student (trained scratch)':<30} {model_param_count(student_scratch):>8,} "
          f"{scratch_acc:>14.2%}")
    print(f"  {'Student (distilled)':<30} {model_param_count(student_distill):>8,} "
          f"{distill_acc:>14.2%}")

    improvement = distill_acc - scratch_acc
    print(f"\n  Distillation improvement: {improvement:+.2%}")
    print("\n✓ Knowledge distillation example completed!")


if __name__ == "__main__":
    main()
