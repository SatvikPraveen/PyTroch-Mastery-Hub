"""
Example: Model Checkpointing
==============================
Demonstrates saving/loading model checkpoints, managing training state,
and resuming interrupted training using pytorch_mastery_hub.utils.io_utils
and the resumable pytorch_mastery_hub.neural_networks.training.Trainer.
Run: python examples/model_checkpointing.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_mastery_hub.neural_networks.models import SimpleMLP
from pytorch_mastery_hub.neural_networks.training import Trainer, TrainerConfig, train_epoch
from pytorch_mastery_hub.utils.io_utils import (
    ModelCheckpointManager,
    load_checkpoint,
    load_model,
    save_model,
)
from pytorch_mastery_hub.utils.reproducibility import seed_everything

DEVICE = "cpu"  # tiny model; keeps the saved files device-agnostic


def make_model():
    return SimpleMLP(input_size=20, hidden_sizes=[64, 32], output_size=3)


def main():
    print("=" * 60)
    print("PyTorch Mastery Hub — Model Checkpointing")
    print("=" * 60)
    seed_everything(0)

    checkpoint_dir = tempfile.mkdtemp(prefix="pytorch_hub_checkpoints_")
    print(f"\nCheckpoint directory: {checkpoint_dir}")

    # ── Setup ─────────────────────────────────────────────────────
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x_data = torch.randn(100, 20)
    y_data = torch.randint(0, 3, (100,))
    loader = DataLoader(TensorDataset(x_data, y_data), batch_size=16, shuffle=True)

    # ── 1. ModelCheckpointManager: one checkpoint per epoch, keep the last 3 ──
    ckpt_manager = ModelCheckpointManager(
        checkpoint_dir=checkpoint_dir, max_checkpoints=3, monitor="loss", mode="min"
    )

    print("\n1. Training for 5 epochs with ModelCheckpointManager...")
    for epoch in range(1, 6):
        metrics = train_epoch(model, loader, criterion, optimizer, DEVICE)
        path = ckpt_manager.save_checkpoint(model, optimizer, epoch=epoch, metrics=metrics)
        print(f"  Epoch {epoch}/5  loss: {metrics['loss']:.4f}  → {path.name}")

    print(f"\n  Files on disk : {sorted(p.name for p in ckpt_manager.checkpoint_dir.iterdir())}")
    print(
        "  (max_checkpoints=3 only prunes non-best files; every epoch improved here, so all stay)"
    )
    print(f"  Best checkpoint  : {ckpt_manager.get_best_checkpoint().name}")
    print(f"  Latest checkpoint: {ckpt_manager.get_latest_checkpoint().name}")

    # ── 2. Load the best checkpoint into a fresh model ────────────
    print("\n2. Loading best checkpoint into a fresh model...")
    new_model = make_model()
    ckpt = load_checkpoint(ckpt_manager.get_best_checkpoint(), model=new_model)
    print(f"  Loaded from epoch {ckpt['epoch']}  (loss {ckpt['loss']:.4f})")

    same = all(torch.allclose(p1, p2) for p1, p2 in zip(model.parameters(), new_model.parameters()))
    print(f"  Weights match the trained model: {same} ✓")

    # ── 3. Simple save / load ─────────────────────────────────────
    print("\n3. Simple model save/load (save_model / load_model)...")
    save_path = os.path.join(checkpoint_dir, "final_model.pth")
    save_model(model, save_path, metadata={"epochs": 5, "note": "checkpointing demo"})
    print(f"  Saved to: {save_path}")

    restored = make_model()
    info = load_model(restored, save_path)
    print(f"  Metadata: {info.get('metadata')}")

    model.eval()
    restored.eval()
    with torch.no_grad():
        x_test = torch.randn(4, 20)
        match = torch.allclose(model(x_test), restored(x_test), atol=1e-5)
    print(f"  Outputs match after restore: {match} ✓")

    # ── 4. Resuming an interrupted Trainer run ────────────────────
    print("\n4. Resuming training with Trainer.save_checkpoint / load_checkpoint...")
    resume_path = os.path.join(checkpoint_dir, "trainer_state.pt")

    first_model = make_model()
    trainer = Trainer(
        first_model,
        criterion,
        torch.optim.Adam(first_model.parameters(), lr=1e-3),
        DEVICE,
        config=TrainerConfig(epochs=2),
    )
    trainer.fit(loader)  # "interrupted" after 2 epochs
    trainer.save_checkpoint(resume_path, extra={"note": "stopped after epoch 2"})
    print(
        f"  Saved trainer state at epoch {trainer.current_epoch} → {os.path.basename(resume_path)}"
    )

    fresh_model = make_model()
    resumed = Trainer(
        fresh_model,
        criterion,
        torch.optim.Adam(fresh_model.parameters(), lr=1e-3),
        DEVICE,
        config=TrainerConfig(epochs=4),
    )
    extra = resumed.load_checkpoint(resume_path)
    print(f"  Restored ({extra['note']}); continuing from epoch {resumed.current_epoch + 1}...")
    history = resumed.fit(loader)  # runs only epochs 3 and 4
    print(f"  History now covers {len(history['loss'])} epochs (2 before + 2 after resume)")

    print("\n✓ Checkpointing example completed!")


if __name__ == "__main__":
    main()
