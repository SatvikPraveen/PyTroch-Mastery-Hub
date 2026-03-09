"""
Example: Model Checkpointing
==============================
Demonstrates saving/loading model checkpoints, managing training state,
and resuming interrupted training using src.utils.io_utils.
Run: python examples/model_checkpointing.py
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.neural_networks.models import SimpleMLP
from src.utils.io_utils import save_model, load_model, ModelCheckpointManager


def main():
    print("=" * 60)
    print("PyTorch Mastery Hub — Model Checkpointing")
    print("=" * 60)

    device = torch.device("cpu")
    checkpoint_dir = tempfile.mkdtemp(prefix="pytorch_hub_checkpoints_")
    print(f"\nCheckpoint directory: {checkpoint_dir}")

    # ── Setup ─────────────────────────────────────────────────────
    model = SimpleMLP(input_dim=20, hidden_dims=[64, 32], output_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x_data = torch.randn(100, 20)
    y_data = torch.randint(0, 3, (100,))
    loader = DataLoader(TensorDataset(x_data, y_data), batch_size=16, shuffle=True)

    # ── Checkpoint manager ────────────────────────────────────────
    ckpt_manager = ModelCheckpointManager(
        checkpoint_dir=checkpoint_dir,
        model_name="mlp_demo",
        max_checkpoints=3,       # Keep last 3 checkpoints
    )

    # ── Training with checkpoints ─────────────────────────────────
    print("\nTraining for 5 epochs with checkpointing...")
    best_loss = float("inf")

    for epoch in range(1, 6):
        model.train()
        epoch_loss = 0.0
        for x_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(x_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        is_best = avg_loss < best_loss
        best_loss = min(best_loss, avg_loss)

        # Save checkpoint each epoch
        ckpt_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=avg_loss,
            is_best=is_best,
        )
        print(f"  Epoch {epoch}/5  loss: {avg_loss:.4f}  "
              f"{'★ BEST' if is_best else '      '}")

    print(f"\nSaved checkpoints: {ckpt_manager.list_checkpoints()}")

    # ── Load the best checkpoint ─────────────────────────────────
    print("\nLoading best checkpoint...")
    new_model = SimpleMLP(input_dim=20, hidden_dims=[64, 32], output_dim=3)
    loaded_epoch = ckpt_manager.load_best(new_model)
    print(f"  Loaded from epoch: {loaded_epoch}")

    # Verify weights match
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), new_model.named_parameters()):
        if n1 == n2 and not torch.allclose(p1.data, p2.data):
            print(f"  WARNING: {n1} weights differ!")

    # ── Simple save / load ────────────────────────────────────────
    print("\nSimple model save/load...")
    save_path = os.path.join(checkpoint_dir, "final_model.pth")
    save_model(model, save_path)
    print(f"  Saved to: {save_path}")

    restored = SimpleMLP(input_dim=20, hidden_dims=[64, 32], output_dim=3)
    load_model(restored, save_path)

    # Test restored model gives identical output
    model.eval()
    restored.eval()
    with torch.no_grad():
        x_test = torch.randn(4, 20)
        out_original = model(x_test)
        out_restored = restored(x_test)

    match = torch.allclose(out_original, out_restored, atol=1e-5)
    print(f"  Outputs match after restore: {match} ✓")

    print("\n✓ Checkpointing example completed!")


if __name__ == "__main__":
    main()
