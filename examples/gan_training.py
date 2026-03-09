"""
Example: GAN Training Loop
============================
Demonstrates training a simple GAN using src.advanced.gan_utils.
Run: python examples/gan_training.py [--epochs 5]
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.advanced.gan_utils import Generator, Discriminator, GANTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="GAN Training Example")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    return parser.parse_args()


def make_synthetic_dataset(n_samples=1000, data_dim=64):
    """Simulate a simple 1D image-like dataset."""
    # Real samples: mix of two Gaussians (simulating diverse real data)
    x = torch.cat([
        torch.randn(n_samples // 2, data_dim) + 2,
        torch.randn(n_samples // 2, data_dim) - 2,
    ])
    return DataLoader(TensorDataset(x), batch_size=64, shuffle=True)


def main():
    args = parse_args()
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("=" * 60)
    print("PyTorch Mastery Hub — GAN Training Example")
    print("=" * 60)
    print(f"\nDevice      : {device}")
    print(f"Epochs      : {args.epochs}")
    print(f"Latent dim  : {args.latent_dim}")

    data_dim = 64
    dataloader = make_synthetic_dataset(n_samples=1000, data_dim=data_dim)

    # ── Models ───────────────────────────────────────────────────
    generator = Generator(
        latent_dim=args.latent_dim,
        output_dim=data_dim,
    ).to(device)
    discriminator = Discriminator(
        input_dim=data_dim,
    ).to(device)

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"\nGenerator params     : {g_params:,}")
    print(f"Discriminator params : {d_params:,}")

    # ── Trainer ──────────────────────────────────────────────────
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        latent_dim=args.latent_dim,
        lr=args.lr,
        device=device,
    )

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        g_losses, d_losses = [], []

        for (real_batch,) in dataloader:
            real_batch = real_batch.to(device)
            d_loss = trainer.train_discriminator_step(real_batch)
            g_loss = trainer.train_generator_step(real_batch.size(0))
            d_losses.append(d_loss)
            g_losses.append(g_loss)

        avg_g = sum(g_losses) / len(g_losses)
        avg_d = sum(d_losses) / len(d_losses)
        print(f"  Epoch {epoch}/{args.epochs}  |  G_loss: {avg_g:.4f}  |  D_loss: {avg_d:.4f}")

    # ── Sample from trained generator ────────────────────────────
    print("\nGenerating samples from trained generator...")
    generator.eval()
    with torch.no_grad():
        z = torch.randn(4, args.latent_dim, device=device)
        fake_samples = generator(z)
    print(f"  Generated sample shape : {fake_samples.shape}")
    print(f"  Mean  : {fake_samples.mean().item():.4f}")
    print(f"  Std   : {fake_samples.std().item():.4f}")

    print("\n✓ GAN training example completed!")


if __name__ == "__main__":
    main()
