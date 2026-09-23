"""
Example: GAN Training Loop
============================
Demonstrates training a simple GAN using pytorch_mastery_hub.advanced.gan_utils.
Run: python examples/gan_training.py [--epochs 5] [--gan-type vanilla|lsgan]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_mastery_hub.advanced.gan_utils import Discriminator, GANTrainer, Generator
from pytorch_mastery_hub.utils.device_utils import get_device
from pytorch_mastery_hub.utils.model_utils import count_parameters
from pytorch_mastery_hub.utils.reproducibility import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="GAN Training Example")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--noise-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gan-type", choices=["vanilla", "lsgan"], default="vanilla")
    return parser.parse_args()


def make_synthetic_dataset(n_samples=1024, data_dim=64, batch_size=64):
    """Real data: two Gaussian blobs, scaled into the generator's tanh range [-1, 1]."""
    half = n_samples // 2
    x = torch.cat(
        [
            0.5 + 0.15 * torch.randn(half, data_dim),
            -0.5 + 0.15 * torch.randn(half, data_dim),
        ]
    ).clamp(-1, 1)
    # drop_last: the generator uses BatchNorm, which needs more than one sample per batch
    return DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True, drop_last=True)


def main():
    args = parse_args()
    seed_everything(0)
    device = get_device()
    print("=" * 60)
    print("PyTorch Mastery Hub — GAN Training Example")
    print("=" * 60)
    print(f"\nDevice      : {device}")
    print(f"GAN type    : {args.gan_type}")
    print(f"Epochs      : {args.epochs}")
    print(f"Noise dim   : {args.noise_dim}")

    data_dim = 64
    dataloader = make_synthetic_dataset(data_dim=data_dim, batch_size=args.batch_size)
    real_all = dataloader.dataset.tensors[0]

    # ── Models ───────────────────────────────────────────────────
    generator = Generator(noise_dim=args.noise_dim, output_dim=data_dim, hidden_dims=[128, 256])
    discriminator = Discriminator(input_dim=data_dim, hidden_dims=[256, 128])
    generator, discriminator = generator.to(device), discriminator.to(device)
    print(f"\nGenerator params     : {count_parameters(generator):,}")
    print(f"Discriminator params : {count_parameters(discriminator):,}")

    # ── Trainer ──────────────────────────────────────────────────
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        device=device,
        gan_type=args.gan_type,
    )

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        g_losses, d_losses = [], []
        for (real_batch,) in dataloader:
            real_batch = real_batch.to(device)
            noise = torch.randn(real_batch.size(0), args.noise_dim, device=device)
            d_loss, g_loss = trainer.train_step(real_batch, noise)  # one D step + one G step
            d_losses.append(d_loss)
            g_losses.append(g_loss)

        avg_g = sum(g_losses) / len(g_losses)
        avg_d = sum(d_losses) / len(d_losses)
        print(f"  Epoch {epoch}/{args.epochs}  |  G_loss: {avg_g:.4f}  |  D_loss: {avg_d:.4f}")

    # ── Sample from trained generator ────────────────────────────
    print("\nGenerating samples from trained generator...")
    generator.eval()
    with torch.no_grad():
        z = torch.randn(256, args.noise_dim, device=device)
        fake_samples = generator(z).cpu()
    print(f"  Generated sample shape : {tuple(fake_samples.shape)}")
    print(f"  Real  data  mean / std : {real_all.mean():+.3f} / {real_all.std():.3f}")
    print(f"  Fake  data  mean / std : {fake_samples.mean():+.3f} / {fake_samples.std():.3f}")
    print(f"  Fake samples in [-1, 1]: {bool(fake_samples.abs().max() <= 1.0)} (tanh output)")

    print("\n✓ GAN training example completed!")


if __name__ == "__main__":
    main()
