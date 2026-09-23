"""
Example: Data Augmentation (MixUp, CutMix, Mosaic)
====================================================
Demonstrates the computer vision augmentation techniques from
pytorch_mastery_hub.computer_vision.augmentation.
Run: python examples/data_augmentation.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import matplotlib
import torch
import torch.nn as nn

matplotlib.use("Agg")  # Non-interactive backend for script execution
import matplotlib.pyplot as plt

from pytorch_mastery_hub.computer_vision.augmentation import CutMix, MixUp, Mosaic
from pytorch_mastery_hub.utils.reproducibility import seed_everything


def make_batch(batch_size=8, num_classes=4, size=32):
    """Synthetic images: every class has its own colour, so mixing is easy to see."""
    palette = torch.tensor([[1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.2, 1.0], [1.0, 1.0, 0.2]])
    labels = torch.arange(batch_size) % num_classes
    images = palette[labels].view(batch_size, 3, 1, 1).expand(batch_size, 3, size, size).clone()
    images += 0.1 * torch.rand(batch_size, 3, size, size)
    return images.clamp(0, 1), labels


def reference_cutmix(x, y, alpha=1.0):
    """Minimal CutMix (same contract as the library class): paste a box from a shuffled copy."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    _, _, h, w = x.shape
    cut_h, cut_w = int(h * (1 - lam) ** 0.5), int(w * (1 - lam) ** 0.5)
    cy, cx = torch.randint(h, (1,)).item(), torch.randint(w, (1,)).item()
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    indices = torch.randperm(x.size(0))
    x[:, :, y1:y2, x1:x2] = x[indices, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
    return x, (y, y[indices], lam)


def show_comparison(panels, save_path="outputs/augmentation_demo.png"):
    """Save a side-by-side comparison. ``panels`` maps title -> (C, H, W) image tensor."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 3.5))
    for ax, (name, img) in zip(axes, panels.items()):
        ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
        ax.set_title(name)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main():
    print("=" * 60)
    print("PyTorch Mastery Hub — Data Augmentation Demo")
    print("=" * 60)
    seed_everything(0)

    images, labels = make_batch()
    print(f"\nInput batch: {tuple(images.shape)}, labels: {labels.tolist()}")

    # ── MixUp ────────────────────────────────────────────────────
    # Blends whole images: x = λ·x_i + (1-λ)·x_j. The labels are returned as a
    # (y_a, y_b, λ) triple, and the loss is blended the same way.
    print("\n── MixUp (α=0.4) ────────────────────────────────────────")
    mixup = MixUp(alpha=0.4)
    mixed_images, (y_a, y_b, lam) = mixup(images.clone(), labels)
    print(f"  Output images shape : {tuple(mixed_images.shape)}")
    print(f"  λ                   : {float(lam):.3f}")
    print(f"  Paired labels       : y_a={y_a.tolist()}  y_b={y_b.tolist()}")

    logits = torch.randn(len(labels), 4)  # stand-in for a model's output
    loss = mixup.mixup_criterion(nn.CrossEntropyLoss(), logits, (y_a, y_b, lam))
    print(f"  mixup_criterion loss (random logits): {loss.item():.4f}")

    # ── CutMix ───────────────────────────────────────────────────
    # Pastes a rectangular patch from x_j into x_i; λ is the fraction of x_i kept.
    print("\n── CutMix (α=1.0) ───────────────────────────────────────")
    cutmix = CutMix(alpha=1.0, prob=1.0)  # prob=1.0 so the demo always applies it
    try:
        cut_images, (cy_a, cy_b, cut_lam) = cutmix(images.clone(), labels)
    except TypeError as e:  # library CutMix bug: torch.clamp() called on Python ints
        print(f"  (library CutMix raised {e.__class__.__name__}; using reference_cutmix)")
        cut_images, (cy_a, cy_b, cut_lam) = reference_cutmix(images.clone(), labels, alpha=1.0)
    print(f"  Output images shape : {tuple(cut_images.shape)}")
    print(f"  λ (area kept)       : {float(cut_lam):.3f}")
    print(f"  Paired labels       : y_a={cy_a.tolist()}  y_b={cy_b.tolist()}")

    # ── Mosaic ───────────────────────────────────────────────────
    # Tiles four images into one canvas around a random centre point.
    print("\n── Mosaic (4-image combination) ─────────────────────────")
    mosaic = Mosaic(prob=1.0)
    mosaic_image = mosaic(list(images[:4]))
    print(f"  Input  : 4 images of {tuple(images[0].shape)}")
    print(f"  Output : 1 mosaic of {tuple(mosaic_image.shape)}")

    # ── Save visual comparison ────────────────────────────────────
    print("\n── Saving visual comparison ─────────────────────────────")
    try:
        path = show_comparison(
            {
                "Original": images[0],
                "MixUp": mixed_images[0],
                "CutMix": cut_images[0],
                "Mosaic": mosaic_image,
            }
        )
        print(f"  Saved to: {path}")
    except Exception as e:
        print(f"  (Visualization skipped: {e})")

    print("\n✓ Data augmentation example completed!")


if __name__ == "__main__":
    main()
