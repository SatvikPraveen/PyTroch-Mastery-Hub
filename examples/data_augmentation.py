"""
Example: Data Augmentation (MixUp, CutMix, Mosaic)
====================================================
Demonstrates the computer vision augmentation techniques from
src.computer_vision.augmentation.
Run: python examples/data_augmentation.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script execution
import matplotlib.pyplot as plt

from src.computer_vision.augmentation import MixUp, CutMix, Mosaic


def show_comparison(original, augmented_dict, save_path="outputs/augmentation_demo.png"):
    """Save a side-by-side comparison of augmentation techniques."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = 1 + len(augmented_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    # Original (take the first image, clamp to [0,1])
    img = original[0].permute(1, 2, 0).clamp(0, 1).numpy()
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, (name, aug) in enumerate(augmented_dict.items(), start=1):
        img = aug[0].permute(1, 2, 0).clamp(0, 1).numpy()
        axes[i].imshow(img)
        axes[i].set_title(name)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    return save_path


def main():
    print("=" * 60)
    print("PyTorch Mastery Hub — Data Augmentation Demo")
    print("=" * 60)

    # Synthetic batch: 8 images of size 3×32×32, 4 classes
    batch_size, num_classes = 8, 4
    images = torch.rand(batch_size, 3, 32, 32)
    labels = torch.randint(0, num_classes, (batch_size,))

    print(f"\nInput batch: {images.shape}, labels: {labels.tolist()}")

    # ── MixUp ────────────────────────────────────────────────────
    print("\n── MixUp (α=0.4) ────────────────────────────────────────")
    mixup = MixUp(alpha=0.4, num_classes=num_classes)
    mixed_images, mixed_labels = mixup(images.clone(), labels.clone())
    print(f"  Output images shape : {mixed_images.shape}")
    print(f"  Output labels shape : {mixed_labels.shape}  (soft labels)")
    print(f"  Labels sum per sample: {mixed_labels.sum(dim=1).tolist()}")  # should be ~1

    # ── CutMix ───────────────────────────────────────────────────
    print("\n── CutMix (α=1.0) ───────────────────────────────────────")
    cutmix = CutMix(alpha=1.0, num_classes=num_classes)
    cut_images, cut_labels = cutmix(images.clone(), labels.clone())
    print(f"  Output images shape : {cut_images.shape}")
    print(f"  Output labels shape : {cut_labels.shape}  (soft labels)")

    # ── Mosaic ───────────────────────────────────────────────────
    print("\n── Mosaic (4-image combination) ─────────────────────────")
    mosaic = Mosaic(num_classes=num_classes)
    mosaic_images, mosaic_labels = mosaic(images.clone(), labels.clone())
    print(f"  Output images shape : {mosaic_images.shape}")
    print(f"  Output labels shape : {mosaic_labels.shape}  (soft labels)")

    # ── Save visual comparison ────────────────────────────────────
    print("\n── Saving visual comparison ─────────────────────────────")
    try:
        path = show_comparison(
            original=images,
            augmented_dict={
                "MixUp": mixed_images,
                "CutMix": cut_images,
                "Mosaic": mosaic_images,
            },
        )
        print(f"  Saved to: {path}")
    except Exception as e:
        print(f"  (Visualization skipped: {e})")

    print("\n✓ Data augmentation example completed!")


if __name__ == "__main__":
    main()
