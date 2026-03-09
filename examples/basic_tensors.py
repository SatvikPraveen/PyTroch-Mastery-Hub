"""
Example: Basic PyTorch Tensor Operations
=========================================
Demonstrates the core tensor operations available in PyTorch Mastery Hub.
Run: python examples/basic_tensors.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.fundamentals.tensor_ops import safe_divide, batch_matrix_multiply, tensor_stats


def main():
    print("=" * 60)
    print("PyTorch Mastery Hub — Basic Tensor Operations")
    print("=" * 60)

    # ── 1. Tensor creation ──────────────────────────────────────
    print("\n1. Tensor Creation")
    zeros = torch.zeros(3, 4)
    ones = torch.ones(3, 4)
    rand = torch.rand(3, 4)
    randn = torch.randn(3, 4)

    print(f"   zeros shape : {zeros.shape}")
    print(f"   ones  shape : {ones.shape}")
    print(f"   rand  shape : {rand.shape}")
    print(f"   randn mean  : {randn.mean():.4f} (should be ~0)")

    # ── 2. Tensor statistics ─────────────────────────────────────
    print("\n2. Tensor Statistics (via src.fundamentals.tensor_ops)")
    x = torch.randn(100, 10)
    stats = tensor_stats(x)
    for key, val in stats.items():
        print(f"   {key:10s}: {val:.4f}")

    # ── 3. Safe division ─────────────────────────────────────────
    print("\n3. Safe Division (handles zero denominator)")
    numerator = torch.tensor([1.0, 2.0, 3.0, 4.0])
    denominator = torch.tensor([2.0, 0.0, 1.0, 0.0])  # contains zeros
    result = safe_divide(numerator, denominator)
    print(f"   numerator  : {numerator.tolist()}")
    print(f"   denominator: {denominator.tolist()}")
    print(f"   result     : {result.tolist()}")

    # ── 4. Batch matrix multiply ─────────────────────────────────
    print("\n4. Batch Matrix Multiply")
    batch_a = torch.randn(8, 4, 6)   # 8 matrices of shape 4×6
    batch_b = torch.randn(8, 6, 3)   # 8 matrices of shape 6×3
    result = batch_matrix_multiply(batch_a, batch_b)
    print(f"   input shapes : {batch_a.shape}, {batch_b.shape}")
    print(f"   output shape : {result.shape}  (expected: torch.Size([8, 4, 3]))")

    # ── 5. Device placement ──────────────────────────────────────
    print("\n5. Device Information")
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Best available device: {device}")
    t = torch.randn(3, 3).to(device)
    print(f"   Tensor on {t.device}")

    # ── 6. Autograd ──────────────────────────────────────────────
    print("\n6. Autograd Example")
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2 + 2 * x + 1   # y = (x+1)^2
    y.backward()
    print(f"   y  = x² + 2x + 1  at x=3  → y = {y.item():.1f}")
    print(f"   dy/dx at x=3      → {x.grad.item():.1f}  (expected: 8.0 = 2*3+2)")

    print("\n✓ All tensor operations completed successfully!")


if __name__ == "__main__":
    main()
