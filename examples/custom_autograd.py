"""
Example: Custom Autograd Functions
=====================================
Demonstrates how to implement and use custom backward passes via
torch.autograd.Function. Uses helpers from src.fundamentals.autograd_helpers.
Run: python examples/custom_autograd.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.fundamentals.autograd_helpers import (
    LinearFunction,
    ReLUFunction,
    SigmoidFunction,
    gradient_check,
    GradientClipping,
)


def demo_custom_linear():
    """Show custom linear forward/backward vs built-in."""
    print("\n── Custom LinearFunction ──────────────────────────────")
    x = torch.randn(4, 5, requires_grad=True)
    w = torch.randn(3, 5, requires_grad=True)
    b = torch.randn(3, requires_grad=True)

    out_custom = LinearFunction.apply(x, w, b)
    out_builtin = x @ w.t() + b

    match = torch.allclose(out_custom, out_builtin, atol=1e-5)
    print(f"  Output shape   : {out_custom.shape}")
    print(f"  Matches nn.Linear: {match} ✓")

    # Backward
    loss = out_custom.sum()
    loss.backward()
    print(f"  x.grad shape   : {x.grad.shape}")
    print(f"  w.grad shape   : {w.grad.shape}")


def demo_custom_relu():
    """Show custom ReLU forward/backward."""
    print("\n── Custom ReLUFunction ────────────────────────────────")
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

    out_custom = ReLUFunction.apply(x)
    out_builtin = torch.relu(x.detach().clone().requires_grad_(True))

    print(f"  Input          : {x.data.tolist()}")
    print(f"  Custom ReLU    : {out_custom.data.tolist()}")
    print(f"  Built-in ReLU  : {out_builtin.data.tolist()}")
    print(f"  Match          : {torch.allclose(out_custom, out_builtin)} ✓")

    out_custom.sum().backward()
    print(f"  Gradient       : {x.grad.tolist()}  (0 where x≤0, 1 elsewhere)")


def demo_gradient_check():
    """Numerical gradient checking."""
    print("\n── Gradient Check ─────────────────────────────────────")

    def simple_func(x):
        return (x ** 3).sum()

    x = torch.randn(4, requires_grad=True, dtype=torch.float64)
    passed = gradient_check(simple_func, x, eps=1e-5, tol=1e-4)
    print(f"  Gradient check passed: {passed} ✓")


def demo_gradient_clipping():
    """Gradient clipping during training."""
    print("\n── Gradient Clipping ──────────────────────────────────")
    model = nn.Linear(10, 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    clipper = GradientClipping(max_norm=1.0)

    x = torch.randn(32, 10)
    y = torch.randn(32, 5)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()

    # Check norms before and after clipping
    norm_before = sum(
        p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
    ) ** 0.5
    clipper.clip(model.parameters())
    norm_after = sum(
        p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
    ) ** 0.5

    print(f"  Gradient norm before clipping : {norm_before:.4f}")
    print(f"  Gradient norm after  clipping : {norm_after:.4f}")
    print(f"  Clipped to ≤ 1.0             : {norm_after <= 1.0 + 1e-5} ✓")


def main():
    print("=" * 60)
    print("PyTorch Mastery Hub — Custom Autograd Functions")
    print("=" * 60)

    demo_custom_linear()
    demo_custom_relu()
    demo_gradient_check()
    demo_gradient_clipping()

    print("\n✓ All autograd demos completed successfully!")


if __name__ == "__main__":
    main()
