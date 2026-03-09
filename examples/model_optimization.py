"""
Example: Model Quantization & Pruning
=======================================
Demonstrates post-training quantization and magnitude pruning using
src.advanced.optimization utilities to reduce model size.
Run: python examples/model_optimization.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.neural_networks.models import SimpleMLP
from src.advanced.optimization import ModelQuantizer, ModelPruner, calculate_sparsity


def model_size_kb(model: nn.Module) -> float:
    """Return approximate model size in KB."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / 1024


def evaluate_model(model, x, y):
    """Simple evaluation on a fixed batch."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item()
    return acc


def main():
    print("=" * 60)
    print("PyTorch Mastery Hub — Model Quantization & Pruning")
    print("=" * 60)

    # ── Baseline model ────────────────────────────────────────────
    model = SimpleMLP(input_dim=784, hidden_dims=[512, 256], output_dim=10)
    x_sample = torch.randn(64, 784)
    y_sample = torch.randint(0, 10, (64,))

    base_acc = evaluate_model(model, x_sample, y_sample)
    base_size = model_size_kb(model)
    print(f"\nBaseline Model")
    print(f"  Size     : {base_size:.1f} KB")
    print(f"  Accuracy : {base_acc:.2%}  (random weights, for illustration)")

    # ── Dynamic quantization ─────────────────────────────────────
    print("\n── Dynamic Quantization ─────────────────────────────────")
    quantizer = ModelQuantizer(model)
    quant_model = quantizer.dynamic_quantize()
    quant_size = model_size_kb(quant_model)
    quant_acc = evaluate_model(quant_model, x_sample, y_sample)
    print(f"  Quantized size   : {quant_size:.1f} KB")
    print(f"  Size reduction   : {100*(1 - quant_size/base_size):.1f}%")
    print(f"  Accuracy         : {quant_acc:.2%}")

    # ── Magnitude pruning ────────────────────────────────────────
    print("\n── Magnitude Pruning (50% sparsity) ─────────────────────")
    # Work on a fresh model copy
    prunable = SimpleMLP(input_dim=784, hidden_dims=[512, 256], output_dim=10)
    pruner = ModelPruner(prunable)
    pruner.magnitude_prune(sparsity=0.5)

    sparsity = calculate_sparsity(prunable)
    pruned_acc = evaluate_model(prunable, x_sample, y_sample)
    print(f"  Overall sparsity : {sparsity:.2%}")
    print(f"  Accuracy         : {pruned_acc:.2%}")

    # ── Structured pruning ───────────────────────────────────────
    print("\n── Structured Pruning (30% channels) ────────────────────")
    struct_model = SimpleMLP(input_dim=784, hidden_dims=[512, 256], output_dim=10)
    struct_pruner = ModelPruner(struct_model)
    struct_pruner.structured_prune(sparsity=0.3)

    struct_sparsity = calculate_sparsity(struct_model)
    struct_acc = evaluate_model(struct_model, x_sample, y_sample)
    print(f"  Overall sparsity : {struct_sparsity:.2%}")
    print(f"  Accuracy         : {struct_acc:.2%}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n── Optimization Summary ─────────────────────────────────")
    print(f"  {'Method':<25} {'Size (KB)':>10} {'Accuracy':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Baseline':<25} {base_size:>10.1f} {base_acc:>10.2%}")
    print(f"  {'Dynamic Quantization':<25} {quant_size:>10.1f} {quant_acc:>10.2%}")
    print(f"  {'Magnitude Pruning 50%':<25} {model_size_kb(prunable):>10.1f} {pruned_acc:>10.2%}")
    print(f"  {'Structured Pruning 30%':<25} {model_size_kb(struct_model):>10.1f} {struct_acc:>10.2%}")

    print("\n✓ Model optimization example completed!")


if __name__ == "__main__":
    main()
