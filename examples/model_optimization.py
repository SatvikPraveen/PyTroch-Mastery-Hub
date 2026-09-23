"""
Example: Model Quantization & Pruning
=======================================
Demonstrates post-training quantization and magnitude / global / structured
pruning using pytorch_mastery_hub.advanced.optimization utilities.
Run: python examples/model_optimization.py
"""

from __future__ import annotations

import copy
import io
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import torch.nn as nn

from pytorch_mastery_hub.advanced.optimization import ModelPruner, ModelQuantizer
from pytorch_mastery_hub.neural_networks.models import SimpleMLP
from pytorch_mastery_hub.utils.model_utils import count_parameters
from pytorch_mastery_hub.utils.reproducibility import seed_everything

INPUT_SIZE, NUM_CLASSES = 784, 10


def serialized_size_kb(model: nn.Module) -> float:
    """Size of the model's state_dict on disk (what deployment cares about)."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1024


def make_data(n: int, centers: torch.Tensor):
    """Toy task: one Gaussian blob per class (MNIST-sized inputs, 784 features)."""
    y = torch.randint(0, NUM_CLASSES, (n,))
    x = centers[y] + torch.randn(n, INPUT_SIZE)
    return x, y


def evaluate_model(model, x, y) -> float:
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(dim=-1) == y).float().mean().item()


def main():
    print("=" * 60)
    print("PyTorch Mastery Hub — Model Quantization & Pruning")
    print("=" * 60)
    seed_everything(0)

    # ── Baseline model (quick-trained on CPU so accuracy is meaningful) ──
    centers = 0.2 * torch.randn(NUM_CLASSES, INPUT_SIZE)
    x_train, y_train = make_data(4096, centers)
    x_test, y_test = make_data(512, centers)
    model = SimpleMLP(INPUT_SIZE, hidden_sizes=[256, 128], output_size=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(5):
        for i in range(0, len(x_train), 64):
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(x_train[i : i + 64]), y_train[i : i + 64]).backward()
            optimizer.step()

    base_acc = evaluate_model(model, x_test, y_test)
    base_size = serialized_size_kb(model)
    print("\nBaseline Model")
    print(f"  Params   : {count_parameters(model):,}")
    print(f"  Size     : {base_size:.1f} KB")
    print(f"  Accuracy : {base_acc:.1%}")

    # ── Dynamic quantization (int8 weights for nn.Linear, CPU only) ──
    print("\n── Dynamic Quantization (int8) ──────────────────────────")
    # PyTorch needs a quantized kernel backend: fbgemm (x86) or qnnpack (ARM/mobile).
    engines = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    if torch.backends.quantized.engine == "none" and engines:
        torch.backends.quantized.engine = engines[0]
    print(f"  Quantized engine : {torch.backends.quantized.engine}")
    warnings.filterwarnings("ignore", message=".*quantize_per_tensor.*")  # torch deprecation note
    try:
        quant_model = ModelQuantizer(copy.deepcopy(model)).dynamic_quantize()
        quant_size = serialized_size_kb(quant_model)
        quant_acc = evaluate_model(quant_model, x_test, y_test)
        print(f"  Quantized size   : {quant_size:.1f} KB")
        print(f"  Size reduction   : {100 * (1 - quant_size / base_size):.1f}%")
        print(f"  Accuracy         : {quant_acc:.1%}")
    except RuntimeError as e:  # no int8 backend on this build
        print(f"  Skipped: {e}")
        quant_size, quant_acc = float("nan"), float("nan")

    # ── Magnitude pruning: drop the 50% smallest weights of every layer ──
    print("\n── Magnitude Pruning (50% per layer) ────────────────────")
    prunable = copy.deepcopy(model)
    pruner = ModelPruner(prunable)
    pruner.magnitude_pruning(amount=0.5)
    sparsity = pruner.get_sparsity()  # per-module + "global_sparsity"
    pruned_acc = evaluate_model(prunable, x_test, y_test)
    for name, value in sparsity.items():
        print(f"  {name:<18}: {value:.1%} zeros")
    print(f"  Accuracy         : {pruned_acc:.1%}")
    pruner.remove_pruning_masks()  # bake the mask into the weights
    print(
        f"  Size after mask removal: {serialized_size_kb(prunable):.1f} KB "
        "(zeros are still stored; savings need sparse kernels/formats)"
    )

    # ── Global pruning: one threshold across all layers ───────────
    print("\n── Global Pruning (30% of all weights) ──────────────────")
    global_model = copy.deepcopy(model)
    global_pruner = ModelPruner(global_model)
    global_pruner.global_pruning(amount=0.3)
    global_sparsity = global_pruner.get_sparsity()
    global_acc = evaluate_model(global_model, x_test, y_test)
    for name, value in global_sparsity.items():
        print(f"  {name:<18}: {value:.1%} zeros")
    print(f"  Accuracy         : {global_acc:.1%}  (note: per-layer sparsity differs)")

    # ── Structured pruning: removes whole output channels of Conv2d ─
    print("\n── Structured Pruning (50% of conv channels) ────────────")
    cnn = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, NUM_CLASSES),
    )
    ModelPruner(cnn).structured_pruning(amount=0.5)
    conv2 = cnn[2]
    dead_channels = int((conv2.weight.abs().sum(dim=(1, 2, 3)) == 0).sum())
    print(f"  conv2 output channels zeroed: {dead_channels}/{conv2.out_channels}")
    print("  (whole filters removed → real speed-ups on dense hardware)")

    # ── Summary ──────────────────────────────────────────────────
    print("\n── Optimization Summary ─────────────────────────────────")
    print(f"  {'Method':<25} {'Size (KB)':>10} {'Sparsity':>10} {'Accuracy':>10}")
    print(f"  {'-' * 58}")
    rows = [
        ("Baseline", base_size, 0.0, base_acc),
        ("Dynamic Quantization", quant_size, 0.0, quant_acc),
        ("Magnitude Pruning 50%", base_size, sparsity["global_sparsity"], pruned_acc),
        ("Global Pruning 30%", base_size, global_sparsity["global_sparsity"], global_acc),
    ]
    for name, size, sp, acc in rows:
        print(f"  {name:<25} {size:>10.1f} {sp:>10.1%} {acc:>10.1%}")

    print("\n✓ Model optimization example completed!")


if __name__ == "__main__":
    main()
