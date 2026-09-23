# PyTorch Mastery Hub

[![CI](https://github.com/SatvikPraveen/PyTroch-Mastery-Hub/actions/workflows/ci.yml/badge.svg)](https://github.com/SatvikPraveen/PyTroch-Mastery-Hub/actions/workflows/ci.yml)
[![CodeQL](https://github.com/SatvikPraveen/PyTroch-Mastery-Hub/actions/workflows/codeql.yml/badge.svg)](https://github.com/SatvikPraveen/PyTroch-Mastery-Hub/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **PyTorch learning and reference implementation hub**: 27 progressive Jupyter notebooks
plus an installable, typed, tested library (`pytorch_mastery_hub`) whose modules are small
enough to read end-to-end yet built the way production code is, with mixed precision,
EMA, grouped-query attention on fused kernels, LoRA, DDP helpers, resumable checkpoints,
and a CLI that runs the whole pipeline from a YAML file.

```bash
pip install -e ".[dev]"
pytorch-hub info                                           # device / version report
pytorch-hub train --config configs/train_synthetic.yaml    # full training run in seconds
pytest                                                     # 290 tests, property-based included
```

## What is inside

| Area | Module | Highlights |
|------|--------|------------|
| Training engine | `neural_networks.training` | `Trainer` with device-agnostic autocast (CUDA fp16/bf16, CPU bf16), gradient accumulation with partial-window flush, grad-norm clipping, per-step/per-epoch schedulers, `torch.compile`, typed `Callback` protocol, `EarlyStopping` that restores best weights, `ModelCheckpoint` that writes resumable state (model/optimizer/scheduler/scaler/EMA/RNG). |
| Weight averaging | `neural_networks.ema` | `ModelEMA` with warm-up decay, `swap()` context for evaluation, full `state_dict`. |
| Modern transformer | `neural_networks.attention` | `RMSNorm`, rotary embeddings (`RotaryEmbedding`, NTK scaling), `MultiHeadAttention` with multi-query / grouped-query heads on `scaled_dot_product_attention`, `KVCache`, `SwiGLU`, pre-norm `DecoderBlock`, `TransformerLM.generate()` with top-k / top-p / EOS. |
| Parameter-efficient fine-tuning | `advanced.lora` | `LoRALinear`, `apply_lora()` by glob, adapter-only `lora_state_dict()`, lossless `merge_lora()` / `unmerge_lora()`. |
| Distributed | `utils.distributed` | torchrun-aware `init_distributed()`, `reduce_dict()`, uneven `all_gather_tensors()`, `wrap_ddp()`, `main_process_first()`, `spawn()` for in-process multi-rank tests. The `Trainer` is DDP-aware. |
| Reproducibility & devices | `utils.reproducibility`, `utils.device_utils` | `seed_everything(deterministic=True)`, `isolated_rng()`, `get_device()` (CUDA > MPS > CPU with env override), `autocast_dtype()`, recursive `move_to_device()`. |
| Introspection | `utils.model_utils`, `utils.memory_utils` | Hook-based `model_summary()`, `count_parameters()`, `freeze()` by pattern, `param_groups_with_weight_decay()`, `MemoryTracker` across CPU/CUDA/MPS. |
| Logging | `utils.logging_utils` | Idempotent `setup_logger()` (safe to re-run in notebooks), `MetricsLogger` writing JSON lines with `to_dataframe()`. |
| Classic building blocks | `fundamentals`, `neural_networks.{layers,models,optimizers}`, `computer_vision`, `nlp`, `advanced.{optimization,gan_utils,deployment}` | Custom autograd functions and gradient checks, MLP/CNN/ResNet/RNN/LSTM/GRU/VAE/Seq2Seq, hand-written SGD/Adam/AdamW and schedulers, MixUp/CutMix/Mosaic, tokenizers and embeddings, quantization/pruning/distillation, TorchScript/ONNX export, GAN trainers. |

## Quick tour

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_mastery_hub import seed_everything, get_device
from pytorch_mastery_hub.neural_networks import (
    Trainer, TrainerConfig, EarlyStopping, ModelCheckpoint, SimpleMLP,
)
from pytorch_mastery_hub.utils.model_utils import param_groups_with_weight_decay

seed_everything(42)
x, y = torch.randn(2048, 32), torch.randint(0, 4, (2048,))
train = DataLoader(TensorDataset(x[:1600], y[:1600]), batch_size=64, shuffle=True)
val = DataLoader(TensorDataset(x[1600:], y[1600:]), batch_size=256)

model = SimpleMLP(32, [128, 64], 4, dropout=0.1)
optimizer = torch.optim.AdamW(param_groups_with_weight_decay(model, 0.01), lr=2e-3)

trainer = Trainer(
    model, nn.CrossEntropyLoss(), optimizer, get_device(),
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10),
    config=TrainerConfig(epochs=10, precision="auto", ema_decay=0.999, clip_grad_norm=1.0),
    callbacks=[EarlyStopping(patience=3), ModelCheckpoint("ckpt/best.pt")],
)
history = trainer.fit(train, val)          # dict of per-epoch curves
trainer.load_checkpoint("ckpt/best.pt")    # resumable: optimizer, scaler, EMA, RNG, step
```

A LLaMA-style language model with grouped-query attention, KV-cached generation, and LoRA:

```python
from pytorch_mastery_hub.neural_networks.attention import TransformerConfig, TransformerLM
from pytorch_mastery_hub.advanced.lora import apply_lora, mark_only_lora_trainable, merge_lora

lm = TransformerLM(TransformerConfig(vocab_size=8000, d_model=256, num_layers=6, num_heads=8, num_kv_heads=2))
apply_lora(lm, ["q_proj", "v_proj"], r=8, alpha=16)
print(mark_only_lora_trainable(lm), "trainable parameters")   # ~1% of the model
# ... fine-tune ...
merge_lora(lm)                                                  # zero inference overhead
tokens = lm.generate(prompt_ids, max_new_tokens=64, temperature=0.8, top_p=0.9)
```

Multi-GPU with `torchrun --nproc_per_node=4 train.py` needs no code changes beyond:

```python
from pytorch_mastery_hub.utils import distributed as du
du.init_distributed()
model = du.wrap_ddp(model.to(du.local_device()))
loader = DataLoader(ds, sampler=du.make_sampler(ds), batch_size=64)
```

## Command-line interface

```
pytorch-hub info                     environment, device, CUDA/MPS/bf16 support
pytorch-hub train [--config y.yaml]  YAML/flag-driven training with AMP, EMA, early stopping, checkpoints
pytorch-hub benchmark                matmul TFLOP/s and fused-attention throughput per device/dtype
pytorch-hub test [--module nlp]      run the test suite
pytorch-hub download-data            fetch MNIST / CIFAR
```

`train` writes `train.log`, `metrics.jsonl`, `best.pt` and `summary.json` to `--output-dir`;
see `configs/` for reference configurations.

## Notebooks (learning path)

| # | Section | Notebooks |
|---|---------|-----------|
| 1 | `01_fundamentals/` | tensors, autograd, custom `Function`s, backprop visualisation |
| 2 | `02_neural_networks/` | MLP from scratch, architectures, training techniques |
| 3 | `03_computer_vision/` | CNN fundamentals, modern CNNs, projects |
| 4 | `04_natural_language_processing/` | RNN/LSTM, seq2seq, sentiment, transformer from scratch |
| 5 | `05_generative_models/` | GANs, advanced GANs & VAEs |
| 6 | `06_optimization_deployment/` | quantization/pruning, serving, MLOps, cloud |
| 7 | `07_advanced_projects/` | image classification, text generation, recommenders |
| 8 | `08_advanced_topics/` | advanced techniques, research applications |
| 9 | `capstone_projects/` | multimodal system, production MLOps |

```bash
pip install -e ".[notebooks]" && jupyter lab notebooks/
```

Runnable, dependency-light scripts for each topic live in [`examples/`](examples/README.md).

## Repository layout

```
src/pytorch_mastery_hub/   installable library (src layout, py.typed)
  fundamentals/            tensor ops, autograd helpers, math utilities
  neural_networks/         training engine, EMA, modern attention, layers, models, optimizers
  computer_vision/         models, datasets, transforms, augmentation
  nlp/                     tokenization, embeddings, models, text utils
  advanced/                LoRA, quantization/pruning/distillation, deployment, GANs
  utils/                   device, reproducibility, logging, model/memory utils, distributed, CLI
tests/                     unit, integration, property-based (Hypothesis) and distributed tests
examples/                  standalone scripts (executed in CI)
configs/                   reference configs for `pytorch-hub train`
notebooks/                 27 tutorial notebooks
docs/                      Sphinx documentation with generated API reference
```

## Development

```bash
pip install -e ".[dev]" && pre-commit install
make lint            # ruff check (lint + import order)
make format          # ruff format + autofix
make type-check      # mypy (new modules are strictly typed; legacy modules are exempt via overrides)
make test            # pytest; `pytest -m "not slow"` skips the multi-process test
make docs            # Sphinx HTML with API reference
```

CI runs lint/type/security gates first, then tests on Python 3.10-3.13 (Linux) plus
macOS/Windows, executes every example script, builds and smoke-installs the wheel, and
builds the docs. Tagging `vX.Y.Z` triggers the release workflow (PyPI Trusted Publishing
+ GitHub release from the changelog).

## Requirements

- Python 3.10+
- PyTorch 2.x (CPU, CUDA or Apple Silicon MPS; mixed precision picks the right dtype automatically)

## License

MIT. See [LICENSE](LICENSE).
