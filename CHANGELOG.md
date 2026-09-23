# Changelog

All notable changes to PyTorch Mastery Hub will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.1] - 2026-09-23

### Fixed
- The package failed to import on Windows: `utils.memory_utils` required the POSIX-only
  `resource` module. It is now optional, with a Win32 `GetProcessMemoryInfo` fallback.
- `utils.data_utils.normalize_data(method="minmax")` could return values marginally above
  1.0 in float32; the scaler now clips to the feature range.
- Bandit findings resolved: download requests have a timeout, `serve_model` binds
  loopback by default, `load_model`/`load_checkpoint` use `torch.load(weights_only=True)`.
- mypy targets Python 3.12 semantics so numpy >= 2.5 stubs type-check; runtime 3.10
  support is unchanged.

### Changed
- GitHub Actions dependencies bumped (checkout v7, setup-python v7, artifacts v7/v8,
  codecov v7, CodeQL v4, gh-release v3).
- The PyPI publish job runs only when the `PYPI_PUBLISH` repository variable is `true`.

## [1.1.0] - 2026-09-23

### Added
- **Training engine rewrite** (`neural_networks.training`): device-agnostic autocast
  (CUDA fp16/bf16, CPU bf16), gradient accumulation with partial-window flush, grad-norm
  clipping reported as a metric, automatic per-step vs per-epoch scheduler handling
  (`ReduceLROnPlateau` receives the monitored metric), optional `torch.compile`, seeded
  runs, dict-style batches, custom per-batch metrics, typed `Callback` protocol,
  `LambdaCallback`, `EarlyStopping` that restores best weights, `ModelCheckpoint` writing
  resumable state (model, optimizer, scheduler, scaler, EMA, RNG, epoch, step, history).
- `neural_networks.ema.ModelEMA` with warm-up decay, `swap()` context and `state_dict`.
- `neural_networks.attention`: `RMSNorm`, `RotaryEmbedding` (NTK scaling), grouped-query
  `MultiHeadAttention` on fused SDPA with padding/causal/explicit masks, `KVCache`,
  `SwiGLU`, pre-norm `DecoderBlock`, `TransformerLM` with KV-cached `generate()`
  (greedy, temperature, top-k, top-p, EOS).
- `advanced.lora`: `LoRALinear`, `apply_lora()`, `mark_only_lora_trainable()`,
  `lora_state_dict()`, `merge_lora()` / `unmerge_lora()`.
- `utils.distributed`: torchrun-aware process-group setup, metric reduction, uneven
  all-gather, DDP wrapping/unwrapping, samplers, `main_process_first()`, `spawn()`.
  `Trainer` is DDP-aware (sampler epochs, cross-rank metric averaging).
- `utils.device_utils`, `utils.reproducibility`, `utils.logging_utils`,
  `utils.model_utils`, `utils.memory_utils`: the helper modules notebooks imported but
  which never existed (`get_device`, `seed_everything`, `setup_logger`,
  `count_parameters`, `model_summary`, `MemoryTracker`, ...).
- CLI: `pytorch-hub train` (YAML/flag-driven end-to-end pipeline writing logs, metrics
  and checkpoints) and `pytorch-hub benchmark`; reference configs in `configs/`.
- Property-based tests with Hypothesis, a real two-process gloo DDP test, CLI
  end-to-end tests (~130 new tests; 290 total).
- CI: lint/type/security gate, Python 3.10-3.13 matrix with CPU torch wheels,
  example-script execution, wheel build + smoke install, docs build, CodeQL,
  Dependabot, tag-triggered release workflow, Read the Docs config, API reference pages.

### Changed
- Library moved to a proper src layout: `import pytorch_mastery_hub` replaces the old
  `src.*` / bare `utils.*` imports. `setup.py` removed; `pyproject.toml` is the single
  source of packaging truth with a dynamic version and `py.typed`.
- Toolchain: ruff replaces black/isort/flake8; mypy with gradual per-module strictness;
  refreshed pre-commit hooks with conventional-commit enforcement.
- Minimum Python is 3.10 (3.8/3.9 are end-of-life); classifiers cover 3.10-3.13.
- Example scripts rewritten against the real library API and executed in CI.
- `from __future__ import annotations` enforced in every module.

### Fixed
- `utils.metrics.accuracy(topk>1)` crashed on non-contiguous tensors.
- `EarlyStoppingCallback` never restored weights and `ModelCheckpointCallback` never
  saved anything; both now work (kept as aliases of the new classes).
- `tests/pytest.ini` used the wrong section header and silently shadowed the
  pyproject configuration (unregistered markers, missing coverage settings).
- Console-script entry point pointed at a non-installed module path.

## [1.0.0] - 2026-03-09

### Added

#### Core Infrastructure
- Complete project structure with `src/`, `tests/`, `notebooks/`, `docs/`, `examples/`, `scripts/` directories
- `pyproject.toml` for modern Python packaging (PEP 517/518)
- `setup.py` with comprehensive package metadata and extras
- `requirements.txt` with full dependency specification
- `.flake8` configuration file
- Docker support: `Dockerfile`, `Dockerfile.gpu`, `docker-compose.yml`
- GitHub Actions CI/CD pipeline (`.github/workflows/ci.yml`)
- Pre-commit hooks configuration (`.pre-commit-config.yaml`)
- `Makefile` for automation of common development tasks
- `tox.ini` for multi-environment testing
- `.editorconfig` for consistent editor formatting
- `.vscode/` settings for VS Code users

#### Source Modules (`src/`)
- **`src/fundamentals/`**
  - `tensor_ops.py`: Safe tensor operations, batch matrix multiply, tensor statistics
  - `autograd_helpers.py`: Custom autograd functions (Linear, ReLU, Sigmoid), gradient checking, gradient clipping, Jacobian/Hessian computation
  - `math_utils.py`: Mathematical utilities for deep learning
- **`src/neural_networks/`**
  - `layers.py`: Custom layers (Linear, Conv, Attention, Dropout, BatchNorm, LayerNorm, ResidualBlock)
  - `models.py`: 11 model architectures (MLP, CNN, ResNet, RNN, LSTM, GRU, Transformer, AutoEncoder, VAE, Seq2Seq)
  - `training.py`: Training loops with gradient accumulation, checkpointing, mixed precision
  - `optimizers.py`: Custom optimizers (SGD, Adam, AdamW) and LR schedulers (Polynomial, WarmupCosine)
- **`src/computer_vision/`**
  - `models.py`: Vision architectures (SimpleCNN, ResNetCV)
  - `augmentation.py`: Advanced augmentation (MixUp, CutMix, Mosaic)
  - `datasets.py`: Dataset classes (Image, Segmentation, ObjectDetection, CSV) with custom samplers
  - `transforms.py`: Transform utilities
- **`src/nlp/`**
  - `models.py`: NLP models (RNNClassifier, TransformerClassifier)
  - `embeddings.py`: Word and positional embeddings with pretrained loading
  - `tokenization.py`: Tokenizer with vocabulary building
  - `text_utils.py`: Text preprocessing utilities
- **`src/advanced/`**
  - `gan_utils.py`: GAN implementations (Vanilla, DCGAN, WGAN), training utilities, Inception score
  - `optimization.py`: Model quantization, pruning, knowledge distillation
  - `deployment.py`: Model deployment utilities
- **`src/utils/`**
  - `data_utils.py`: Dataset loading (MNIST, CIFAR-10, Fashion-MNIST, Iris), synthetic data generation
  - `metrics.py`: Comprehensive metrics (accuracy, precision, recall, F1, confusion matrix)
  - `io_utils.py`: Model I/O, checkpoint management, config handling, logging
  - `visualization.py`: Plotting and visualization utilities
  - `cli.py`: Command-line interface

#### Jupyter Notebooks (`notebooks/`)
- **Section 01 - Fundamentals** (4 notebooks)
  - `01_introduction_to_tensors.ipynb`
  - `02_gradient_computation.ipynb`
  - `03_custom_autograd_functions.ipynb`
  - `04_backpropagation_visualization.ipynb`
- **Section 02 - Neural Networks** (3 notebooks)
  - `05_mlp_from_scratch.ipynb`
  - `06_advanced_architectures.ipynb`
  - `07_training_techniques.ipynb`
- **Section 03 - Computer Vision** (3 notebooks)
  - `08_cnn_fundamentals.ipynb`
  - `09_modern_cnn_architectures.ipynb`
  - `10_computer_vision_projects.ipynb`
- **Section 04 - NLP** (4 notebooks)
  - `11_rnn_lstm_fundamentals.ipynb`
  - `12_sequence_to_sequence.ipynb`
  - `13_sentiment_analysis_project.ipynb`
  - `14_transformer_from_scratch.ipynb`
- **Section 05 - Generative Models** (2 notebooks)
  - `15_gan_fundamentals.ipynb`
  - `16_advanced_gans_vaes.ipynb`
- **Section 06 - Optimization & Deployment** (4 notebooks)
  - `17_model_optimization.ipynb`
  - `18_model_serving_apis.ipynb`
  - `19_monitoring_mlops.ipynb`
  - `20_cloud_deployment.ipynb`
- **Section 07 - Advanced Projects** (3 notebooks)
  - `21_image_classification_project.ipynb`
  - `22_text_generation_project.ipynb`
  - `23_recommendation_system.ipynb`
- **Section 08 - Advanced Topics** (2 notebooks)
  - `24_advanced_techniques.ipynb`
  - `25_research_applications.ipynb`
- **Capstone Projects** (2 notebooks)
  - `26_Capstone_part1_multimodal_system.ipynb`
  - `27_Capstone_part2_production_mlops.ipynb`

#### Test Suite (`tests/`)
- `conftest.py`: Shared fixtures (device, tensors, batches, text)
- `pytest.ini`: Pytest configuration with markers
- `run_tests.py`: Test runner script
- `test_integration.py`: End-to-end pipeline tests
- Tests for all src modules

#### Documentation (`docs/`)
- Sphinx documentation structure
- API reference stubs
- Tutorial guides

#### Examples (`examples/`)
- Standalone runnable Python scripts for key concepts
- Quick-start examples for each module

#### Scripts (`scripts/`)
- `setup_env.sh`: Automated environment setup
- `download_datasets.py`: Dataset download utilities
- `run_all_notebooks.sh`: Batch notebook execution

#### Community Files
- `README.md`: Comprehensive project documentation with badges
- `CONTRIBUTING.md`: Contribution guidelines
- `CODE_OF_CONDUCT.md`: Community standards
- `LICENSE`: MIT License

---

[Unreleased]: https://github.com/SatvikPraveen/pytorch-mastery-hub/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/SatvikPraveen/pytorch-mastery-hub/releases/tag/v1.0.0
