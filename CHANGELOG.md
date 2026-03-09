# Changelog

All notable changes to PyTorch Mastery Hub will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Pending items for next release

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
