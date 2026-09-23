# Examples

Standalone Python scripts that demonstrate each key concept from the `src/` library.
They run directly without Jupyter, need **no internet access** (anything that would be
downloaded falls back to synthetic data), and each finishes in well under a minute on CPU,
which makes them ideal for quick testing or CI validation.

## Running Examples

All examples should be run from the **project root**. Each script adds `src/` to
`sys.path`, so they also work without `pip install -e .`:

```bash
# Activate virtual environment first
source venv/bin/activate

# Tensor basics
python examples/basic_tensors.py

# Train MLP on MNIST with the Trainer (AMP, EMA, early stopping, checkpoints).
# Downloads MNIST if it can; otherwise (or with --synthetic) uses an MNIST-shaped stand-in.
python examples/train_mnist.py --epochs 3 --batch-size 64
python examples/train_mnist.py --synthetic

# Custom autograd functions
python examples/custom_autograd.py

# Transfer learning with ResNet-18 (pretrained weights if available, random init otherwise)
python examples/transfer_learning.py

# Transformer text classification
python examples/transformer_text_classification.py

# Model quantization and pruning
python examples/model_optimization.py

# GAN training
python examples/gan_training.py --epochs 5 --gan-type vanilla   # or lsgan

# Knowledge distillation
python examples/knowledge_distillation.py

# Model checkpointing (ModelCheckpointManager, save/load, resumable Trainer)
python examples/model_checkpointing.py

# Data augmentation (MixUp, CutMix, Mosaic) — writes outputs/augmentation_demo.png
python examples/data_augmentation.py
```

## Example Index

| Script | Topics | Key `src/` Modules |
|--------|--------|-------------------|
| `basic_tensors.py` | Tensor ops, autograd, device placement | `fundamentals.tensor_ops`, `utils.device_utils` |
| `train_mnist.py` | `Trainer` + `TrainerConfig`, `EarlyStopping`, `ModelCheckpoint`, mixed precision, EMA | `neural_networks.training`, `neural_networks.models`, `utils.data_utils`, `utils.io_utils` |
| `custom_autograd.py` | Custom `Function`, gradient checking, gradient clipping | `fundamentals.autograd_helpers` |
| `transfer_learning.py` | Pretrained CNN fine-tuning, freeze/unfreeze, `Trainer` | `neural_networks.training`, `utils.model_utils` |
| `transformer_text_classification.py` | Tokenization, Transformer, padding masks, metrics | `nlp.tokenization`, `nlp.models`, `utils.metrics` |
| `model_optimization.py` | Dynamic quantization, magnitude / global / structured pruning | `advanced.optimization` |
| `gan_training.py` | GAN training loop (`GANTrainer`, vanilla / LSGAN) | `advanced.gan_utils` |
| `knowledge_distillation.py` | Teacher-student distillation, functional `train_epoch` / `validate_epoch` | `advanced.optimization`, `neural_networks.training` |
| `model_checkpointing.py` | `ModelCheckpointManager`, `save_model`/`load_model`, `Trainer.save_checkpoint`/`load_checkpoint` | `utils.io_utils`, `neural_networks.training` |
| `data_augmentation.py` | MixUp, CutMix, Mosaic | `computer_vision.augmentation` |

Common helpers used across the scripts: `utils.device_utils.get_device`,
`utils.reproducibility.seed_everything`, `utils.model_utils.count_parameters`.
