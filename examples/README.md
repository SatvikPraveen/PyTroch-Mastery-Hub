# Examples

Standalone Python scripts that demonstrate each key concept from the `src/` library.
These can be run directly without Jupyter, making them ideal for quick testing or
CI validation.

## Running Examples

All examples should be run from the **project root**:

```bash
# Activate virtual environment first
source venv/bin/activate

# Tensor basics
python examples/basic_tensors.py

# Train MLP on MNIST
python examples/train_mnist.py --epochs 5 --batch-size 64

# Custom autograd functions
python examples/custom_autograd.py

# Transfer learning with ResNet
python examples/transfer_learning.py

# Transformer text classification
python examples/transformer_text_classification.py

# Model quantization and pruning
python examples/model_optimization.py

# GAN training
python examples/gan_training.py --epochs 5

# Knowledge distillation
python examples/knowledge_distillation.py

# Model checkpointing
python examples/model_checkpointing.py

# Data augmentation (MixUp, CutMix, Mosaic)
python examples/data_augmentation.py
```

## Example Index

| Script | Topics | Key `src/` Modules |
|--------|--------|-------------------|
| `basic_tensors.py` | Tensor ops, autograd, device placement | `fundamentals.tensor_ops` |
| `train_mnist.py` | Full training loop, checkpointing | `neural_networks.models`, `utils.data_utils` |
| `custom_autograd.py` | Custom `Function`, gradient checking | `fundamentals.autograd_helpers` |
| `transfer_learning.py` | Pretrained CNN fine-tuning, MixUp | `computer_vision.*`, `neural_networks.training` |
| `transformer_text_classification.py` | Tokenization, Transformer, metrics | `nlp.*`, `utils.metrics` |
| `model_optimization.py` | Quantization, pruning | `advanced.optimization` |
| `gan_training.py` | GAN training loop | `advanced.gan_utils` |
| `knowledge_distillation.py` | Teacher-student distillation | `advanced.optimization` |
| `model_checkpointing.py` | Save/load, `ModelCheckpointManager` | `utils.io_utils` |
| `data_augmentation.py` | MixUp, CutMix, Mosaic | `computer_vision.augmentation` |
