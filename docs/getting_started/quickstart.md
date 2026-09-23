# Quick Start

This guide gets you running with PyTorch Mastery Hub in minutes.

## 5-Minute Start

```python
import torch
import sys
sys.path.insert(0, '..')  # if running from notebooks/

from pytorch_mastery_hub.fundamentals.tensor_ops import tensor_stats
from pytorch_mastery_hub.neural_networks.models import SimpleMLP
from pytorch_mastery_hub.utils.data_utils import load_dataset

# 1. Create tensors
x = torch.randn(32, 10)
print(tensor_stats(x))

# 2. Build a model
model = SimpleMLP(input_dim=10, hidden_dims=[64, 32], output_dim=1)
print(model)

# 3. Forward pass
output = model(x)
print(f"Output shape: {output.shape}")

# 4. Load a dataset
train_loader, test_loader = load_dataset('mnist', batch_size=64)
print(f"Training batches: {len(train_loader)}")
```

## Explore Notebooks

Navigate notebooks in order for a structured learning path:

| Section | Topics |
|---------|--------|
| `01_fundamentals/` | Tensors, autograd, backpropagation |
| `02_neural_networks/` | MLP, training loops, regularization |
| `03_computer_vision/` | CNNs, ResNets, image classification |
| `04_nlp/` | RNNs, LSTMs, Transformers |
| `05_generative_models/` | GANs, VAEs |
| `06_optimization_deployment/` | Quantization, pruning, serving |
| `07_advanced_projects/` | End-to-end projects |
| `08_advanced_topics/` | Research techniques |
| `capstone_projects/` | Full multimodal system |

```bash
jupyter lab notebooks/
```

## Running Examples

Standalone example scripts in `examples/`:

```bash
# Tensor basics
python examples/basic_tensors.py

# Train a simple MLP on MNIST
python examples/train_mnist.py

# Fine-tune a transformer
python examples/fine_tune_transformer.py
```

## Using Source Modules in Your Code

```python
# Fundamentals
from pytorch_mastery_hub.fundamentals.tensor_ops import safe_divide, batch_matrix_multiply
from pytorch_mastery_hub.fundamentals.autograd_helpers import gradient_check, GradientClipping

# Neural Networks
from pytorch_mastery_hub.neural_networks.models import SimpleMLP, SimpleTransformer
from pytorch_mastery_hub.neural_networks.training import train_epoch, validate_epoch
from pytorch_mastery_hub.neural_networks.optimizers import CustomAdam, WarmupCosineAnnealingLR

# Computer Vision
from pytorch_mastery_hub.computer_vision.models import SimpleCNN, ResNetCV
from pytorch_mastery_hub.computer_vision.augmentation import MixUp, CutMix

# NLP
from pytorch_mastery_hub.nlp.models import RNNClassifier, TransformerClassifier
from pytorch_mastery_hub.nlp.tokenization import SimpleTokenizer

# Utilities
from pytorch_mastery_hub.utils.data_utils import load_dataset
from pytorch_mastery_hub.utils.metrics import accuracy, classification_report
from pytorch_mastery_hub.utils.io_utils import save_model, load_model, ModelCheckpointManager
from pytorch_mastery_hub.utils.visualization import plot_training_curves
```
