# Quick Start

This guide gets you running with PyTorch Mastery Hub in minutes.

## 5-Minute Start

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_mastery_hub import get_device, seed_everything
from pytorch_mastery_hub.fundamentals.tensor_ops import tensor_stats
from pytorch_mastery_hub.neural_networks import EarlyStopping, SimpleMLP, Trainer, TrainerConfig

seed_everything(0)

# 1. Tensors
x = torch.randn(512, 10)
print(tensor_stats(x))

# 2. Model (input_size, hidden_sizes, output_size)
model = SimpleMLP(10, [64, 32], 3)

# 3. Data
y = (x @ torch.randn(10, 3)).argmax(1)
train = DataLoader(TensorDataset(x[:400], y[:400]), batch_size=32, shuffle=True)
val = DataLoader(TensorDataset(x[400:], y[400:]), batch_size=64)

# 4. Train with mixed precision, EMA and early stopping
trainer = Trainer(
    model, nn.CrossEntropyLoss(), torch.optim.AdamW(model.parameters(), 1e-3), get_device(),
    config=TrainerConfig(epochs=20, precision="auto", ema_decay=0.99),
    callbacks=[EarlyStopping(patience=3)],
)
history = trainer.fit(train, val)
print(f"best val_loss: {min(history['val_loss']):.4f}")
```

Or from the shell, without writing any code:

```bash
pytorch-hub train --config configs/train_synthetic.yaml
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

# Transformer text classification
python examples/transformer_text_classification.py
```

## Using Source Modules in Your Code

```python
# Fundamentals
from pytorch_mastery_hub.fundamentals.tensor_ops import safe_divide, batch_matrix_multiply
from pytorch_mastery_hub.fundamentals.autograd_helpers import gradient_check, GradientClipping

# Neural Networks
from pytorch_mastery_hub.neural_networks.models import SimpleMLP, SimpleTransformer
from pytorch_mastery_hub.neural_networks.training import Trainer, TrainerConfig, train_epoch, validate_epoch
from pytorch_mastery_hub.neural_networks.ema import ModelEMA
from pytorch_mastery_hub.neural_networks.attention import TransformerLM, MultiHeadAttention, KVCache
from pytorch_mastery_hub.neural_networks.optimizers import CustomAdam, WarmupCosineAnnealingLR

# Parameter-efficient fine-tuning and distributed training
from pytorch_mastery_hub.advanced.lora import apply_lora, merge_lora
from pytorch_mastery_hub.utils import distributed

# Computer Vision
from pytorch_mastery_hub.computer_vision.models import SimpleCNN, ResNetCV
from pytorch_mastery_hub.computer_vision.augmentation import MixUp, CutMix

# NLP
from pytorch_mastery_hub.nlp.models import RNNClassifier, TransformerClassifier
from pytorch_mastery_hub.nlp.tokenization import SimpleTokenizer

# Utilities
from pytorch_mastery_hub.utils.device_utils import get_device
from pytorch_mastery_hub.utils.reproducibility import seed_everything
from pytorch_mastery_hub.utils.model_utils import model_summary, count_parameters
from pytorch_mastery_hub.utils.data_utils import load_dataset
from pytorch_mastery_hub.utils.metrics import accuracy, classification_report
from pytorch_mastery_hub.utils.io_utils import save_model, load_model, ModelCheckpointManager
from pytorch_mastery_hub.utils.visualization import plot_training_curves
```
