# docs/docs/api/index.md

# API Reference Overview

The ML Training Framework provides a comprehensive API organized into specialized modules for different aspects of machine learning development.

## 📦 Module Structure

### Core Modules

| Module                                  | Description                                | Key Classes                  |
| --------------------------------------- | ------------------------------------------ | ---------------------------- |
| **[Fundamentals](#fundamentals)**       | Basic tensor operations and math utilities | `TensorOps`, `MathUtils`     |
| **[Neural Networks](#neural-networks)** | Core neural network components             | `Trainer`, `Layer`, `Model`  |
| **[Computer Vision](#computer-vision)** | Vision models and utilities                | `ResNet`, `DataAugmentation` |
| **[NLP](#natural-language-processing)** | Natural language processing                | `Transformer`, `Tokenizer`   |
| **[Advanced](#advanced-modules)**       | Specialized techniques                     | `GANTrainer`, `Optimization` |
| **[Utils](#utilities)**                 | Common utilities and helpers               | `DataUtils`, `Metrics`       |

## 🔧 Fundamentals

Core tensor operations and mathematical utilities.

```python
from src.fundamentals import tensor_ops, math_utils, autograd_helpers

# Basic tensor operations
result = tensor_ops.batch_matrix_multiply(a, b)

# Mathematical utilities
loss = math_utils.focal_loss(predictions, targets)

# Custom autograd functions
output = autograd_helpers.ReLUFunction.apply(input)
```

**Key Features:**

- Custom tensor operations
- Mathematical functions for ML
- Autograd function helpers
- Memory-efficient implementations

## 🧠 Neural Networks

Core neural network building blocks and training utilities.

```python
from src.neural_networks import models, layers, training, optimizers

# Create custom models
model = models.FlexibleMLP(input_dim=784, hidden_dims=[512, 256], output_dim=10)

# Advanced training
trainer = training.Trainer(
    model=model,
    optimizer='adamw',
    scheduler='cosine',
    use_mixed_precision=True
)

# Custom optimizers
optimizer = optimizers.AdamW(model.parameters(), lr=1e-3)
```

**Key Features:**

- Flexible model architectures
- Advanced training loops
- Custom optimizers and schedulers
- Mixed precision support

## 👁️ Computer Vision

Specialized components for computer vision tasks.

```python
from src.computer_vision import models, transforms, datasets, augmentation

# Pre-built models
model = models.EfficientNet(num_classes=1000, variant='b0')

# Data preprocessing
transform = transforms.ImageNetTransforms(size=224, mode='train')

# Advanced augmentation
aug = augmentation.AdvancedAugmentation(
    geometric=True,
    color=True,
    noise=True
)
```

**Key Features:**

- Modern CNN architectures
- Advanced data augmentation
- Custom datasets and transforms
- Pre-trained model support

## 🔤 Natural Language Processing

Tools and models for text processing and NLP tasks.

```python
from src.nlp import models, tokenization, embeddings, text_utils

# Transformer models
model = models.TransformerClassifier(
    vocab_size=30000,
    num_classes=2,
    d_model=512
)

# Tokenization
tokenizer = tokenization.SimpleTokenizer(vocab_size=30000)
tokens = tokenizer.encode("Hello, world!")

# Embeddings
embedder = embeddings.PositionalEmbedding(d_model=512, max_len=1024)
```

**Key Features:**

- Transformer architectures
- Flexible tokenization
- Positional embeddings
- Text preprocessing utilities

## 🚀 Advanced Modules

Specialized techniques for advanced ML applications.

```python
from src.advanced import gan_utils, optimization, deployment

# GAN training
gan = gan_utils.DCGAN(noise_dim=100, img_channels=3)
trainer = gan_utils.GANTrainer(gan.generator, gan.discriminator)

# Model optimization
optimizer = optimization.ModelOptimizer()
optimized_model = optimizer.quantize(model, method='dynamic')

# Deployment utilities
api = deployment.ModelAPI(model, preprocessor=transform)
```

**Key Features:**

- GAN implementations
- Model optimization techniques
- Deployment utilities
- Advanced training strategies

## 🔧 Utilities

Common utilities and helper functions.

```python
from src.utils import data_utils, metrics, visualization, io_utils

# Data utilities
loader = data_utils.create_dataloader(dataset, batch_size=32, shuffle=True)

# Metrics
accuracy = metrics.accuracy_score(predictions, targets)
f1 = metrics.f1_score(predictions, targets, average='weighted')

# Visualization
visualization.plot_training_curves(train_losses, val_losses)

# I/O utilities
model_state = io_utils.load_checkpoint('model.pt')
```

**Key Features:**

- Data loading and preprocessing
- Comprehensive metrics
- Training visualization
- Model saving/loading

## 🎯 Quick API Examples

### Training a Model

```python
from src.neural_networks.training import Trainer
from src.computer_vision.models import SimpleResNet
from src.utils.data_utils import get_cifar10_loaders

# Setup
model = SimpleResNet(num_classes=10)
train_loader, val_loader = get_cifar10_loaders(batch_size=32)

# Train
trainer = Trainer(model, train_loader, val_loader)
trainer.fit(epochs=10)
```

### Custom Data Pipeline

```python
from src.utils.data_utils import create_dataloader
from src.computer_vision.transforms import ImageNetTransforms

# Create transforms
transform = ImageNetTransforms(size=224, mode='train')

# Create data loader
loader = create_dataloader(
    dataset,
    batch_size=32,
    transform=transform,
    shuffle=True
)
```

### Model Evaluation

```python
from src.utils.metrics import ModelEvaluator

evaluator = ModelEvaluator(model)
results = evaluator.evaluate(test_loader)

print(f"Accuracy: {results['accuracy']:.3f}")
print(f"F1 Score: {results['f1_score']:.3f}")
```

## 📖 Detailed Documentation

For complete API documentation with all parameters, methods, and examples:

**[→ Full API Reference](reference.md)**

## 🔍 Finding What You Need

### By Task Type

=== "Image Classification" - [`computer_vision.models`](reference.md#computer-vision-models) - [`computer_vision.transforms`](reference.md#computer-vision-transforms) - [`neural_networks.training`](reference.md#training)

=== "Text Processing" - [`nlp.models`](reference.md#nlp-models) - [`nlp.tokenization`](reference.md#tokenization) - [`nlp.text_utils`](reference.md#text-utilities)

=== "Generative Models" - [`advanced.gan_utils`](reference.md#gan-utilities) - [`advanced.optimization`](reference.md#optimization)

=== "Training & Evaluation" - [`neural_networks.training`](reference.md#training) - [`utils.metrics`](reference.md#metrics) - [`utils.visualization`](reference.md#visualization)

### By Experience Level

=== "Beginner"
Start with: - [`neural_networks.models`](reference.md#neural-network-models) - [`neural_networks.training`](reference.md#training) - [`utils.data_utils`](reference.md#data-utilities)

=== "Intermediate"  
 Explore: - [`computer_vision.models`](reference.md#computer-vision-models) - [`nlp.models`](reference.md#nlp-models) - [`utils.metrics`](reference.md#metrics)

=== "Advanced"
Deep dive into: - [`advanced.gan_utils`](reference.md#gan-utilities) - [`advanced.optimization`](reference.md#optimization) - [`fundamentals.autograd_helpers`](reference.md#autograd-helpers)

---

Ready to explore the complete API? **[View Full Reference →](reference.md)**
