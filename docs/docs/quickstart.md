# docs/docs/quickstart.md

# Quick Start Guide

Get up and running with the ML Training Framework in just 5 minutes!

## Installation

### Option 1: Using pip (Recommended)

```bash
# Install the framework
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Option 2: Using Docker (Production Ready)

```bash
# Clone the repository
git clone https://github.com/ml-training/framework.git
cd framework

# Start with Docker Compose
docker-compose up -d

# Access Jupyter Lab
open http://localhost:8888
```

### Option 3: Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/ml-training/framework.git
cd framework
pip install -e .

# Run tests to verify installation
pytest tests/
```

## Requirements

- **Python**: 3.9, 3.10, or 3.11
- **PyTorch**: >= 2.0.0
- **CUDA**: Optional, for GPU acceleration
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 2GB free space

## Your First Training Run

### 1. Basic Image Classification

```python
from src.computer_vision.models import SimpleResNet
from src.neural_networks.training import Trainer
from src.utils.data_utils import get_cifar10_loaders

# Load data
train_loader, val_loader = get_cifar10_loaders(batch_size=32)

# Create model
model = SimpleResNet(num_classes=10)

# Setup trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Train the model
trainer.fit(epochs=10)
```

### 2. Text Classification

```python
from src.nlp.models import TextClassifier
from src.nlp.tokenization import SimpleTokenizer

# Setup tokenizer and model
tokenizer = SimpleTokenizer(vocab_size=10000)
model = TextClassifier(vocab_size=10000, num_classes=2)

# Your training code here
trainer = Trainer(model=model, ...)
trainer.fit(epochs=5)
```

### 3. GAN Training

```python
from src.advanced.gan_utils import DCGAN, GANTrainer

# Create GAN
gan = DCGAN(noise_dim=100, img_channels=3)

# Setup GAN trainer
gan_trainer = GANTrainer(
    generator=gan.generator,
    discriminator=gan.discriminator
)

# Train GAN
gan_trainer.train(epochs=50)
```

## Docker Quick Start

### Development Environment

```bash
# Start development environment with hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Access services:
# - Jupyter Lab: http://localhost:8888
# - TensorBoard: http://localhost:6006
# - MLflow: http://localhost:5000
```

### Production Environment

```bash
# Start production environment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Monitor with Grafana: http://localhost:3000
```

## Interactive Notebooks

### Launch Jupyter Lab

=== "Local Installation"
`bash
    jupyter lab notebooks/
    `

=== "Docker Environment"
`bash
    docker-compose up jupyter
    # Open http://localhost:8888
    `

### Recommended Starting Notebooks

1. **[Introduction to Tensors](../notebooks/01_fundamentals/01_introduction_to_tensors.ipynb)**

   - Basic tensor operations
   - PyTorch fundamentals
   - GPU acceleration

2. **[MLP from Scratch](../notebooks/02_neural_networks/05_mlp_from_scratch.ipynb)**

   - Build neural networks
   - Training loops
   - Optimization

3. **[CNN Fundamentals](../notebooks/03_computer_vision/08_cnn_fundamentals.ipynb)**
   - Convolutional layers
   - Image classification
   - Data augmentation

## Verification

### Check Installation

```python
import torch
from src import fundamentals, neural_networks, computer_vision

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("✅ Installation verified!")
```

### Run Sample Tests

```bash
# Quick smoke test
python -c "from src.utils.data_utils import generate_sample_data; print('✅ Data utils working')"

# Run specific test suite
pytest tests/test_fundamentals/ -v

# Run all tests (takes ~2 minutes)
pytest tests/ --tb=short
```

### GPU Check

```python
import torch

if torch.cuda.is_available():
    print(f"✅ GPU available: {torch.cuda.get_device_name()}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  GPU not available, using CPU")
```

## Common Issues & Solutions

### Installation Issues

??? question "ImportError: No module named 'torch'"
`bash
    # Install PyTorch with CUDA support
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    `

??? question "CUDA out of memory"
```python # Reduce batch size
train_loader = DataLoader(dataset, batch_size=16) # instead of 32

    # Enable gradient accumulation
    trainer = Trainer(..., accumulate_grad_batches=4)
    ```

??? question "Docker permission denied"
`bash
    # Add user to docker group
    sudo usermod -aG docker $USER
    # Logout and login again
    `

### Performance Issues

??? question "Training too slow"
```python # Enable mixed precision
trainer = Trainer(..., use_mixed_precision=True)

    # Increase num_workers
    train_loader = DataLoader(..., num_workers=4)
    ```

??? question "Docker containers won't start"
```bash # Check Docker resources
docker system df
docker system prune # if needed

    # Check ports
    netstat -tulpn | grep :8888
    ```

## Next Steps

### Explore Examples

- Browse [Code Examples](examples.md) for practical use cases
- Try the [Computer Vision Project](../notebooks/03_computer_vision/10_computer_vision_projects.ipynb)
- Experiment with [GAN Training](../notebooks/05_generative_models/15_gan_fundamentals.ipynb)

### Deep Dive

- Read the [API Reference](api/index.md) for detailed documentation
- Check out [Advanced Techniques](../notebooks/08_advanced_topics/24_advanced_techniques.ipynb)
- Explore [Deployment Options](../notebooks/06_optimization_deployment/20_cloud_deployment.ipynb)

### Get Help

- View [Contributing Guide](contributing.md) for development setup
- Check existing [GitHub Issues](https://github.com/ml-training/framework/issues)
- Join our [Discord Community](https://discord.gg/ml-training)

---

**🎉 Congratulations!** You're ready to build amazing ML models. Start with the [notebooks](../notebooks/) or dive into the [examples](examples.md)!
