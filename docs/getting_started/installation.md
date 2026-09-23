# Installation

## Prerequisites

- **Python** 3.10 or higher
- **pip** 21.0+
- **Git**
- **CUDA** (optional, for GPU acceleration)
- **8GB RAM** (16GB+ recommended for large models)

## Quick Install

### Option 1: Clone and Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/SatvikPraveen/PyTroch-Mastery-Hub.git
cd PyTroch-Mastery-Hub

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install the library with dev + notebook extras
pip install -e ".[dev,notebooks]"

# Start Jupyter Lab
jupyter lab
```

### Option 2: Install via pip

```bash
pip install pytorch-mastery-hub
```

### Option 3: Docker (No Setup Required)

```bash
# CPU environment
docker-compose --profile cpu up -d

# Navigate to http://localhost:8888
```

## Install Options

### Core Only

```bash
pip install -e .
```

### Development (tests, linting)

```bash
pip install -e ".[dev]"
```

### Notebooks Support

```bash
pip install -e ".[notebooks]"
```

### Advanced (transformers, wandb, lightning)

```bash
pip install -e ".[advanced]"
```

### Computer Vision

```bash
pip install -e ".[cv]"
```

### NLP

```bash
pip install -e ".[nlp]"
```

### Everything

```bash
pip install -e ".[all]"
```

## Verify Installation

```bash
pytorch-hub info        # versions, best device, CUDA/MPS/bf16 support
pytorch-hub benchmark   # optional: matmul / attention throughput
pytest -m "not slow"    # run the test suite
```

## GPU Setup

For NVIDIA GPU support, install PyTorch with CUDA:

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Check GPU availability:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))  # Name of the first GPU
```

## Apple Silicon (M1/M2/M3)

PyTorch natively supports Apple Silicon via MPS backend:

```python
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
```

## Troubleshooting

**ImportError: No module named 'torch'**
```bash
pip install torch torchvision torchaudio
```

**CUDA not available**
- Ensure NVIDIA drivers are installed
- Install PyTorch with correct CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/)

**Out of memory errors**
- Reduce batch size, or raise `accumulation_steps` in `TrainerConfig` to keep the effective batch
- Use mixed precision: `TrainerConfig(precision="auto")` picks bf16/fp16 per device
- Track allocations with `pytorch_mastery_hub.utils.memory_utils.MemoryTracker`
