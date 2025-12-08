# PyTorch Mastery Hub

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red.svg)](https://github.com/SatvikPraveen)

A comprehensive **PyTorch Learning and Implementation Reference** showcasing core concepts, architectural patterns, and best practices across the entire machine learning spectrum. This repository is designed as an educational resource for understanding PyTorch fundamentals and implementing deep learning models from scratch.

## 🎯 Project Overview

This is a **learning-focused repository** that demonstrates how to implement PyTorch models and understand deep learning concepts through practical code examples. Rather than a production system, this codebase emphasizes clarity, educational value, and transferable patterns you can use in your own projects.

### Learning Content Coverage

- **Core Fundamentals**: Tensor operations, autograd mechanics, custom gradients, and backpropagation visualization
- **Neural Network Architectures**: MLPs, CNNs, RNNs, LSTMs, and Transformers implemented from scratch
- **Computer Vision**: CNN architectures, attention mechanisms, and classification systems
- **Natural Language Processing**: Text processing, sequence models, attention mechanisms, and transformers
- **Generative Models**: GANs and VAEs with detailed implementations and training strategies
- **Optimization Techniques**: Learning rate scheduling, gradient clipping, and training strategies
- **Implementation Patterns**: Reusable components and best practices for building neural networks

## 🚀 Setup & Installation

### System Requirements

- Python 3.8+
- CUDA-compatible GPU (optional but recommended)
- 8GB+ RAM recommended

### Quick Start

```bash
git clone https://github.com/SatvikPraveen/PyTorch-Mastery-Hub.git
cd PyTorch-Mastery-Hub

# Create virtual environment
python -m venv pytorch_env
source pytorch_env/bin/activate  # Windows: pytorch_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Start Jupyter and open the notebooks
jupyter notebook notebooks/
```

## 📁 Repository Structure

```
PyTorch-Mastery-Hub/
├── notebooks/                    # 27 Jupyter notebooks with implementations
│   ├── 01_fundamentals/          # Tensor ops, autograd, backprop visualization
│   │   ├── 01_introduction_to_tensors.ipynb
│   │   ├── 02_gradient_computation.ipynb
│   │   ├── 03_custom_autograd_functions.ipynb
│   │   └── 04_backpropagation_visualization.ipynb
│   ├── 02_neural_networks/       # MLP, advanced architectures, training
│   │   ├── 05_mlp_from_scratch.ipynb
│   │   ├── 06_advanced_architectures.ipynb
│   │   └── 07_training_techniques.ipynb
│   ├── 03_computer_vision/       # CNN, vision architectures
│   │   ├── 08_cnn_fundamentals.ipynb
│   │   ├── 09_modern_cnn_architectures.ipynb
│   │   └── 10_computer_vision_projects.ipynb
│   ├── 04_natural_language_processing/ # RNN, LSTM, Transformers
│   │   ├── 11_rnn_lstm_fundamentals.ipynb
│   │   ├── 12_sequence_to_sequence.ipynb
│   │   ├── 13_sentiment_analysis_project.ipynb
│   │   └── 14_transformer_from_scratch.ipynb
│   ├── 05_generative_models/     # GANs, VAEs
│   │   ├── 15_gan_fundamentals.ipynb
│   │   └── 16_advanced_gans_vaes.ipynb
│   └── 06_optimization_deployment/ # Model optimization
│       └── 17_model_optimization.ipynb
├── src/                          # Reusable source code modules
│   ├── fundamentals/             # Tensor utilities and autograd helpers
│   │   ├── tensor_ops.py
│   │   ├── autograd_helpers.py
│   │   └── math_utils.py
│   ├── neural_networks/          # Network components and training loops
│   │   ├── layers.py
│   │   ├── models.py
│   │   ├── optimizers.py
│   │   └── training.py
│   ├── computer_vision/          # Vision models and utilities
│   │   ├── models.py
│   │   ├── augmentation.py
│   │   ├── datasets.py
│   │   └── transforms.py
│   ├── nlp/                      # NLP models and utilities
│   │   ├── models.py
│   │   └── embeddings.py
│   └── utils/                    # Common utilities
├── tests/                        # Unit and integration tests
├── requirements.txt              # Dependencies
├── setup.py                      # Package configuration
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## 💡 Key Features

### **Comprehensive Coverage**
- 27 detailed Jupyter notebooks spanning beginner to advanced topics
- Each notebook includes theory, implementation, and practical examples
- Modular code in `src/` for reuse and reference

### **Learning-Focused Design**
- Well-commented code explaining PyTorch concepts
- Progressive difficulty from fundamentals to advanced topics
- Each notebook builds on previous concepts

### **Implementation Patterns**
- Clean, readable code following PyTorch best practices
- Reusable classes and functions for common tasks
- Documented design choices and architecture decisions

### **Educational Value**
- Understand how to implement models from first principles
- Learn PyTorch APIs and their proper usage
- See how different components fit together

## 🎓 Learning Path

**Recommended order for learning:**

1. **Fundamentals** (Notebooks 1-4): Master PyTorch basics
   - Tensor operations and properties
   - Automatic differentiation and gradients
   - Custom autograd functions

2. **Neural Networks** (Notebooks 5-7): Build essential models
   - Implement MLPs from scratch
   - Understand modern architectures
   - Learn training strategies

3. **Computer Vision** (Notebooks 8-10): Vision models
   - CNN fundamentals
   - Advanced architectures (ResNet, EfficientNet, etc.)
   - Practical CV implementations

4. **Natural Language Processing** (Notebooks 11-14): NLP models
   - RNNs and LSTMs
   - Sequence-to-sequence models
   - Transformers from scratch

5. **Generative Models** (Notebooks 15-16): Advanced topics
   - GAN implementations
   - VAE architectures
   - Training techniques for generative models

6. **Optimization** (Notebook 17): Production considerations
   - Model optimization techniques
   - Efficient implementations
   - Performance improvements

## 🚀 Getting Started

### Open Notebooks

```bash
jupyter notebook notebooks/01_fundamentals/
```

Start with notebook 1 and progress sequentially through each section.

### Using Source Code

Import modules for your own projects:

```python
from src.fundamentals.tensor_ops import create_tensor, advanced_operations
from src.neural_networks.models import MLP, train_model
from src.computer_vision.models import CNN
from src.nlp.models import RNNTextClassifier
```

### Running Tests

```bash
python -m pytest tests/
```

## 📝 Code Examples

### Basic Tensor Operations

```python
import torch
from src.fundamentals.tensor_ops import create_tensor

# Create tensor with gradient tracking
x = create_tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
loss = y.sum()
loss.backward()
print(x.grad)
```

### Simple Neural Network

```python
from src.neural_networks.models import SimpleMLP
import torch.optim as optim

# Create model
model = SimpleMLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)

# Setup training
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### CNN Implementation

```python
from src.computer_vision.models import SimpleCNN

# Create CNN model
model = SimpleCNN(num_classes=10)

# Forward pass
output = model(images)
```

### LSTM Text Classification

```python
from src.nlp.models import LSTMClassifier

# Create LSTM model
model = LSTMClassifier(vocab_size=10000, embedding_dim=128, hidden_dim=256)

# Process sequences
output = model(sequences)
```

## 📚 Dependencies

**Core:**
- PyTorch >= 2.0.0
- NumPy >= 1.21.0

**Data & ML:**
- Pandas >= 1.3.0
- scikit-learn >= 0.24.0

**Visualization:**
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0

**Testing:**
- pytest >= 7.0.0

See `requirements.txt` for exact versions.

## ❓ FAQ

**Q: Can I use this code in my projects?**
A: Yes! All code is MIT licensed and free to use and modify.

**Q: Should I memorize all the code?**
A: No, focus on understanding the patterns and concepts. Use this as a reference.

**Q: In what order should I study?**
A: Follow the numbered notebooks (01 → 27). Each builds on previous knowledge.

**Q: Can I skip some notebooks?**
A: Fundamentals (1-4) and Neural Networks (5-7) are essential. Others can be explored based on interest.

**Q: Do I need a GPU?**
A: No, but it makes training faster. All code works on CPU.

**Q: How do I apply this to my own projects?**
A: Copy patterns from `src/` modules and adapt them to your problem.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

This is a personal learning reference. For issues or suggestions, open a GitHub issue or submit a pull request.

---

**A comprehensive learning resource for PyTorch and deep learning implementation.**
