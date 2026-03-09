# Contributing to PyTorch Mastery Hub

First off, thank you for considering contributing to PyTorch Mastery Hub! 🎉 It's people like you that make this an excellent learning resource for the PyTorch community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Improving Documentation](#improving-documentation)
  - [Contributing Code](#contributing-code)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Commit Message Guidelines](#commit-message-guidelines)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the [existing issues](https://github.com/SatvikPraveen/pytorch-mastery-hub/issues) to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, notebook cells)
- **Describe the behavior you observed** and what you expected
- **Include screenshots** if relevant
- **Mention your environment**: OS, Python version, PyTorch version

**Bug Report Template:**
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run notebook '...'
2. Execute cell '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
 - OS: [e.g., Ubuntu 22.04]
 - Python Version: [e.g., 3.10]
 - PyTorch Version: [e.g., 2.0.1]
 - CUDA Version: [if applicable]

**Additional context**
Add any other context about the problem.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested enhancement
- **Explain why this enhancement would be useful** to most users
- **List some examples** of how the enhancement would be used
- **Mention any related projects** that implement similar features

### Improving Documentation

Documentation improvements are always welcome! This includes:

- Fixing typos or grammatical errors
- Improving clarity of explanations
- Adding examples to notebooks
- Expanding docstrings in source code
- Creating new tutorials or guides
- Improving README or other markdown files

### Contributing Code

We welcome code contributions! Here are some areas where you can help:

- **Adding new notebooks**: Cover advanced topics or emerging PyTorch features
- **Improving existing code**: Optimize implementations, add features
- **Adding tests**: Increase code coverage
- **Fixing bugs**: Check the [issues](https://github.com/SatvikPraveen/pytorch-mastery-hub/issues)
- **Adding utilities**: Create reusable helper functions

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- (Optional) CUDA-compatible GPU for GPU-accelerated notebooks

### Local Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/pytorch-mastery-hub.git
   cd pytorch-mastery-hub
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e ".[dev,notebooks]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Run tests to ensure everything works**
   ```bash
   pytest tests/
   ```

### Docker Setup (Alternative)

```bash
# CPU-only environment
docker-compose --profile cpu up -d

# GPU-enabled environment (requires nvidia-docker)
docker-compose --profile gpu up -d
```

Access Jupyter at `http://localhost:8888`

## Pull Request Process

1. **Create a new branch** from `main` or `develop`
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes**
   - Write clear, commented code
   - Follow the style guidelines (below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Run the test suite**
   ```bash
   # Run all tests
   pytest tests/
   
   # Run with coverage
   pytest tests/ --cov=src --cov-report=html
   
   # Run specific test file
   pytest tests/test_fundamentals/test_tensor_ops.py
   ```

4. **Format your code**
   ```bash
   # The pre-commit hooks will do this automatically, but you can run manually:
   black src tests
   isort src tests
   flake8 src tests
   ```

5. **Commit your changes**
   - Follow our [commit message guidelines](#commit-message-guidelines)
   - Pre-commit hooks will run automatically

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Use a clear, descriptive title
   - Reference any related issues
   - Describe your changes in detail
   - Add screenshots for UI changes
   - Request review from maintainers

### Pull Request Checklist

- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Comments added to complex code sections
- [ ] Documentation updated (docstrings, README, etc.)
- [ ] No new warnings introduced
- [ ] Tests added for new functionality
- [ ] All tests pass locally
- [ ] Commit messages follow guidelines

## Style Guidelines

### Python Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Docstring format**: NumPy style
- **Import sorting**: Managed by `isort` with black profile
- **Formatting**: Automated by `black`

**Example:**
```python
import torch
import torch.nn as nn
from typing import Optional, Tuple

class MyModel(nn.Module):
    """
    Brief description of the model.
    
    Parameters
    ----------
    input_dim : int
        Input dimension size
    hidden_dim : int
        Hidden layer dimension size
    output_dim : int
        Output dimension size
    dropout : float, optional
        Dropout probability (default: 0.5)
    
    Attributes
    ----------
    fc1 : nn.Linear
        First fully connected layer
    fc2 : nn.Linear
        Second fully connected layer
    
    Examples
    --------
    >>> model = MyModel(input_dim=10, hidden_dim=20, output_dim=5)
    >>> x = torch.randn(32, 10)
    >>> output = model(x)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.5
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### Jupyter Notebook Style

- **Clear markdown headers** for sections
- **Explanatory text** before code cells
- **Comments in code** for complex operations
- **Visualizations** where appropriate
- **Clear outputs** showing results
- **Clean cell execution** (restart kernel and run all before committing)

### Documentation Style

- Use **clear, concise language**
- Include **code examples** in docstrings
- Add **type hints** to function signatures
- Follow **NumPy docstring format**
- Keep README and documentation **up-to-date**

## Testing Guidelines

### Test Structure

```
tests/
├── test_fundamentals/
├── test_neural_networks/
├── test_computer_vision/
├── test_nlp/
├── test_advanced/
├── test_utils/
└── test_integration.py
```

### Writing Tests

```python
import pytest
import torch
from src.fundamentals.tensor_ops import safe_divide

class TestTensorOperations:
    """Test suite for tensor operations."""
    
    def test_safe_divide_normal_case(self):
        """Test safe division with normal inputs."""
        numerator = torch.tensor([10.0, 20.0, 30.0])
        denominator = torch.tensor([2.0, 4.0, 5.0])
        result = safe_divide(numerator, denominator)
        expected = torch.tensor([5.0, 5.0, 6.0])
        assert torch.allclose(result, expected)
    
    def test_safe_divide_zero_denominator(self):
        """Test safe division handles zero denominator."""
        numerator = torch.tensor([10.0, 20.0])
        denominator = torch.tensor([0.0, 4.0])
        result = safe_divide(numerator, denominator, epsilon=1e-8)
        assert torch.isfinite(result).all()
    
    @pytest.mark.gpu
    def test_safe_divide_gpu(self, device):
        """Test safe division on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        numerator = torch.tensor([10.0, 20.0, 30.0], device=device)
        denominator = torch.tensor([2.0, 4.0, 5.0], device=device)
        result = safe_divide(numerator, denominator)
        assert result.device == device
```

### Test Markers

- `@pytest.mark.unit`: Unit tests (fast)
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.gpu`: Tests requiring GPU

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

### Examples

```
feat(nlp): add BERT tokenizer implementation

- Implement WordPiece tokenization
- Add vocabulary loading from pretrained models
- Include tests for tokenization

Closes #123
```

```
fix(training): resolve gradient accumulation bug

The gradient accumulation was not resetting properly
between batches, causing incorrect weight updates.
```

```
docs(notebooks): improve transformer notebook explanations

- Add more detailed comments on attention mechanism
- Include visualization of attention weights
- Fix typos in mathematical notation
```

## Recognition

Contributors will be recognized in several ways:

- Listed in README.md contributors section
- Mentioned in release notes for significant contributions
- GitHub contributor badge on profile
- Special recognition for major features or improvements

## Questions?

Feel free to:
- Open an issue for questions
- Start a discussion in GitHub Discussions
- Contact maintainers directly

Thank you for contributing to PyTorch Mastery Hub! Your efforts help make deep learning education more accessible to everyone. 🚀
