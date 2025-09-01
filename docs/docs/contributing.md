# docs/docs/contributing.md

# Contributing Guide

Welcome to the ML Training Framework! We're excited to have you contribute to making this framework better for everyone.

## 🚀 Quick Start for Contributors

### Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/yourusername/ml-training-framework.git
cd ml-training-framework

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# 4. Verify installation
pytest tests/ --tb=short
```

### Docker Development Setup

```bash
# Start development environment
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Access development container
docker exec -it ml-training-app-dev bash

# Run tests inside container
pytest tests/ -v
```

## 🛠️ Development Workflow

### 1. Create a Feature Branch

```bash
# Create branch from main
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name

# For bug fixes
git checkout -b fix/issue-description
```

### 2. Make Your Changes

Follow our coding standards:

- Write clear, documented code
- Add type hints where appropriate
- Follow PEP 8 style guidelines
- Add tests for new functionality

### 3. Test Your Changes

```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/test_fundamentals/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Code quality checks
black --check src/ tests/
flake8 src/ tests/
mypy src/
```

### 4. Submit Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Create pull request on GitHub
# - Describe your changes
# - Link related issues
# - Add screenshots if relevant
```

## 📝 Coding Standards

### Code Style

We use **Black** for code formatting and **flake8** for linting.

```bash
# Format code
black src/ tests/

# Check formatting
black --check src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Documentation Style

````python
def train_model(model: nn.Module, data_loader: DataLoader, epochs: int = 10) -> Dict[str, float]:
    """Train a PyTorch model.

    Args:
        model: The neural network model to train
        data_loader: DataLoader containing training data
        epochs: Number of training epochs (default: 10)

    Returns:
        Dictionary containing training metrics

    Example:
        ```python
        model = SimpleResNet(num_classes=10)
        loader = get_cifar10_loaders(batch_size=32)[0]
        metrics = train_model(model, loader, epochs=5)
        print(f"Final accuracy: {metrics['accuracy']:.3f}")
        ```
    """
    # Implementation here
    pass
````

### Test Writing

```python
import pytest
import torch
from src.neural_networks.models import FlexibleMLP

class TestFlexibleMLP:
    """Test FlexibleMLP model."""

    def test_initialization(self):
        """Test model initialization."""
        model = FlexibleMLP(784, [512, 256], 10)
        assert isinstance(model, FlexibleMLP)

    def test_forward_pass(self):
        """Test forward pass."""
        model = FlexibleMLP(784, [512, 256], 10)
        x = torch.randn(32, 784)
        output = model(x)
        assert output.shape == (32, 10)

    def test_different_architectures(self):
        """Test different architecture configurations."""
        # Test various hidden layer configurations
        configs = [
            ([128], "single hidden layer"),
            ([512, 256, 128], "three hidden layers"),
            ([1024], "large single layer")
        ]

        for hidden_dims, description in configs:
            model = FlexibleMLP(784, hidden_dims, 10)
            x = torch.randn(4, 784)
            output = model(x)
            assert output.shape == (4, 10), f"Failed for {description}"
```

## 🎯 Types of Contributions

### 🐛 Bug Fixes

- Fix existing functionality that doesn't work as expected
- Improve error handling and edge cases
- Performance optimizations

### ✨ New Features

- Add new model architectures
- Implement new training techniques
- Add new utilities and helpers

### 📚 Documentation

- Improve existing documentation
- Add new tutorials and examples
- Fix typos and clarify explanations

### 🧪 Tests

- Add missing test coverage
- Improve existing tests
- Add integration tests

### 🔧 Infrastructure

- Improve CI/CD pipelines
- Docker improvements
- Deployment enhancements

## 📋 Contribution Guidelines

### Issue Reporting

When reporting bugs:

1. **Search existing issues** first
2. **Use issue templates** provided
3. **Include reproduction steps**
4. **Provide environment details**

```markdown
**Bug Description:**
Clear description of the issue

**Reproduction Steps:**

1. Run `python script.py`
2. Call `function_name()`
3. Error occurs

**Environment:**

- OS: Ubuntu 20.04
- Python: 3.11
- PyTorch: 2.0.1
- Framework version: 1.0.0

**Expected vs Actual:**
Expected: Should return tensor shape (32, 10)
Actual: Returns shape (32, 1)
```

### Feature Requests

For new features:

1. **Check existing features** and issues
2. **Describe the use case** clearly
3. **Provide example usage**
4. **Consider implementation complexity**

### Pull Request Guidelines

#### Before Submitting

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated (if applicable)

#### PR Description Template

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing

- [ ] Added new tests
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## 🧪 Testing

### Test Structure

```
tests/
├── conftest.py                 # Shared fixtures
├── test_fundamentals/          # Unit tests for fundamentals
├── test_neural_networks/       # Unit tests for neural networks
├── test_computer_vision/       # Unit tests for computer vision
├── test_nlp/                   # Unit tests for NLP
├── test_advanced/              # Unit tests for advanced features
├── test_utils/                 # Unit tests for utilities
└── test_integration.py         # Integration tests
```

### Writing Tests

```python
# Use descriptive test names
def test_model_trains_successfully_with_valid_data():
    """Test that model training completes without errors."""
    pass

# Test edge cases
def test_model_handles_empty_batch():
    """Test model behavior with empty input batch."""
    model = FlexibleMLP(784, [128], 10)
    empty_batch = torch.empty(0, 784)

    with pytest.raises(RuntimeError):
        model(empty_batch)

# Use parametrized tests for multiple scenarios
@pytest.mark.parametrize("batch_size,input_dim,output_dim", [
    (1, 784, 10),
    (32, 784, 10),
    (64, 1024, 100),
])
def test_model_output_shapes(batch_size, input_dim, output_dim):
    """Test model outputs correct shapes for different configurations."""
    model = FlexibleMLP(input_dim, [256], output_dim)
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    assert output.shape == (batch_size, output_dim)
```

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_neural_networks/

# With coverage
pytest --cov=src

# Parallel execution
pytest -n 4

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Run only failed tests from last run
pytest --lf
```

## 📚 Documentation

### Building Documentation

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build and serve locally
cd docs/
mkdocs serve

# Build for production
mkdocs build
```

### Documentation Standards

- Use clear, concise language
- Include code examples
- Add screenshots for UI changes
- Link to related documentation
- Keep examples up-to-date

### Adding API Documentation

API documentation is auto-generated from docstrings:

````python
def my_function(param1: int, param2: str = "default") -> bool:
    """Short description of the function.

    Longer description with more details about what the function does,
    when to use it, and any important considerations.

    Args:
        param1: Description of first parameter
        param2: Description of second parameter with default value

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is negative
        RuntimeError: When something goes wrong

    Example:
        ```python
        result = my_function(42, "hello")
        assert result is True
        ```
    """
    pass
````

## 🔄 Release Process

### Version Management

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Creating Releases

Releases are handled by maintainers:

1. Update version in `setup.py`
2. Update `CHANGELOG.md`
3. Create release branch
4. Tag release: `git tag v1.2.3`
5. Push tag: `git push origin v1.2.3`
6. GitHub Actions creates release automatically

## 🌟 Recognition

### Contributors

All contributors are recognized in:

- `CONTRIBUTORS.md` file
- GitHub contributors page
- Release notes

### Special Recognition

Outstanding contributions receive:

- Mention in documentation
- Social media shout-outs
- Conference talk opportunities

## 📞 Getting Help

### Community Support

- **Discord**: [Join our Discord](https://discord.gg/ml-training)
- **GitHub Discussions**: [Discuss ideas](https://github.com/ml-training/framework/discussions)
- **Stack Overflow**: Tag questions with `ml-training-framework`

### Direct Contact

- **Email**: maintainers@ml-training.org
- **Twitter**: [@MLTrainingFramework](https://twitter.com/mltrainingframework)

### Mentorship Program

New contributors can request mentorship:

- Pair programming sessions
- Code review guidance
- Architecture discussions

---

**Ready to contribute?** 🚀

1. Check our [good first issues](https://github.com/ml-training/framework/labels/good%20first%20issue)
2. Join our [Discord community](https://discord.gg/ml-training)
3. Start with the [development setup](#development-setup)

Thank you for making the ML Training Framework better for everyone! 🙏
