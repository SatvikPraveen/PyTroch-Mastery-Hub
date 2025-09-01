# tests/conftest.py
"""
Pytest configuration and shared fixtures for PyTorch Mastery Hub tests
"""

import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import os


@pytest.fixture
def device():
    """Get available device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_tensor():
    """Sample tensor for testing."""
    return torch.randn(10, 5)


@pytest.fixture
def sample_image_tensor():
    """Sample image tensor for testing."""
    return torch.randn(3, 32, 32)


@pytest.fixture
def sample_batch():
    """Sample batch for testing."""
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def sample_text():
    """Sample text for NLP testing."""
    return "This is a sample text for testing natural language processing functions."


@pytest.fixture
def sample_texts():
    """Sample texts list for testing."""
    return [
        "This is the first sample text.",
        "Here is another example sentence.",
        "Testing natural language processing is important.",
        "PyTorch makes deep learning accessible.",
        "Machine learning models need proper testing."
    ]


@pytest.fixture
def sample_vocab():
    """Sample vocabulary for testing."""
    return {
        '<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3,
        'this': 4, 'is': 5, 'a': 6, 'test': 7, 'sentence': 8
    }


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    return torch.utils.data.TensorDataset(X, y)


@pytest.fixture
def sample_image():
    """Sample PIL image for testing."""
    return Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))


@pytest.fixture
def temp_dir():
    """Temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_model():
    """Simple model for testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )


@pytest.fixture
def classification_data():
    """Sample classification data."""
    X = torch.randn(50, 10)
    y = torch.randint(0, 3, (50,))
    return X, y


@pytest.fixture
def regression_data():
    """Sample regression data."""
    X = torch.randn(50, 10)
    y = torch.randn(50, 1)
    return X, y


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Reset seeds after each test
    yield
    
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        'model': {
            'type': 'SimpleMLP',
            'input_size': 10,
            'hidden_sizes': [20, 15],
            'output_size': 5
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 10
        }
    }


# Skip GPU tests if CUDA is not available
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)