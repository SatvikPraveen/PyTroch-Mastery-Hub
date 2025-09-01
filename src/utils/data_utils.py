# src/utils/data_utils.py
"""
Data loading and preprocessing utilities for PyTorch Mastery Hub
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union
import requests
import hashlib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


def load_dataset(name: str, data_dir: str = "data", download: bool = True) -> Dict[str, Any]:
    """
    Load common datasets for PyTorch tutorials.
    
    Args:
        name: Dataset name ('mnist', 'cifar10', 'iris', etc.)
        data_dir: Data directory path
        download: Whether to download if not exists
        
    Returns:
        Dictionary containing train/test data and metadata
    """
    data_dir = Path(data_dir)
    name = name.lower()
    
    if name == 'mnist':
        return _load_mnist(data_dir, download)
    elif name == 'cifar10':
        return _load_cifar10(data_dir, download)
    elif name == 'iris':
        return _load_iris(data_dir)
    elif name == 'boston':
        return _load_boston(data_dir)
    elif name == 'fashion_mnist':
        return _load_fashion_mnist(data_dir, download)
    else:
        raise ValueError(f"Dataset '{name}' not supported")


def _load_mnist(data_dir: Path, download: bool = True) -> Dict[str, Any]:
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=download, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=download, transform=transform
    )
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'num_classes': 10,
        'input_shape': (1, 28, 28),
        'classes': [str(i) for i in range(10)]
    }


def _load_cifar10(data_dir: Path, download: bool = True) -> Dict[str, Any]:
    """Load CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=transform
    )
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'num_classes': 10,
        'input_shape': (3, 32, 32),
        'classes': classes
    }


def _load_fashion_mnist(data_dir: Path, download: bool = True) -> Dict[str, Any]:
    """Load Fashion-MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=True, download=download, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=False, download=download, transform=transform
    )
    
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'num_classes': 10,
        'input_shape': (1, 28, 28),
        'classes': classes
    }


def _load_iris(data_dir: Path) -> Dict[str, Any]:
    """Load Iris dataset from CSV."""
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'num_classes': 3,
        'input_shape': (4,),
        'classes': iris.target_names.tolist(),
        'feature_names': iris.feature_names
    }


def _load_boston(data_dir: Path) -> Dict[str, Any]:
    """Load Boston Housing dataset."""
    from sklearn.datasets import load_boston
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    boston = load_boston()
    X, y = boston.data, boston.target
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'num_classes': 1,  # Regression
        'input_shape': (13,),
        'feature_names': boston.feature_names.tolist(),
        'scaler': scaler
    }


def create_data_loaders(
    train_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_train: bool = True
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset (optional)
        val_dataset: Validation dataset (optional)
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Dictionary of data loaders
    """
    loaders = {}
    
    # Training loader
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Validation loader
    if val_dataset is not None:
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    # Test loader
    if test_dataset is not None:
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    return loaders


def train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.2,
    random_seed: Optional[int] = 42
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into training and validation sets.
    
    Args:
        dataset: PyTorch dataset to split
        val_ratio: Validation set ratio
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset


def normalize_data(
    X: Union[torch.Tensor, np.ndarray],
    method: str = 'standard',
    fitted_scaler: Optional[Any] = None
) -> Tuple[Union[torch.Tensor, np.ndarray], Any]:
    """
    Normalize data using different methods.
    
    Args:
        X: Input data
        method: Normalization method ('standard', 'minmax', 'robust')
        fitted_scaler: Pre-fitted scaler (for test data)
        
    Returns:
        Tuple of (normalized_data, scaler)
    """
    is_tensor = isinstance(X, torch.Tensor)
    
    # Convert to numpy if tensor
    if is_tensor:
        X_np = X.numpy()
    else:
        X_np = X
    
    if fitted_scaler is None:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        X_normalized = scaler.fit_transform(X_np)
    else:
        scaler = fitted_scaler
        X_normalized = scaler.transform(X_np)
    
    # Convert back to tensor if needed
    if is_tensor:
        X_normalized = torch.FloatTensor(X_normalized)
    
    return X_normalized, scaler


def generate_synthetic_data(
    task: str = 'classification',
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    noise: float = 0.1,
    random_seed: Optional[int] = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic datasets for learning.
    
    Args:
        task: Task type ('classification', 'regression')
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes (for classification)
        noise: Noise level
        random_seed: Random seed
        
    Returns:
        Tuple of (X, y) tensors
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    if task == 'classification':
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=max(2, n_features // 2),
            n_redundant=n_features // 4,
            noise=noise,
            random_state=random_seed
        )
    elif task == 'regression':
        from sklearn.datasets import make_regression
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise * 10,
            random_state=random_seed
        )
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return torch.FloatTensor(X), torch.LongTensor(y) if task == 'classification' else torch.FloatTensor(y)


def get_dataset_info(dataset: Dataset) -> Dict[str, Any]:
    """
    Get information about a dataset.
    
    Args:
        dataset: PyTorch dataset
        
    Returns:
        Dictionary with dataset information
    """
    info = {
        'size': len(dataset),
        'sample_shape': None,
        'data_type': None
    }
    
    if len(dataset) > 0:
        sample = dataset[0]
        if isinstance(sample, (tuple, list)):
            info['sample_shape'] = [x.shape if hasattr(x, 'shape') else type(x) for x in sample]
            info['data_type'] = [type(x) for x in sample]
        else:
            info['sample_shape'] = sample.shape if hasattr(sample, 'shape') else type(sample)
            info['data_type'] = type(sample)
    
    return info


class CustomDataset(Dataset):
    """
    Custom PyTorch dataset for educational purposes.
    """
    
    def __init__(self, data: torch.Tensor, targets: torch.Tensor, transform=None):
        """
        Initialize custom dataset.
        
        Args:
            data: Input data tensor
            targets: Target tensor
            transform: Optional transform function
        """
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, target


def download_dataset(name: str, data_dir: str = "data", verify_hash: bool = True) -> str:
    """
    Download dataset from URL with caching and verification.
    
    Args:
        name: Dataset name
        data_dir: Directory to save data
        verify_hash: Whether to verify file hash
        
    Returns:
        Path to downloaded file
    """
    # Dataset URLs and hashes (example)
    datasets_info = {
        'sample_text': {
            'url': 'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/penn/train.txt',
            'filename': 'penn_train.txt',
            'hash': None
        }
    }
    
    if name not in datasets_info:
        raise ValueError(f"Dataset '{name}' not available for download")
    
    info = datasets_info[name]
    data_dir = Path(data_dir) / "external"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = data_dir / info['filename']
    
    # Check if file already exists
    if file_path.exists():
        if not verify_hash or info['hash'] is None:
            return str(file_path)
        
        # Verify hash
        if _verify_file_hash(file_path, info['hash']):
            return str(file_path)
    
    # Download file
    print(f"Downloading {name} dataset...")
    response = requests.get(info['url'], stream=True)
    response.raise_for_status()
    
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    # Verify hash if provided
    if verify_hash and info['hash'] is not None:
        if not _verify_file_hash(file_path, info['hash']):
            raise ValueError(f"Hash verification failed for {name}")
    
    print(f"Downloaded {name} to {file_path}")
    return str(file_path)


def _verify_file_hash(file_path: Path, expected_hash: str) -> bool:
    """Verify file hash."""
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() == expected_hash