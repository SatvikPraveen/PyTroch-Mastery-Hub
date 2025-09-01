# tests/test_utils/test_data_utils.py
"""
Tests for data utilities
"""

import pytest
import torch
import numpy as np
from torch.utils.data import TensorDataset
from utils.data_utils import (
    load_dataset, create_data_loaders, train_val_split,
    normalize_data, generate_synthetic_data, get_dataset_info
)


class TestLoadDataset:
    """Test dataset loading functions."""
    
    def test_load_iris_dataset(self):
        """Test loading Iris dataset."""
        dataset_info = load_dataset('iris')
        
        assert 'train_dataset' in dataset_info
        assert 'test_dataset' in dataset_info
        assert dataset_info['num_classes'] == 3
        assert dataset_info['input_shape'] == (4,)
        assert len(dataset_info['classes']) == 3
    
    @pytest.mark.slow
    def test_load_mnist_dataset(self):
        """Test loading MNIST dataset."""
        dataset_info = load_dataset('mnist', download=False)
        
        assert dataset_info['num_classes'] == 10
        assert dataset_info['input_shape'] == (1, 28, 28)
        assert len(dataset_info['classes']) == 10
    
    def test_load_unknown_dataset(self):
        """Test loading unknown dataset raises error."""
        with pytest.raises(ValueError, match="Dataset 'unknown' not supported"):
            load_dataset('unknown')


class TestCreateDataLoaders:
    """Test data loader creation."""
    
    def test_create_train_loader(self, sample_dataset):
        """Test creating training data loader."""
        loaders = create_data_loaders(sample_dataset, batch_size=16)
        
        assert 'train' in loaders
        assert loaders['train'].batch_size == 16
    
    def test_create_all_loaders(self, sample_dataset):
        """Test creating train, val, and test loaders."""
        train_dataset, val_dataset = train_val_split(sample_dataset, val_ratio=0.2)
        
        loaders = create_data_loaders(
            train_dataset, 
            test_dataset=val_dataset, 
            val_dataset=val_dataset,
            batch_size=8
        )
        
        assert 'train' in loaders
        assert 'test' in loaders
        assert 'val' in loaders
        assert all(loader.batch_size == 8 for loader in loaders.values())


class TestTrainValSplit:
    """Test train/validation split functionality."""
    
    def test_split_ratio(self, sample_dataset):
        """Test splitting with specific ratio."""
        train_dataset, val_dataset = train_val_split(sample_dataset, val_ratio=0.3)
        
        total_size = len(sample_dataset)
        expected_val_size = int(total_size * 0.3)
        expected_train_size = total_size - expected_val_size
        
        assert len(train_dataset) == expected_train_size
        assert len(val_dataset) == expected_val_size
    
    def test_split_reproducibility(self, sample_dataset):
        """Test that split is reproducible with same seed."""
        train1, val1 = train_val_split(sample_dataset, random_seed=42)
        train2, val2 = train_val_split(sample_dataset, random_seed=42)
        
        # Check that indices are the same
        assert train1.indices == train2.indices
        assert val1.indices == val2.indices


class TestNormalizeData:
    """Test data normalization functions."""
    
    def test_standard_normalization_tensor(self):
        """Test standard normalization with tensor input."""
        data = torch.randn(100, 5) * 10 + 5  # Mean ~5, std ~10
        normalized, scaler = normalize_data(data, method='standard')
        
        assert isinstance(normalized, torch.Tensor)
        assert normalized.shape == data.shape
        assert abs(normalized.mean().item()) < 0.1  # Close to 0
        assert abs(normalized.std().item() - 1.0) < 0.1  # Close to 1
    
    def test_minmax_normalization(self):
        """Test min-max normalization."""
        data = torch.randn(100, 5) * 10 + 5
        normalized, scaler = normalize_data(data, method='minmax')
        
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
    
    def test_normalization_with_fitted_scaler(self):
        """Test using pre-fitted scaler."""
        train_data = torch.randn(100, 5)
        test_data = torch.randn(50, 5)
        
        # Fit on training data
        _, scaler = normalize_data(train_data, method='standard')
        
        # Apply to test data
        normalized_test, _ = normalize_data(test_data, method='standard', fitted_scaler=scaler)
        
        assert normalized_test.shape == test_data.shape


class TestGenerateSyntheticData:
    """Test synthetic data generation."""
    
    def test_generate_classification_data(self):
        """Test generating synthetic classification data."""
        X, y = generate_synthetic_data(
            task='classification', 
            n_samples=100, 
            n_features=5, 
            n_classes=3
        )
        
        assert X.shape == (100, 5)
        assert y.shape == (100,)
        assert y.dtype == torch.long
        assert y.min() >= 0
        assert y.max() < 3
    
    def test_generate_regression_data(self):
        """Test generating synthetic regression data."""
        X, y = generate_synthetic_data(
            task='regression', 
            n_samples=50, 
            n_features=3
        )
        
        assert X.shape == (50, 3)
        assert y.shape == (50,)
        assert y.dtype == torch.float32
    
    def test_invalid_task_raises_error(self):
        """Test that invalid task raises error."""
        with pytest.raises(ValueError, match="Unknown task"):
            generate_synthetic_data(task='invalid_task')


class TestGetDatasetInfo:
    """Test dataset information extraction."""
    
    def test_tensor_dataset_info(self, sample_dataset):
        """Test getting info from tensor dataset."""
        info = get_dataset_info(sample_dataset)
        
        assert 'size' in info
        assert info['size'] == len(sample_dataset)
        assert 'sample_shape' in info
        assert 'data_type' in info
    
    def test_empty_dataset_info(self):
        """Test getting info from empty dataset."""
        empty_dataset = TensorDataset(torch.empty(0, 5), torch.empty(0))
        info = get_dataset_info(empty_dataset)
        
        assert info['size'] == 0
        assert info['sample_shape'] is None