# tests/test_fundamentals/test_tensor_ops.py
"""
Tests for tensor operations
"""

import pytest
import torch
import numpy as np
from fundamentals.tensor_ops import (
    safe_divide, batch_matrix_multiply, tensor_stats, tensor_summary,
    reshape_tensor, tensor_to_numpy, numpy_to_tensor, tensor_memory_usage
)


class TestSafeDivide:
    """Test safe division operations."""
    
    def test_safe_divide_normal(self):
        """Test safe division with normal inputs."""
        a = torch.tensor([4.0, 6.0, 8.0])
        b = torch.tensor([2.0, 3.0, 2.0])
        
        result = safe_divide(a, b)
        expected = torch.tensor([2.0, 2.0, 4.0])
        
        assert torch.allclose(result, expected)
    
    def test_safe_divide_with_zero(self):
        """Test safe division with zero denominator."""
        a = torch.tensor([4.0, 6.0, 8.0])
        b = torch.tensor([2.0, 0.0, 2.0])
        
        result = safe_divide(a, b, epsilon=1e-8)
        
        # Should handle division by zero gracefully
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert torch.isfinite(result).all()
    
    def test_safe_divide_custom_epsilon(self):
        """Test safe division with custom epsilon."""
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([0.0, 0.0])
        epsilon = 0.1
        
        result = safe_divide(a, b, epsilon=epsilon)
        expected = torch.tensor([1.0/epsilon, 2.0/epsilon])
        
        assert torch.allclose(result, expected)


class TestBatchMatrixMultiply:
    """Test batch matrix multiplication."""
    
    def test_batch_matrix_multiply_valid(self):
        """Test valid batch matrix multiplication."""
        batch_size = 4
        a = torch.randn(batch_size, 3, 5)
        b = torch.randn(batch_size, 5, 2)
        
        result = batch_matrix_multiply(a, b)
        
        assert result.shape == (batch_size, 3, 2)
        
        # Compare with torch.bmm
        expected = torch.bmm(a, b)
        assert torch.allclose(result, expected, atol=1e-6)
    
    def test_batch_matrix_multiply_dimension_error(self):
        """Test error with wrong dimensions."""
        a = torch.randn(4, 3)  # 2D instead of 3D
        b = torch.randn(4, 3, 2)
        
        with pytest.raises(ValueError, match="Input tensors must be 3-dimensional"):
            batch_matrix_multiply(a, b)
    
    def test_batch_matrix_multiply_batch_mismatch(self):
        """Test error with mismatched batch sizes."""
        a = torch.randn(4, 3, 5)
        b = torch.randn(3, 5, 2)  # Different batch size
        
        with pytest.raises(ValueError, match="Batch sizes must match"):
            batch_matrix_multiply(a, b)
    
    def test_batch_matrix_multiply_incompatible_dims(self):
        """Test error with incompatible matrix dimensions."""
        a = torch.randn(4, 3, 5)
        b = torch.randn(4, 4, 2)  # Inner dimensions don't match
        
        with pytest.raises(ValueError, match="Matrix dimensions incompatible"):
            batch_matrix_multiply(a, b)


class TestTensorStats:
    """Test tensor statistics computation."""
    
    def test_tensor_stats_basic(self, sample_tensor):
        """Test basic tensor statistics."""
        stats = tensor_stats(sample_tensor)
        
        assert isinstance(stats, dict)
        
        # Check required keys
        required_keys = ['shape', 'dtype', 'device', 'numel', 'mean', 'std', 'min', 'max']
        for key in required_keys:
            assert key in stats
        
        # Verify values make sense
        assert stats['shape'] == list(sample_tensor.shape)
        assert stats['numel'] == sample_tensor.numel()
        assert isinstance(stats['mean'], float)
        assert isinstance(stats['std'], float)
    
    def test_tensor_stats_percentiles(self, sample_tensor):
        """Test percentile calculations in tensor stats."""
        stats = tensor_stats(sample_tensor)
        
        # Check percentiles exist
        percentiles = [25, 50, 75, 90, 95, 99]
        for p in percentiles:
            assert f'p{p}' in stats
            assert isinstance(stats[f'p{p}'], float)
        
        # Percentiles should be ordered
        assert stats['p25'] <= stats['p50'] <= stats['p75']
    
    def test_tensor_stats_sparsity(self):
        """Test sparsity calculation."""
        # Create tensor with some zeros
        tensor = torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0])
        stats = tensor_stats(tensor)
        
        expected_sparsity = (3 / 5) * 100  # 3 zeros out of 5 elements
        assert abs(stats['sparsity'] - expected_sparsity) < 1e-6
    
    def test_tensor_stats_memory_usage(self, sample_tensor):
        """Test memory usage calculation."""
        stats = tensor_stats(sample_tensor)
        
        assert 'memory_mb' in stats
        assert stats['memory_mb'] > 0
        
        # Rough check - should be reasonable
        expected_bytes = sample_tensor.numel() * sample_tensor.element_size()
        expected_mb = expected_bytes / (1024 * 1024)
        
        assert abs(stats['memory_mb'] - expected_mb) < 1e-6


class TestTensorSummary:
    """Test tensor summary printing."""
    
    def test_tensor_summary_runs(self, sample_tensor, capsys):
        """Test that tensor summary runs without error."""
        tensor_summary(sample_tensor, name="Test Tensor")
        
        captured = capsys.readouterr()
        assert "Test Tensor Summary:" in captured.out
        assert "Shape:" in captured.out
        assert "Mean:" in captured.out


class TestReshapeTensor:
    """Test tensor reshaping utilities."""
    
    def test_reshape_tensor_valid(self):
        """Test valid tensor reshaping."""
        tensor = torch.randn(2, 3, 4)
        new_shape = (6, 4)
        
        result = reshape_tensor(tensor, new_shape)
        
        assert result.shape == torch.Size(new_shape)
        assert result.numel() == tensor.numel()
    
    def test_reshape_tensor_invalid_size(self):
        """Test reshaping with invalid size."""
        tensor = torch.randn(2, 3, 4)
        new_shape = (5, 5)  # 25 != 24 elements
        
        with pytest.raises(ValueError, match="Cannot reshape tensor"):
            reshape_tensor(tensor, new_shape, validate=True)
    
    def test_reshape_tensor_no_validation(self):
        """Test reshaping without validation."""
        tensor = torch.randn(2, 3, 4)
        new_shape = (8, 3)  # Same number of elements
        
        # Should work without validation
        result = reshape_tensor(tensor, new_shape, validate=False)
        assert result.shape == torch.Size(new_shape)


class TestTensorConversions:
    """Test tensor conversion utilities."""
    
    def test_tensor_to_numpy(self, sample_tensor):
        """Test tensor to numpy conversion."""
        numpy_array = tensor_to_numpy(sample_tensor)
        
        assert isinstance(numpy_array, np.ndarray)
        assert numpy_array.shape == sample_tensor.shape
        
        # Values should be the same
        assert np.allclose(numpy_array, sample_tensor.detach().cpu().numpy())
    
    def test_numpy_to_tensor(self):
        """Test numpy to tensor conversion."""
        numpy_array = np.random.randn(3, 4)
        
        tensor = numpy_to_tensor(numpy_array)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == torch.Size(numpy_array.shape)
        assert torch.allclose(tensor, torch.from_numpy(numpy_array).float())
    
    def test_numpy_to_tensor_with_options(self):
        """Test numpy to tensor conversion with options."""
        numpy_array = np.random.randn(3, 4).astype(np.float32)
        
        tensor = numpy_to_tensor(
            numpy_array,
            dtype=torch.float64,
            device=torch.device('cpu'),
            requires_grad=True
        )
        
        assert tensor.dtype == torch.float64
        assert tensor.device.type == 'cpu'
        assert tensor.requires_grad == True
    
    def test_tensor_gpu_to_numpy(self, device):
        """Test converting GPU tensor to numpy."""
        if device.type == 'cuda':
            gpu_tensor = torch.randn(3, 4, device=device)
            numpy_array = tensor_to_numpy(gpu_tensor)
            
            assert isinstance(numpy_array, np.ndarray)
            assert numpy_array.shape == (3, 4)


class TestTensorMemoryUsage:
    """Test tensor memory usage utilities."""
    
    def test_tensor_memory_usage(self, sample_tensor):
        """Test tensor memory usage calculation."""
        memory_info = tensor_memory_usage(sample_tensor)
        
        assert isinstance(memory_info, dict)
        
        required_keys = [
            'element_size_bytes', 'num_elements', 'total_bytes',
            'total_kb', 'total_mb', 'total_gb', 'dtype', 'shape'
        ]
        for key in required_keys:
            assert key in memory_info
        
        # Verify calculations
        expected_bytes = sample_tensor.numel() * sample_tensor.element_size()
        assert memory_info['total_bytes'] == expected_bytes
        assert memory_info['total_kb'] == expected_bytes / 1024
        assert memory_info['total_mb'] == expected_bytes / (1024 * 1024)
    
    def test_tensor_memory_usage_different_dtypes(self):
        """Test memory usage with different data types."""
        float32_tensor = torch.randn(10, 10, dtype=torch.float32)
        float64_tensor = torch.randn(10, 10, dtype=torch.float64)
        
        float32_info = tensor_memory_usage(float32_tensor)
        float64_info = tensor_memory_usage(float64_tensor)
        
        # float64 should use twice as much memory as float32
        assert float64_info['total_bytes'] == 2 * float32_info['total_bytes']
        assert float64_info['element_size_bytes'] == 2 * float32_info['element_size_bytes']


class TestTensorOperationsEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_operations_with_empty_tensor(self):
        """Test operations with empty tensors."""
        empty_tensor = torch.empty(0, 5)
        
        stats = tensor_stats(empty_tensor)
        assert stats['numel'] == 0
        
        memory_info = tensor_memory_usage(empty_tensor)
        assert memory_info['total_bytes'] == 0
    
    def test_operations_with_scalar_tensor(self):
        """Test operations with scalar tensors."""
        scalar_tensor = torch.tensor(42.0)
        
        stats = tensor_stats(scalar_tensor)
        assert stats['shape'] == []
        assert stats['numel'] == 1
        assert stats['mean'] == 42.0
    
    def test_safe_divide_broadcast(self):
        """Test safe divide with broadcasting."""
        a = torch.randn(3, 4)
        b = torch.randn(4)
        
        result = safe_divide(a, b)
        
        assert result.shape == (3, 4)
        assert not torch.isnan(result).any()