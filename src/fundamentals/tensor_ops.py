# src/fundamentals/tensor_ops.py
"""
Tensor operation helpers for PyTorch Mastery Hub
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional, List, Dict, Any


def safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Perform safe division avoiding division by zero.
    
    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor  
        epsilon: Small value to add to denominator
        
    Returns:
        Result of safe division
    """
    return numerator / (denominator + epsilon)


def batch_matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Batch matrix multiplication with error checking.
    
    Args:
        a: First tensor [batch_size, m, k]
        b: Second tensor [batch_size, k, n]
        
    Returns:
        Result tensor [batch_size, m, n]
    """
    if a.dim() != 3 or b.dim() != 3:
        raise ValueError("Input tensors must be 3-dimensional")
    
    if a.size(0) != b.size(0):
        raise ValueError("Batch sizes must match")
    
    if a.size(2) != b.size(1):
        raise ValueError(f"Matrix dimensions incompatible: {a.size(2)} != {b.size(1)}")
    
    return torch.bmm(a, b)


def tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    """
    Compute comprehensive statistics for a tensor.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Dictionary with tensor statistics
    """
    with torch.no_grad():
        flat_tensor = tensor.view(-1)
        
        stats = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
            'numel': tensor.numel(),
            'mean': float(tensor.mean()),
            'std': float(tensor.std()),
            'min': float(tensor.min()),
            'max': float(tensor.max()),
            'median': float(tensor.median()),
            'sum': float(tensor.sum()),
            'norm': float(tensor.norm()),
        }
        
        # Percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        sorted_values, _ = torch.sort(flat_tensor)
        for p in percentiles:
            idx = int((p / 100.0) * (len(sorted_values) - 1))
            stats[f'p{p}'] = float(sorted_values[idx])
        
        # Sparsity (percentage of zeros)
        zero_count = (tensor == 0).sum().item()
        stats['sparsity'] = (zero_count / tensor.numel()) * 100
        
        # Memory usage (approximate)
        stats['memory_mb'] = (tensor.numel() * tensor.element_size()) / (1024 * 1024)
        
        return stats


def tensor_summary(tensor: torch.Tensor, name: str = "Tensor") -> None:
    """
    Print a formatted summary of tensor statistics.
    
    Args:
        tensor: Input tensor
        name: Name for the tensor
    """
    stats = tensor_stats(tensor)
    
    print(f"\n{name} Summary:")
    print("=" * 50)
    print(f"Shape: {stats['shape']}")
    print(f"Data Type: {stats['dtype']}")
    print(f"Device: {stats['device']}")
    print(f"Total Elements: {stats['numel']:,}")
    print(f"Memory Usage: {stats['memory_mb']:.2f} MB")
    print(f"Sparsity: {stats['sparsity']:.2f}%")
    print()
    print(f"Mean: {stats['mean']:.6f}")
    print(f"Std:  {stats['std']:.6f}")
    print(f"Min:  {stats['min']:.6f}")
    print(f"Max:  {stats['max']:.6f}")
    print(f"Norm: {stats['norm']:.6f}")
    print()
    print("Percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  P{p}: {stats[f'p{p}']:.6f}")


def reshape_tensor(
    tensor: torch.Tensor,
    new_shape: Union[List[int], Tuple[int, ...]],
    validate: bool = True
) -> torch.Tensor:
    """
    Reshape tensor with validation.
    
    Args:
        tensor: Input tensor
        new_shape: New shape
        validate: Whether to validate the reshape
        
    Returns:
        Reshaped tensor
    """
    if validate:
        original_numel = tensor.numel()
        new_numel = np.prod(new_shape)
        
        if original_numel != new_numel:
            raise ValueError(
                f"Cannot reshape tensor: original size {original_numel} "
                f"does not match new size {new_numel}"
            )
    
    return tensor.view(new_shape)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array.
    
    Args:
        tensor: Input tensor
        
    Returns:
        NumPy array
    """
    return tensor.detach().cpu().numpy()


def numpy_to_tensor(
    array: np.ndarray,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    requires_grad: bool = False
) -> torch.Tensor:
    """
    Convert NumPy array to PyTorch tensor.
    
    Args:
        array: Input NumPy array
        dtype: Desired tensor dtype
        device: Target device
        requires_grad: Whether tensor requires gradients
        
    Returns:
        PyTorch tensor
    """
    tensor = torch.from_numpy(array.copy())
    
    if dtype is not None:
        tensor = tensor.to(dtype)
    
    if device is not None:
        tensor = tensor.to(device)
    
    if requires_grad:
        tensor.requires_grad_(True)
    
    return tensor


def create_tensor_like(
    reference: torch.Tensor,
    fill_value: Optional[float] = None,
    requires_grad: Optional[bool] = None
) -> torch.Tensor:
    """
    Create a new tensor with same shape and properties as reference.
    
    Args:
        reference: Reference tensor
        fill_value: Value to fill tensor with
        requires_grad: Whether new tensor requires gradients
        
    Returns:
        New tensor
    """
    if fill_value is None:
        tensor = torch.empty_like(reference)
    elif fill_value == 0:
        tensor = torch.zeros_like(reference)
    elif fill_value == 1:
        tensor = torch.ones_like(reference)
    else:
        tensor = torch.full_like(reference, fill_value)
    
    if requires_grad is not None:
        tensor.requires_grad_(requires_grad)
    
    return tensor


def concatenate_tensors(
    tensors: List[torch.Tensor],
    dim: int = 0,
    validate_shapes: bool = True
) -> torch.Tensor:
    """
    Concatenate tensors along specified dimension.
    
    Args:
        tensors: List of tensors to concatenate
        dim: Dimension to concatenate along
        validate_shapes: Whether to validate tensor shapes
        
    Returns:
        Concatenated tensor
    """
    if not tensors:
        raise ValueError("Cannot concatenate empty list of tensors")
    
    if len(tensors) == 1:
        return tensors[0]
    
    if validate_shapes:
        reference_shape = list(tensors[0].shape)
        for i, tensor in enumerate(tensors[1:], 1):
            current_shape = list(tensor.shape)
            
            # Check all dimensions except concatenation dimension
            for d in range(len(reference_shape)):
                if d != dim and reference_shape[d] != current_shape[d]:
                    raise ValueError(
                        f"Tensor {i} has incompatible shape {current_shape} "
                        f"with reference {reference_shape} at dimension {d}"
                    )
    
    return torch.cat(tensors, dim=dim)


def stack_tensors(
    tensors: List[torch.Tensor],
    dim: int = 0,
    validate_shapes: bool = True
) -> torch.Tensor:
    """
    Stack tensors along new dimension.
    
    Args:
        tensors: List of tensors to stack
        dim: Dimension to stack along
        validate_shapes: Whether to validate tensor shapes
        
    Returns:
        Stacked tensor
    """
    if not tensors:
        raise ValueError("Cannot stack empty list of tensors")
    
    if validate_shapes:
        reference_shape = tensors[0].shape
        for i, tensor in enumerate(tensors[1:], 1):
            if tensor.shape != reference_shape:
                raise ValueError(
                    f"Tensor {i} has incompatible shape {tensor.shape} "
                    f"with reference {reference_shape}"
                )
    
    return torch.stack(tensors, dim=dim)


def split_tensor(
    tensor: torch.Tensor,
    split_size_or_sections: Union[int, List[int]],
    dim: int = 0
) -> List[torch.Tensor]:
    """
    Split tensor into chunks.
    
    Args:
        tensor: Input tensor
        split_size_or_sections: Size of each chunk or list of sizes
        dim: Dimension to split along
        
    Returns:
        List of tensor chunks
    """
    if isinstance(split_size_or_sections, int):
        return list(torch.split(tensor, split_size_or_sections, dim=dim))
    else:
        return list(torch.split(tensor, split_size_or_sections, dim=dim))


def tensor_memory_usage(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Get detailed memory usage information for tensor.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Dictionary with memory usage details
    """
    element_size = tensor.element_size()
    numel = tensor.numel()
    total_bytes = element_size * numel
    
    return {
        'element_size_bytes': element_size,
        'num_elements': numel,
        'total_bytes': total_bytes,
        'total_kb': total_bytes / 1024,
        'total_mb': total_bytes / (1024 * 1024),
        'total_gb': total_bytes / (1024 * 1024 * 1024),
        'dtype': str(tensor.dtype),
        'shape': list(tensor.shape)
    }


def flatten_tensor(
    tensor: torch.Tensor,
    start_dim: int = 0,
    end_dim: int = -1
) -> torch.Tensor:
    """
    Flatten tensor dimensions.
    
    Args:
        tensor: Input tensor
        start_dim: First dim to flatten
        end_dim: Last dim to flatten
        
    Returns:
        Flattened tensor
    """
    return torch.flatten(tensor, start_dim, end_dim)


def expand_tensor(
    tensor: torch.Tensor,
    sizes: Union[List[int], Tuple[int, ...]]
) -> torch.Tensor:
    """
    Expand tensor to larger size.
    
    Args:
        tensor: Input tensor
        sizes: Target sizes
        
    Returns:
        Expanded tensor
    """
    return tensor.expand(*sizes)


def squeeze_tensor(
    tensor: torch.Tensor,
    dim: Optional[int] = None
) -> torch.Tensor:
    """
    Remove single-dimensional entries.
    
    Args:
        tensor: Input tensor
        dim: Dimension to squeeze (optional)
        
    Returns:
        Squeezed tensor
    """
    if dim is not None:
        return tensor.squeeze(dim)
    return tensor.squeeze()


def unsqueeze_tensor(
    tensor: torch.Tensor,
    dim: int
) -> torch.Tensor:
    """
    Add single-dimensional entry.
    
    Args:
        tensor: Input tensor
        dim: Position to add dimension
        
    Returns:
        Unsqueezed tensor
    """
    return tensor.unsqueeze(dim)