# src/advanced/optimization.py
"""
Model optimization techniques for PyTorch Mastery Hub
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Tuple, Optional, Any
import time
import numpy as np


class ModelQuantizer:
    """Model quantization utilities."""
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def dynamic_quantize(self, dtype: torch.dtype = torch.qint8) -> nn.Module:
        """Apply dynamic quantization."""
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=dtype
        )
        return quantized_model
    
    def static_quantize(self, calibration_data_loader) -> nn.Module:
        """Apply static quantization with calibration."""
        # Prepare model for quantization
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        prepared_model = torch.quantization.prepare(self.model)
        
        # Calibrate with sample data
        prepared_model.eval()
        with torch.no_grad():
            for batch, _ in calibration_data_loader:
                prepared_model(batch)
                break  # Single batch for demo
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        return quantized_model
    
    def qat_quantize(self, train_loader, num_epochs: int = 3) -> nn.Module:
        """Quantization-aware training."""
        # Prepare model for QAT
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        prepared_model = torch.quantization.prepare_qat(self.model)
        
        # Training loop (simplified)
        optimizer = torch.optim.Adam(prepared_model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            for batch, targets in train_loader:
                optimizer.zero_grad()
                outputs = prepared_model(batch)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Convert to quantized model
        prepared_model.eval()
        quantized_model = torch.quantization.convert(prepared_model)
        return quantized_model


class ModelPruner:
    """Model pruning utilities."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.pruned_modules = []
    
    def magnitude_pruning(self, amount: float = 0.2) -> nn.Module:
        """Apply magnitude-based pruning."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.l1_unstructured(module, name='weight', amount=amount)
                self.pruned_modules.append((module, 'weight'))
        
        return self.model
    
    def structured_pruning(self, amount: float = 0.2) -> nn.Module:
        """Apply structured pruning (channel-wise)."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                self.pruned_modules.append((module, 'weight'))
        
        return self.model
    
    def global_pruning(self, amount: float = 0.2) -> nn.Module:
        """Apply global magnitude pruning."""
        parameters_to_prune = []
        
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
        
        self.pruned_modules = parameters_to_prune
        return self.model
    
    def remove_pruning_masks(self):
        """Remove pruning masks to make pruning permanent."""
        for module, param_name in self.pruned_modules:
            prune.remove(module, param_name)
    
    def get_sparsity(self) -> Dict[str, float]:
        """Calculate model sparsity."""
        sparsity_info = {}
        total_params = 0
        total_zeros = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight
                module_zeros = (weight == 0).sum().item()
                module_total = weight.numel()
                
                sparsity_info[name] = module_zeros / module_total
                total_zeros += module_zeros
                total_params += module_total
        
        sparsity_info['global_sparsity'] = total_zeros / total_params
        return sparsity_info


class KnowledgeDistillation:
    """Knowledge distillation trainer."""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 3.0,
        alpha: float = 0.5
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def distillation_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate knowledge distillation loss."""
        # Soft targets from teacher
        teacher_soft = torch.softmax(teacher_outputs / self.temperature, dim=1)
        student_soft = torch.log_softmax(student_outputs / self.temperature, dim=1)
        
        # KL divergence loss
        kd_loss = nn.KLDivLoss(reduction='batchmean')(student_soft, teacher_soft)
        kd_loss = kd_loss * (self.temperature ** 2)
        
        # Standard cross-entropy loss
        ce_loss = nn.CrossEntropyLoss()(student_outputs, targets)
        
        # Combined loss
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        
        return total_loss, kd_loss, ce_loss
    
    def train_step(
        self,
        batch: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Perform one training step with knowledge distillation."""
        self.student_model.train()
        
        # Forward pass
        with torch.no_grad():
            teacher_outputs = self.teacher_model(batch)
        
        student_outputs = self.student_model(batch)
        
        # Calculate losses
        total_loss, kd_loss, ce_loss = self.distillation_loss(
            student_outputs, teacher_outputs, targets
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'kd_loss': kd_loss.item(),
            'ce_loss': ce_loss.item()
        }


def optimize_model(
    model: nn.Module,
    optimization_type: str = 'quantization',
    **kwargs
) -> nn.Module:
    """
    Apply model optimization.
    
    Args:
        model: PyTorch model
        optimization_type: Type of optimization ('quantization', 'pruning', 'distillation')
        **kwargs: Additional optimization parameters
        
    Returns:
        Optimized model
    """
    if optimization_type == 'quantization':
        quantizer = ModelQuantizer(model)
        if kwargs.get('method', 'dynamic') == 'dynamic':
            return quantizer.dynamic_quantize()
        elif kwargs.get('method') == 'static':
            return quantizer.static_quantize(kwargs.get('calibration_data'))
    
    elif optimization_type == 'pruning':
        pruner = ModelPruner(model)
        method = kwargs.get('method', 'magnitude')
        amount = kwargs.get('amount', 0.2)
        
        if method == 'magnitude':
            return pruner.magnitude_pruning(amount)
        elif method == 'structured':
            return pruner.structured_pruning(amount)
        elif method == 'global':
            return pruner.global_pruning(amount)
    
    else:
        raise ValueError(f"Unknown optimization type: {optimization_type}")


def profile_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device = torch.device('cpu'),
    num_runs: int = 100
) -> Dict[str, Any]:
    """
    Profile model performance.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to profile on
        num_runs: Number of inference runs
        
    Returns:
        Profiling results
    """
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Profile inference time
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / num_runs
    
    # Calculate model size and parameters
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32
    
    # Memory usage (approximate)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated()
        _ = model(dummy_input)
        memory_after = torch.cuda.memory_allocated()
        memory_usage_mb = (memory_after - memory_before) / (1024 * 1024)
    else:
        memory_usage_mb = 0  # CPU memory profiling is more complex
    
    return {
        'avg_inference_time': avg_inference_time,
        'throughput_fps': 1.0 / avg_inference_time,
        'parameter_count': param_count,
        'model_size_mb': model_size_mb,
        'memory_usage_mb': memory_usage_mb,
        'device': str(device)
    }


def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_sizes: List[int] = [1, 8, 16, 32],
    devices: Optional[List[torch.device]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Comprehensive model benchmarking.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch dimension)
        batch_sizes: List of batch sizes to test
        devices: List of devices to test on
        
    Returns:
        Benchmarking results
    """
    if devices is None:
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
    
    results = {}
    
    for device in devices:
        device_results = {}
        
        for batch_size in batch_sizes:
            full_input_shape = (batch_size,) + input_shape
            
            try:
                profile_result = profile_model(model, full_input_shape, device)
                device_results[f'batch_{batch_size}'] = profile_result
            except Exception as e:
                device_results[f'batch_{batch_size}'] = {'error': str(e)}
        
        results[str(device)] = device_results
    
    return results


class ModelOptimizer:
    """
    Comprehensive model optimizer combining multiple techniques.
    """
    
    def __init__(self, model: nn.Module):
        self.original_model = model
        self.current_model = model
        self.optimization_history = []
    
    def apply_pruning(self, amount: float = 0.2, method: str = 'magnitude') -> 'ModelOptimizer':
        """Apply pruning optimization."""
        pruner = ModelPruner(self.current_model)
        
        if method == 'magnitude':
            self.current_model = pruner.magnitude_pruning(amount)
        elif method == 'structured':
            self.current_model = pruner.structured_pruning(amount)
        elif method == 'global':
            self.current_model = pruner.global_pruning(amount)
        
        self.optimization_history.append({
            'type': 'pruning',
            'method': method,
            'amount': amount,
            'sparsity': pruner.get_sparsity()
        })
        
        return self
    
    def apply_quantization(self, method: str = 'dynamic') -> 'ModelOptimizer':
        """Apply quantization optimization."""
        quantizer = ModelQuantizer(self.current_model)
        
        if method == 'dynamic':
            self.current_model = quantizer.dynamic_quantize()
        
        self.optimization_history.append({
            'type': 'quantization',
            'method': method
        })
        
        return self
    
    def get_optimized_model(self) -> nn.Module:
        """Get the optimized model."""
        return self.current_model
    
    def compare_models(self, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Compare original and optimized models."""
        original_profile = profile_model(self.original_model, input_shape)
        optimized_profile = profile_model(self.current_model, input_shape)
        
        size_reduction = (
            1 - optimized_profile['model_size_mb'] / original_profile['model_size_mb']
        ) * 100
        
        speed_improvement = (
            original_profile['avg_inference_time'] / optimized_profile['avg_inference_time'] - 1
        ) * 100
        
        return {
            'original': original_profile,
            'optimized': optimized_profile,
            'improvements': {
                'size_reduction_percent': size_reduction,
                'speed_improvement_percent': speed_improvement
            },
            'optimization_history': self.optimization_history
        }