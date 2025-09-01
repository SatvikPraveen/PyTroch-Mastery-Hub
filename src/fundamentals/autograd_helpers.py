# src/fundamentals/autograd_helpers.py
"""
Custom autograd functions and gradient utilities for PyTorch Mastery Hub
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Tuple, Optional, Any, Callable


class LinearFunction(Function):
    """
    Custom linear function implementation for educational purposes.
    """
    
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        """Forward pass of linear function."""
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of linear function."""
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        
        return grad_input, grad_weight, grad_bias


class ReLUFunction(Function):
    """
    Custom ReLU function implementation.
    """
    
    @staticmethod
    def forward(ctx, input):
        """Forward pass of ReLU."""
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of ReLU."""
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


class SigmoidFunction(Function):
    """
    Custom Sigmoid function implementation.
    """
    
    @staticmethod
    def forward(ctx, input):
        """Forward pass of Sigmoid."""
        output = torch.sigmoid(input)
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of Sigmoid."""
        output, = ctx.saved_tensors
        grad_input = grad_output * output * (1 - output)
        return grad_input


class DropoutFunction(Function):
    """
    Custom Dropout function implementation.
    """
    
    @staticmethod
    def forward(ctx, input, p=0.5, training=True):
        """Forward pass of Dropout."""
        if training:
            mask = torch.bernoulli(torch.full_like(input, 1 - p))
            ctx.save_for_backward(mask)
            ctx.p = p
            return input * mask / (1 - p)
        else:
            return input
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of Dropout."""
        mask, = ctx.saved_tensors
        return grad_output * mask / (1 - ctx.p), None, None


class CustomFunction(Function):
    """
    Template for creating custom autograd functions.
    """
    
    @staticmethod
    def forward(ctx, *inputs):
        """
        Forward pass - implement your custom operation here.
        
        Args:
            ctx: Context object to save information for backward pass
            *inputs: Input tensors
            
        Returns:
            Output tensor(s)
        """
        # Save tensors needed for backward pass
        ctx.save_for_backward(*inputs)
        
        # Implement your forward operation
        # Example: simple element-wise square
        output = inputs[0] ** 2
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass - compute gradients.
        
        Args:
            ctx: Context object with saved information
            grad_output: Gradient of loss w.r.t. output
            
        Returns:
            Gradients w.r.t. inputs
        """
        # Retrieve saved tensors
        inputs = ctx.saved_tensors
        
        # Compute gradients
        # Example: gradient of x^2 is 2x
        grad_input = grad_output * 2 * inputs[0]
        
        return grad_input


def compute_gradients(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    grad_outputs: Optional[torch.Tensor] = None,
    retain_graph: bool = False,
    create_graph: bool = False
) -> torch.Tensor:
    """
    Compute gradients of outputs with respect to inputs.
    
    Args:
        outputs: Output tensor
        inputs: Input tensor
        grad_outputs: Gradient of outputs (optional)
        retain_graph: Whether to retain computational graph
        create_graph: Whether to create graph for higher-order derivatives
        
    Returns:
        Gradients of outputs w.r.t. inputs
    """
    return torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
        only_inputs=True
    )[0]


def gradient_check(
    func: Callable,
    inputs: torch.Tensor,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3
) -> bool:
    """
    Numerical gradient checking for custom functions.
    
    Args:
        func: Function to check
        inputs: Input tensor
        eps: Epsilon for numerical differentiation
        atol: Absolute tolerance
        rtol: Relative tolerance
        
    Returns:
        True if gradients match within tolerance
    """
    inputs.requires_grad_(True)
    
    # Analytical gradient
    output = func(inputs)
    if output.dim() > 0:
        output = output.sum()  # Reduce to scalar for gradient computation
    
    analytical_grad = torch.autograd.grad(output, inputs, create_graph=False)[0]
    
    # Numerical gradient
    numerical_grad = torch.zeros_like(inputs)
    flat_inputs = inputs.view(-1)
    flat_grad = numerical_grad.view(-1)
    
    for i in range(flat_inputs.numel()):
        # f(x + eps)
        flat_inputs[i] += eps
        output_pos = func(inputs)
        if output_pos.dim() > 0:
            output_pos = output_pos.sum()
        
        # f(x - eps)
        flat_inputs[i] -= 2 * eps
        output_neg = func(inputs)
        if output_neg.dim() > 0:
            output_neg = output_neg.sum()
        
        # Numerical derivative
        flat_grad[i] = (output_pos - output_neg) / (2 * eps)
        
        # Restore original value
        flat_inputs[i] += eps
    
    # Check if gradients match
    return torch.allclose(analytical_grad, numerical_grad, atol=atol, rtol=rtol)


class GradientClipping:
    """
    Gradient clipping utilities.
    """
    
    @staticmethod
    def clip_grad_norm(parameters, max_norm: float, norm_type: float = 2.0):
        """
        Clip gradients by norm.
        
        Args:
            parameters: Model parameters
            max_norm: Maximum gradient norm
            norm_type: Type of norm to compute
            
        Returns:
            Total norm before clipping
        """
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)
    
    @staticmethod
    def clip_grad_value(parameters, clip_value: float):
        """
        Clip gradients by value.
        
        Args:
            parameters: Model parameters
            clip_value: Maximum gradient value
        """
        torch.nn.utils.clip_grad_value_(parameters, clip_value)


def hook_gradient(tensor: torch.Tensor, hook_fn: Callable[[torch.Tensor], torch.Tensor]):
    """
    Register a backward hook on a tensor.
    
    Args:
        tensor: Tensor to hook
        hook_fn: Hook function that modifies gradients
        
    Returns:
        Hook handle
    """
    return tensor.register_hook(hook_fn)


def zero_gradients(parameters):
    """
    Zero out gradients for given parameters.
    
    Args:
        parameters: Model parameters or optimizer
    """
    if hasattr(parameters, 'zero_grad'):
        parameters.zero_grad()
    else:
        for param in parameters:
            if param.grad is not None:
                param.grad.zero_()


class GradientAccumulator:
    """
    Utility for gradient accumulation.
    """
    
    def __init__(self, model: nn.Module, accumulation_steps: int = 1):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.accumulated_loss = 0.0
    
    def accumulate(self, loss: torch.Tensor) -> bool:
        """
        Accumulate gradients from loss.
        
        Args:
            loss: Loss tensor
            
        Returns:
            True if ready to update parameters
        """
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        self.accumulated_loss += loss.item()
        self.current_step += 1
        
        # Check if ready to update
        if self.current_step >= self.accumulation_steps:
            return True
        return False
    
    def reset(self):
        """Reset accumulator."""
        self.current_step = 0
        self.accumulated_loss = 0.0
    
    def get_averaged_loss(self) -> float:
        """Get accumulated loss averaged over steps."""
        if self.current_step == 0:
            return 0.0
        return self.accumulated_loss / self.current_step


def compute_jacobian(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    create_graph: bool = False
) -> torch.Tensor:
    """
    Compute Jacobian matrix of outputs w.r.t. inputs.
    
    Args:
        outputs: Output tensor [batch_size, output_dim]
        inputs: Input tensor [batch_size, input_dim]
        create_graph: Whether to create computational graph
        
    Returns:
        Jacobian matrix [batch_size, output_dim, input_dim]
    """
    batch_size = outputs.size(0)
    output_dim = outputs.size(1)
    input_dim = inputs.size(1)
    
    jacobian = torch.zeros(batch_size, output_dim, input_dim)
    
    for i in range(output_dim):
        grad_outputs = torch.zeros_like(outputs)
        grad_outputs[:, i] = 1.0
        
        grads = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=create_graph,
            only_inputs=True
        )[0]
        
        jacobian[:, i, :] = grads
    
    return jacobian


def compute_hessian(
    output: torch.Tensor,
    inputs: torch.Tensor
) -> torch.Tensor:
    """
    Compute Hessian matrix of scalar output w.r.t. inputs.
    
    Args:
        output: Scalar output tensor
        inputs: Input tensor [input_dim]
        
    Returns:
        Hessian matrix [input_dim, input_dim]
    """
    if output.numel() != 1:
        raise ValueError("Output must be a scalar for Hessian computation")
    
    input_dim = inputs.numel()
    hessian = torch.zeros(input_dim, input_dim)
    
    # Compute first-order gradients
    first_grads = torch.autograd.grad(
        outputs=output,
        inputs=inputs,
        create_graph=True,
        retain_graph=True
    )[0].view(-1)
    
    # Compute second-order gradients
    for i in range(input_dim):
        if first_grads[i].requires_grad:
            second_grads = torch.autograd.grad(
                outputs=first_grads[i],
                inputs=inputs,
                retain_graph=True,
                allow_unused=True
            )[0]
            
            if second_grads is not None:
                hessian[i, :] = second_grads.view(-1)
    
    return hessian