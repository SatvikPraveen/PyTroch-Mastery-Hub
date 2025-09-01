# src/neural_networks/optimizers.py
"""
Custom optimizers and scheduling utilities for PyTorch Mastery Hub
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
import math
from typing import Dict, Any, Optional, Union


class CustomSGD(Optimizer):
    """
    Custom SGD optimizer implementation for educational purposes.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                       weight_decay=weight_decay, nesterov=nesterov)
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
        super(CustomSGD, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)
                
                if momentum != 0:
                    param_state = self.state[p]
                    if len(param_state) == 0:
                        param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                
                p.data.add_(d_p, alpha=-group['lr'])
        
        return loss


class CustomAdam(Optimizer):
    """
    Custom Adam optimizer implementation.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(CustomAdam, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if group['amsgrad']:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                step_size = group['lr'] / bias_correction1
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


class CustomAdamW(Optimizer):
    """
    Custom AdamW optimizer implementation.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(CustomAdamW, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if group['amsgrad']:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # Weight decay (decoupled)
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                
                # Apply update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


class PolynomialDecayLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Polynomial decay learning rate scheduler.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        max_decay_steps: int,
        end_learning_rate: float = 0.0001,
        power: float = 1.0,
        last_epoch: int = -1
    ):
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]
        
        return [
            (base_lr - self.end_learning_rate) * 
            ((1 - self.last_epoch / self.max_decay_steps) ** self.power) + 
            self.end_learning_rate
            for base_lr in self.base_lrs
        ]


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with warmup.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            cosine_steps = self.last_epoch - self.warmup_steps
            cosine_max_steps = self.max_steps - self.warmup_steps
            return [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * cosine_steps / cosine_max_steps)) / 2
                for base_lr in self.base_lrs
            ]


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adam',
    lr: float = 1e-3,
    weight_decay: float = 0,
    **kwargs
) -> Optimizer:
    """
    Get optimizer by name.
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    params = model.parameters()
    
    if optimizer_name.lower() == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        nesterov = kwargs.get('nesterov', False)
        return torch.optim.SGD(params, lr=lr, momentum=momentum, 
                              weight_decay=weight_decay, nesterov=nesterov)
    
    elif optimizer_name.lower() == 'adam':
        betas = kwargs.get('betas', (0.9, 0.999))
        eps = kwargs.get('eps', 1e-8)
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, 
                               weight_decay=weight_decay)
    
    elif optimizer_name.lower() == 'adamw':
        betas = kwargs.get('betas', (0.9, 0.999))
        eps = kwargs.get('eps', 1e-8)
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, 
                                weight_decay=weight_decay)
    
    elif optimizer_name.lower() == 'rmsprop':
        alpha = kwargs.get('alpha', 0.99)
        eps = kwargs.get('eps', 1e-8)
        momentum = kwargs.get('momentum', 0)
        return torch.optim.RMSprop(params, lr=lr, alpha=alpha, eps=eps,
                                  weight_decay=weight_decay, momentum=momentum)
    
    elif optimizer_name.lower() == 'adagrad':
        eps = kwargs.get('eps', 1e-10)
        return torch.optim.Adagrad(params, lr=lr, eps=eps, weight_decay=weight_decay)
    
    elif optimizer_name.lower() == 'custom_sgd':
        return CustomSGD(params, lr=lr, weight_decay=weight_decay, **kwargs)
    
    elif optimizer_name.lower() == 'custom_adam':
        return CustomAdam(params, lr=lr, weight_decay=weight_decay, **kwargs)
    
    elif optimizer_name.lower() == 'custom_adamw':
        return CustomAdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(
    optimizer: Optimizer,
    scheduler_name: str = 'cosine',
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Get learning rate scheduler by name.
    
    Args:
        optimizer: Optimizer instance
        scheduler_name: Name of scheduler
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance or None
    """
    if scheduler_name.lower() == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_name.lower() == 'multistep':
        milestones = kwargs.get('milestones', [30, 80])
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    elif scheduler_name.lower() == 'exponential':
        gamma = kwargs.get('gamma', 0.95)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_name.lower() == 'cosine':
        T_max = kwargs.get('T_max', 50)
        eta_min = kwargs.get('eta_min', 0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif scheduler_name.lower() == 'reduce_on_plateau':
        mode = kwargs.get('mode', 'min')
        factor = kwargs.get('factor', 0.1)
        patience = kwargs.get('patience', 10)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )
    
    elif scheduler_name.lower() == 'polynomial':
        max_decay_steps = kwargs.get('max_decay_steps', 100)
        end_learning_rate = kwargs.get('end_learning_rate', 0.0001)
        power = kwargs.get('power', 1.0)
        return PolynomialDecayLR(optimizer, max_decay_steps, end_learning_rate, power)
    
    elif scheduler_name.lower() == 'warmup_cosine':
        warmup_steps = kwargs.get('warmup_steps', 10)
        max_steps = kwargs.get('max_steps', 100)
        eta_min = kwargs.get('eta_min', 0)
        return WarmupCosineAnnealingLR(optimizer, warmup_steps, max_steps, eta_min)
    
    elif scheduler_name.lower() == 'none':
        return None
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def configure_optimizer_and_scheduler(
    model: nn.Module,
    config: Dict[str, Any]
) -> tuple:
    """
    Configure optimizer and scheduler from config.
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Optimizer configuration
    opt_config = config.get('optimizer', {})
    optimizer = get_optimizer(model, **opt_config)
    
    # Scheduler configuration
    sched_config = config.get('scheduler', {})
    scheduler = get_scheduler(optimizer, **sched_config)
    
    return optimizer, scheduler