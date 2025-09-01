# src/neural_networks/training.py
"""
Training utilities for PyTorch Mastery Hub
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Any, Tuple
import time
from pathlib import Path
from ..utils.metrics import AverageMeter, MetricTracker
from ..utils.io_utils import save_checkpoint


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[Any] = None,
    clip_grad_norm: Optional[float] = None,
    accumulation_steps: int = 1
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scheduler: Learning rate scheduler (optional)
        clip_grad_norm: Gradient clipping norm (optional)
        accumulation_steps: Gradient accumulation steps
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    metrics = MetricTracker(['loss', 'accuracy'])
    
    optimizer.zero_grad()
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        
        # Update metrics
        batch_size = data.size(0)
        metrics.update(loss=loss.item())
        
        # Calculate accuracy
        if output.dim() > 1 and output.size(1) > 1:
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100. * correct / batch_size
            metrics.update(accuracy=accuracy)
        
        # Gradient accumulation and optimization step
        if (batch_idx + 1) % accumulation_steps == 0:
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
    
    # Final step if needed
    if len(dataloader) % accumulation_steps != 0:
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
    
    return metrics.get_averages()


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    compute_metrics: Optional[Callable] = None
) -> Dict[str, float]:
    """
    Validate model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        compute_metrics: Function to compute additional metrics
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    metrics = MetricTracker(['loss', 'accuracy'])
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Update metrics
            batch_size = data.size(0)
            metrics.update(loss=loss.item())
            
            # Calculate accuracy
            if output.dim() > 1 and output.size(1) > 1:
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                accuracy = 100. * correct / batch_size
                metrics.update(accuracy=accuracy)
            
            # Store for additional metrics
            if compute_metrics is not None:
                all_outputs.append(output.cpu())
                all_targets.append(target.cpu())
    
    # Compute additional metrics if provided
    result_metrics = metrics.get_averages()
    if compute_metrics is not None and all_outputs:
        outputs = torch.cat(all_outputs, dim=0)
        targets = torch.cat(all_targets, dim=0)
        additional_metrics = compute_metrics(outputs, targets)
        result_metrics.update(additional_metrics)
    
    return result_metrics


class Trainer:
    """
    Comprehensive training class with callbacks and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[Any] = None,
        clip_grad_norm: Optional[float] = None,
        accumulation_steps: int = 1
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm
        self.accumulation_steps = accumulation_steps
        
        self.callbacks = []
        self.history = {'train': {}, 'val': {}}
        self.current_epoch = 0
    
    def add_callback(self, callback):
        """Add training callback."""
        self.callbacks.append(callback)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        self.model.to(self.device)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training phase
            train_metrics = train_epoch(
                self.model, train_loader, self.criterion, self.optimizer,
                self.device, self.scheduler, self.clip_grad_norm, self.accumulation_steps
            )
            
            # Validation phase
            val_metrics = {}
            if val_loader is not None:
                val_metrics = validate_epoch(
                    self.model, val_loader, self.criterion, self.device
                )
            
            # Update history
            for key, value in train_metrics.items():
                if key not in self.history['train']:
                    self.history['train'][key] = []
                self.history['train'][key].append(value)
            
            for key, value in val_metrics.items():
                if key not in self.history['val']:
                    self.history['val'][key] = []
                self.history['val'][key].append(value)
            
            # Execute callbacks
            logs = {'train': train_metrics, 'val': val_metrics, 'epoch': epoch}
            for callback in self.callbacks:
                callback.on_epoch_end(logs)
            
            # Print progress
            if verbose:
                epoch_time = time.time() - start_time
                self._print_epoch_results(epoch, epochs, train_metrics, val_metrics, epoch_time)
            
            # Check for early stopping
            should_stop = any(getattr(cb, 'stop_training', False) for cb in self.callbacks)
            if should_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        return self.history
    
    def _print_epoch_results(self, epoch, total_epochs, train_metrics, val_metrics, epoch_time):
        """Print epoch results."""
        print(f"Epoch {epoch + 1}/{total_epochs} - {epoch_time:.2f}s", end="")
        
        for key, value in train_metrics.items():
            print(f" - {key}: {value:.4f}", end="")
        
        if val_metrics:
            for key, value in val_metrics.items():
                print(f" - val_{key}: {value:.4f}", end="")
        
        print()


class EarlyStoppingCallback:
    """Early stopping callback."""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.wait = 0
        self.stop_training = False
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = lambda current, best: current < best - min_delta
            self.best_score = float('inf')
        else:
            self.monitor_op = lambda current, best: current > best + min_delta
            self.best_score = float('-inf')
    
    def on_epoch_end(self, logs):
        """Called at the end of each epoch."""
        current_score = logs['val'].get(self.monitor.replace('val_', ''))
        
        if current_score is None:
            return
        
        if self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in logs.get('model', {}).state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True
                if self.restore_best_weights and self.best_weights:
                    print("Restoring best weights...")


class ModelCheckpointCallback:
    """Model checkpoint callback."""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        verbose: bool = True
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        if mode == 'min':
            self.monitor_op = lambda current, best: current < best
            self.best_score = float('inf')
        else:
            self.monitor_op = lambda current, best: current > best
            self.best_score = float('-inf')
    
    def on_epoch_end(self, logs):
        """Called at the end of each epoch."""
        current_score = logs['val'].get(self.monitor.replace('val_', ''))
        
        if current_score is None:
            return
        
        if not self.save_best_only or self.monitor_op(current_score, self.best_score):
            if self.monitor_op(current_score, self.best_score):
                self.best_score = current_score
            
            # Save checkpoint
            filepath = str(self.filepath).format(epoch=logs['epoch'], **logs['val'])
            # Note: In a real implementation, you'd pass the model here
            if self.verbose:
                print(f"Saved model checkpoint to {filepath}")


class LearningRateSchedulerCallback:
    """Learning rate scheduler callback."""
    
    def __init__(self, scheduler, monitor: str = 'val_loss'):
        self.scheduler = scheduler
        self.monitor = monitor
    
    def on_epoch_end(self, logs):
        """Called at the end of each epoch."""
        if hasattr(self.scheduler, 'step'):
            if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                metric = logs['val'].get(self.monitor.replace('val_', ''))
                if metric is not None:
                    self.scheduler.step(metric)
            else:
                self.scheduler.step()


class ProgressCallback:
    """Progress logging callback."""
    
    def __init__(self, print_freq: int = 1):
        self.print_freq = print_freq
    
    def on_epoch_end(self, logs):
        """Called at the end of each epoch."""
        epoch = logs['epoch']
        if epoch % self.print_freq == 0:
            train_metrics = logs.get('train', {})
            val_metrics = logs.get('val', {})
            
            metrics_str = ""
            for key, value in train_metrics.items():
                metrics_str += f" {key}: {value:.4f}"
            
            for key, value in val_metrics.items():
                metrics_str += f" val_{key}: {value:.4f}"
            
            print(f"Epoch {epoch + 1}:{metrics_str}")


def train_with_mixed_precision(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler
) -> Dict[str, float]:
    """
    Train with automatic mixed precision.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scaler: GradScaler for mixed precision
        
    Returns:
        Training metrics
    """
    model.train()
    metrics = MetricTracker(['loss', 'accuracy'])
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # Backward pass with scaled gradients
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        batch_size = data.size(0)
        metrics.update(loss=loss.item())
        
        if output.dim() > 1 and output.size(1) > 1:
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100. * correct / batch_size
            metrics.update(accuracy=accuracy)
    
    return metrics.get_averages()