# src/utils/metrics.py
"""
Custom metrics and evaluation functions for PyTorch Mastery Hub
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, topk: int = 1) -> float:
    """
    Calculate accuracy for classification tasks.
    
    Args:
        y_pred: Predicted logits/probabilities [batch_size, num_classes]
        y_true: True labels [batch_size]
        topk: Top-k accuracy (default: 1)
        
    Returns:
        Accuracy as float
    """
    with torch.no_grad():
        if topk == 1:
            _, predicted = torch.max(y_pred, 1)
            correct = (predicted == y_true).sum().item()
            total = y_true.size(0)
            return correct / total
        else:
            _, predicted = y_pred.topk(topk, 1, True, True)
            predicted = predicted.t()
            correct = predicted.eq(y_true.view(1, -1).expand_as(predicted))
            correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
            return correct_k.item() / y_true.size(0)


def top_k_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, k: int = 5) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        y_pred: Predicted logits/probabilities
        y_true: True labels
        k: k for top-k accuracy
        
    Returns:
        Top-k accuracy
    """
    return accuracy(y_pred, y_true, topk=k)


def precision_recall_f1(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    average: str = 'macro',
    num_classes: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1-score.
    
    Args:
        y_pred: Predicted logits/probabilities
        y_true: True labels
        average: Averaging method ('macro', 'micro', 'weighted')
        num_classes: Number of classes (for multi-class)
        
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    with torch.no_grad():
        if y_pred.dim() > 1:
            _, y_pred = torch.max(y_pred, 1)
        
        y_pred_np = y_pred.cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_np, y_pred_np, average=average, zero_division=0
        )
        
        return float(precision), float(recall), float(f1)


def classification_report(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive classification report.
    
    Args:
        y_pred: Predicted logits/probabilities
        y_true: True labels
        class_names: Optional class names
        
    Returns:
        Dictionary with classification metrics
    """
    with torch.no_grad():
        if y_pred.dim() > 1:
            _, y_pred = torch.max(y_pred, 1)
        
        y_pred_np = y_pred.cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        
        # Basic metrics
        acc = accuracy_score(y_true_np, y_pred_np)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_np, y_pred_np, average=None, zero_division=0
        )
        
        # Macro averages
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()
        
        # Micro averages
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true_np, y_pred_np, average='micro', zero_division=0
        )
        
        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true_np, y_pred_np, average='weighted', zero_division=0
        )
        
        num_classes = len(np.unique(y_true_np))
        if class_names is None:
            class_names = [f"Class {i}" for i in range(num_classes)]
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(class_names[:len(precision)]):
            per_class_metrics[class_name] = {
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1_score': float(f1[i]) if i < len(f1) else 0.0,
                'support': int(support[i]) if i < len(support) else 0
            }
        
        return {
            'accuracy': float(acc),
            'macro_avg': {
                'precision': float(macro_precision),
                'recall': float(macro_recall),
                'f1_score': float(macro_f1)
            },
            'micro_avg': {
                'precision': float(micro_precision),
                'recall': float(micro_recall),
                'f1_score': float(micro_f1)
            },
            'weighted_avg': {
                'precision': float(weighted_precision),
                'recall': float(weighted_recall),
                'f1_score': float(weighted_f1)
            },
            'per_class': per_class_metrics
        }


def regression_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_pred: Predicted values
        y_true: True values
        
    Returns:
        Dictionary with regression metrics
    """
    with torch.no_grad():
        y_pred_np = y_pred.cpu().numpy().flatten()
        y_true_np = y_true.cpu().numpy().flatten()
        
        mse = mean_squared_error(y_true_np, y_pred_np)
        mae = mean_absolute_error(y_true_np, y_pred_np)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_np, y_pred_np)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true_np - y_pred_np) / np.maximum(np.abs(y_true_np), 1e-8))) * 100
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'mape': float(mape)
        }


def confusion_matrix_torch(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: Optional[int] = None
) -> torch.Tensor:
    """
    Calculate confusion matrix using PyTorch.
    
    Args:
        y_pred: Predicted logits/probabilities
        y_true: True labels
        num_classes: Number of classes
        
    Returns:
        Confusion matrix as tensor
    """
    with torch.no_grad():
        if y_pred.dim() > 1:
            _, y_pred = torch.max(y_pred, 1)
        
        if num_classes is None:
            num_classes = max(y_true.max().item(), y_pred.max().item()) + 1
        
        # Create confusion matrix
        cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        
        for t, p in zip(y_true.view(-1), y_pred.view(-1)):
            cm[t.long(), p.long()] += 1
        
        return cm


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    
    def __init__(self, name: str = "", fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class MetricTracker:
    """
    Track multiple metrics during training.
    """
    
    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names
        self.metrics = {name: AverageMeter(name) for name in metric_names}
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for name, value in kwargs.items():
            if name in self.metrics:
                self.metrics[name].update(value)
    
    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()
    
    def get_averages(self) -> Dict[str, float]:
        """Get average values for all metrics."""
        return {name: metric.avg for name, metric in self.metrics.items()}
    
    def get_current(self) -> Dict[str, float]:
        """Get current values for all metrics."""
        return {name: metric.val for name, metric in self.metrics.items()}
    
    def __str__(self):
        return ', '.join([str(metric) for metric in self.metrics.values()])


def dice_coefficient(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1e-8) -> float:
    """
    Calculate Dice coefficient for segmentation tasks.
    
    Args:
        y_pred: Predicted probabilities [batch_size, num_classes, H, W]
        y_true: True labels [batch_size, num_classes, H, W] or [batch_size, H, W]
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient
    """
    with torch.no_grad():
        if y_pred.dim() == 4 and y_true.dim() == 3:
            # Convert to one-hot if needed
            num_classes = y_pred.size(1)
            y_true = F.one_hot(y_true.long(), num_classes).permute(0, 3, 1, 2).float()
        
        # Flatten tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        intersection = (y_pred * y_true).sum()
        dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
        
        return dice.item()


def iou_score(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1e-8) -> float:
    """
    Calculate Intersection over Union (IoU) score.
    
    Args:
        y_pred: Predicted probabilities
        y_true: True labels
        smooth: Smoothing factor
        
    Returns:
        IoU score
    """
    with torch.no_grad():
        # Flatten tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()


def calculate_class_weights(y: torch.Tensor, num_classes: Optional[int] = None) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y: Target labels
        num_classes: Number of classes
        
    Returns:
        Class weights tensor
    """
    if num_classes is None:
        num_classes = y.max().item() + 1
    
    # Count samples per class
    class_counts = torch.bincount(y, minlength=num_classes).float()
    
    # Calculate weights (inverse frequency)
    total_samples = len(y)
    weights = total_samples / (num_classes * class_counts)
    
    # Handle zero counts
    weights[class_counts == 0] = 0
    
    return weights


def pearson_correlation(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate Pearson correlation coefficient.
    
    Args:
        y_pred: Predicted values
        y_true: True values
        
    Returns:
        Pearson correlation coefficient
    """
    with torch.no_grad():
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        # Calculate means
        mean_pred = y_pred.mean()
        mean_true = y_true.mean()
        
        # Calculate correlation
        numerator = ((y_pred - mean_pred) * (y_true - mean_true)).sum()
        denominator = torch.sqrt(((y_pred - mean_pred) ** 2).sum() * ((y_true - mean_true) ** 2).sum())
        
        correlation = numerator / (denominator + 1e-8)
        return correlation.item()


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, model: nn.Module):
        """Save model checkpoint."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()