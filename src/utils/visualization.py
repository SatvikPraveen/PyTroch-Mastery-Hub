"""
Visualization utilities for PyTorch Mastery Hub
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Union

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot training and validation curves for loss and accuracy.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses (optional)
        train_accs: Training accuracies (optional)
        val_accs: Validation accuracies (optional)
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    n_plots = 1 + (1 if train_accs is not None else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses is not None:
        axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if train_accs is not None:
        axes[1].plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
        
        if val_accs is not None:
            axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_tensor_as_image(
    tensor: torch.Tensor,
    title: str = "Tensor as Image",
    cmap: str = 'gray',
    normalize: bool = True,
    figsize: Tuple[int, int] = (6, 6)
) -> plt.Figure:
    """
    Plot a tensor as an image.
    
    Args:
        tensor: Input tensor (C, H, W) or (H, W)
        title: Plot title
        cmap: Colormap
        normalize: Whether to normalize values to [0, 1]
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Convert tensor to numpy
    if isinstance(tensor, torch.Tensor):
        img = tensor.detach().cpu().numpy()
    else:
        img = tensor
    
    # Handle different tensor shapes
    if len(img.shape) == 3:
        if img.shape[0] in [1, 3]:  # (C, H, W)
            img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:  # (H, W, 1)
            img = img.squeeze(2)
    
    # Normalize if requested
    if normalize and img.max() > 1.0:
        img = (img - img.min()) / (img.max() - img.min())
    
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')
    
    return fig


def plot_gradient_flow(named_parameters, title: str = "Gradient Flow") -> plt.Figure:
    """
    Plot gradient flow through network layers.
    
    Args:
        named_parameters: Model's named parameters
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    ave_grads = []
    max_grads = []
    layers = []
    
    for name, param in named_parameters:
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().cpu().item())
            max_grads.append(param.grad.abs().max().cpu().item())
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x_pos = np.arange(len(layers))
    ax.bar(x_pos, max_grads, alpha=0.5, label="Max Gradient", color='blue')
    ax.bar(x_pos, ave_grads, alpha=0.8, label="Average Gradient", color='red')
    
    ax.set_xlabel("Layers")
    ax.set_ylabel("Gradient Magnitude")
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        normalize: Whether to normalize
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)
    plt.tight_layout()
    
    return fig


def visualize_model_architecture(model: nn.Module, input_size: Tuple[int, ...]) -> None:
    """
    Print a summary of model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (without batch dimension)
    """
    try:
        from torchsummary import summary
        summary(model, input_size)
    except ImportError:
        print("torchsummary not installed. Install with: pip install torchsummary")
        print("\nModel Architecture:")
        print(model)


def plot_learning_rate_schedule(
    optimizer_history: List[float],
    title: str = "Learning Rate Schedule",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot learning rate schedule over training.
    
    Args:
        optimizer_history: List of learning rates
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    epochs = range(1, len(optimizer_history) + 1)
    ax.plot(epochs, optimizer_history, 'b-', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    return fig