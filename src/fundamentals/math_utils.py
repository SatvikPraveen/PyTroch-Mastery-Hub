# src/fundamentals/math_utils.py
"""
Mathematical utilities for PyTorch Mastery Hub
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union


def softmax(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    """
    Compute softmax with temperature scaling.
    
    Args:
        x: Input tensor
        dim: Dimension to apply softmax
        temperature: Temperature parameter for scaling
        
    Returns:
        Softmax probabilities
    """
    scaled_x = x / temperature
    return F.softmax(scaled_x, dim=dim)


def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log softmax (numerically stable).
    
    Args:
        x: Input tensor
        dim: Dimension to apply log softmax
        
    Returns:
        Log softmax values
    """
    return F.log_softmax(x, dim=dim)


def cross_entropy(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean',
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    Compute cross entropy loss.
    
    Args:
        pred: Predicted logits [batch_size, num_classes]
        target: Target labels [batch_size]
        reduction: Reduction method ('mean', 'sum', 'none')
        label_smoothing: Label smoothing parameter
        
    Returns:
        Cross entropy loss
    """
    return F.cross_entropy(pred, target, reduction=reduction, label_smoothing=label_smoothing)


def mse_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Compute mean squared error loss.
    
    Args:
        pred: Predicted values
        target: Target values
        reduction: Reduction method
        
    Returns:
        MSE loss
    """
    return F.mse_loss(pred, target, reduction=reduction)


def mae_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Compute mean absolute error loss.
    
    Args:
        pred: Predicted values
        target: Target values
        reduction: Reduction method
        
    Returns:
        MAE loss
    """
    return F.l1_loss(pred, target, reduction=reduction)


def huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute Huber loss (smooth L1 loss).
    
    Args:
        pred: Predicted values
        target: Target values
        delta: Threshold parameter
        reduction: Reduction method
        
    Returns:
        Huber loss
    """
    return F.smooth_l1_loss(pred, target, reduction=reduction, beta=delta)


def cosine_similarity(
    x1: torch.Tensor,
    x2: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute cosine similarity between vectors.
    
    Args:
        x1: First tensor
        x2: Second tensor
        dim: Dimension along which to compute similarity
        eps: Small value to avoid division by zero
        
    Returns:
        Cosine similarity values
    """
    return F.cosine_similarity(x1, x2, dim=dim, eps=eps)


def euclidean_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Compute Euclidean distance between tensors.
    
    Args:
        x1: First tensor
        x2: Second tensor
        
    Returns:
        Euclidean distances
    """
    return torch.norm(x1 - x2, p=2, dim=-1)


def manhattan_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Compute Manhattan (L1) distance between tensors.
    
    Args:
        x1: First tensor
        x2: Second tensor
        
    Returns:
        Manhattan distances
    """
    return torch.norm(x1 - x2, p=1, dim=-1)


def normalize(
    x: torch.Tensor,
    p: float = 2.0,
    dim: int = -1,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Normalize tensor along specified dimension.
    
    Args:
        x: Input tensor
        p: Order of norm
        dim: Dimension to normalize
        eps: Small value to avoid division by zero
        
    Returns:
        Normalized tensor
    """
    return F.normalize(x, p=p, dim=dim, eps=eps)


def standardize(x: torch.Tensor, dim: Optional[int] = None, eps: float = 1e-8) -> torch.Tensor:
    """
    Standardize tensor (zero mean, unit variance).
    
    Args:
        x: Input tensor
        dim: Dimension(s) to standardize over
        eps: Small value for numerical stability
        
    Returns:
        Standardized tensor
    """
    if dim is None:
        mean = x.mean()
        std = x.std()
    else:
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True)
    
    return (x - mean) / (std + eps)


def gelu(x: torch.Tensor, approximate: bool = False) -> torch.Tensor:
    """
    Gaussian Error Linear Unit activation.
    
    Args:
        x: Input tensor
        approximate: Whether to use approximation
        
    Returns:
        GELU activated tensor
    """
    if approximate:
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Swish activation function.
    
    Args:
        x: Input tensor
        beta: Beta parameter
        
    Returns:
        Swish activated tensor
    """
    return x * torch.sigmoid(beta * x)


def mish(x: torch.Tensor) -> torch.Tensor:
    """
    Mish activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Mish activated tensor
    """
    return x * torch.tanh(F.softplus(x))


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    gamma: float = 2.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute focal loss for addressing class imbalance.
    
    Args:
        pred: Predicted logits
        target: Target labels
        alpha: Weighting factor
        gamma: Focusing parameter
        reduction: Reduction method
        
    Returns:
        Focal loss
    """
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-8
) -> torch.Tensor:
    """
    Compute Dice loss for segmentation.
    
    Args:
        pred: Predicted probabilities
        target: Target masks
        smooth: Smoothing factor
        
    Returns:
        Dice loss
    """
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    reduction: str = 'batchmean'
) -> torch.Tensor:
    """
    Compute KL divergence between distributions.
    
    Args:
        p: First distribution (log probabilities)
        q: Second distribution (log probabilities)
        reduction: Reduction method
        
    Returns:
        KL divergence
    """
    return F.kl_div(p, q, reduction=reduction, log_target=True)


def jensen_shannon_divergence(
    p: torch.Tensor,
    q: torch.Tensor
) -> torch.Tensor:
    """
    Compute Jensen-Shannon divergence.
    
    Args:
        p: First distribution
        q: Second distribution
        
    Returns:
        JS divergence
    """
    m = 0.5 * (p + q)
    js = 0.5 * F.kl_div(p.log(), m, reduction='batchmean', log_target=False) + \
         0.5 * F.kl_div(q.log(), m, reduction='batchmean', log_target=False)
    return js


def entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute entropy of probability distribution.
    
    Args:
        p: Probability distribution
        dim: Dimension to compute entropy over
        
    Returns:
        Entropy values
    """
    log_p = torch.log(p + 1e-8)
    return -torch.sum(p * log_p, dim=dim)


def mutual_information(
    x: torch.Tensor,
    y: torch.Tensor,
    bins: int = 10
) -> float:
    """
    Estimate mutual information between two variables.
    
    Args:
        x: First variable
        y: Second variable
        bins: Number of bins for histogram
        
    Returns:
        Mutual information estimate
    """
    # Convert to numpy for histogram computation
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    
    # Compute joint and marginal histograms
    xy_hist, _, _ = torch.histogramdd(torch.stack([x.flatten(), y.flatten()]).T, bins=bins)
    x_hist, _ = torch.histogram(x.flatten(), bins=bins)
    y_hist, _ = torch.histogram(y.flatten(), bins=bins)
    
    # Normalize to probabilities
    xy_prob = xy_hist / xy_hist.sum()
    x_prob = x_hist / x_hist.sum()
    y_prob = y_hist / y_hist.sum()
    
    # Compute mutual information
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if xy_prob[i, j] > 0:
                mi += xy_prob[i, j] * torch.log(xy_prob[i, j] / (x_prob[i] * y_prob[j] + 1e-8))
    
    return mi.item()


def pearson_correlation_coefficient(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute Pearson correlation coefficient.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Correlation coefficient
    """
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    
    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
    
    return numerator / (denominator + 1e-8)


def spearman_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute Spearman rank correlation coefficient.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Spearman correlation
    """
    # Convert to ranks
    x_ranks = torch.argsort(torch.argsort(x)).float()
    y_ranks = torch.argsort(torch.argsort(y)).float()
    
    return pearson_correlation_coefficient(x_ranks, y_ranks)


def gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float = 1.0
) -> torch.Tensor:
    """
    Compute Gaussian (RBF) kernel between tensors.
    
    Args:
        x: First tensor
        y: Second tensor
        sigma: Kernel bandwidth
        
    Returns:
        Kernel matrix
    """
    # Compute pairwise squared distances
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True)
    
    dist_sq = x_norm + y_norm.t() - 2 * torch.mm(x, y.t())
    
    # Apply Gaussian kernel
    return torch.exp(-dist_sq / (2 * sigma ** 2))


def polynomial_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    degree: int = 2,
    gamma: float = 1.0,
    coef0: float = 0.0
) -> torch.Tensor:
    """
    Compute polynomial kernel between tensors.
    
    Args:
        x: First tensor
        y: Second tensor
        degree: Polynomial degree
        gamma: Scaling parameter
        coef0: Independent term
        
    Returns:
        Kernel matrix
    """
    return (gamma * torch.mm(x, y.t()) + coef0) ** degree