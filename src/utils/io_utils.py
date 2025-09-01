# src/utils/io_utils.py
"""
File I/O utilities for PyTorch Mastery Hub
"""

import torch
import torch.nn as nn
import json
import yaml
import pickle
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from datetime import datetime


def save_model(
    model: nn.Module,
    filepath: Union[str, Path],
    save_architecture: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save PyTorch model with optional metadata.
    
    Args:
        model: PyTorch model to save
        filepath: Path to save the model
        save_architecture: Whether to save model architecture info
        metadata: Optional metadata dictionary
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'timestamp': datetime.now().isoformat()
    }
    
    if save_architecture:
        save_dict['model_str'] = str(model)
    
    if metadata:
        save_dict['metadata'] = metadata
    
    torch.save(save_dict, filepath)
    print(f"Model saved to {filepath}")


def load_model(
    model: nn.Module,
    filepath: Union[str, Path],
    map_location: Optional[str] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load PyTorch model from file.
    
    Args:
        model: Model instance to load weights into
        filepath: Path to saved model
        map_location: Device to load model on
        strict: Whether to strictly enforce key matching
        
    Returns:
        Dictionary with loaded metadata
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=map_location)
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        # Assume the file contains only state dict
        model.load_state_dict(checkpoint, strict=strict)
    
    print(f"Model loaded from {filepath}")
    
    # Return metadata if available
    metadata = {}
    for key in ['model_class', 'timestamp', 'metadata']:
        if key in checkpoint:
            metadata[key] = checkpoint[key]
    
    return metadata


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: Union[str, Path],
    scheduler: Optional[Any] = None,
    metrics: Optional[Dict[str, float]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Checkpoint file path
        scheduler: Learning rate scheduler (optional)
        metrics: Training metrics (optional)
        metadata: Additional metadata (optional)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        filepath: Checkpoint file path
        model: Model to load weights into (optional)
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        map_location: Device to load on
        
    Returns:
        Checkpoint dictionary with metadata
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=map_location)
    
    # Load model weights
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint


def save_results(
    results: Dict[str, Any],
    filepath: Union[str, Path],
    format: str = 'json'
) -> None:
    """
    Save experiment results to file.
    
    Args:
        results: Results dictionary
        filepath: Output file path
        format: Output format ('json', 'yaml', 'pickle')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp if not present
    if 'timestamp' not in results:
        results['timestamp'] = datetime.now().isoformat()
    
    if format.lower() == 'json':
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format.lower() == 'yaml':
        with open(filepath, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    elif format.lower() == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Results saved to {filepath}")


def load_results(
    filepath: Union[str, Path],
    format: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load experiment results from file.
    
    Args:
        filepath: Input file path
        format: File format (auto-detected if None)
        
    Returns:
        Results dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    # Auto-detect format if not specified
    if format is None:
        format = filepath.suffix[1:].lower()
    
    if format == 'json':
        with open(filepath, 'r') as f:
            results = json.load(f)
    elif format in ['yaml', 'yml']:
        with open(filepath, 'r') as f:
            results = yaml.safe_load(f)
    elif format in ['pickle', 'pkl']:
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return results


def load_config(
    filepath: Union[str, Path],
    format: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        filepath: Configuration file path
        format: File format (auto-detected if None)
        
    Returns:
        Configuration dictionary
    """
    return load_results(filepath, format)


def save_config(
    config: Dict[str, Any],
    filepath: Union[str, Path],
    format: str = 'yaml'
) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        filepath: Output file path
        format: Output format
    """
    save_results(config, filepath, format)


class ModelCheckpointManager:
    """
    Manage model checkpoints with automatic cleanup.
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_best_only: bool = False,
        monitor: str = 'val_loss',
        mode: str = 'min'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.checkpoints = []
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        scheduler: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Path]:
        """
        Save checkpoint with automatic management.
        
        Returns:
            Path to saved checkpoint or None if not saved
        """
        current_score = metrics.get(self.monitor, 0)
        
        # Check if we should save
        should_save = True
        is_best = False
        
        if self.save_best_only:
            if self.mode == 'min':
                is_best = current_score < self.best_score
            else:
                is_best = current_score > self.best_score
            
            should_save = is_best
        
        if not should_save:
            return None
        
        # Update best score
        if is_best or not self.save_best_only:
            if self.mode == 'min':
                if current_score < self.best_score:
                    self.best_score = current_score
                    is_best = True
            else:
                if current_score > self.best_score:
                    self.best_score = current_score
                    is_best = True
        
        # Create checkpoint filename
        filename = f"checkpoint_epoch_{epoch:03d}"
        if is_best:
            filename += "_best"
        filename += ".pt"
        
        filepath = self.checkpoint_dir / filename
        
        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=metrics.get('loss', 0),
            filepath=filepath,
            scheduler=scheduler,
            metrics=metrics,
            metadata=metadata
        )
        
        # Add to checkpoint list
        self.checkpoints.append({
            'filepath': filepath,
            'epoch': epoch,
            'score': current_score,
            'is_best': is_best
        })
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        return filepath
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to stay within max_checkpoints limit."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by score (keep best) or epoch (keep latest)
        if self.save_best_only:
            key_func = lambda x: x['score']
            reverse = self.mode == 'max'
        else:
            key_func = lambda x: x['epoch']
            reverse = True
        
        # Keep best checkpoints
        self.checkpoints.sort(key=key_func, reverse=reverse)
        
        # Remove excess checkpoints
        to_remove = self.checkpoints[self.max_checkpoints:]
        self.checkpoints = self.checkpoints[:self.max_checkpoints]
        
        # Delete files
        for checkpoint in to_remove:
            if not checkpoint['is_best']:  # Don't delete best checkpoint
                try:
                    checkpoint['filepath'].unlink()
                    print(f"Removed old checkpoint: {checkpoint['filepath']}")
                except FileNotFoundError:
                    pass
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        best_checkpoints = [cp for cp in self.checkpoints if cp['is_best']]
        return best_checkpoints[0]['filepath'] if best_checkpoints else None
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        if not self.checkpoints:
            return None
        
        latest = max(self.checkpoints, key=lambda x: x['epoch'])
        return latest['filepath']


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    level: str = 'INFO',
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Log file path (optional)
        level: Logging level
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[logging.StreamHandler()]
    )
    
    # Add file handler if specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        
        logger = logging.getLogger()
        logger.addHandler(file_handler)
    
    return logging.getLogger(__name__)