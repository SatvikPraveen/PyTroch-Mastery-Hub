# src/__init__.py
"""
PyTorch Mastery Hub - Source Package
A comprehensive PyTorch learning resource with hands-on examples.
"""

__version__ = "1.0.0"
__author__ = "PyTorch Mastery Hub Contributors"

# Make key utilities available at package level
from .utils.visualization import plot_training_curves, plot_tensor_as_image
from .utils.data_utils import load_dataset, create_data_loaders
from .utils.metrics import accuracy, precision_recall_f1

__all__ = [
    "plot_training_curves",
    "plot_tensor_as_image", 
    "load_dataset",
    "create_data_loaders",
    "accuracy",
    "precision_recall_f1",
]