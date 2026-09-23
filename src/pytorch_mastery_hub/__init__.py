"""
PyTorch Mastery Hub - Source Package
A comprehensive PyTorch learning resource with hands-on examples.
"""

__version__ = "1.0.0"
__author__ = "Satvik Praveen"

# Make key utilities available at package level
from .utils.data_utils import create_data_loaders, load_dataset
from .utils.device_utils import get_device
from .utils.metrics import accuracy, precision_recall_f1
from .utils.model_utils import count_parameters
from .utils.reproducibility import seed_everything
from .utils.visualization import plot_tensor_as_image, plot_training_curves

__all__ = [
    "plot_training_curves",
    "plot_tensor_as_image",
    "load_dataset",
    "create_data_loaders",
    "accuracy",
    "precision_recall_f1",
    "get_device",
    "seed_everything",
    "count_parameters",
]
