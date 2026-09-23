# pytorch_mastery_hub/utils/__init__.py
"""
Utility functions for PyTorch Mastery Hub
"""

from .data_utils import *
from .device_utils import *
from .io_utils import *
from .logging_utils import *
from .memory_utils import *
from .metrics import *
from .model_utils import *
from .reproducibility import *
from .visualization import *

__all__ = [
    # data_utils
    "load_dataset",
    "create_data_loaders",
    "train_val_split",
    "normalize_data",
    "download_dataset",
    "get_dataset_info",
    # visualization
    "plot_training_curves",
    "plot_tensor_as_image",
    "plot_gradient_flow",
    "plot_confusion_matrix",
    "visualize_model_architecture",
    # metrics
    "accuracy",
    "precision_recall_f1",
    "classification_report",
    "regression_metrics",
    "top_k_accuracy",
    # device_utils
    "get_device",
    "autocast_dtype",
    "move_to_device",
    "device_info",
    "print_system_info",
    "synchronize",
    # reproducibility
    "seed_everything",
    "seed_worker",
    "make_generator",
    "isolated_rng",
    # logging_utils
    "setup_logger",
    "get_logger",
    "MetricsLogger",
    # model_utils
    "count_parameters",
    "get_model_size",
    "model_summary",
    "summarize",
    "freeze",
    "unfreeze",
    "init_weights",
    "param_groups_with_weight_decay",
    # memory_utils
    "get_memory_usage",
    "clear_memory",
    "MemoryTracker",
    "tensor_bytes",
    # io_utils
    "save_model",
    "load_model",
    "save_checkpoint",
    "load_checkpoint",
    "save_results",
    "load_config",
]
