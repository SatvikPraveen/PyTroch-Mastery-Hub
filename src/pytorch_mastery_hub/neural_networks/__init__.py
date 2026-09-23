# pytorch_mastery_hub/neural_networks/__init__.py
"""
Neural network components and utilities for PyTorch Mastery Hub
"""

from .layers import *
from .models import *
from .optimizers import *
from .training import *

__all__ = [
    # layers
    "LinearLayer",
    "ConvLayer",
    "AttentionLayer",
    "DropoutLayer",
    "BatchNormLayer",
    "LayerNormLayer",
    "ResidualBlock",
    # models
    "SimpleMLP",
    "DeepMLP",
    "CustomCNN",
    "ResNet",
    "SimpleRNN",
    "SimpleLSTM",
    "SimpleGRU",
    "TransformerBlock",
    # training
    "Trainer",
    "train_epoch",
    "validate_epoch",
    "EarlyStoppingCallback",
    "ModelCheckpointCallback",
    "LearningRateSchedulerCallback",
    # optimizers
    "CustomSGD",
    "CustomAdam",
    "CustomAdamW",
    "get_optimizer",
    "get_scheduler",
    "PolynomialDecayLR",
]
