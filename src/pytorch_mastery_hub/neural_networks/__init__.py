# pytorch_mastery_hub/neural_networks/__init__.py
"""
Neural network components and utilities for PyTorch Mastery Hub
"""

from .attention import *
from .ema import *
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
    "TrainerConfig",
    "Callback",
    "LambdaCallback",
    "train_epoch",
    "validate_epoch",
    "EarlyStopping",
    "ModelCheckpoint",
    "ProgressCallback",
    "EarlyStoppingCallback",
    "ModelCheckpointCallback",
    "LearningRateSchedulerCallback",
    # attention (modern)
    "MultiHeadAttention",
    "DecoderBlock",
    "RMSNorm",
    "RotaryEmbedding",
    "KVCache",
    "SwiGLU",
    "TransformerConfig",
    "TransformerLM",
    "build_causal_mask",
    # ema
    "ModelEMA",
    "steps_to_reach",
    # optimizers
    "CustomSGD",
    "CustomAdam",
    "CustomAdamW",
    "get_optimizer",
    "get_scheduler",
    "PolynomialDecayLR",
]
