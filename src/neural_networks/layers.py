# src/neural_networks/layers.py
"""
Custom layer implementations for PyTorch Mastery Hub
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union


class LinearLayer(nn.Module):
    """
    Custom linear layer implementation for educational purposes.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.linear(x, self.weight, self.bias)
        return output
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class ConvLayer(nn.Module):
    """
    Custom convolutional layer with configurable activation and normalization.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        activation: Optional[str] = 'relu',
        norm: Optional[str] = None,
        dropout: float = 0.0
    ):
        super(ConvLayer, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        # Normalization
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'group':
            self.norm = nn.GroupNorm(8, out_channels)
        else:
            self.norm = None
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = None
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        
        if self.norm is not None:
            x = self.norm(x)
        
        if self.activation is not None:
            x = self.activation(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class AttentionLayer(nn.Module):
    """
    Multi-head attention layer implementation.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super(AttentionLayer, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.out_linear = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attention = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear transformation
        output = self.out_linear(attention)
        return output
    
    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        return output


class DropoutLayer(nn.Module):
    """
    Custom dropout layer with different dropout types.
    """
    
    def __init__(self, p: float = 0.5, dropout_type: str = 'standard'):
        super(DropoutLayer, self).__init__()
        self.p = p
        self.dropout_type = dropout_type
        
        if dropout_type == 'standard':
            self.dropout = nn.Dropout(p)
        elif dropout_type == '2d':
            self.dropout = nn.Dropout2d(p)
        elif dropout_type == '3d':
            self.dropout = nn.Dropout3d(p)
        elif dropout_type == 'alpha':
            self.dropout = nn.AlphaDropout(p)
        else:
            raise ValueError(f"Unknown dropout type: {dropout_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


class BatchNormLayer(nn.Module):
    """
    Batch normalization with different variants.
    """
    
    def __init__(
        self,
        num_features: int,
        dim: int = 2,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True
    ):
        super(BatchNormLayer, self).__init__()
        
        if dim == 1:
            self.norm = nn.BatchNorm1d(num_features, eps, momentum, affine)
        elif dim == 2:
            self.norm = nn.BatchNorm2d(num_features, eps, momentum, affine)
        elif dim == 3:
            self.norm = nn.BatchNorm3d(num_features, eps, momentum, affine)
        else:
            raise ValueError("dim must be 1, 2, or 3")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class LayerNormLayer(nn.Module):
    """
    Layer normalization implementation.
    """
    
    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        super(LayerNormLayer, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class ResidualBlock(nn.Module):
    """
    Residual block with customizable layers.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        activation: str = 'relu'
    ):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = ConvLayer(in_channels, out_channels, 3, stride, 1, norm='batch', activation=activation)
        self.conv2 = ConvLayer(out_channels, out_channels, 3, 1, 1, norm='batch', activation=None)
        
        self.downsample = downsample
        
        if activation == 'relu':
            self.final_activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.final_activation = nn.GELU()
        else:
            self.final_activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.final_activation(out)
        
        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        # Squeeze
        y = self.squeeze(x).view(b, c)
        
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        
        # Scale
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """
    Channel attention mechanism.
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        # Average pooling branch
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        
        # Max pooling branch
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism.
    """
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute average and max across channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    """
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]