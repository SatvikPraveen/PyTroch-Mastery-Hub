# src/neural_networks/models.py
"""
Model architectures for PyTorch Mastery Hub
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Union, Tuple
from .layers import *


class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron implementation.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: str = 'relu',
        dropout: float = 0.0,
        batch_norm: bool = False
    ):
        super(SimpleMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DeepMLP(nn.Module):
    """
    Deep MLP with residual connections and advanced features.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        residual: bool = True,
        dropout: float = 0.1,
        layer_norm: bool = True
    ):
        super(DeepMLP, self).__init__()
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList()
        
        for _ in range(num_layers):
            layer = DeepMLPBlock(hidden_size, dropout, layer_norm, residual)
            self.layers.append(layer)
        
        self.output_proj = nn.Linear(hidden_size, output_size)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.output_proj(x)
        return x


class DeepMLPBlock(nn.Module):
    """Block for DeepMLP with residual connection."""
    
    def __init__(self, hidden_size: int, dropout: float, layer_norm: bool, residual: bool):
        super(DeepMLPBlock, self).__init__()
        self.residual = residual
        
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        if layer_norm:
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
        else:
            self.norm1 = self.norm2 = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.norm1(x)
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        if self.residual:
            x = x + residual
        
        x = self.norm2(x)
        return x


class CustomCNN(nn.Module):
    """
    Custom CNN for image classification.
    """
    
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        base_filters: int = 64,
        num_blocks: int = 4
    ):
        super(CustomCNN, self).__init__()
        
        self.features = self._make_feature_extractor(input_channels, base_filters, num_blocks)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(base_filters * (2 ** (num_blocks - 1)), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _make_feature_extractor(self, in_channels: int, base_filters: int, num_blocks: int):
        layers = []
        current_channels = in_channels
        current_filters = base_filters
        
        for i in range(num_blocks):
            # Convolutional block
            layers.extend([
                ConvLayer(current_channels, current_filters, 3, 1, 1, norm='batch', activation='relu'),
                ConvLayer(current_filters, current_filters, 3, 1, 1, norm='batch', activation='relu'),
            ])
            
            if i < num_blocks - 1:  # No pooling in the last block
                layers.append(nn.MaxPool2d(2, 2))
            
            current_channels = current_filters
            current_filters *= 2
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResNet(nn.Module):
    """
    Simplified ResNet implementation.
    """
    
    def __init__(
        self,
        num_classes: int,
        layers: List[int] = [2, 2, 2, 2],
        input_channels: int = 3
    ):
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, channels: int, num_blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.in_channels != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        
        layers = []
        layers.append(ResidualBlock(self.in_channels, channels, stride, downsample))
        self.in_channels = channels
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(channels, channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class SimpleRNN(nn.Module):
    """
    Simple RNN implementation.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers,
            dropout=dropout, bidirectional=bidirectional, batch_first=True
        )
        
        multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * multiplier, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Initialize hidden state
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(x.device)
        
        # Forward pass
        out, _ = self.rnn(x, h0)
        
        # Use last output for classification
        out = self.fc(out[:, -1, :])
        
        return out


class SimpleLSTM(nn.Module):
    """
    Simple LSTM implementation.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        super(SimpleLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, bidirectional=bidirectional, batch_first=True
        )
        
        multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * multiplier, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Initialize hidden and cell states
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(x.device)
        
        # Forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Use last output for classification
        out = self.fc(out[:, -1, :])
        
        return out


class SimpleGRU(nn.Module):
    """
    Simple GRU implementation.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        super(SimpleGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            dropout=dropout, bidirectional=bidirectional, batch_first=True
        )
        
        multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * multiplier, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Initialize hidden state
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(x.device)
        
        # Forward pass
        out, _ = self.gru(x, h0)
        
        # Use last output for classification
        out = self.fc(out[:, -1, :])
        
        return out


class TransformerBlock(nn.Module):
    """
    Single Transformer block implementation.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super(TransformerBlock, self).__init__()
        
        self.attention = AttentionLayer(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_out)
        
        # Feedforward with residual connection
        ff_out = self.feedforward(x)
        x = self.norm2(x + ff_out)
        
        return x


class SimpleTransformer(nn.Module):
    """
    Simple Transformer for sequence classification.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        max_seq_len: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super(SimpleTransformer, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Global average pooling and classification
        x = x.mean(dim=1)  # Average over sequence length
        x = self.classifier(x)
        
        return x


class AutoEncoder(nn.Module):
    """
    Simple AutoEncoder implementation.
    """
    
    def __init__(
        self,
        input_size: int,
        encoding_dims: List[int],
        activation: str = 'relu'
    ):
        super(AutoEncoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_size = input_size
        
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_size, dim),
                nn.ReLU(inplace=True) if activation == 'relu' else nn.Tanh()
            ])
            prev_size = dim
        
        self.encoder = nn.Sequential(*encoder_layers[:-1])  # Remove last activation
        
        # Decoder (reverse of encoder)
        decoder_layers = []
        decoding_dims = list(reversed(encoding_dims[:-1])) + [input_size]
        prev_size = encoding_dims[-1]
        
        for i, dim in enumerate(decoding_dims):
            decoder_layers.append(nn.Linear(prev_size, dim))
            if i < len(decoding_dims) - 1:  # No activation on output layer
                decoder_layers.append(nn.ReLU(inplace=True) if activation == 'relu' else nn.Tanh())
            prev_size = dim
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class VariationalAutoEncoder(nn.Module):
    """
    Variational AutoEncoder implementation.
    """
    
    def __init__(self, input_size: int, hidden_size: int, latent_size: int):
        super(VariationalAutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        
        # Latent space
        self.mu_layer = nn.Linear(hidden_size, latent_size)
        self.logvar_layer = nn.Linear(hidden_size, latent_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class Seq2SeqModel(nn.Module):
    """
    Simple Sequence-to-Sequence model with attention.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super(Seq2SeqModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encoder
        self.encoder = nn.LSTM(
            embedding_dim, hidden_size, num_layers,
            dropout=dropout, batch_first=True, bidirectional=True
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            embedding_dim + hidden_size * 2, hidden_size, num_layers,
            dropout=dropout, batch_first=True
        )
        
        # Attention
        self.attention = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        
        # Output projection
        self.out = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # Encode
        src_emb = self.embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder(src_emb)
        
        # Decode with attention
        tgt_emb = self.embedding(tgt)
        batch_size, seq_len = tgt.size()
        
        outputs = []
        decoder_hidden = hidden[-1:], cell[-1:]  # Use last layer
        
        for i in range(seq_len):
            # Attention mechanism
            query = decoder_hidden[0].expand(encoder_outputs.size(1), -1, -1).transpose(0, 1)
            energy = torch.tanh(self.attention(torch.cat([query, encoder_outputs], dim=2)))
            attention_weights = F.softmax(torch.sum(energy * self.v, dim=2), dim=1)
            context = torch.sum(attention_weights.unsqueeze(2) * encoder_outputs, dim=1)
            
            # Decoder step
            decoder_input = torch.cat([tgt_emb[:, i:i+1, :], context.unsqueeze(1)], dim=2)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # Output projection
            output = self.out(decoder_output.squeeze(1))
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)