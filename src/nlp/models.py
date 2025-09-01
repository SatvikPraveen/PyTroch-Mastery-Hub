# src/nlp/models.py
"""
NLP model architectures for PyTorch Mastery Hub
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .embeddings import WordEmbedding, PositionalEncoding


class RNNClassifier(nn.Module):
    """RNN-based text classifier."""
    
    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int,
        num_classes: int, num_layers: int = 1, dropout: float = 0.1,
        rnn_type: str = 'LSTM', bidirectional: bool = True
    ):
        super(RNNClassifier, self).__init__()
        
        self.embedding = WordEmbedding(vocab_size, embedding_dim)
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                              dropout=dropout, bidirectional=bidirectional, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers,
                             dropout=dropout, bidirectional=bidirectional, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_dim, num_classes)
        )
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)
        rnn_out, _ = self.rnn(emb)
        
        # Use last output (or mean pooling)
        output = rnn_out[:, -1, :]  # Last timestep
        return self.classifier(output)


class TransformerClassifier(nn.Module):
    """Transformer-based text classifier."""
    
    def __init__(
        self, vocab_size: int, d_model: int = 512, num_heads: int = 8,
        num_layers: int = 6, num_classes: int = 2, max_len: int = 512,
        dropout: float = 0.1
    ):
        super(TransformerClassifier, self).__init__()
        
        self.embedding = WordEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(input_ids) * torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=torch.float))
        emb = self.pos_encoding(emb.transpose(0, 1)).transpose(0, 1)
        
        # Create padding mask
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
            
        transformer_out = self.transformer(emb, src_key_padding_mask=src_key_padding_mask)
        
        # Global average pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(transformer_out.size()).float()
            sum_embeddings = torch.sum(transformer_out * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = transformer_out.mean(1)
        
        return self.classifier(pooled)


class LanguageModel(nn.Module):
    """Simple language model."""
    
    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int,
        num_layers: int = 2, dropout: float = 0.1
    ):
        super(LanguageModel, self).__init__()
        
        self.embedding = WordEmbedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)
        lstm_out, _ = self.lstm(emb)
        return self.output(lstm_out)


class Seq2SeqModel(nn.Module):
    """Sequence-to-sequence model with attention."""
    
    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int,
        num_layers: int = 1, dropout: float = 0.1
    ):
        super(Seq2SeqModel, self).__init__()
        
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.decoder = AttentionDecoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        encoder_outputs, hidden = self.encoder(src)
        return self.decoder(tgt, hidden, encoder_outputs)


class Encoder(nn.Module):
    """Encoder for Seq2Seq."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super(Encoder, self).__init__()
        
        self.embedding = WordEmbedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embedding(x)
        outputs, hidden = self.lstm(emb)
        return outputs, hidden


class AttentionDecoder(nn.Module):
    """Decoder with attention mechanism."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super(AttentionDecoder, self).__init__()
        
        self.embedding = WordEmbedding(vocab_size, embedding_dim)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor], 
                encoder_outputs: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        
        outputs = []
        for i in range(x.size(1)):
            # Attention
            context = self.attention(hidden[0][-1], encoder_outputs)
            
            # LSTM step
            lstm_input = torch.cat([emb[:, i:i+1, :], context.unsqueeze(1)], dim=2)
            output, hidden = self.lstm(lstm_input, hidden)
            outputs.append(output)
        
        outputs = torch.cat(outputs, dim=1)
        return self.output(outputs)


class Attention(nn.Module):
    """Attention mechanism."""
    
    def __init__(self, hidden_dim: int):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        # Calculate attention weights
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        attention = torch.sum(self.v * energy, dim=2)
        attention = F.softmax(attention, dim=1).unsqueeze(1)
        
        # Apply attention
        context = torch.bmm(attention, encoder_outputs).squeeze(1)
        return context