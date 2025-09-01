# src/nlp/embeddings.py
"""
Embedding utilities for PyTorch Mastery Hub
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Dict, Tuple, Union
from pathlib import Path


class WordEmbedding(nn.Module):
    """
    Word embedding layer with optional pre-trained weights.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: Optional[int] = 0,
        max_norm: Optional[float] = None,
        freeze: bool = False
    ):
        super(WordEmbedding, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, 
            padding_idx=padding_idx, max_norm=max_norm
        )
        
        if freeze:
            self.embedding.weight.requires_grad = False
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)
    
    def load_pretrained(self, pretrained_weights: torch.Tensor):
        """Load pre-trained embedding weights."""
        if pretrained_weights.shape != self.embedding.weight.shape:
            raise ValueError(f"Shape mismatch: expected {self.embedding.weight.shape}, got {pretrained_weights.shape}")
        
        with torch.no_grad():
            self.embedding.weight.copy_(pretrained_weights)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


class LearnedEmbedding(nn.Module):
    """
    Learned positional embedding.
    """
    
    def __init__(self, max_len: int, d_model: int):
        super(LearnedEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.embedding(positions)


def load_pretrained_embeddings(
    filepath: Union[str, Path],
    word_to_idx: Dict[str, int],
    embedding_dim: int,
    format: str = 'glove'
) -> torch.Tensor:
    """
    Load pre-trained word embeddings.
    
    Args:
        filepath: Path to embedding file
        word_to_idx: Vocabulary mapping
        embedding_dim: Embedding dimension
        format: Embedding format ('glove', 'word2vec', 'fasttext')
    
    Returns:
        Embedding matrix tensor
    """
    vocab_size = len(word_to_idx)
    embedding_matrix = torch.zeros(vocab_size, embedding_dim)
    
    found_words = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if format == 'glove':
                values = line.strip().split()
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)
            elif format == 'word2vec':
                # Skip header line for word2vec format
                if len(line.strip().split()) == 2:
                    continue
                values = line.strip().split()
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            if word in word_to_idx and len(vector) == embedding_dim:
                idx = word_to_idx[word]
                embedding_matrix[idx] = torch.from_numpy(vector)
                found_words += 1
    
    print(f"Loaded embeddings for {found_words}/{vocab_size} words")
    return embedding_matrix


def create_embedding_matrix(
    word_to_idx: Dict[str, int],
    pretrained_embeddings: Dict[str, np.ndarray],
    embedding_dim: int
) -> torch.Tensor:
    """
    Create embedding matrix from pre-trained embeddings dictionary.
    
    Args:
        word_to_idx: Vocabulary mapping
        pretrained_embeddings: Dictionary of word -> embedding
        embedding_dim: Embedding dimension
    
    Returns:
        Embedding matrix tensor
    """
    vocab_size = len(word_to_idx)
    embedding_matrix = torch.zeros(vocab_size, embedding_dim)
    
    found_words = 0
    for word, idx in word_to_idx.items():
        if word in pretrained_embeddings:
            embedding_matrix[idx] = torch.from_numpy(pretrained_embeddings[word])
            found_words += 1
        else:
            # Initialize with random values
            embedding_matrix[idx] = torch.randn(embedding_dim) * 0.1
    
    print(f"Found embeddings for {found_words}/{vocab_size} words")
    return embedding_matrix


class CharacterEmbedding(nn.Module):
    """
    Character-level embedding with CNN.
    """
    
    def __init__(
        self,
        char_vocab_size: int,
        char_embedding_dim: int = 16,
        word_embedding_dim: int = 100,
        filters: list = [25, 50, 75, 100],
        filter_sizes: list = [2, 3, 4, 5]
    ):
        super(CharacterEmbedding, self).__init__()
        
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)
        
        # CNN layers
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embedding_dim, num_filters, kernel_size=filter_size)
            for num_filters, filter_size in zip(filters, filter_sizes)
        ])
        
        self.highway = Highway(sum(filters), word_embedding_dim)
        
    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: (batch_size, seq_len, word_len)
        batch_size, seq_len, word_len = char_ids.shape
        
        # Reshape for processing
        char_ids = char_ids.view(-1, word_len)  # (batch_size * seq_len, word_len)
        
        # Character embeddings
        char_emb = self.char_embedding(char_ids)  # (batch_size * seq_len, word_len, char_dim)
        char_emb = char_emb.transpose(1, 2)  # (batch_size * seq_len, char_dim, word_len)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(char_emb))  # (batch_size * seq_len, num_filters, new_len)
            conv_out = torch.max(conv_out, dim=2)[0]  # Max pooling
            conv_outputs.append(conv_out)
        
        # Concatenate all conv outputs
        word_emb = torch.cat(conv_outputs, dim=1)  # (batch_size * seq_len, total_filters)
        
        # Highway network
        word_emb = self.highway(word_emb)
        
        # Reshape back
        word_emb = word_emb.view(batch_size, seq_len, -1)
        
        return word_emb


class Highway(nn.Module):
    """
    Highway network for character embeddings.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super(Highway, self).__init__()
        
        self.projection = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.transform = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.projection(x)
        gate = torch.sigmoid(self.gate(x))
        transform = torch.relu(self.transform(x))
        
        return gate * transform + (1 - gate) * proj


class SubwordEmbedding(nn.Module):
    """
    Subword embedding using character n-grams.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        n_grams: list = [3, 4, 5, 6],
        bucket_size: int = 2000000
    ):
        super(SubwordEmbedding, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.n_grams = n_grams
        self.bucket_size = bucket_size
        
        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Subword embeddings
        self.subword_embeddings = nn.Embedding(bucket_size, embedding_dim)
        
    def _hash(self, ngram: str) -> int:
        """Simple hash function for n-grams."""
        h = 0
        for c in ngram:
            h = h * 37 + ord(c)
        return h % self.bucket_size
    
    def _get_subwords(self, word: str) -> list:
        """Extract character n-grams from word."""
        word = '<' + word + '>'
        subwords = []
        
        for n in self.n_grams:
            for i in range(len(word) - n + 1):
                ngram = word[i:i+n]
                subwords.append(self._hash(ngram))
        
        return subwords
    
    def forward(self, input_ids: torch.Tensor, words: list = None) -> torch.Tensor:
        # Word embeddings
        word_emb = self.word_embeddings(input_ids)
        
        if words is None:
            return word_emb
        
        # Subword embeddings
        batch_size, seq_len = input_ids.shape
        subword_emb = torch.zeros_like(word_emb)
        
        for i in range(batch_size):
            for j in range(seq_len):
                if j < len(words[i]):
                    word = words[i][j]
                    subword_ids = self._get_subwords(word)
                    if subword_ids:
                        subword_vectors = self.subword_embeddings(
                            torch.LongTensor(subword_ids).to(input_ids.device)
                        )
                        subword_emb[i, j] = subword_vectors.mean(dim=0)
        
        return word_emb + subword_emb


class ContextualEmbedding(nn.Module):
    """
    Simple bidirectional LSTM for contextual embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super(ContextualEmbedding, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        self.output_dim = hidden_dim * 2  # Bidirectional
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Word embeddings
        emb = self.embedding(input_ids)
        
        # Contextual embeddings
        lstm_out, _ = self.lstm(emb)
        
        return lstm_out


def compute_embedding_similarity(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    metric: str = 'cosine'
) -> torch.Tensor:
    """
    Compute similarity between embeddings.
    
    Args:
        emb1: First embedding tensor
        emb2: Second embedding tensor  
        metric: Similarity metric ('cosine', 'dot', 'euclidean')
    
    Returns:
        Similarity scores
    """
    if metric == 'cosine':
        return torch.cosine_similarity(emb1, emb2, dim=-1)
    elif metric == 'dot':
        return torch.sum(emb1 * emb2, dim=-1)
    elif metric == 'euclidean':
        return -torch.norm(emb1 - emb2, dim=-1)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def get_nearest_neighbors(
    query_embedding: torch.Tensor,
    embedding_matrix: torch.Tensor,
    k: int = 5,
    metric: str = 'cosine'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find k nearest neighbors for query embedding.
    
    Args:
        query_embedding: Query embedding
        embedding_matrix: Matrix of all embeddings
        k: Number of neighbors
        metric: Distance metric
    
    Returns:
        Tuple of (distances, indices)
    """
    # Compute similarities
    if metric == 'cosine':
        similarities = torch.cosine_similarity(
            query_embedding.unsqueeze(0), embedding_matrix, dim=1
        )
    elif metric == 'dot':
        similarities = torch.matmul(embedding_matrix, query_embedding)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    # Get top-k
    values, indices = torch.topk(similarities, k)
    
    return values, indices