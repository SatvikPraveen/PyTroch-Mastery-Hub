# src/nlp/tokenization.py
"""
Text preprocessing and tokenization utilities for PyTorch Mastery Hub
"""

import torch
import re
import string
from collections import Counter, OrderedDict
from typing import List, Dict, Optional, Tuple, Union, Set
import pickle
from pathlib import Path


class SimpleTokenizer:
    """
    Simple word-based tokenizer.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        min_freq: int = 2,
        special_tokens: Optional[List[str]] = None
    ):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.special_tokens = special_tokens or ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_built = False
    
    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from texts."""
        # Count words
        word_counts = Counter()
        for text in texts:
            words = self.basic_tokenize(text)
            word_counts.update(words)
        
        # Create vocabulary
        vocab = OrderedDict()
        
        # Add special tokens
        for token in self.special_tokens:
            vocab[token] = len(vocab)
        
        # Add frequent words
        for word, count in word_counts.most_common():
            if count >= self.min_freq and len(vocab) < self.vocab_size:
                vocab[word] = len(vocab)
            elif len(vocab) >= self.vocab_size:
                break
        
        self.word_to_idx = vocab
        self.idx_to_word = {idx: word for word, idx in vocab.items()}
        self.vocab_built = True
        
        print(f"Built vocabulary with {len(vocab)} tokens")
    
    def basic_tokenize(self, text: str) -> List[str]:
        """Basic tokenization."""
        # Lowercase and remove punctuation
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return self.basic_tokenize(text)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text to token indices."""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = ['<BOS>'] + tokens + ['<EOS>']
        
        indices = []
        for token in tokens:
            if token in self.word_to_idx:
                indices.append(self.word_to_idx[token])
            else:
                indices.append(self.word_to_idx['<UNK>'])
        
        return indices
    
    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token indices back to text."""
        tokens = []
        for idx in indices:
            if idx in self.idx_to_word:
                token = self.idx_to_word[idx]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save tokenizer."""
        data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq,
            'special_tokens': self.special_tokens,
            'vocab_built': self.vocab_built
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: Union[str, Path]) -> None:
        """Load tokenizer."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = data['idx_to_word']
        self.vocab_size = data['vocab_size']
        self.min_freq = data['min_freq']
        self.special_tokens = data['special_tokens']
        self.vocab_built = data['vocab_built']
    
    def __len__(self) -> int:
        return len(self.word_to_idx)


class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer.
    """
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_vocab = {}
        self.bpe_codes = []
        self.encoder = {}
        self.decoder = {}
        
    def get_word_tokens(self, text: str) -> Dict[str, int]:
        """Get word frequency dictionary."""
        words = text.lower().split()
        word_counts = Counter(words)
        
        # Add end-of-word token
        word_vocab = {}
        for word, count in word_counts.items():
            word_tokens = ' '.join(list(word)) + ' </w>'
            word_vocab[word_tokens] = count
        
        return word_vocab
    
    def get_pairs(self, word_vocab: Dict[str, int]) -> Set[Tuple[str, str]]:
        """Get all pairs of consecutive tokens."""
        pairs = set()
        for word in word_vocab:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs.add((symbols[i], symbols[i + 1]))
        return pairs
    
    def merge_vocab(self, pair: Tuple[str, str], word_vocab: Dict[str, int]) -> Dict[str, int]:
        """Merge most frequent pair in vocabulary."""
        new_word_vocab = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in word_vocab:
            new_word = p.sub(''.join(pair), word)
            new_word_vocab[new_word] = word_vocab[word]
        
        return new_word_vocab
    
    def train(self, texts: List[str]) -> None:
        """Train BPE tokenizer."""
        # Combine all texts
        corpus = ' '.join(texts)
        
        # Get initial word vocabulary
        word_vocab = self.get_word_tokens(corpus)
        
        # Learn BPE codes
        num_merges = self.vocab_size - len(set(''.join(word_vocab.keys())))
        
        for i in range(num_merges):
            pairs = self.get_pairs(word_vocab)
            if not pairs:
                break
            
            # Count pair frequencies
            pair_counts = {}
            for word, count in word_vocab.items():
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pair = (symbols[j], symbols[j + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + count
            
            # Get most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            
            # Merge the best pair
            word_vocab = self.merge_vocab(best_pair, word_vocab)
            self.bpe_codes.append(best_pair)
        
        # Build encoder/decoder
        vocab = set()
        for word in word_vocab:
            vocab.update(word.split())
        
        vocab = sorted(list(vocab))
        self.encoder = {token: i for i, token in enumerate(vocab)}
        self.decoder = {i: token for i, token in enumerate(vocab)}
        
        print(f"Learned {len(self.bpe_codes)} BPE codes")
    
    def encode(self, text: str) -> List[int]:
        """Encode text using learned BPE."""
        words = text.lower().split()
        tokens = []
        
        for word in words:
            word_tokens = ' '.join(list(word)) + ' </w>'
            
            # Apply BPE codes
            for pair in self.bpe_codes:
                bigram = re.escape(' '.join(pair))
                p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
                word_tokens = p.sub(''.join(pair), word_tokens)
            
            tokens.extend(word_tokens.split())
        
        # Convert to indices
        return [self.encoder.get(token, 0) for token in tokens]
    
    def decode(self, indices: List[int]) -> str:
        """Decode indices back to text."""
        tokens = [self.decoder.get(idx, '') for idx in indices]
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()


class SubwordTokenizer:
    """
    Simple subword tokenizer using character-level fallback.
    """
    
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.word_vocab = {}
        self.char_vocab = set()
        self.encoder = {}
        self.decoder = {}
    
    def train(self, texts: List[str]) -> None:
        """Train subword tokenizer."""
        # Collect word and character vocabularies
        word_counts = Counter()
        char_set = set()
        
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
            char_set.update(text.lower())
        
        # Start with character vocabulary
        vocab = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        vocab.extend(sorted(char_set))
        
        # Add frequent words up to vocab size
        for word, count in word_counts.most_common():
            if len(vocab) >= self.vocab_size:
                break
            if len(word) > 1 and count >= 2:  # Only multi-character words
                vocab.append(word)
        
        # Build encoder/decoder
        self.encoder = {token: i for i, token in enumerate(vocab)}
        self.decoder = {i: token for i, token in enumerate(vocab)}
        
        print(f"Built subword vocabulary with {len(vocab)} tokens")
    
    def encode(self, text: str) -> List[int]:
        """Encode text with subword fallback."""
        text = text.lower()
        words = text.split()
        tokens = ['<BOS>']
        
        for word in words:
            if word in self.encoder:
                tokens.append(word)
            else:
                # Fall back to character level
                tokens.extend(list(word))
        
        tokens.append('<EOS>')
        
        # Convert to indices
        indices = []
        for token in tokens:
            indices.append(self.encoder.get(token, self.encoder['<UNK>']))
        
        return indices
    
    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """Decode indices to text."""
        tokens = []
        for idx in indices:
            token = self.decoder.get(idx, '<UNK>')
            if skip_special_tokens and token in ['<PAD>', '<BOS>', '<EOS>']:
                continue
            tokens.append(token)
        
        # Reconstruct text
        text = ''
        for token in tokens:
            if len(token) == 1 and token.isalpha():
                text += token  # Character
            else:
                text += ' ' + token  # Word
        
        return text.strip()


class WordTokenizer:
    """
    Advanced word tokenizer with preprocessing.
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        min_freq: int = 2,
        lowercase: bool = True,
        remove_punct: bool = True
    ):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        
        self.vocab = {}
        self.word_counts = Counter()
    
    def preprocess(self, text: str) -> str:
        """Preprocess text."""
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punct:
            # Remove punctuation but keep apostrophes
            text = re.sub(r"[^\w\s']", '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize preprocessed text."""
        text = self.preprocess(text)
        
        # Simple word splitting
        words = text.split()
        
        # Handle contractions
        processed_words = []
        for word in words:
            if "'" in word:
                # Simple contraction splitting
                parts = word.split("'")
                processed_words.extend(parts)
            else:
                processed_words.append(word)
        
        return processed_words
    
    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from texts."""
        # Count all words
        for text in texts:
            words = self.tokenize_text(text)
            self.word_counts.update(words)
        
        # Build vocabulary
        vocab = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        
        # Add frequent words
        for word, count in self.word_counts.most_common():
            if count >= self.min_freq and len(vocab) < self.vocab_size:
                vocab.append(word)
        
        self.vocab = {word: i for i, word in enumerate(vocab)}
        
        print(f"Built vocabulary with {len(self.vocab)} words")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token indices."""
        words = self.tokenize_text(text)
        indices = [self.vocab['<BOS>']]
        
        for word in words:
            indices.append(self.vocab.get(word, self.vocab['<UNK>']))
        
        indices.append(self.vocab['<EOS>'])
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Decode indices to text."""
        idx_to_word = {i: w for w, i in self.vocab.items()}
        words = []
        
        for idx in indices:
            word = idx_to_word.get(idx, '<UNK>')
            if word not in ['<PAD>', '<BOS>', '<EOS>']:
                words.append(word)
        
        return ' '.join(words)


def tokenize_text(text: str, method: str = 'simple') -> List[str]:
    """
    Tokenize text using specified method.
    
    Args:
        text: Input text
        method: Tokenization method ('simple', 'regex', 'nltk')
    
    Returns:
        List of tokens
    """
    if method == 'simple':
        return text.lower().split()
    
    elif method == 'regex':
        # Regex-based tokenization
        pattern = r'\b\w+\b'
        return re.findall(pattern, text.lower())
    
    elif method == 'nltk':
        try:
            import nltk
            return nltk.word_tokenize(text.lower())
        except ImportError:
            print("NLTK not available, falling back to simple tokenization")
            return text.lower().split()
    
    else:
        raise ValueError(f"Unknown tokenization method: {method}")


def build_vocabulary(
    texts: List[str],
    vocab_size: int = 10000,
    min_freq: int = 2,
    special_tokens: Optional[List[str]] = None
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build vocabulary from texts.
    
    Args:
        texts: List of texts
        vocab_size: Maximum vocabulary size
        min_freq: Minimum frequency threshold
        special_tokens: Special tokens to include
    
    Returns:
        Tuple of (word_to_idx, idx_to_word)
    """
    if special_tokens is None:
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    
    # Count words
    word_counts = Counter()
    for text in texts:
        tokens = tokenize_text(text)
        word_counts.update(tokens)
    
    # Build vocabulary
    vocab = OrderedDict()
    
    # Add special tokens first
    for token in special_tokens:
        vocab[token] = len(vocab)
    
    # Add frequent words
    for word, count in word_counts.most_common():
        if count >= min_freq and len(vocab) < vocab_size:
            vocab[word] = len(vocab)
        elif len(vocab) >= vocab_size:
            break
    
    word_to_idx = dict(vocab)
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    return word_to_idx, idx_to_word