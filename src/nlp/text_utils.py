# src/nlp/text_utils.py
"""
Text processing utilities for PyTorch Mastery Hub
"""

import torch
import re
import string
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter
import numpy as np


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_punct: bool = True,
    remove_numbers: bool = False,
    remove_extra_spaces: bool = True
) -> str:
    """
    Preprocess text with various cleaning options.
    
    Args:
        text: Input text
        lowercase: Convert to lowercase
        remove_punct: Remove punctuation
        remove_numbers: Remove numbers
        remove_extra_spaces: Remove extra whitespace
        
    Returns:
        Preprocessed text
    """
    if lowercase:
        text = text.lower()
    
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    if remove_extra_spaces:
        text = ' '.join(text.split())
    
    return text


def clean_text(text: str) -> str:
    """
    Basic text cleaning.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def text_to_sequences(
    texts: List[str],
    word_to_idx: Dict[str, int],
    max_length: Optional[int] = None,
    padding: str = 'post',
    truncating: str = 'post'
) -> List[List[int]]:
    """
    Convert texts to sequences of token indices.
    
    Args:
        texts: List of texts
        word_to_idx: Vocabulary mapping
        max_length: Maximum sequence length
        padding: Padding strategy ('pre' or 'post')
        truncating: Truncating strategy ('pre' or 'post')
        
    Returns:
        List of sequences
    """
    sequences = []
    
    for text in texts:
        tokens = text.lower().split()
        sequence = [word_to_idx.get(token, word_to_idx.get('<UNK>', 1)) for token in tokens]
        sequences.append(sequence)
    
    if max_length is not None:
        sequences = pad_sequences(sequences, max_length, padding, truncating)
    
    return sequences


def pad_sequences(
    sequences: List[List[int]],
    max_length: int,
    padding: str = 'post',
    truncating: str = 'post',
    pad_value: int = 0
) -> List[List[int]]:
    """
    Pad sequences to uniform length.
    
    Args:
        sequences: List of sequences
        max_length: Target length
        padding: Padding strategy
        truncating: Truncating strategy  
        pad_value: Padding value
        
    Returns:
        Padded sequences
    """
    padded_sequences = []
    
    for seq in sequences:
        # Truncate if necessary
        if len(seq) > max_length:
            if truncating == 'pre':
                seq = seq[-max_length:]
            else:
                seq = seq[:max_length]
        
        # Pad if necessary
        if len(seq) < max_length:
            pad_length = max_length - len(seq)
            if padding == 'pre':
                seq = [pad_value] * pad_length + seq
            else:
                seq = seq + [pad_value] * pad_length
        
        padded_sequences.append(seq)
    
    return padded_sequences


def compute_text_stats(texts: List[str]) -> Dict[str, Union[int, float]]:
    """
    Compute statistics about text data.
    
    Args:
        texts: List of texts
        
    Returns:
        Dictionary with statistics
    """
    if not texts:
        return {}
    
    # Tokenize all texts
    all_tokens = []
    lengths = []
    
    for text in texts:
        tokens = text.lower().split()
        all_tokens.extend(tokens)
        lengths.append(len(tokens))
    
    # Compute statistics
    vocab_size = len(set(all_tokens))
    total_tokens = len(all_tokens)
    
    stats = {
        'num_texts': len(texts),
        'total_tokens': total_tokens,
        'vocab_size': vocab_size,
        'avg_text_length': np.mean(lengths),
        'min_text_length': min(lengths),
        'max_text_length': max(lengths),
        'std_text_length': np.std(lengths),
        'median_text_length': np.median(lengths)
    }
    
    # Most common words
    word_counts = Counter(all_tokens)
    stats['most_common_words'] = word_counts.most_common(10)
    
    return stats


def create_ngrams(text: str, n: int = 2) -> List[str]:
    """
    Create n-grams from text.
    
    Args:
        text: Input text
        n: N-gram size
        
    Returns:
        List of n-grams
    """
    words = text.lower().split()
    
    if len(words) < n:
        return []
    
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams


def calculate_bleu_score(
    reference: List[str],
    candidate: List[str],
    weights: List[float] = [0.25, 0.25, 0.25, 0.25]
) -> float:
    """
    Calculate BLEU score for text generation evaluation.
    
    Args:
        reference: Reference text tokens
        candidate: Candidate text tokens
        weights: N-gram weights
        
    Returns:
        BLEU score
    """
    if len(candidate) == 0:
        return 0.0
    
    # Brevity penalty
    ref_len = len(reference)
    cand_len = len(candidate)
    
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0
    
    # N-gram precisions
    precisions = []
    
    for n in range(1, len(weights) + 1):
        ref_ngrams = Counter(create_ngrams(' '.join(reference), n))
        cand_ngrams = Counter(create_ngrams(' '.join(candidate), n))
        
        overlap = 0
        total = 0
        
        for ngram, count in cand_ngrams.items():
            overlap += min(count, ref_ngrams.get(ngram, 0))
            total += count
        
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(overlap / total)
    
    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        geo_mean = np.exp(sum(w * np.log(p) for w, p in zip(weights, precisions)))
    else:
        geo_mean = 0.0
    
    return bp * geo_mean


def calculate_rouge_l(reference: List[str], candidate: List[str]) -> float:
    """
    Calculate ROUGE-L score.
    
    Args:
        reference: Reference text tokens
        candidate: Candidate text tokens
        
    Returns:
        ROUGE-L F1 score
    """
    if not reference or not candidate:
        return 0.0
    
    # Find LCS length
    def lcs_length(x, y):
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    lcs_len = lcs_length(reference, candidate)
    
    if lcs_len == 0:
        return 0.0
    
    # Calculate precision, recall, and F1
    precision = lcs_len / len(candidate)
    recall = lcs_len / len(reference)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def extract_keywords(
    text: str,
    num_keywords: int = 10,
    min_word_length: int = 3
) -> List[Tuple[str, float]]:
    """
    Extract keywords using TF-IDF-like scoring.
    
    Args:
        text: Input text
        num_keywords: Number of keywords to extract
        min_word_length: Minimum word length
        
    Returns:
        List of (keyword, score) tuples
    """
    # Simple preprocessing
    words = text.lower().split()
    words = [w.strip(string.punctuation) for w in words]
    words = [w for w in words if len(w) >= min_word_length and w.isalpha()]
    
    if not words:
        return []
    
    # Calculate word frequencies
    word_freq = Counter(words)
    max_freq = max(word_freq.values())
    
    # Normalize frequencies (simple TF)
    word_scores = {}
    for word, freq in word_freq.items():
        word_scores[word] = freq / max_freq
    
    # Sort by score
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_words[:num_keywords]


def sentence_similarity(sent1: str, sent2: str) -> float:
    """
    Calculate similarity between two sentences using word overlap.
    
    Args:
        sent1: First sentence
        sent2: Second sentence
        
    Returns:
        Similarity score (0-1)
    """
    words1 = set(sent1.lower().split())
    words2 = set(sent2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def detect_language(text: str) -> str:
    """
    Simple language detection based on character patterns.
    
    Args:
        text: Input text
        
    Returns:
        Detected language code
    """
    # Very basic language detection
    # In practice, use proper libraries like langdetect
    
    # Count character types
    latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    total_chars = sum(1 for c in text if c.isalpha())
    
    if total_chars == 0:
        return 'unknown'
    
    latin_ratio = latin_chars / total_chars
    
    if latin_ratio > 0.8:
        return 'en'  # English (or other Latin-based)
    else:
        return 'other'


def generate_summary(
    text: str,
    num_sentences: int = 3,
    sentence_separator: str = '.'
) -> str:
    """
    Simple extractive summarization.
    
    Args:
        text: Input text
        num_sentences: Number of sentences in summary
        sentence_separator: Sentence separator
        
    Returns:
        Generated summary
    """
    # Split into sentences
    sentences = [s.strip() for s in text.split(sentence_separator) if s.strip()]
    
    if len(sentences) <= num_sentences:
        return text
    
    # Score sentences by word frequency
    words = text.lower().split()
    word_freq = Counter(words)
    
    sentence_scores = []
    for sent in sentences:
        sent_words = sent.lower().split()
        score = sum(word_freq.get(word, 0) for word in sent_words)
        sentence_scores.append((score, sent))
    
    # Select top sentences
    top_sentences = sorted(sentence_scores, key=lambda x: x[0], reverse=True)[:num_sentences]
    
    # Maintain original order
    selected = [sent for _, sent in top_sentences]
    summary_sentences = []
    
    for sent in sentences:
        if sent in selected:
            summary_sentences.append(sent)
            if len(summary_sentences) == num_sentences:
                break
    
    return sentence_separator.join(summary_sentences) + sentence_separator


def tokenize_sentences(text: str) -> List[str]:
    """
    Simple sentence tokenization.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def remove_stopwords(
    tokens: List[str],
    stopwords: Optional[List[str]] = None
) -> List[str]:
    """
    Remove stopwords from token list.
    
    Args:
        tokens: List of tokens
        stopwords: List of stopwords (uses default if None)
        
    Returns:
        Filtered tokens
    """
    if stopwords is None:
        # Common English stopwords
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'their', 'time',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'has', 'more',
            'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call',
            'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
            'come', 'made', 'may', 'part'
        }
    else:
        stopwords = set(stopwords)
    
    return [token for token in tokens if token.lower() not in stopwords]


def calculate_readability_score(text: str) -> Dict[str, float]:
    """
    Calculate simple readability metrics.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with readability scores
    """
    sentences = tokenize_sentences(text)
    words = text.split()
    
    if not sentences or not words:
        return {'avg_sentence_length': 0, 'avg_word_length': 0}
    
    # Average sentence length
    avg_sentence_length = len(words) / len(sentences)
    
    # Average word length
    avg_word_length = sum(len(word.strip(string.punctuation)) for word in words) / len(words)
    
    # Simple complexity score (higher = more complex)
    complexity = (avg_sentence_length * 0.5) + (avg_word_length * 0.5)
    
    return {
        'avg_sentence_length': avg_sentence_length,
        'avg_word_length': avg_word_length,
        'complexity_score': complexity,
        'num_sentences': len(sentences),
        'num_words': len(words)
    }