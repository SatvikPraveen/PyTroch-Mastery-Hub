# tests/test_nlp/test_tokenization.py
"""
Tests for NLP tokenization utilities
"""

import pytest
import tempfile
from pathlib import Path
from nlp.tokenization import (
    SimpleTokenizer, BPETokenizer, SubwordTokenizer, WordTokenizer,
    tokenize_text, build_vocabulary
)


class TestSimpleTokenizer:
    """Test SimpleTokenizer functionality."""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = SimpleTokenizer(vocab_size=1000, min_freq=2)
        
        assert tokenizer.vocab_size == 1000
        assert tokenizer.min_freq == 2
        assert not tokenizer.vocab_built
    
    def test_build_vocab(self, sample_texts):
        """Test vocabulary building."""
        tokenizer = SimpleTokenizer(vocab_size=100, min_freq=1)
        tokenizer.build_vocab(sample_texts)
        
        assert tokenizer.vocab_built
        assert len(tokenizer.word_to_idx) > 0
        assert len(tokenizer.idx_to_word) == len(tokenizer.word_to_idx)
        
        # Check special tokens
        for token in tokenizer.special_tokens:
            assert token in tokenizer.word_to_idx
    
    def test_tokenize_text(self, sample_text):
        """Test text tokenization."""
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.tokenize(sample_text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
    
    def test_encode_text(self, sample_texts):
        """Test text encoding."""
        tokenizer = SimpleTokenizer(vocab_size=100)
        tokenizer.build_vocab(sample_texts)
        
        text = sample_texts[0]
        encoded = tokenizer.encode(text, add_special_tokens=True)
        
        assert isinstance(encoded, list)
        assert all(isinstance(idx, int) for idx in encoded)
        assert encoded[0] == tokenizer.word_to_idx['<BOS>']
        assert encoded[-1] == tokenizer.word_to_idx['<EOS>']
    
    def test_decode_text(self, sample_texts):
        """Test text decoding."""
        tokenizer = SimpleTokenizer(vocab_size=100)
        tokenizer.build_vocab(sample_texts)
        
        text = sample_texts[0]
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        
        assert isinstance(decoded, str)
        # Decoded text might not be identical due to preprocessing
        assert len(decoded) > 0
    
    def test_encode_without_vocab_raises_error(self):
        """Test encoding without built vocabulary raises error."""
        tokenizer = SimpleTokenizer()
        
        with pytest.raises(ValueError, match="Vocabulary not built"):
            tokenizer.encode("test text")
    
    def test_save_and_load_tokenizer(self, sample_texts, temp_dir):
        """Test saving and loading tokenizer."""
        tokenizer = SimpleTokenizer(vocab_size=100)
        tokenizer.build_vocab(sample_texts)
        
        # Save tokenizer
        save_path = temp_dir / "tokenizer.pkl"
        tokenizer.save(save_path)
        
        # Load tokenizer
        new_tokenizer = SimpleTokenizer()
        new_tokenizer.load(save_path)
        
        assert new_tokenizer.vocab_built
        assert new_tokenizer.word_to_idx == tokenizer.word_to_idx
        assert new_tokenizer.vocab_size == tokenizer.vocab_size


class TestBPETokenizer:
    """Test BPE tokenization functionality."""
    
    def test_bpe_initialization(self):
        """Test BPE tokenizer initialization."""
        tokenizer = BPETokenizer(vocab_size=1000)
        
        assert tokenizer.vocab_size == 1000
        assert len(tokenizer.bpe_codes) == 0
    
    def test_bpe_training(self, sample_texts):
        """Test BPE training."""
        tokenizer = BPETokenizer(vocab_size=200)
        tokenizer.train(sample_texts)
        
        assert len(tokenizer.bpe_codes) > 0
        assert len(tokenizer.encoder) > 0
        assert len(tokenizer.decoder) == len(tokenizer.encoder)
    
    def test_bpe_encode_decode(self, sample_texts):
        """Test BPE encoding and decoding."""
        tokenizer = BPETokenizer(vocab_size=200)
        tokenizer.train(sample_texts)
        
        text = sample_texts[0]
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        assert isinstance(encoded, list)
        assert isinstance(decoded, str)
        assert len(encoded) > 0
    
    def test_get_word_tokens(self):
        """Test word token extraction."""
        tokenizer = BPETokenizer()
        text = "hello world hello"
        
        word_vocab = tokenizer.get_word_tokens(text)
        
        assert isinstance(word_vocab, dict)
        assert len(word_vocab) > 0
        # Should contain end-of-word markers
        assert any('</w>' in word for word in word_vocab.keys())


class TestSubwordTokenizer:
    """Test Subword tokenization."""
    
    def test_subword_initialization(self):
        """Test subword tokenizer initialization."""
        tokenizer = SubwordTokenizer(vocab_size=1000)
        
        assert tokenizer.vocab_size == 1000
    
    def test_subword_training(self, sample_texts):
        """Test subword tokenizer training."""
        tokenizer = SubwordTokenizer(vocab_size=200)
        tokenizer.train(sample_texts)
        
        assert len(tokenizer.encoder) > 0
        assert len(tokenizer.decoder) == len(tokenizer.encoder)
    
    def test_subword_encode_decode(self, sample_texts):
        """Test subword encoding and decoding."""
        tokenizer = SubwordTokenizer(vocab_size=200)
        tokenizer.train(sample_texts)
        
        text = "hello world"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        assert isinstance(encoded, list)
        assert isinstance(decoded, str)


class TestWordTokenizer:
    """Test advanced word tokenization."""
    
    def test_word_tokenizer_initialization(self):
        """Test word tokenizer initialization."""
        tokenizer = WordTokenizer(vocab_size=1000, min_freq=2)
        
        assert tokenizer.vocab_size == 1000
        assert tokenizer.min_freq == 2
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        tokenizer = WordTokenizer(lowercase=True, remove_punct=True)
        
        text = "Hello, World! How are you?"
        processed = tokenizer.preprocess(text)
        
        assert processed.islower()
        assert ',' not in processed
        assert '!' not in processed
        assert '?' not in processed
    
    def test_tokenize_with_contractions(self):
        """Test tokenization with contractions."""
        tokenizer = WordTokenizer()
        
        text = "I'm happy you're here"
        tokens = tokenizer.tokenize_text(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
    
    def test_build_vocab_with_frequency_threshold(self, sample_texts):
        """Test vocabulary building with frequency threshold."""
        tokenizer = WordTokenizer(vocab_size=100, min_freq=2)
        tokenizer.build_vocab(sample_texts)
        
        assert len(tokenizer.vocab) > 0
        # All words should meet minimum frequency requirement
        for word, count in tokenizer.word_counts.items():
            if word in tokenizer.vocab and word not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                assert count >= 2 or len(tokenizer.vocab) < tokenizer.vocab_size


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_tokenize_text_simple(self, sample_text):
        """Test simple text tokenization."""
        tokens = tokenize_text(sample_text, method='simple')
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(token.islower() for token in tokens)
    
    def test_tokenize_text_regex(self, sample_text):
        """Test regex-based tokenization."""
        tokens = tokenize_text(sample_text, method='regex')
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
    
    def test_tokenize_text_unknown_method(self, sample_text):
        """Test tokenization with unknown method raises error."""
        with pytest.raises(ValueError, match="Unknown tokenization method"):
            tokenize_text(sample_text, method='unknown')
    
    def test_build_vocabulary_function(self, sample_texts):
        """Test build_vocabulary utility function."""
        word_to_idx, idx_to_word = build_vocabulary(
            sample_texts, vocab_size=100, min_freq=1
        )
        
        assert isinstance(word_to_idx, dict)
        assert isinstance(idx_to_word, dict)
        assert len(word_to_idx) == len(idx_to_word)
        
        # Check special tokens
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        for token in special_tokens:
            assert token in word_to_idx
    
    def test_build_vocabulary_with_custom_special_tokens(self, sample_texts):
        """Test vocabulary building with custom special tokens."""
        custom_special = ['<START>', '<END>', '<MASK>']
        
        word_to_idx, idx_to_word = build_vocabulary(
            sample_texts, 
            vocab_size=100, 
            special_tokens=custom_special
        )
        
        for token in custom_special:
            assert token in word_to_idx


class TestTokenizerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_text_tokenization(self):
        """Test tokenization of empty text."""
        tokenizer = SimpleTokenizer()
        
        tokens = tokenizer.tokenize("")
        
        assert isinstance(tokens, list)
        assert len(tokens) == 0
    
    def test_tokenizer_with_empty_corpus(self):
        """Test tokenizer training with empty corpus."""
        tokenizer = SimpleTokenizer(vocab_size=100)
        
        # Should handle empty corpus gracefully
        tokenizer.build_vocab([])
        
        # Should only have special tokens
        assert len(tokenizer.word_to_idx) == len(tokenizer.special_tokens)
    
    def test_very_small_vocab_size(self, sample_texts):
        """Test tokenizer with very small vocabulary size."""
        # Vocab size smaller than special tokens
        tokenizer = SimpleTokenizer(vocab_size=2)
        tokenizer.build_vocab(sample_texts)
        
        # Should still contain special tokens
        for token in tokenizer.special_tokens[:2]:
            assert token in tokenizer.word_to_idx
    
    def test_encode_unknown_words(self, sample_texts):
        """Test encoding text with unknown words."""
        tokenizer = SimpleTokenizer(vocab_size=10, min_freq=10)  # Very restrictive
        tokenizer.build_vocab(sample_texts)
        
        # Encode text with likely unknown words
        encoded = tokenizer.encode("unknown words that probably arent in vocab")
        
        # Should contain UNK tokens
        unk_idx = tokenizer.word_to_idx['<UNK>']
        assert unk_idx in encoded