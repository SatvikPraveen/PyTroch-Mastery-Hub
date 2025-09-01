# tests/test_neural_networks/test_models.py
"""
Tests for neural network models
"""

import pytest
import torch
import torch.nn as nn
from neural_networks.models import (
    SimpleMLP, DeepMLP, CustomCNN, ResNet, SimpleRNN, 
    SimpleLSTM, SimpleGRU, TransformerBlock, SimpleTransformer,
    AutoEncoder, VariationalAutoEncoder
)


class TestSimpleMLP:
    """Test SimpleMLP model."""
    
    def test_mlp_initialization(self):
        """Test MLP initialization."""
        model = SimpleMLP(
            input_size=10,
            hidden_sizes=[20, 15],
            output_size=5
        )
        
        assert model.input_size == 10
        assert model.hidden_sizes == [20, 15]
        assert model.output_size == 5
    
    def test_mlp_forward_pass(self):
        """Test MLP forward pass."""
        model = SimpleMLP(10, [20, 15], 5)
        batch_size = 8
        x = torch.randn(batch_size, 10)
        
        output = model(x)
        
        assert output.shape == (batch_size, 5)
        assert not torch.isnan(output).any()
    
    def test_mlp_with_batch_norm(self):
        """Test MLP with batch normalization."""
        model = SimpleMLP(10, [20, 15], 5, batch_norm=True)
        x = torch.randn(8, 10)
        
        output = model(x)
        
        assert output.shape == (8, 5)
    
    def test_mlp_with_dropout(self):
        """Test MLP with dropout."""
        model = SimpleMLP(10, [20, 15], 5, dropout=0.5)
        x = torch.randn(8, 10)
        
        # Training mode (dropout active)
        model.train()
        output_train = model(x)
        
        # Eval mode (dropout inactive)
        model.eval()
        output_eval = model(x)
        
        assert output_train.shape == output_eval.shape == (8, 5)


class TestDeepMLP:
    """Test DeepMLP model."""
    
    def test_deep_mlp_initialization(self):
        """Test DeepMLP initialization."""
        model = DeepMLP(
            input_size=10,
            hidden_size=64,
            num_layers=3,
            output_size=5
        )
        
        assert len(model.layers) == 3
    
    def test_deep_mlp_forward_pass(self):
        """Test DeepMLP forward pass."""
        model = DeepMLP(10, 64, 3, 5)
        x = torch.randn(8, 10)
        
        output = model(x)
        
        assert output.shape == (8, 5)
    
    def test_deep_mlp_without_residual(self):
        """Test DeepMLP without residual connections."""
        model = DeepMLP(10, 64, 2, 5, residual=False)
        x = torch.randn(8, 10)
        
        output = model(x)
        
        assert output.shape == (8, 5)


class TestCustomCNN:
    """Test CustomCNN model."""
    
    def test_cnn_initialization(self):
        """Test CNN initialization."""
        model = CustomCNN(num_classes=10)
        
        assert hasattr(model, 'features')
        assert hasattr(model, 'classifier')
    
    def test_cnn_forward_pass(self):
        """Test CNN forward pass."""
        model = CustomCNN(num_classes=10)
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        
        output = model(x)
        
        assert output.shape == (batch_size, 10)
    
    def test_cnn_different_input_channels(self):
        """Test CNN with different input channels."""
        model = CustomCNN(num_classes=5, input_channels=1)
        x = torch.randn(4, 1, 32, 32)
        
        output = model(x)
        
        assert output.shape == (4, 5)


class TestResNet:
    """Test ResNet model."""
    
    def test_resnet_initialization(self):
        """Test ResNet initialization."""
        model = ResNet(num_classes=10)
        
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'layer1')
        assert hasattr(model, 'fc')
    
    def test_resnet_forward_pass(self):
        """Test ResNet forward pass."""
        model = ResNet(num_classes=10, layers=[1, 1, 1, 1])  # Smaller for testing
        x = torch.randn(2, 3, 64, 64)
        
        output = model(x)
        
        assert output.shape == (2, 10)
    
    def test_resnet_custom_layers(self):
        """Test ResNet with custom layer configuration."""
        model = ResNet(num_classes=5, layers=[2, 2, 2, 2])
        x = torch.randn(2, 3, 64, 64)
        
        output = model(x)
        
        assert output.shape == (2, 5)


class TestRNNModels:
    """Test RNN-based models."""
    
    def test_simple_rnn(self):
        """Test SimpleRNN model."""
        model = SimpleRNN(
            input_size=10,
            hidden_size=20,
            num_layers=2,
            output_size=5
        )
        
        batch_size, seq_len = 4, 15
        x = torch.randn(batch_size, seq_len, 10)
        
        output = model(x)
        
        assert output.shape == (batch_size, 5)
    
    def test_simple_lstm(self):
        """Test SimpleLSTM model."""
        model = SimpleLSTM(10, 20, 2, 5)
        x = torch.randn(4, 15, 10)
        
        output = model(x)
        
        assert output.shape == (4, 5)
    
    def test_simple_gru(self):
        """Test SimpleGRU model."""
        model = SimpleGRU(10, 20, 2, 5)
        x = torch.randn(4, 15, 10)
        
        output = model(x)
        
        assert output.shape == (4, 5)
    
    def test_bidirectional_rnn(self):
        """Test bidirectional RNN."""
        model = SimpleRNN(10, 20, 1, 5, bidirectional=True)
        x = torch.randn(4, 15, 10)
        
        output = model(x)
        
        assert output.shape == (4, 5)


class TestTransformerModels:
    """Test Transformer-based models."""
    
    def test_transformer_block(self):
        """Test TransformerBlock."""
        block = TransformerBlock(d_model=64, num_heads=4)
        x = torch.randn(4, 10, 64)  # batch_size, seq_len, d_model
        
        output = block(x)
        
        assert output.shape == (4, 10, 64)
    
    def test_simple_transformer(self):
        """Test SimpleTransformer."""
        model = SimpleTransformer(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            num_layers=2,
            num_classes=5
        )
        
        batch_size, seq_len = 4, 10
        x = torch.randint(0, 100, (batch_size, seq_len))
        
        output = model(x)
        
        assert output.shape == (batch_size, 5)
    
    def test_transformer_with_mask(self):
        """Test Transformer with attention mask."""
        model = SimpleTransformer(100, 64, 4, 2, 5)
        x = torch.randint(0, 100, (4, 10))
        mask = torch.ones(4, 10)
        mask[:, -2:] = 0  # Mask last 2 positions
        
        output = model(x, mask)
        
        assert output.shape == (4, 5)


class TestAutoEncoders:
    """Test AutoEncoder models."""
    
    def test_autoencoder(self):
        """Test basic AutoEncoder."""
        model = AutoEncoder(
            input_size=784,
            encoding_dims=[256, 128, 64],
            activation='relu'
        )
        
        batch_size = 8
        x = torch.randn(batch_size, 784)
        
        encoded, decoded = model(x)
        
        assert encoded.shape == (batch_size, 64)
        assert decoded.shape == (batch_size, 784)
    
    def test_autoencoder_encode_decode(self):
        """Test separate encode/decode methods."""
        model = AutoEncoder(100, [50, 25])
        x = torch.randn(8, 100)
        
        encoded = model.encode(x)
        decoded = model.decode(encoded)
        
        assert encoded.shape == (8, 25)
        assert decoded.shape == (8, 100)
    
    def test_variational_autoencoder(self):
        """Test Variational AutoEncoder."""
        model = VariationalAutoEncoder(
            input_size=784,
            hidden_size=256,
            latent_size=32
        )
        
        batch_size = 8
        x = torch.randn(batch_size, 784)
        
        recon, mu, logvar = model(x)
        
        assert recon.shape == (batch_size, 784)
        assert mu.shape == (batch_size, 32)
        assert logvar.shape == (batch_size, 32)
    
    def test_vae_reparameterize(self):
        """Test VAE reparameterization trick."""
        model = VariationalAutoEncoder(100, 128, 16)
        mu = torch.randn(8, 16)
        logvar = torch.randn(8, 16)
        
        z = model.reparameterize(mu, logvar)
        
        assert z.shape == (8, 16)


class TestModelParameters:
    """Test model parameter properties."""
    
    def test_model_has_trainable_parameters(self):
        """Test that models have trainable parameters."""
        model = SimpleMLP(10, [20], 5)
        
        params = list(model.parameters())
        assert len(params) > 0
        
        # Check if parameters require gradients
        for param in params:
            assert param.requires_grad
    
    def test_model_parameter_count(self):
        """Test parameter count calculation."""
        model = SimpleMLP(10, [20], 5)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Expected: (10*20 + 20) + (20*5 + 5) = 200 + 20 + 100 + 5 = 325
        expected_params = (10 * 20 + 20) + (20 * 5 + 5)
        assert total_params == expected_params
    
    def test_model_eval_mode(self):
        """Test model evaluation mode."""
        model = SimpleMLP(10, [20], 5, dropout=0.5)
        
        # Switch to eval mode
        model.eval()
        assert not model.training
        
        # Switch back to training mode
        model.train()
        assert model.training