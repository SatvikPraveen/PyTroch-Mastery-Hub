# tests/test_integration.py
"""
Integration tests for PyTorch Mastery Hub
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from neural_networks.models import SimpleMLP
from neural_networks.training import train_epoch, validate_epoch
from utils.data_utils import generate_synthetic_data
from utils.metrics import accuracy


class TestEndToEndTraining:
    """Test end-to-end training pipeline."""
    
    def test_complete_training_pipeline(self):
        """Test complete training pipeline from data to trained model."""
        # Generate synthetic data
        X, y = generate_synthetic_data(
            task='classification',
            n_samples=200,
            n_features=10,
            n_classes=3,
            random_seed=42
        )
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Create model
        model = SimpleMLP(
            input_size=10,
            hidden_sizes=[20, 15],
            output_size=3,
            activation='relu',
            dropout=0.1
        )
        
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device('cpu')
        
        # Train for one epoch
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Check that training completed
        assert 'loss' in train_metrics
        assert 'loss' in val_metrics
        assert train_metrics['loss'] > 0
        assert val_metrics['loss'] > 0
        
        # Check that model can make predictions
        model.eval()
        with torch.no_grad():
            sample_input = X[:5]
            predictions = model(sample_input)
            assert predictions.shape == (5, 3)


class TestModelCompatibility:
    """Test compatibility between different components."""
    
    def test_model_with_different_optimizers(self):
        """Test model training with different optimizers."""
        model = SimpleMLP(10, [20], 5)
        X = torch.randn(50, 10)
        y = torch.randint(0, 5, (50,))
        
        optimizers = [
            torch.optim.SGD(model.parameters(), lr=0.01),
            torch.optim.Adam(model.parameters(), lr=0.001),
            torch.optim.RMSprop(model.parameters(), lr=0.001)
        ]
        
        for optimizer in optimizers:
            # Reset model parameters
            for param in model.parameters():
                param.data.fill_(0.1)
            
            # Single training step
            optimizer.zero_grad()
            outputs = model(X)
            loss = nn.CrossEntropyLoss()(outputs, y)
            loss.backward()
            optimizer.step()
            
            # Check that parameters were updated
            param_sum = sum(p.sum().item() for p in model.parameters())
            assert param_sum != 0.1 * sum(p.numel() for p in model.parameters())


class TestDataFlowIntegration:
    """Test data flow through the entire pipeline."""
    
    def test_data_preprocessing_to_model(self):
        """Test data flow from preprocessing to model prediction."""
        from utils.data_utils import normalize_data, create_data_loaders
        
        # Generate raw data
        raw_data = torch.randn(100, 8) * 10 + 5  # Non-normalized
        labels = torch.randint(0, 2, (100,))
        
        # Normalize data
        normalized_data, scaler = normalize_data(raw_data, method='standard')
        
        # Create dataset and loader
        dataset = torch.utils.data.TensorDataset(normalized_data, labels)
        loaders = create_data_loaders(dataset, batch_size=16)
        
        # Create and use model
        model = SimpleMLP(8, [16], 2)
        model.eval()
        
        # Test on a batch
        for batch_data, batch_labels in loaders['train']:
            predictions = model(batch_data)
            
            assert predictions.shape == (batch_data.size(0), 2)
            assert not torch.isnan(predictions).any()
            break  # Just test one batch


class TestCrossModuleIntegration:
    """Test integration between different modules."""
    
    def test_cv_transforms_with_models(self):
        """Test computer vision transforms with models."""
        try:
            from computer_vision.transforms import get_val_transforms
            from computer_vision.models import SimpleCNN
            from PIL import Image
            import numpy as np
            
            # Create sample image
            img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            # Apply transforms
            transforms = get_val_transforms(input_size=32, normalize=False)
            transformed = transforms(img)
            
            # Use with model
            model = SimpleCNN(num_classes=10, filters=[16, 32])
            model.eval()
            
            # Add batch dimension
            batch = transformed.unsqueeze(0)
            
            with torch.no_grad():
                output = model(batch)
            
            assert output.shape == (1, 10)
            
        except ImportError:
            pytest.skip("PIL not available")
    
    def test_nlp_tokenization_with_models(self):
        """Test NLP tokenization with models."""
        from nlp.tokenization import SimpleTokenizer
        from nlp.models import RNNClassifier
        
        # Create tokenizer and build vocabulary
        texts = ["hello world", "this is test", "another example"]
        tokenizer = SimpleTokenizer(vocab_size=50)
        tokenizer.build_vocab(texts)
        
        # Tokenize and encode
        encoded = tokenizer.encode("hello test")
        
        # Create model
        model = RNNClassifier(
            vocab_size=len(tokenizer),
            embedding_dim=16,
            hidden_dim=32,
            num_classes=2,
            rnn_type='LSTM'
        )
        
        # Test with sequence
        sequence = torch.tensor(encoded).unsqueeze(0)  # Add batch dim
        
        model.eval()
        with torch.no_grad():
            output = model(sequence)
        
        assert output.shape == (1, 2)


class TestAdvancedFeatures:
    """Test advanced features integration."""
    
    def test_gan_training_integration(self):
        """Test basic GAN training integration."""
        from advanced.gan_utils import Generator, Discriminator, GANTrainer
        
        # Create GAN components
        generator = Generator(noise_dim=10, output_dim=20, hidden_dims=[16])
        discriminator = Discriminator(input_dim=20, hidden_dims=[16])
        
        # Create optimizers
        g_opt = torch.optim.Adam(generator.parameters(), lr=0.0002)
        d_opt = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
        
        # Create trainer
        trainer = GANTrainer(
            generator, discriminator, g_opt, d_opt,
            device=torch.device('cpu'),
            gan_type='vanilla'
        )
        
        # Test training step
        batch_size = 8
        real_data = torch.randn(batch_size, 20)
        noise = torch.randn(batch_size, 10)
        
        d_loss, g_loss = trainer.train_step(real_data, noise)
        
        assert isinstance(d_loss, float)
        assert isinstance(g_loss, float)
        assert d_loss >= 0
        assert g_loss >= 0
    
    def test_model_optimization_integration(self):
        """Test model optimization features."""
        from advanced.optimization import ModelOptimizer
        
        # Create a simple model
        model = SimpleMLP(10, [20, 15], 5)
        
        # Create optimizer
        optimizer = ModelOptimizer(model)
        
        # Apply optimizations
        optimized_model = (optimizer
                          .apply_pruning(amount=0.1, method='magnitude')
                          .get_optimized_model())
        
        # Test that optimized model still works
        test_input = torch.randn(4, 10)
        
        original_output = model(test_input)
        optimized_output = optimized_model(test_input)
        
        assert original_output.shape == optimized_output.shape
        # Outputs might be different due to pruning, but should be valid
        assert not torch.isnan(optimized_output).any()


class TestErrorHandlingIntegration:
    """Test error handling across modules."""
    
    def test_dimension_mismatch_errors(self):
        """Test proper error handling for dimension mismatches."""
        model = SimpleMLP(10, [20], 5)
        
        # Wrong input dimension
        wrong_input = torch.randn(4, 8)  # Should be 10
        
        with pytest.raises((RuntimeError, ValueError)):
            model(wrong_input)
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        from utils.data_utils import create_data_loaders
        
        # Empty dataset
        empty_X = torch.empty(0, 10)
        empty_y = torch.empty(0, dtype=torch.long)
        empty_dataset = torch.utils.data.TensorDataset(empty_X, empty_y)
        
        # Should not raise error when creating loader
        loaders = create_data_loaders(empty_dataset, batch_size=16)
        
        # But should be empty
        assert len(list(loaders['train'])) == 0


class TestPerformanceIntegration:
    """Test performance-related integration."""
    
    def test_gpu_cpu_compatibility(self, device):
        """Test model works on both CPU and GPU."""
        model = SimpleMLP(10, [20], 5)
        data = torch.randn(16, 10)
        
        # Test on CPU
        model_cpu = model.to('cpu')
        data_cpu = data.to('cpu')
        
        output_cpu = model_cpu(data_cpu)
        assert output_cpu.shape == (16, 5)
        
        # Test on GPU if available
        if device.type == 'cuda':
            model_gpu = model.to(device)
            data_gpu = data.to(device)
            
            output_gpu = model_gpu(data_gpu)
            assert output_gpu.shape == (16, 5)
            assert output_gpu.device.type == 'cuda'
    
    @pytest.mark.slow
    def test_training_convergence(self):
        """Test that training actually improves model performance."""
        # Generate separable data
        X, y = generate_synthetic_data(
            task='classification',
            n_samples=500,
            n_features=10,
            n_classes=2,
            noise=0.1,  # Low noise for easier learning
            random_seed=42
        )
        
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model = SimpleMLP(10, [20], 2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Initial accuracy
        model.eval()
        with torch.no_grad():
            initial_output = model(X)
            initial_acc = accuracy(initial_output, y)
        
        # Train for several epochs
        model.train()
        for epoch in range(5):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Final accuracy
        model.eval()
        with torch.no_grad():
            final_output = model(X)
            final_acc = accuracy(final_output, y)
        
        # Should improve with separable data
        assert final_acc > initial_acc or final_acc > 0.8  # Either improved or already good