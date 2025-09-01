"""
Tests for GAN utilities
"""

import pytest
import torch
import torch.nn as nn
from advanced.gan_utils import (
    Generator, Discriminator, DCGAN, WGAN, WGANCritic,
    compute_gradient_penalty, GANTrainer, gan_loss
)


class TestGenerator:
    """Test Generator model."""
    
    def test_generator_initialization(self):
        """Test Generator initialization."""
        generator = Generator(
            noise_dim=100,
            output_dim=784,
            hidden_dims=[256, 512]
        )
        
        assert hasattr(generator, 'model')
        
        # Check input/output dimensions
        noise = torch.randn(4, 100)
        output = generator(noise)
        assert output.shape == (4, 784)
    
    def test_generator_forward_pass(self):
        """Test Generator forward pass."""
        generator = Generator(noise_dim=50, output_dim=200)
        
        batch_size = 8
        noise = torch.randn(batch_size, 50)
        
        output = generator(noise)
        
        assert output.shape == (batch_size, 200)
        assert not torch.isnan(output).any()
        
        # Output should be in tanh range [-1, 1]
        assert output.min() >= -1.0
        assert output.max() <= 1.0
    
    def test_generator_different_architectures(self):
        """Test Generator with different architectures."""
        # Small architecture
        small_gen = Generator(10, 50, [20])
        noise = torch.randn(4, 10)
        output = small_gen(noise)
        assert output.shape == (4, 50)
        
        # Large architecture
        large_gen = Generator(100, 1000, [256, 512, 1024])
        noise = torch.randn(2, 100)
        output = large_gen(noise)
        assert output.shape == (2, 1000)


class TestDiscriminator:
    """Test Discriminator model."""
    
    def test_discriminator_initialization(self):
        """Test Discriminator initialization."""
        discriminator = Discriminator(
            input_dim=784,
            hidden_dims=[512, 256]
        )
        
        assert hasattr(discriminator, 'model')
    
    def test_discriminator_forward_pass(self):
        """Test Discriminator forward pass."""
        discriminator = Discriminator(input_dim=100)
        
        batch_size = 8
        x = torch.randn(batch_size, 100)
        
        output = discriminator(x)
        
        assert output.shape == (batch_size,)
        assert not torch.isnan(output).any()
    
    def test_discriminator_with_real_fake_data(self):
        """Test Discriminator with real and fake data."""
        discriminator = Discriminator(input_dim=50)
        
        # Real data (from some distribution)
        real_data = torch.randn(4, 50)
        real_output = discriminator(real_data)
        
        # Fake data (from generator)
        fake_data = torch.randn(4, 50)
        fake_output = discriminator(fake_data)
        
        assert real_output.shape == fake_output.shape == (4,)


class TestDCGAN:
    """Test DCGAN implementation."""
    
    def test_dcgan_initialization(self):
        """Test DCGAN initialization."""
        dcgan = DCGAN(noise_dim=100, img_channels=3, feature_maps=64)
        
        assert hasattr(dcgan, 'generator')
        assert hasattr(dcgan, 'discriminator')
    
    def test_dcgan_generator_forward(self):
        """Test DCGAN generator forward pass."""
        dcgan = DCGAN(noise_dim=100, img_channels=3, feature_maps=32)
        
        batch_size = 4
        noise = torch.randn(batch_size, 100, 1, 1)
        
        generated_images = dcgan.generator(noise)
        
        # Should generate 64x64 images
        assert generated_images.shape == (batch_size, 3, 64, 64)
        assert not torch.isnan(generated_images).any()
    
    def test_dcgan_discriminator_forward(self):
        """Test DCGAN discriminator forward pass."""
        dcgan = DCGAN(noise_dim=100, img_channels=3, feature_maps=32)
        
        batch_size = 4
        images = torch.randn(batch_size, 3, 64, 64)
        
        output = dcgan.discriminator(images)
        
        assert output.shape == (batch_size,)
        assert not torch.isnan(output).any()
    
    def test_dcgan_with_different_channels(self):
        """Test DCGAN with different image channels."""
        # Grayscale images
        dcgan_gray = DCGAN(noise_dim=50, img_channels=1)
        noise = torch.randn(2, 50, 1, 1)
        
        generated = dcgan_gray.generator(noise)
        assert generated.shape == (2, 1, 64, 64)


class TestWGAN:
    """Test WGAN implementation."""
    
    def test_wgan_initialization(self):
        """Test WGAN initialization."""
        wgan = WGAN(noise_dim=100, output_dim=784)
        
        assert hasattr(wgan, 'generator')
        assert hasattr(wgan, 'critic')
        assert isinstance(wgan.critic, WGANCritic)
    
    def test_wgan_critic_forward(self):
        """Test WGAN critic forward pass."""
        critic = WGANCritic(input_dim=100)
        
        batch_size = 8
        x = torch.randn(batch_size, 100)
        
        output = critic(x)
        
        assert output.shape == (batch_size,)
        # Critic output is not constrained like discriminator
        assert not torch.isnan(output).any()
    
    def test_wgan_train_step(self):
        """Test WGAN training step."""
        generator = Generator(20, 50)
        critic = WGANCritic(50)
        
        g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=0.0001)
        d_optimizer = torch.optim.RMSprop(critic.parameters(), lr=0.0001)
        
        trainer = GANTrainer(
            generator, critic, 
            g_optimizer, d_optimizer,
            torch.device('cpu'),
            gan_type='wgan'
        )
        
        batch_size = 8
        real_data = torch.randn(batch_size, 50)
        noise = torch.randn(batch_size, 20)
        
        d_loss, g_loss = trainer.train_step(real_data, noise)
        
        assert isinstance(d_loss, float)
        assert isinstance(g_loss, float)


class TestComputeGradientPenalty:
    """Test gradient penalty computation for WGAN-GP."""
    
    def test_gradient_penalty_computation(self):
        """Test gradient penalty computation."""
        critic = WGANCritic(input_dim=50)
        
        batch_size = 4
        real_samples = torch.randn(batch_size, 50)
        fake_samples = torch.randn(batch_size, 50)
        device = torch.device('cpu')
        
        gp = compute_gradient_penalty(
            critic, real_samples, fake_samples, device, lambda_gp=10.0
        )
        
        assert isinstance(gp, torch.Tensor)
        assert gp.shape == torch.Size([])  # Scalar
        assert gp.item() >= 0.0
        assert not torch.isnan(gp).any()
    
    @pytest.mark.gpu
    def test_gradient_penalty_gpu(self):
        """Test gradient penalty computation on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        critic = WGANCritic(input_dim=50).to(device)
        
        batch_size = 4
        real_samples = torch.randn(batch_size, 50).to(device)
        fake_samples = torch.randn(batch_size, 50).to(device)
        
        gp = compute_gradient_penalty(
            critic, real_samples, fake_samples, device
        )
        
        assert gp.device.type == 'cuda'
        assert not torch.isnan(gp).any()
    
    def test_gradient_penalty_with_different_lambda(self):
        """Test gradient penalty with different lambda values."""
        critic = WGANCritic(input_dim=30)
        
        real_samples = torch.randn(4, 30)
        fake_samples = torch.randn(4, 30)
        device = torch.device('cpu')
        
        # Test different lambda values
        for lambda_gp in [1.0, 5.0, 10.0, 20.0]:
            gp = compute_gradient_penalty(
                critic, real_samples, fake_samples, device, lambda_gp=lambda_gp
            )
            assert isinstance(gp, torch.Tensor)
            assert not torch.isnan(gp).any()


class TestGANLoss:
    """Test GAN loss functions."""
    
    def test_vanilla_gan_loss_real(self):
        """Test vanilla GAN loss for real samples."""
        discriminator_output = torch.randn(4, 1)
        
        loss = gan_loss(discriminator_output, target_is_real=True, gan_mode='vanilla')
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss).any()
    
    def test_vanilla_gan_loss_fake(self):
        """Test vanilla GAN loss for fake samples."""
        discriminator_output = torch.randn(4, 1)
        
        loss = gan_loss(discriminator_output, target_is_real=False, gan_mode='vanilla')
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss).any()
    
    def test_lsgan_loss(self):
        """Test LSGAN loss."""
        discriminator_output = torch.randn(4, 1)
        
        real_loss = gan_loss(discriminator_output, target_is_real=True, gan_mode='lsgan')
        fake_loss = gan_loss(discriminator_output, target_is_real=False, gan_mode='lsgan')
        
        assert isinstance(real_loss, torch.Tensor)
        assert isinstance(fake_loss, torch.Tensor)
        assert not torch.isnan(real_loss).any()
        assert not torch.isnan(fake_loss).any()
    
    def test_wgan_loss(self):
        """Test WGAN loss."""
        discriminator_output = torch.randn(4, 1)
        
        real_loss = gan_loss(discriminator_output, target_is_real=True, gan_mode='wgan')
        fake_loss = gan_loss(discriminator_output, target_is_real=False, gan_mode='wgan')
        
        assert isinstance(real_loss, torch.Tensor)
        assert isinstance(fake_loss, torch.Tensor)
        # WGAN loss can be negative
        assert not torch.isnan(real_loss).any()
        assert not torch.isnan(fake_loss).any()
    
    def test_invalid_gan_mode(self):
        """Test invalid GAN mode raises error."""
        discriminator_output = torch.randn(4, 1)
        
        with pytest.raises(ValueError, match="Unsupported GAN mode"):
            gan_loss(discriminator_output, target_is_real=True, gan_mode='invalid')
    
    def test_loss_tensor_shapes(self):
        """Test loss function with different tensor shapes."""
        # Test with different output shapes
        for shape in [(4,), (4, 1), (4, 2, 1)]:
            discriminator_output = torch.randn(shape)
            
            loss = gan_loss(discriminator_output, target_is_real=True, gan_mode='vanilla')
            assert isinstance(loss, torch.Tensor)
            assert loss.shape == torch.Size([])


class TestGANTrainer:
    """Test GAN training utilities."""
    
    def test_gan_trainer_initialization(self):
        """Test GANTrainer initialization."""
        generator = Generator(50, 100)
        discriminator = Discriminator(100)
        
        g_optimizer = torch.optim.Adam(generator.parameters())
        d_optimizer = torch.optim.Adam(discriminator.parameters())
        
        trainer = GANTrainer(
            generator, discriminator, 
            g_optimizer, d_optimizer,
            torch.device('cpu'),
            gan_type='vanilla'
        )
        
        assert trainer.generator == generator
        assert trainer.discriminator == discriminator
        assert trainer.gan_type == 'vanilla'
    
    def test_vanilla_gan_train_step(self):
        """Test vanilla GAN training step."""
        generator = Generator(20, 50)
        discriminator = Discriminator(50)
        
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
        
        trainer = GANTrainer(
            generator, discriminator, 
            g_optimizer, d_optimizer,
            torch.device('cpu'),
            gan_type='vanilla'
        )
        
        batch_size = 8
        real_data = torch.randn(batch_size, 50)
        noise = torch.randn(batch_size, 20)
        
        d_loss, g_loss = trainer.train_step(real_data, noise)
        
        assert isinstance(d_loss, float)
        assert isinstance(g_loss, float)
        assert d_loss >= 0.0
        assert g_loss >= 0.0
    
    def test_wgan_train_step(self):
        """Test WGAN training step."""
        generator = Generator(20, 50)
        critic = WGANCritic(50)
        
        g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=0.0001)
        d_optimizer = torch.optim.RMSprop(critic.parameters(), lr=0.0001)
        
        trainer = GANTrainer(
            generator, critic, 
            g_optimizer, d_optimizer,
            torch.device('cpu'),
            gan_type='wgan'
        )
        
        batch_size = 8
        real_data = torch.randn(batch_size, 50)
        noise = torch.randn(batch_size, 20)
        
        d_loss, g_loss = trainer.train_step(real_data, noise)
        
        assert isinstance(d_loss, float)
        assert isinstance(g_loss, float)
    
    def test_wgan_gp_train_step(self):
        """Test WGAN-GP training step."""
        generator = Generator(20, 50)
        critic = WGANCritic(50)
        
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
        d_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0001)
        
        trainer = GANTrainer(
            generator, critic, 
            g_optimizer, d_optimizer,
            torch.device('cpu'),
            gan_type='wgan-gp'
        )
        
        batch_size = 8
        real_data = torch.randn(batch_size, 50)
        noise = torch.randn(batch_size, 20)
        
        d_loss, g_loss = trainer.train_step(real_data, noise)
        
        assert isinstance(d_loss, float)
        assert isinstance(g_loss, float)
    
    def test_trainer_with_different_optimizers(self):
        """Test trainer with different optimizer types."""
        generator = Generator(20, 50)
        discriminator = Discriminator(50)
        
        # Test with different optimizers
        optimizers = [
            (torch.optim.Adam, {'lr': 0.0002, 'betas': (0.5, 0.999)}),
            (torch.optim.RMSprop, {'lr': 0.0001}),
            (torch.optim.SGD, {'lr': 0.01, 'momentum': 0.9}),
        ]
        
        for optimizer_class, optimizer_kwargs in optimizers:
            g_optimizer = optimizer_class(generator.parameters(), **optimizer_kwargs)
            d_optimizer = optimizer_class(discriminator.parameters(), **optimizer_kwargs)
            
            trainer = GANTrainer(
                generator, discriminator,
                g_optimizer, d_optimizer,
                torch.device('cpu'),
                gan_type='vanilla'
            )
            
            real_data = torch.randn(4, 50)
            noise = torch.randn(4, 20)
            
            d_loss, g_loss = trainer.train_step(real_data, noise)
            
            assert isinstance(d_loss, float)
            assert isinstance(g_loss, float)


class TestGANIntegration:
    """Test GAN integration and end-to-end functionality."""
    
    def test_generator_discriminator_compatibility(self):
        """Test generator and discriminator work together."""
        noise_dim = 50
        data_dim = 100
        
        generator = Generator(noise_dim, data_dim)
        discriminator = Discriminator(data_dim)
        
        batch_size = 4
        noise = torch.randn(batch_size, noise_dim)
        
        # Generate fake data
        fake_data = generator(noise)
        
        # Discriminate fake data
        fake_output = discriminator(fake_data)
        
        assert fake_data.shape == (batch_size, data_dim)
        assert fake_output.shape == (batch_size,)
    
    def test_dcgan_end_to_end(self):
        """Test DCGAN end-to-end generation and discrimination."""
        dcgan = DCGAN(noise_dim=64, img_channels=1, feature_maps=32)
        
        batch_size = 2
        noise = torch.randn(batch_size, 64, 1, 1)
        
        # Generate images
        generated_images = dcgan.generator(noise)
        
        # Discriminate images
        discriminator_output = dcgan.discriminator(generated_images)
        
        assert generated_images.shape == (batch_size, 1, 64, 64)
        assert discriminator_output.shape == (batch_size,)
        
        # Check that generated images are in valid range
        assert generated_images.min() >= -1.0
        assert generated_images.max() <= 1.0
    
    def test_full_training_loop(self):
        """Test a few iterations of full GAN training loop."""
        generator = Generator(20, 50)
        discriminator = Discriminator(50)
        
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
        
        trainer = GANTrainer(
            generator, discriminator,
            g_optimizer, d_optimizer,
            torch.device('cpu'),
            gan_type='vanilla'
        )
        
        # Run a few training iterations
        for _ in range(5):
            real_data = torch.randn(8, 50)
            noise = torch.randn(8, 20)
            
            d_loss, g_loss = trainer.train_step(real_data, noise)
            
            assert isinstance(d_loss, float)
            assert isinstance(g_loss, float)
            assert not torch.isnan(torch.tensor(d_loss)).any()
            assert not torch.isnan(torch.tensor(g_loss)).any()


class TestGANUtilities:
    """Test GAN utility functions."""
    
    def test_model_weight_initialization(self):
        """Test that models have proper weight initialization."""
        generator = Generator(50, 100)
        discriminator = Discriminator(100)
        
        # Check that weights are not all zeros or ones
        for model in [generator, discriminator]:
            for param in model.parameters():
                if param.dim() > 1:  # Weight matrices
                    assert not torch.allclose(param, torch.zeros_like(param))
                    assert not torch.allclose(param, torch.ones_like(param))
    
    def test_gradient_flow_in_gan(self):
        """Test gradient flow in GAN training."""
        generator = Generator(20, 50)
        discriminator = Discriminator(50)
        
        # Create optimizers
        g_optimizer = torch.optim.Adam(generator.parameters())
        d_optimizer = torch.optim.Adam(discriminator.parameters())
        
        batch_size = 4
        real_data = torch.randn(batch_size, 50)
        noise = torch.randn(batch_size, 20)
        
        # Train discriminator
        d_optimizer.zero_grad()
        
        real_output = discriminator(real_data)
        fake_data = generator(noise).detach()
        fake_output = discriminator(fake_data)
        
        d_loss = -torch.mean(real_output) + torch.mean(fake_output)
        d_loss.backward()
        
        # Check that discriminator has gradients
        d_has_grad = any(p.grad is not None for p in discriminator.parameters())
        assert d_has_grad
        
        d_optimizer.step()
        
        # Train generator
        g_optimizer.zero_grad()
        
        fake_data = generator(noise)
        fake_output = discriminator(fake_data)
        g_loss = -torch.mean(fake_output)
        g_loss.backward()
        
        # Check that generator has gradients
        g_has_grad = any(p.grad is not None for p in generator.parameters())
        assert g_has_grad
        
        g_optimizer.step()
    
    def test_model_modes(self):
        """Test switching between train and eval modes."""
        generator = Generator(20, 50)
        discriminator = Discriminator(50)
        
        # Test train mode
        generator.train()
        discriminator.train()
        
        assert generator.training
        assert discriminator.training
        
        # Test eval mode
        generator.eval()
        discriminator.eval()
        
        assert not generator.training
        assert not discriminator.training
    
    def test_model_parameter_counts(self):
        """Test that models have reasonable parameter counts."""
        generator = Generator(100, 784, [256, 512])
        discriminator = Discriminator(784, [512, 256])
        
        gen_params = sum(p.numel() for p in generator.parameters())
        disc_params = sum(p.numel() for p in discriminator.parameters())
        
        # Should have reasonable number of parameters
        assert gen_params > 1000  # At least 1k parameters
        assert disc_params > 1000
        assert gen_params < 10**7  # Less than 10M parameters
        assert disc_params < 10**7


class TestErrorHandling:
    """Test error handling in GAN utilities."""
    
    def test_invalid_dimensions(self):
        """Test handling of invalid dimensions."""
        # Test generator with invalid dimensions
        with pytest.raises((ValueError, RuntimeError)):
            generator = Generator(0, 100)  # Invalid noise_dim
    
    def test_dimension_mismatch(self):
        """Test dimension mismatch between generator and discriminator."""
        generator = Generator(50, 100)
        discriminator = Discriminator(200)  # Mismatched input dimension
        
        noise = torch.randn(4, 50)
        fake_data = generator(noise)
        
        # This should raise an error due to dimension mismatch
        with pytest.raises(RuntimeError):
            _ = discriminator(fake_data)
    
    def test_empty_batch(self):
        """Test handling of empty batches."""
        generator = Generator(50, 100)
        
        # Empty batch
        empty_noise = torch.randn(0, 50)
        
        output = generator(empty_noise)
        assert output.shape == (0, 100)


if __name__ == "__main__":
    pytest.main([__file__])