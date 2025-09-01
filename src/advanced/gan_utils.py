# src/advanced/gan_utils.py
"""
GAN utilities for PyTorch Mastery Hub
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np


class Generator(nn.Module):
    """Basic Generator for GAN."""
    
    def __init__(self, noise_dim: int, output_dim: int, hidden_dims: list = [256, 512, 1024]):
        super(Generator, self).__init__()
        
        layers = []
        input_dim = noise_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.extend([
            nn.Linear(input_dim, output_dim),
            nn.Tanh()
        ])
        
        self.model = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.model(noise)


class Discriminator(nn.Module):
    """Basic Discriminator for GAN."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [1024, 512, 256]):
        super(Discriminator, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ])
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, 1))
        
        self.model = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DCGAN(nn.Module):
    """Deep Convolutional GAN."""
    
    def __init__(self, noise_dim: int = 100, img_channels: int = 3, feature_maps: int = 64):
        super(DCGAN, self).__init__()
        self.generator = DCGANGenerator(noise_dim, img_channels, feature_maps)
        self.discriminator = DCGANDiscriminator(img_channels, feature_maps)
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.generator(noise)


class DCGANGenerator(nn.Module):
    """DCGAN Generator."""
    
    def __init__(self, noise_dim: int, img_channels: int, feature_maps: int):
        super(DCGANGenerator, self).__init__()
        
        self.model = nn.Sequential(
            # Input: noise_dim
            nn.ConvTranspose2d(noise_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(inplace=True),
            
            # State: (feature_maps*8) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(inplace=True),
            
            # State: (feature_maps*4) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(inplace=True),
            
            # State: (feature_maps*2) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(inplace=True),
            
            # State: feature_maps x 32 x 32
            nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: img_channels x 64 x 64
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        # Reshape noise for convolution
        if noise.dim() == 2:
            noise = noise.view(noise.size(0), noise.size(1), 1, 1)
        return self.model(noise)


class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator."""
    
    def __init__(self, img_channels: int, feature_maps: int):
        super(DCGANDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Input: img_channels x 64 x 64
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: feature_maps x 32 x 32
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (feature_maps*2) x 16 x 16
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (feature_maps*4) x 8 x 8
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (feature_maps*8) x 4 x 4
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).view(-1, 1).squeeze(1)


class WGAN(nn.Module):
    """Wasserstein GAN implementation."""
    
    def __init__(self, noise_dim: int, output_dim: int):
        super(WGAN, self).__init__()
        self.generator = Generator(noise_dim, output_dim)
        self.critic = WGANCritic(output_dim)
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.generator(noise)


class WGANCritic(nn.Module):
    """WGAN Critic (replaces discriminator)."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [1024, 512, 256]):
        super(WGANCritic, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, 1))
        
        self.model = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def compute_gradient_penalty(
    critic: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.
    
    Args:
        critic: Critic network
        real_samples: Real data samples
        fake_samples: Generated samples
        device: Device to compute on
        lambda_gp: Gradient penalty coefficient
        
    Returns:
        Gradient penalty loss
    """
    batch_size = real_samples.size(0)
    
    # Random interpolation between real and fake samples
    alpha = torch.rand(batch_size, 1, device=device)
    
    # Expand alpha to match sample dimensions
    for _ in range(len(real_samples.shape) - 2):
        alpha = alpha.unsqueeze(-1)
    
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates = interpolates.requires_grad_(True)
    
    # Compute critic output for interpolated samples
    critic_interpolates = critic(interpolates)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return lambda_gp * gradient_penalty


class GANTrainer:
    """
    Trainer for GANs with various loss functions.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        g_optimizer: torch.optim.Optimizer,
        d_optimizer: torch.optim.Optimizer,
        device: torch.device,
        gan_type: str = 'vanilla'
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.device = device
        self.gan_type = gan_type
        
        if gan_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif gan_type == 'lsgan':
            self.criterion = nn.MSELoss()
    
    def train_step(self, real_data: torch.Tensor, noise: torch.Tensor) -> Tuple[float, float]:
        """
        Perform one training step.
        
        Args:
            real_data: Real data batch
            noise: Noise for generator
            
        Returns:
            Tuple of (discriminator_loss, generator_loss)
        """
        batch_size = real_data.size(0)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Real samples
        real_output = self.discriminator(real_data)
        
        # Fake samples
        fake_data = self.generator(noise).detach()
        fake_output = self.discriminator(fake_data)
        
        # Discriminator loss
        if self.gan_type == 'vanilla':
            real_labels = torch.ones_like(real_output)
            fake_labels = torch.zeros_like(fake_output)
            
            d_loss_real = self.criterion(real_output, real_labels)
            d_loss_fake = self.criterion(fake_output, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2
            
        elif self.gan_type == 'lsgan':
            d_loss_real = self.criterion(real_output, torch.ones_like(real_output))
            d_loss_fake = self.criterion(fake_output, torch.zeros_like(fake_output))
            d_loss = (d_loss_real + d_loss_fake) / 2
            
        elif self.gan_type == 'wgan':
            d_loss = -torch.mean(real_output) + torch.mean(fake_output)
            
        d_loss.backward()
        self.d_optimizer.step()
        
        # Clip weights for WGAN
        if self.gan_type == 'wgan':
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        fake_data = self.generator(noise)
        fake_output = self.discriminator(fake_data)
        
        # Generator loss
        if self.gan_type == 'vanilla':
            g_loss = self.criterion(fake_output, torch.ones_like(fake_output))
        elif self.gan_type == 'lsgan':
            g_loss = self.criterion(fake_output, torch.ones_like(fake_output))
        elif self.gan_type == 'wgan':
            g_loss = -torch.mean(fake_output)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return d_loss.item(), g_loss.item()


def gan_loss(
    discriminator_output: torch.Tensor,
    target_is_real: bool,
    gan_mode: str = 'vanilla'
) -> torch.Tensor:
    """
    Calculate GAN loss for generator and discriminator.
    
    Args:
        discriminator_output: Output from discriminator
        target_is_real: Whether target should be real (True) or fake (False)
        gan_mode: Type of GAN loss ('vanilla', 'lsgan', 'wgan')
        
    Returns:
        GAN loss
    """
    if gan_mode == 'vanilla':
        if target_is_real:
            loss = F.binary_cross_entropy_with_logits(
                discriminator_output, torch.ones_like(discriminator_output)
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                discriminator_output, torch.zeros_like(discriminator_output)
            )
            
    elif gan_mode == 'lsgan':
        if target_is_real:
            loss = F.mse_loss(discriminator_output, torch.ones_like(discriminator_output))
        else:
            loss = F.mse_loss(discriminator_output, torch.zeros_like(discriminator_output))
            
    elif gan_mode == 'wgan':
        if target_is_real:
            loss = -discriminator_output.mean()
        else:
            loss = discriminator_output.mean()
            
    else:
        raise ValueError(f"Unsupported GAN mode: {gan_mode}")
    
    return loss


def calculate_inception_score(
    images: torch.Tensor,
    inception_model: nn.Module,
    splits: int = 10
) -> Tuple[float, float]:
    """
    Calculate Inception Score for generated images.
    
    Args:
        images: Generated images
        inception_model: Pre-trained Inception model
        splits: Number of splits for calculation
        
    Returns:
        Tuple of (mean_score, std_score)
    """
    inception_model.eval()
    
    # Get predictions
    with torch.no_grad():
        preds = inception_model(images)
        preds = F.softmax(preds, dim=1)
    
    # Calculate scores
    scores = []
    N = len(preds)
    
    for i in range(splits):
        start_idx = i * N // splits
        end_idx = (i + 1) * N // splits
        
        part = preds[start_idx:end_idx]
        
        # Calculate KL divergence
        py = part.mean(dim=0)
        kl_div = part * (torch.log(part) - torch.log(py))
        kl_div = kl_div.sum(dim=1)
        
        scores.append(torch.exp(kl_div.mean()).item())
    
    return np.mean(scores), np.std(scores)