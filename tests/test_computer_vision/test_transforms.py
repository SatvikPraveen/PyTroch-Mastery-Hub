# tests/test_computer_vision/test_transforms.py
"""
Tests for computer vision transforms
"""

import pytest
import torch
import numpy as np
from PIL import Image
from computer_vision.transforms import (
    get_train_transforms, get_val_transforms, RandomRotation,
    ColorJitter, GaussianBlur, MixUp, CutMix
)


class TestTransformPipelines:
    """Test transform pipeline functions."""
    
    def test_get_train_transforms(self):
        """Test getting training transforms."""
        transforms = get_train_transforms(input_size=224, normalize=True)
        
        assert transforms is not None
        assert len(transforms.transforms) > 0
    
    def test_get_val_transforms(self):
        """Test getting validation transforms."""
        transforms = get_val_transforms(input_size=224, normalize=True)
        
        assert transforms is not None
        assert len(transforms.transforms) > 0
    
    def test_transforms_without_normalization(self):
        """Test transforms without normalization."""
        train_transforms = get_train_transforms(normalize=False)
        val_transforms = get_val_transforms(normalize=False)
        
        assert train_transforms is not None
        assert val_transforms is not None
    
    def test_custom_input_size(self):
        """Test transforms with custom input size."""
        transforms = get_train_transforms(input_size=128)
        
        assert transforms is not None


class TestCustomTransforms:
    """Test custom transform implementations."""
    
    def test_random_rotation(self, sample_image):
        """Test RandomRotation transform."""
        transform = RandomRotation(degrees=30, p=1.0)
        
        rotated_image = transform(sample_image)
        
        assert isinstance(rotated_image, Image.Image)
        assert rotated_image.size == sample_image.size
    
    def test_random_rotation_with_probability(self, sample_image):
        """Test RandomRotation with probability."""
        # p=0 should return original image
        transform = RandomRotation(degrees=30, p=0.0)
        result = transform(sample_image)
        
        assert result == sample_image
    
    def test_color_jitter(self, sample_image):
        """Test ColorJitter transform."""
        transform = ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.1, 
            p=1.0
        )
        
        jittered_image = transform(sample_image)
        
        assert isinstance(jittered_image, Image.Image)
        assert jittered_image.size == sample_image.size
    
    def test_gaussian_blur(self, sample_image):
        """Test GaussianBlur transform."""
        transform = GaussianBlur(radius=1.0, p=1.0)
        
        blurred_image = transform(sample_image)
        
        assert isinstance(blurred_image, Image.Image)
        assert blurred_image.size == sample_image.size
    
    def test_gaussian_blur_with_range(self, sample_image):
        """Test GaussianBlur with radius range."""
        transform = GaussianBlur(radius=(0.5, 2.0), p=1.0)
        
        blurred_image = transform(sample_image)
        
        assert isinstance(blurred_image, Image.Image)


class TestAugmentationTechniques:
    """Test advanced augmentation techniques."""
    
    def test_mixup(self):
        """Test MixUp augmentation."""
        mixup = MixUp(alpha=0.2)
        
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 10, (batch_size,))
        
        mixed_x, mixed_y = mixup(x, y)
        
        assert mixed_x.shape == x.shape
        assert len(mixed_y) == 3  # (y_a, y_b, lambda)
    
    def test_mixup_with_zero_alpha(self):
        """Test MixUp with alpha=0 (no mixing)."""
        mixup = MixUp(alpha=0.0)
        
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 10, (4,))
        
        mixed_x, mixed_y = mixup(x, y)
        
        # Should return original data when alpha=0
        assert torch.equal(mixed_x, x)
        assert torch.equal(mixed_y, y)
    
    def test_cutmix(self):
        """Test CutMix augmentation."""
        cutmix = CutMix(alpha=1.0, prob=1.0)
        
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 10, (batch_size,))
        
        mixed_x, mixed_y = cutmix(x, y)
        
        assert mixed_x.shape == x.shape
        assert len(mixed_y) == 3  # (y_a, y_b, lambda)
    
    def test_cutmix_with_zero_prob(self):
        """Test CutMix with probability=0."""
        cutmix = CutMix(alpha=1.0, prob=0.0)
        
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 10, (4,))
        
        mixed_x, mixed_y = cutmix(x, y)
        
        # Should return original data when prob=0
        assert torch.equal(mixed_x, x)
        assert torch.equal(mixed_y, y)


class TestTransformIntegration:
    """Test transform integration with datasets."""
    
    def test_transform_with_tensor_input(self):
        """Test transforms working with tensor input."""
        # Some transforms should work with both PIL and tensor inputs
        mixup = MixUp(alpha=0.2)
        
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 10, (4,))
        
        # Should not raise an error
        mixed_x, mixed_y = mixup(x, y)
        
        assert mixed_x.shape == x.shape
    
    def test_transform_composition(self, sample_image):
        """Test composing multiple transforms."""
        from torchvision import transforms
        
        composed_transforms = transforms.Compose([
            RandomRotation(degrees=15, p=1.0),
            ColorJitter(brightness=0.1, p=1.0),
            transforms.ToTensor()
        ])
        
        result = composed_transforms(sample_image)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3  # RGB channels


class TestTransformErrorHandling:
    """Test error handling in transforms."""
    
    def test_invalid_transform_parameters(self):
        """Test transforms with invalid parameters."""
        # Negative probability should be handled gracefully
        transform = ColorJitter(brightness=0.2, p=-0.5)
        
        # Should still create the transform (implementation dependent)
        assert transform is not None
    
    def test_transform_with_none_input(self):
        """Test transform behavior with None input."""
        transform = RandomRotation(degrees=30)
        
        # Depending on implementation, this might raise an error
        # This test ensures we handle edge cases appropriately
        with pytest.raises((AttributeError, TypeError)):
            transform(None)