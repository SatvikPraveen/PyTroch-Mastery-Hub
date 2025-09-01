# src/computer_vision/transforms.py
"""
Custom image transforms for PyTorch Mastery Hub
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import random
import numpy as np
from typing import Tuple, Optional, Union, List


def get_train_transforms(input_size: int = 224, normalize: bool = True) -> transforms.Compose:
    """
    Get training transforms with augmentation.
    
    Args:
        input_size: Target image size
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Resize((input_size + 32, input_size + 32)),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    
    return transforms.Compose(transform_list)


def get_val_transforms(input_size: int = 224, normalize: bool = True) -> transforms.Compose:
    """
    Get validation transforms without augmentation.
    
    Args:
        input_size: Target image size
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    
    return transforms.Compose(transform_list)


class RandomRotation(torch.nn.Module):
    """
    Custom random rotation transform.
    """
    
    def __init__(self, degrees: Union[float, Tuple[float, float]], p: float = 1.0):
        super().__init__()
        self.degrees = degrees if isinstance(degrees, tuple) else (-degrees, degrees)
        self.p = p
    
    def forward(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            return TF.rotate(img, angle, interpolation=Image.BILINEAR)
        return img


class RandomCrop(torch.nn.Module):
    """
    Custom random crop transform with padding.
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]], padding: Optional[int] = None):
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else size
        self.padding = padding
    
    def forward(self, img: Image.Image) -> Image.Image:
        if self.padding:
            img = TF.pad(img, self.padding, fill=0, padding_mode='constant')
        
        i, j, h, w = transforms.RandomCrop.get_params(img, self.size)
        return TF.crop(img, i, j, h, w)


class ColorJitter(torch.nn.Module):
    """
    Enhanced color jitter transform.
    """
    
    def __init__(
        self, 
        brightness: float = 0, 
        contrast: float = 0, 
        saturation: float = 0, 
        hue: float = 0,
        p: float = 1.0
    ):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p
        self.color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)
    
    def forward(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return self.color_jitter(img)
        return img


class GaussianBlur(torch.nn.Module):
    """
    Gaussian blur transform.
    """
    
    def __init__(self, radius: Union[float, Tuple[float, float]] = 1.0, p: float = 0.5):
        super().__init__()
        self.radius = radius if isinstance(radius, tuple) else (0.1, radius)
        self.p = p
    
    def forward(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            radius = random.uniform(self.radius[0], self.radius[1])
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class RandomErasing(torch.nn.Module):
    """
    Random erasing augmentation.
    """
    
    def __init__(
        self, 
        p: float = 0.5, 
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: Union[int, str] = 0
    ):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return TF.erase(tensor, *self._get_params(tensor), self.value)
        return tensor
    
    def _get_params(self, tensor: torch.Tensor) -> Tuple[int, int, int, int]:
        img_c, img_h, img_w = tensor.shape
        area = img_h * img_w
        
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w < img_w and h < img_h:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                return i, j, h, w
        
        # Fallback
        return 0, 0, img_h, img_w


class GridMask(torch.nn.Module):
    """
    GridMask augmentation for object detection.
    """
    
    def __init__(
        self, 
        d1: int = 96, 
        d2: int = 224, 
        rotate: int = 45,
        ratio: float = 0.6,
        p: float = 0.5
    ):
        super().__init__()
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.p = p
    
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            h, w = tensor.shape[1:]
            d = random.randint(self.d1, self.d2)
            
            # Create grid mask
            mask = np.ones((h, w), dtype=np.float32)
            
            # Generate grid
            for i in range(0, h, d):
                for j in range(0, w, d):
                    y1, y2 = i, min(i + int(d * self.ratio), h)
                    x1, x2 = j, min(j + int(d * self.ratio), w)
                    mask[y1:y2, x1:x2] = 0
            
            mask = torch.from_numpy(mask).unsqueeze(0)
            return tensor * mask
        return tensor


class MixUp(torch.nn.Module):
    """
    MixUp augmentation.
    """
    
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.beta_dist = torch.distributions.Beta(alpha, alpha)
    
    def forward(self, batch: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch.size(0)
        
        if self.alpha > 0:
            lam = self.beta_dist.sample()
        else:
            lam = 1
        
        # Shuffle indices
        index = torch.randperm(batch_size).to(batch.device)
        
        # Mix images
        mixed_batch = lam * batch + (1 - lam) * batch[index, :]
        
        # Mix targets
        targets_a, targets_b = targets, targets[index]
        mixed_targets = (targets_a, targets_b, lam)
        
        return mixed_batch, mixed_targets


class CutMix(torch.nn.Module):
    """
    CutMix augmentation.
    """
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta_dist = torch.distributions.Beta(alpha, alpha)
    
    def forward(self, batch: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch.size(0)
        
        if self.alpha > 0:
            lam = self.beta_dist.sample()
        else:
            lam = 1
        
        # Get random patch
        W, H = batch.shape[2:]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Shuffle indices
        index = torch.randperm(batch_size).to(batch.device)
        
        # Apply cutmix
        mixed_batch = batch.clone()
        mixed_batch[:, :, bbx1:bbx2, bby1:bby2] = batch[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        targets_a, targets_b = targets, targets[index]
        mixed_targets = (targets_a, targets_b, lam)
        
        return mixed_batch, mixed_targets


class CustomTransform(torch.nn.Module):
    """
    Template for creating custom transforms.
    """
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, img: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        if random.random() < self.p:
            # Implement your custom transformation here
            pass
        return img


class Compose:
    """
    Custom compose class for chaining transforms.
    """
    
    def __init__(self, transforms: List[torch.nn.Module]):
        self.transforms = transforms
    
    def __call__(self, img):
        for transform in self.transforms:
            img = transform(img)
        return img
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n    {0}'.format(t)
        format_string += '\n)'
        return format_string


def get_advanced_transforms(input_size: int = 224, phase: str = 'train') -> transforms.Compose:
    """
    Get advanced transforms with modern augmentation techniques.
    
    Args:
        input_size: Target image size
        phase: 'train' or 'val'
        
    Returns:
        Composed transforms
    """
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(0.5),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
            GaussianBlur(radius=2.0, p=0.5),
            transforms.ToTensor(),
            RandomErasing(p=0.25),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return get_val_transforms(input_size, normalize=True)