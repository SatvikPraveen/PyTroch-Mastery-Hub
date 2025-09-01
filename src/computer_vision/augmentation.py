# src/computer_vision/augmentation.py
"""
Data augmentation utilities for PyTorch Mastery Hub
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import Tuple, List, Optional, Union
from PIL import Image, ImageEnhance, ImageOps
import torchvision.transforms.functional as TF


class MixUp(nn.Module):
    """
    MixUp data augmentation.
    """
    
    def __init__(self, alpha: float = 0.2):
        super(MixUp, self).__init__()
        self.alpha = alpha
        self.beta_dist = torch.distributions.Beta(alpha, alpha) if alpha > 0 else None
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        if self.beta_dist is None:
            return x, y
        
        batch_size = x.size(0)
        lam = self.beta_dist.sample()
        
        # Shuffle indices
        indices = torch.randperm(batch_size).to(x.device)
        
        # Mix inputs and targets
        mixed_x = lam * x + (1 - lam) * x[indices]
        y_a, y_b = y, y[indices]
        
        return mixed_x, (y_a, y_b, lam)
    
    def mixup_criterion(self, criterion, pred, y_mixed):
        y_a, y_b, lam = y_mixed
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class CutMix(nn.Module):
    """
    CutMix data augmentation.
    """
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        super(CutMix, self).__init__()
        self.alpha = alpha
        self.prob = prob
        self.beta_dist = torch.distributions.Beta(alpha, alpha) if alpha > 0 else None
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        if self.beta_dist is None or random.random() > self.prob:
            return x, y
        
        batch_size = x.size(0)
        lam = self.beta_dist.sample()
        
        # Get cut coordinates
        _, _, H, W = x.shape
        cut_rat = torch.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = torch.randint(W, (1,)).item()
        cy = torch.randint(H, (1,)).item()
        
        bbx1 = torch.clamp(cx - cut_w // 2, 0, W).item()
        bby1 = torch.clamp(cy - cut_h // 2, 0, H).item()
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W).item()
        bby2 = torch.clamp(cy + cut_h // 2, 0, H).item()
        
        # Shuffle and mix
        indices = torch.randperm(batch_size).to(x.device)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[indices, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        y_a, y_b = y, y[indices]
        
        return x, (y_a, y_b, lam)


class Mosaic(nn.Module):
    """
    Mosaic data augmentation (4 images combined).
    """
    
    def __init__(self, prob: float = 0.5):
        super(Mosaic, self).__init__()
        self.prob = prob
    
    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        if random.random() > self.prob or len(batch) < 4:
            return torch.stack(batch)
        
        # Take first 4 images
        imgs = batch[:4]
        _, H, W = imgs[0].shape
        
        # Create mosaic
        mosaic = torch.zeros_like(imgs[0])
        
        # Random center point
        cx = random.randint(W // 4, 3 * W // 4)
        cy = random.randint(H // 4, 3 * H // 4)
        
        # Place images in quadrants
        # Top-left
        mosaic[:, :cy, :cx] = TF.resize(imgs[0], (cy, cx))
        
        # Top-right
        mosaic[:, :cy, cx:] = TF.resize(imgs[1], (cy, W - cx))
        
        # Bottom-left
        mosaic[:, cy:, :cx] = TF.resize(imgs[2], (H - cy, cx))
        
        # Bottom-right
        mosaic[:, cy:, cx:] = TF.resize(imgs[3], (H - cy, W - cx))
        
        return mosaic


class RandomAugment(nn.Module):
    """
    Random augmentation with configurable operations.
    """
    
    def __init__(self, n: int = 2, m: int = 9):
        super(RandomAugment, self).__init__()
        self.n = n  # Number of operations
        self.m = m  # Magnitude
        
        self.augment_list = [
            self.auto_contrast,
            self.equalize,
            self.invert,
            self.rotate,
            self.posterize,
            self.solarize,
            self.color,
            self.contrast,
            self.brightness,
            self.sharpness,
            self.shear_x,
            self.shear_y,
            self.translate_x,
            self.translate_y,
        ]
    
    def forward(self, img: Image.Image) -> Image.Image:
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            img = op(img, self.m)
        return img
    
    def auto_contrast(self, img: Image.Image, magnitude: int) -> Image.Image:
        return ImageOps.autocontrast(img)
    
    def equalize(self, img: Image.Image, magnitude: int) -> Image.Image:
        return ImageOps.equalize(img)
    
    def invert(self, img: Image.Image, magnitude: int) -> Image.Image:
        return ImageOps.invert(img)
    
    def rotate(self, img: Image.Image, magnitude: int) -> Image.Image:
        degree = magnitude / 10 * 30
        if random.random() > 0.5:
            degree = -degree
        return img.rotate(degree, Image.BILINEAR)
    
    def posterize(self, img: Image.Image, magnitude: int) -> Image.Image:
        level = int(magnitude / 10 * 4)
        return ImageOps.posterize(img, 4 - level)
    
    def solarize(self, img: Image.Image, magnitude: int) -> Image.Image:
        level = int(magnitude / 10 * 256)
        return ImageOps.solarize(img, 256 - level)
    
    def color(self, img: Image.Image, magnitude: int) -> Image.Image:
        level = magnitude / 10 * 1.8 + 0.1
        return ImageEnhance.Color(img).enhance(level)
    
    def contrast(self, img: Image.Image, magnitude: int) -> Image.Image:
        level = magnitude / 10 * 1.8 + 0.1
        return ImageEnhance.Contrast(img).enhance(level)
    
    def brightness(self, img: Image.Image, magnitude: int) -> Image.Image:
        level = magnitude / 10 * 1.8 + 0.1
        return ImageEnhance.Brightness(img).enhance(level)
    
    def sharpness(self, img: Image.Image, magnitude: int) -> Image.Image:
        level = magnitude / 10 * 1.8 + 0.1
        return ImageEnhance.Sharpness(img).enhance(level)
    
    def shear_x(self, img: Image.Image, magnitude: int) -> Image.Image:
        level = magnitude / 10 * 0.3
        if random.random() > 0.5:
            level = -level
        return img.transform(img.size, Image.AFFINE, (1, level, 0, 0, 1, 0), Image.BILINEAR)
    
    def shear_y(self, img: Image.Image, magnitude: int) -> Image.Image:
        level = magnitude / 10 * 0.3
        if random.random() > 0.5:
            level = -level
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, level, 1, 0), Image.BILINEAR)
    
    def translate_x(self, img: Image.Image, magnitude: int) -> Image.Image:
        level = magnitude / 10 * img.size[0] / 3
        if random.random() > 0.5:
            level = -level
        return img.transform(img.size, Image.AFFINE, (1, 0, level, 0, 1, 0), Image.BILINEAR)
    
    def translate_y(self, img: Image.Image, magnitude: int) -> Image.Image:
        level = magnitude / 10 * img.size[1] / 3
        if random.random() > 0.5:
            level = -level
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, level), Image.BILINEAR)


class TrivialAugment(nn.Module):
    """
    TrivialAugment - simpler version of AutoAugment.
    """
    
    def __init__(self):
        super(TrivialAugment, self).__init__()
        self.augment_list = [
            'identity', 'auto_contrast', 'equalize', 'rotate', 'solarize',
            'color', 'posterize', 'contrast', 'brightness', 'sharpness',
            'shear_x', 'shear_y', 'translate_x', 'translate_y'
        ]
    
    def forward(self, img: Image.Image) -> Image.Image:
        op = random.choice(self.augment_list)
        magnitude = random.randint(1, 31)
        
        if op == 'identity':
            return img
        elif op == 'auto_contrast':
            return ImageOps.autocontrast(img)
        elif op == 'equalize':
            return ImageOps.equalize(img)
        elif op == 'rotate':
            degree = magnitude * 30 / 31
            return img.rotate(degree, Image.BILINEAR)
        elif op == 'solarize':
            threshold = magnitude * 256 // 31
            return ImageOps.solarize(img, threshold)
        elif op == 'color':
            factor = 1 + magnitude * 0.9 / 31
            return ImageEnhance.Color(img).enhance(factor)
        elif op == 'posterize':
            bits = 8 - magnitude * 4 // 31
            return ImageOps.posterize(img, max(1, bits))
        elif op == 'contrast':
            factor = 1 + magnitude * 0.9 / 31
            return ImageEnhance.Contrast(img).enhance(factor)
        elif op == 'brightness':
            factor = 1 + magnitude * 0.9 / 31
            return ImageEnhance.Brightness(img).enhance(factor)
        elif op == 'sharpness':
            factor = 1 + magnitude * 0.9 / 31
            return ImageEnhance.Sharpness(img).enhance(factor)
        else:
            return img


class CopyPaste(nn.Module):
    """
    Copy-paste augmentation for object detection.
    """
    
    def __init__(self, prob: float = 0.5):
        super(CopyPaste, self).__init__()
        self.prob = prob
    
    def forward(self, img1: torch.Tensor, boxes1: torch.Tensor, 
                img2: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.prob:
            return img1, boxes1
        
        # Simple implementation: paste objects from img2 to img1
        # This is a simplified version - real implementation would handle masks
        
        # Random selection of boxes to paste
        if len(boxes2) > 0:
            num_paste = random.randint(1, min(3, len(boxes2)))
            paste_indices = random.sample(range(len(boxes2)), num_paste)
            
            for idx in paste_indices:
                box = boxes2[idx]
                x1, y1, x2, y2 = box.int()
                
                # Extract object region
                obj_region = img2[:, y1:y2, x1:x2]
                
                # Random position in target image
                h, w = img1.shape[1], img1.shape[2]
                new_h, new_w = obj_region.shape[1], obj_region.shape[2]
                
                if new_h < h and new_w < w:
                    new_y = random.randint(0, h - new_h)
                    new_x = random.randint(0, w - new_w)
                    
                    # Paste object
                    img1[:, new_y:new_y+new_h, new_x:new_x+new_w] = obj_region
                    
                    # Add new box
                    new_box = torch.tensor([new_x, new_y, new_x + new_w, new_y + new_h])
                    boxes1 = torch.cat([boxes1, new_box.unsqueeze(0)], dim=0)
        
        return img1, boxes1


class RandAugment(nn.Module):
    """
    RandAugment implementation.
    """
    
    def __init__(self, n: int = 2, m: int = 9):
        super(RandAugment, self).__init__()
        self.n = n
        self.m = m
        self.augment_ops = RandomAugment(n, m)
    
    def forward(self, img: Image.Image) -> Image.Image:
        return self.augment_ops(img)


class AugMix(nn.Module):
    """
    AugMix augmentation for improved robustness.
    """
    
    def __init__(self, width: int = 3, depth: int = -1, alpha: float = 1.0, prob: float = 0.5):
        super(AugMix, self).__init__()
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.prob = prob
        
        self.augment_list = [
            'auto_contrast', 'equalize', 'rotate', 'solarize', 'color',
            'posterize', 'contrast', 'brightness', 'sharpness'
        ]
    
    def forward(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        
        ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))
        
        mix = torch.zeros_like(TF.to_tensor(img))
        
        for i in range(self.width):
            img_aug = img.copy()
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            
            for _ in range(depth):
                op = np.random.choice(self.augment_list)
                img_aug = self._apply_op(img_aug, op)
            
            mix += ws[i] * TF.to_tensor(img_aug)
        
        mixed = (1 - m) * TF.to_tensor(img) + m * mix
        return TF.to_pil_image(mixed)
    
    def _apply_op(self, img: Image.Image, op: str) -> Image.Image:
        magnitude = np.random.randint(1, 11)
        
        if op == 'auto_contrast':
            return ImageOps.autocontrast(img)
        elif op == 'equalize':
            return ImageOps.equalize(img)
        elif op == 'rotate':
            degree = magnitude * 3
            return img.rotate(degree, Image.BILINEAR)
        elif op == 'solarize':
            threshold = 256 - magnitude * 25
            return ImageOps.solarize(img, threshold)
        elif op == 'color':
            factor = 0.1 + magnitude * 0.18
            return ImageEnhance.Color(img).enhance(factor)
        elif op == 'posterize':
            level = 8 - magnitude
            return ImageOps.posterize(img, max(1, level))
        elif op == 'contrast':
            factor = 0.1 + magnitude * 0.18
            return ImageEnhance.Contrast(img).enhance(factor)
        elif op == 'brightness':
            factor = 0.1 + magnitude * 0.18
            return ImageEnhance.Brightness(img).enhance(factor)
        elif op == 'sharpness':
            factor = 0.1 + magnitude * 0.18
            return ImageEnhance.Sharpness(img).enhance(factor)
        
        return img


class AdvancedAugmentationPipeline(nn.Module):
    """
    Complete augmentation pipeline combining multiple techniques.
    """
    
    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        cutmix_prob: float = 0.5,
        randaugment_n: int = 2,
        randaugment_m: int = 9,
        use_mixup: bool = True,
        use_cutmix: bool = True,
        use_randaugment: bool = True
    ):
        super(AdvancedAugmentationPipeline, self).__init__()
        
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.use_randaugment = use_randaugment
        
        if use_mixup:
            self.mixup = MixUp(mixup_alpha)
        
        if use_cutmix:
            self.cutmix = CutMix(cutmix_alpha, cutmix_prob)
        
        if use_randaugment:
            self.randaugment = RandomAugment(randaugment_n, randaugment_m)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, phase: str = 'train') -> Tuple[torch.Tensor, torch.Tensor]:
        if phase != 'train':
            return x, y
        
        # Apply batch-level augmentations
        if self.use_mixup and random.random() < 0.5:
            return self.mixup(x, y)
        elif self.use_cutmix and random.random() < 0.5:
            return self.cutmix(x, y)
        
        return x, y