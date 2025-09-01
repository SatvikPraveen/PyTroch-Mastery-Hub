# src/computer_vision/datasets.py
"""
Custom dataset classes for PyTorch Mastery Hub
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import pandas as pd


class ImageDataset(Dataset):
    """
    Generic image dataset class.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = extensions
        
        self.samples = self._load_samples()
        self.classes = sorted(set(sample['class'] for sample in self.samples))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load image samples from directory structure."""
        samples = []
        
        for class_dir in self.root_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.extensions:
                    samples.append({
                        'path': img_path,
                        'class': class_name
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.class_to_idx[sample['class']]
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets."""
        class_counts = [0] * len(self.classes)
        for sample in self.samples:
            class_idx = self.class_to_idx[sample['class']]
            class_counts[class_idx] += 1
        
        total_samples = len(self.samples)
        weights = [total_samples / (len(self.classes) * count) for count in class_counts]
        return torch.FloatTensor(weights)


class SegmentationDataset(Dataset):
    """
    Dataset for semantic segmentation tasks.
    """
    
    def __init__(
        self,
        images_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        self.extensions = extensions
        
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict[str, Path]]:
        """Load image and mask pairs."""
        samples = []
        
        for img_path in self.images_dir.iterdir():
            if img_path.suffix.lower() in self.extensions:
                mask_path = self.masks_dir / f"{img_path.stem}.png"
                if mask_path.exists():
                    samples.append({
                        'image': img_path,
                        'mask': mask_path
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image and mask
        image = Image.open(sample['image']).convert('RGB')
        mask = Image.open(sample['mask']).convert('L')  # Grayscale
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask


class ObjectDetectionDataset(Dataset):
    """
    Dataset for object detection tasks (COCO-style annotations).
    """
    
    def __init__(
        self,
        images_dir: Union[str, Path],
        annotations_file: Union[str, Path],
        transform: Optional[Callable] = None
    ):
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.coco = json.load(f)
        
        self.image_ids = list(sorted(self.coco['images'], key=lambda x: x['id']))
        self.categories = {cat['id']: cat['name'] for cat in self.coco['categories']}
        
        # Create image_id to annotations mapping
        self.annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_info = self.image_ids[idx]
        img_id = img_info['id']
        
        # Load image
        img_path = self.images_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations
        anns = self.annotations.get(img_id, [])
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))
        
        # Convert to tensors
        boxes = torch.FloatTensor(boxes) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.LongTensor(labels) if labels else torch.zeros((0,), dtype=torch.int64)
        areas = torch.FloatTensor(areas) if areas else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.BoolTensor(iscrowd) if iscrowd else torch.zeros((0,), dtype=torch.bool)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'areas': areas,
            'iscrowd': iscrowd,
            'image_id': torch.tensor([img_id])
        }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


class CSVDataset(Dataset):
    """
    Dataset for loading images from CSV file.
    """
    
    def __init__(
        self,
        csv_file: Union[str, Path],
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        img_col: str = 'filename',
        label_col: str = 'label'
    ):
        self.csv_file = Path(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.img_col = img_col
        self.label_col = label_col
        
        self.data_frame = pd.read_csv(csv_file)
        self.classes = sorted(self.data_frame[label_col].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self) -> int:
        return len(self.data_frame)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.data_frame.iloc[idx]
        img_name = self.root_dir / row[self.img_col]
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.class_to_idx[row[self.label_col]]
        
        return image, label


class CustomImageDataset(Dataset):
    """
    Flexible custom image dataset with multiple annotation formats.
    """
    
    def __init__(
        self,
        data_file: Union[str, Path],
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        mode: str = 'classification'  # 'classification', 'regression', 'multilabel'
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        
        # Load data
        if str(data_file).endswith('.csv'):
            self.data = pd.read_csv(data_file)
        elif str(data_file).endswith('.json'):
            with open(data_file, 'r') as f:
                self.data = json.load(f)
        else:
            raise ValueError("Unsupported data file format")
        
        if self.mode == 'classification' and isinstance(self.data, pd.DataFrame):
            self.classes = sorted(self.data['label'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[int, float, torch.Tensor]]:
        if isinstance(self.data, pd.DataFrame):
            row = self.data.iloc[idx]
            img_path = self.root_dir / row['filename']
            
            if self.mode == 'classification':
                target = self.class_to_idx[row['label']]
            elif self.mode == 'regression':
                target = float(row['target'])
            elif self.mode == 'multilabel':
                target = torch.FloatTensor([float(x) for x in row['labels'].split(',')])
        else:
            # JSON format
            item = self.data[idx]
            img_path = self.root_dir / item['filename']
            target = item['target']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, target


def get_dataset_splits(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: PyTorch dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    torch.manual_seed(random_seed)
    
    return random_split(dataset, [train_size, val_size, test_size])


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Create a DataLoader with common settings.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=None
    )


def collate_fn_detection(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Custom collate function for object detection.
    
    Args:
        batch: List of (image, target) pairs
        
    Returns:
        Tuple of (images, targets)
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    return images, targets


class InMemoryDataset(Dataset):
    """
    Dataset that loads all data into memory for faster access.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        transform: Optional[Callable] = None
    ):
        self.base_dataset = base_dataset
        self.transform = transform
        
        # Load all data into memory
        print("Loading dataset into memory...")
        self.data = []
        for i in range(len(base_dataset)):
            self.data.append(base_dataset[i])
        print(f"Loaded {len(self.data)} samples into memory")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        image, target = self.data[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


class WeightedSampler:
    """
    Create weighted sampler for imbalanced datasets.
    """
    
    def __init__(self, dataset: Dataset, target_attr: str = 'targets'):
        if hasattr(dataset, target_attr):
            targets = getattr(dataset, target_attr)
        else:
            # Extract targets manually
            targets = []
            for _, target in dataset:
                targets.append(target)
        
        class_counts = torch.bincount(torch.LongTensor(targets))
        class_weights = 1. / class_counts.float()
        
        self.sample_weights = torch.zeros(len(targets))
        for i, target in enumerate(targets):
            self.sample_weights[i] = class_weights[target]
    
    def get_sampler(self):
        """Get the weighted random sampler."""
        from torch.utils.data import WeightedRandomSampler
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True
        )