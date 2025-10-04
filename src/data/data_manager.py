"""
CIFAR-10 Data Loading and Augmentation Module

This module handles data loading, preprocessing, and augmentation for the CIFAR-10 dataset.
It uses the Albumentation library for advanced data augmentation techniques.

Features:
- Automatic dataset download and caching
- True mean and std calculation from the entire dataset
- Albumentation-based data augmentation
- Proper data normalization
- Configurable data loaders

Author: EVA5 Student
Date: 2024
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional
from config import DataConfig


class AlbumentationTransform:
    """
    Custom transform wrapper for Albumentation library.
    
    This class bridges the gap between PyTorch's transform interface
    and Albumentation's transform interface.
    
    Args:
        transforms: Albumentation transform composition
    """
    
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms
        
    def __call__(self, image) -> torch.Tensor:
        """
        Apply transforms to the input image.
        
        Args:
            image: Input image as PIL Image or numpy array
            
        Returns:
            Transformed image as PyTorch tensor
        """
        # Convert PIL Image to numpy array if needed
        if hasattr(image, 'mode'):  # PIL Image
            image = np.array(image)
        
        return self.transforms(image=image)['image']


class CIFAR10DataManager:
    """
    CIFAR-10 Data Manager for handling dataset operations.
    
    This class provides methods for calculating dataset statistics,
    creating transforms, loading datasets, and creating data loaders.
    
    Args:
        config: Data configuration parameters
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._mean = None
        self._std = None
        
    def calculate_dataset_statistics(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Calculate true mean and standard deviation from the entire CIFAR-10 dataset.
        
        This method loads the entire dataset (train + test) and calculates
        the true mean and standard deviation for proper normalization.
        
        Returns:
            Tuple of (mean, std) where each is a tuple of (R, G, B) values
        """
        if self._mean is not None and self._std is not None:
            return self._mean, self._std
            
        print("Calculating CIFAR-10 dataset statistics...")
        
        # Simple transform for calculation
        simple_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.config.data_root,
            train=True,
            download=True,
            transform=simple_transform
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.config.data_root,
            train=False,
            download=True,
            transform=simple_transform
        )
        
        # Combine datasets
        train_data = train_dataset.data
        test_data = test_dataset.data
        combined_data = np.concatenate((train_data, test_data), axis=0)
        
        # Reshape to (channels, height, width, samples)
        combined_data = np.transpose(combined_data, (3, 1, 2, 0))
        
        # Calculate mean and std
        mean = (
            np.mean(combined_data[0]) / 255.0,  # Red channel
            np.mean(combined_data[1]) / 255.0,  # Green channel
            np.mean(combined_data[2]) / 255.0   # Blue channel
        )
        
        std = (
            np.std(combined_data[0]) / 255.0,   # Red channel
            np.std(combined_data[1]) / 255.0,   # Green channel
            np.std(combined_data[2]) / 255.0    # Blue channel
        )
        
        # Round to 2 decimal places
        self._mean = tuple(map(lambda x: round(x, 2), mean))
        self._std = tuple(map(lambda x: round(x, 2), std))
        
        print(f"Dataset statistics calculated:")
        print(f"  Mean: {self._mean}")
        print(f"  Std: {self._std}")
        
        return self._mean, self._std
    
    def create_transforms(self) -> Tuple[AlbumentationTransform, AlbumentationTransform]:
        """
        Create training and testing transforms using Albumentation.
        
        Training transforms include:
        - Horizontal flip
        - Shift, scale, and rotate
        - Coarse dropout
        - Normalization
        
        Testing transforms include:
        - Normalization only
        
        Returns:
            Tuple of (train_transform, test_transform)
        """
        # Use calculated statistics or config defaults
        if self._mean is None or self._std is None:
            mean = self.config.mean
            std = self.config.std
        else:
            mean = self._mean
            std = self._std
        
        print(f"Creating transforms with normalization: mean={mean}, std={std}")
        
        # Convert to numpy arrays for Albumentation
        mean_array = np.array(mean)
        std_array = np.array(std)
        
        # Training transforms with augmentation
        train_transform = AlbumentationTransform(A.Compose([
            A.HorizontalFlip(p=self.config.horizontal_flip_prob),
            A.ShiftScaleRotate(
                shift_limit=self.config.shift_limit,
                scale_limit=self.config.scale_limit,
                rotate_limit=self.config.rotate_limit,
                p=self.config.shift_scale_rotate_prob
            ),
            A.CoarseDropout(
                max_holes=self.config.max_holes,
                max_height=self.config.max_height,
                max_width=self.config.max_width,
                min_holes=self.config.min_holes,
                min_height=self.config.min_height,
                min_width=self.config.min_width,
                fill_value=mean_array,
                mask_fill_value=None,
                p=self.config.coarse_dropout_prob
            ),
            A.Normalize(mean=mean_array, std=std_array),
            ToTensorV2()
        ]))
        
        # Testing transforms (no augmentation)
        test_transform = AlbumentationTransform(A.Compose([
            A.Normalize(mean=mean_array, std=std_array),
            ToTensorV2()
        ]))
        
        return train_transform, test_transform
    
    def load_datasets(
        self,
        train_transform: Optional[AlbumentationTransform] = None,
        test_transform: Optional[AlbumentationTransform] = None
    ) -> Tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
        """
        Load CIFAR-10 train and test datasets.
        
        Args:
            train_transform: Transform to apply to training data
            test_transform: Transform to apply to test data
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if train_transform is None or test_transform is None:
            train_transform, test_transform = self.create_transforms()
        
        print("Loading CIFAR-10 datasets...")
        
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.config.data_root,
            train=True,
            download=True,
            transform=train_transform
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.config.data_root,
            train=False,
            download=True,
            transform=test_transform
        )
        
        print(f"Datasets loaded successfully:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Classes: {train_dataset.classes}")
        
        return train_dataset, test_dataset
    
    def create_data_loaders(
        self,
        train_dataset: torchvision.datasets.CIFAR10,
        test_dataset: torchvision.datasets.CIFAR10
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Create data loaders for training and testing.
        
        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        
        # Set reproducibility
        torch.manual_seed(self.config.seed)
        if cuda_available:
            torch.cuda.manual_seed(self.config.seed)
        
        # DataLoader arguments
        dataloader_args = {
            'shuffle': True,
            'batch_size': self.config.batch_size,
            'num_workers': self.config.num_workers if cuda_available else 1,
            'pin_memory': self.config.pin_memory and cuda_available
        }
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
        test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)
        
        print(f"Data loaders created:")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print(f"  Batch size: {self.config.batch_size}")
        
        return train_loader, test_loader
    
    def get_data_info(self) -> dict:
        """
        Get information about the dataset configuration.
        
        Returns:
            Dictionary containing dataset information
        """
        return {
            "dataset_name": self.config.dataset_name,
            "data_root": self.config.data_root,
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
            "normalization_mean": self._mean or self.config.mean,
            "normalization_std": self._std or self.config.std,
            "augmentation_probabilities": {
                "horizontal_flip": self.config.horizontal_flip_prob,
                "shift_scale_rotate": self.config.shift_scale_rotate_prob,
                "coarse_dropout": self.config.coarse_dropout_prob
            }
        }


def create_data_manager(config: DataConfig) -> CIFAR10DataManager:
    """
    Create a CIFAR-10 data manager instance.
    
    Args:
        config: Data configuration parameters
        
    Returns:
        Initialized data manager instance
    """
    return CIFAR10DataManager(config)


if __name__ == "__main__":
    # Test the data manager
    from config import get_config
    
    config = get_config()
    data_manager = create_data_manager(config.data)
    
    # Calculate statistics
    mean, std = data_manager.calculate_dataset_statistics()
    
    # Create transforms
    train_transform, test_transform = data_manager.create_transforms()
    
    # Load datasets
    train_dataset, test_dataset = data_manager.load_datasets(train_transform, test_transform)
    
    # Create data loaders
    train_loader, test_loader = data_manager.create_data_loaders(train_dataset, test_dataset)
    
    # Print data info
    data_info = data_manager.get_data_info()
    print(f"\nData Manager Info:")
    for key, value in data_info.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Data manager test completed successfully!")
