"""
Configuration file for Advanced CIFAR-10 Classification Project

This module contains all hyperparameters, model settings, and configuration
parameters used throughout the project. All settings are centralized here
for easy modification and experimentation.

"""

import os
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any


@dataclass
class ModelConfig:
    """Model architecture configuration parameters"""
    
    # Model architecture
    input_channels: int = 3
    num_classes: int = 10
    dropout_rate: float = 0.1
    
    # Conv Block 1 (C1) - Standard convolutions
    c1_out_channels: int = 8
    
    # Conv Block 2 (C2) - Depthwise Separable Convolution
    c2_out_channels: int = 12
    
    # Conv Block 3 (C3) - Dilated Convolution
    c3_out_channels: int = 16
    c3_dilation: int = 2
    
    # Conv Block 4 (C40) - Stride=2 instead of MaxPooling
    c4_out_channels: int = 24
    c4_stride: int = 2
    
    # Conv Block 5 - Additional layers for RF > 44
    c5_out_channels: int = 36
    
    # Global Average Pooling + FC
    fc_hidden_size: int = 36
    
    # Parameter constraints
    max_parameters: int = 200000
    min_receptive_field: int = 44


@dataclass
class DataConfig:
    """Data loading and augmentation configuration"""
    
    # Dataset settings
    dataset_name: str = "CIFAR10"
    data_root: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    
    # Reproducibility
    seed: int = 1
    
    # Normalization (calculated from CIFAR-10 dataset)
    mean: Tuple[float, float, float] = (0.49, 0.48, 0.45)
    std: Tuple[float, float, float] = (0.25, 0.24, 0.26)
    
    # Albumentation augmentation parameters
    horizontal_flip_prob: float = 0.5
    shift_scale_rotate_prob: float = 0.5
    shift_limit: float = 0.1
    scale_limit: float = 0.1
    rotate_limit: int = 10
    
    # CoarseDropout parameters
    coarse_dropout_prob: float = 0.3
    max_holes: int = 1
    max_height: int = 16
    max_width: int = 16
    min_holes: int = 1
    min_height: int = 16
    min_width: int = 16


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    
    # Training settings
    epochs: int = 50
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # Scheduler settings
    scheduler_type: str = "OneCycleLR"
    max_lr: float = 0.1
    
    # Device settings
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # Reproducibility
    seed: int = 1
    
    # Model saving
    save_best_model: bool = True
    model_save_path: str = "best_model.pth"
    
    # Performance targets
    target_accuracy: float = 85.0
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "training.log"


@dataclass
class VisualizationConfig:
    """Visualization and analysis configuration"""
    
    # Plot settings
    figure_size: Tuple[int, int] = (15, 10)
    dpi: int = 300
    
    # Sample visualization
    num_sample_images: int = 25
    num_misclassified_images: int = 25
    
    # Save paths
    training_curves_path: str = "training_curves.png"
    class_accuracy_path: str = "class_accuracy.png"
    misclassified_path: str = "misclassified_images.png"


@dataclass
class ProjectConfig:
    """Main project configuration combining all configs"""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Project metadata
    project_name: str = "Advanced CIFAR-10 Classification"
    version: str = "1.0.0"
    author: str = "ERA Student"


def get_config() -> ProjectConfig:
    """
    Get the complete project configuration.
    
    Returns:
        ProjectConfig: Complete configuration object
    """
    return ProjectConfig()


def print_config(config: ProjectConfig) -> None:
    """
    Print the current configuration in a readable format.
    
    Args:
        config: Project configuration to print
    """
    print("=" * 60)
    print(f"Project: {config.project_name} v{config.version}")
    print(f"Author: {config.author}")
    print("=" * 60)
    
    print("\nModel Configuration:")
    print(f"  Input Channels: {config.model.input_channels}")
    print(f"  Number of Classes: {config.model.num_classes}")
    print(f"  Dropout Rate: {config.model.dropout_rate}")
    print(f"  Max Parameters: {config.model.max_parameters:,}")
    print(f"  Min Receptive Field: {config.model.min_receptive_field}")
    
    print("\nData Configuration:")
    print(f"  Dataset: {config.data.dataset_name}")
    print(f"  Batch Size: {config.data.batch_size}")
    print(f"  Data Root: {config.data.data_root}")
    print(f"  Normalization Mean: {config.data.mean}")
    print(f"  Normalization Std: {config.data.std}")
    
    print("\nTraining Configuration:")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  Scheduler: {config.training.scheduler_type}")
    print(f"  Target Accuracy: {config.training.target_accuracy}%")
    
    print("\nAugmentation Configuration:")
    print(f"  Horizontal Flip: {config.data.horizontal_flip_prob}")
    print(f"  ShiftScaleRotate: {config.data.shift_scale_rotate_prob}")
    print(f"  CoarseDropout: {config.data.coarse_dropout_prob}")
    
    print("=" * 60)


def validate_config(config: ProjectConfig) -> bool:
    """
    Validate the configuration parameters.
    
    Args:
        config: Project configuration to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    errors = []
    
    # Model validation
    if config.model.max_parameters <= 0:
        errors.append("Max parameters must be positive")
    
    if config.model.min_receptive_field <= 0:
        errors.append("Min receptive field must be positive")
    
    if config.model.dropout_rate < 0 or config.model.dropout_rate > 1:
        errors.append("Dropout rate must be between 0 and 1")
    
    # Data validation
    if config.data.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    if config.data.num_workers < 0:
        errors.append("Number of workers cannot be negative")
    
    # Training validation
    if config.training.epochs <= 0:
        errors.append("Number of epochs must be positive")
    
    if config.training.learning_rate <= 0:
        errors.append("Learning rate must be positive")
    
    if config.training.target_accuracy < 0 or config.training.target_accuracy > 100:
        errors.append("Target accuracy must be between 0 and 100")
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


# Default configuration instance
DEFAULT_CONFIG = get_config()

if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print_config(config)
    
    if validate_config(config):
        print("\n✅ Configuration is valid!")
    else:
        print("\n❌ Configuration has errors!")
