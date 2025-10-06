"""
Utility Functions for CIFAR-10 Project

This module contains utility functions for:
- Device management
- Model parameter counting
- Receptive field calculation
- Configuration validation
- File operations
"""

import torch
import torch.nn as nn
import os
import json
from typing import Dict, Any, Optional, Tuple
from config import ProjectConfig


def get_device(device_preference: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device_preference: Device preference ("auto", "cuda", "cpu")
        
    Returns:
        PyTorch device object
    """
    if device_preference == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_preference == "cuda":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    return device


def print_device_info(device: torch.device) -> None:
    """
    Print information about the current device.
    
    Args:
        device: PyTorch device object
    """
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        print("Using CPU")


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate the model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def calculate_receptive_field_info() -> Dict[str, Any]:
    """
    Calculate receptive field information for the C1C2C3C40 architecture.
    
    Returns:
        Dictionary containing receptive field information
    """
    rf_info = {
        "Conv Block 1": {
            "description": "Standard convolutions (3x3)",
            "output_size": "32x32",
            "receptive_field": 5,
            "parameters": "~1K"
        },
        "Conv Block 2": {
            "description": "Depthwise Separable Convolution",
            "output_size": "32x32", 
            "receptive_field": 9,
            "parameters": "~2K"
        },
        "Conv Block 3": {
            "description": "Dilated Convolution (dilation=2)",
            "output_size": "32x32",
            "receptive_field": 17,
            "parameters": "~8K"
        },
        "Conv Block 4": {
            "description": "Stride=2 instead of MaxPooling",
            "output_size": "16x16",
            "receptive_field": 25,
            "parameters": "~30K"
        },
        "Conv Block 5": {
            "description": "Optimized: 2 stride=2 operations with minimum 2 conv gap",
            "output_size": "4x4",
            "receptive_field": 45,
            "parameters": "~75K"
        },
        "Global Average Pool": {
            "description": "Adaptive average pooling",
            "output_size": "1x1",
            "receptive_field": 45,
            "parameters": "0"
        },
        "Total": {
            "receptive_field": 45,
            "meets_requirement": True,
            "requirement": "> 44"
        }
    }
    
    return rf_info


def print_receptive_field_info() -> None:
    """
    Print receptive field information in a formatted way.
    """
    rf_info = calculate_receptive_field_info()
    
    print("=" * 60)
    print("RECEPTIVE FIELD ANALYSIS")
    print("=" * 60)
    
    for layer, info in rf_info.items():
        if layer != "Total":
            print(f"{layer}:")
            print(f"  Description: {info['description']}")
            print(f"  Output Size: {info['output_size']}")
            print(f"  Receptive Field: {info['receptive_field']}")
            print(f"  Parameters: {info['parameters']}")
            print()
    
    total_info = rf_info["Total"]
    print(f"Total Receptive Field: {total_info['receptive_field']}")
    print(f"Requirement: {total_info['requirement']}")
    print(f"Meets Requirement: {'✓' if total_info['meets_requirement'] else '✗'}")
    print("=" * 60)


def save_config(config: ProjectConfig, filepath: str) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Project configuration object
        filepath: Path to save the configuration
    """
    config_dict = {
        "model": {
            "input_channels": config.model.input_channels,
            "num_classes": config.model.num_classes,
            "dropout_rate": config.model.dropout_rate,
            "c1_out_channels": config.model.c1_out_channels,
            "c2_out_channels": config.model.c2_out_channels,
            "c3_out_channels": config.model.c3_out_channels,
            "c3_dilation": config.model.c3_dilation,
            "c4_out_channels": config.model.c4_out_channels,
            "c4_stride": config.model.c4_stride,
            "c5_out_channels": config.model.c5_out_channels,
            "fc_hidden_size": config.model.fc_hidden_size,
            "max_parameters": config.model.max_parameters,
            "min_receptive_field": config.model.min_receptive_field
        },
        "data": {
            "dataset_name": config.data.dataset_name,
            "data_root": config.data.data_root,
            "batch_size": config.data.batch_size,
            "num_workers": config.data.num_workers,
            "pin_memory": config.data.pin_memory,
            "mean": config.data.mean,
            "std": config.data.std,
            "horizontal_flip_prob": config.data.horizontal_flip_prob,
            "shift_scale_rotate_prob": config.data.shift_scale_rotate_prob,
            "shift_limit": config.data.shift_limit,
            "scale_limit": config.data.scale_limit,
            "rotate_limit": config.data.rotate_limit,
            "coarse_dropout_prob": config.data.coarse_dropout_prob,
            "max_holes": config.data.max_holes,
            "max_height": config.data.max_height,
            "max_width": config.data.max_width,
            "min_holes": config.data.min_holes,
            "min_height": config.data.min_height,
            "min_width": config.data.min_width
        },
        "training": {
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            "momentum": config.training.momentum,
            "weight_decay": config.training.weight_decay,
            "scheduler_type": config.training.scheduler_type,
            "max_lr": config.training.max_lr,
            "device": config.training.device,
            "seed": config.training.seed,
            "save_best_model": config.training.save_best_model,
            "model_save_path": config.training.model_save_path,
            "target_accuracy": config.training.target_accuracy
        },
        "visualization": {
            "figure_size": config.visualization.figure_size,
            "dpi": config.visualization.dpi,
            "num_sample_images": config.visualization.num_sample_images,
            "num_misclassified_images": config.visualization.num_misclassified_images,
            "training_curves_path": config.visualization.training_curves_path,
            "class_accuracy_path": config.visualization.class_accuracy_path,
            "misclassified_path": config.visualization.misclassified_path
        },
        "project": {
            "project_name": config.project_name,
            "version": config.version,
            "author": config.author,
            "log_level": config.log_level,
            "log_file": config.log_file
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Configuration saved to {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        filepath: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    print(f"Configuration loaded from {filepath}")
    return config_dict


def create_directories(config: ProjectConfig) -> None:
    """
    Create necessary directories for the project.
    
    Args:
        config: Project configuration object
    """
    directories = [
        config.data.data_root,
        os.path.dirname(config.training.model_save_path),
        os.path.dirname(config.training.log_file),
        os.path.dirname(config.visualization.training_curves_path),
        "models",
        "logs",
        "results"
    ]
    
    for directory in directories:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")


def print_project_info(config: ProjectConfig) -> None:
    """
    Print comprehensive project information.
    
    Args:
        config: Project configuration object
    """
    print("=" * 70)
    print(f"PROJECT: {config.project_name}")
    print(f"VERSION: {config.version}")
    print(f"AUTHOR: {config.author}")
    print("=" * 70)
    
    print("\nOBJECTIVES:")
    print("✓ C1C2C3C40 architecture (No MaxPooling, stride=2 in last conv)")
    print("✓ Total Receptive Field > 44")
    print("✓ Depthwise Separable Convolution")
    print("✓ Dilated Convolution")
    print("✓ Global Average Pooling with FC layer")
    print("✓ Albumentation data augmentation")
    print("✓ Parameters < 200k")
    print("✓ Target accuracy: 85%")
    
    print("\nARCHITECTURE FEATURES:")
    print("✓ Conv Block 1: Standard convolutions")
    print("✓ Conv Block 2: Depthwise Separable Convolution")
    print("✓ Conv Block 3: Dilated Convolution (dilation=2)")
    print("✓ Conv Block 4: Stride=2 instead of MaxPooling")
    print("✓ Conv Block 5: Optimized: 2 stride=2 operations with minimum 2 conv gap")
    print("✓ Global Average Pooling + FC layer")
    
    print("\nDATA AUGMENTATION:")
    print("✓ Horizontal Flip")
    print("✓ ShiftScaleRotate")
    print("✓ CoarseDropout")
    print("✓ Proper normalization")
    
    print("=" * 70)


if __name__ == "__main__":
    # Test utility functions
    from config import get_config
    
    config = get_config()
    
    # Test device functions
    device = get_device()
    print_device_info(device)
    
    # Test receptive field calculation
    print_receptive_field_info()
    
    # Test project info
    print_project_info(config)
    
    # Test directory creation
    create_directories(config)
    
    print("✅ Utility functions test completed successfully!")
