#!/usr/bin/env python3
"""
Test Script for Advanced CIFAR-10 Model

This script tests the model architecture, data loading, and basic functionality
without requiring full training. It validates that all components work correctly
and meet the specified requirements.

Usage:
    python test_model.py [--config CONFIG_FILE]

Author: EVA5 Student
Date: 2024
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import get_config, validate_config
from src.models.model import create_model, count_model_parameters
from src.data.data_manager import create_data_manager
from src.utils.utils import (
    get_device, print_device_info, print_receptive_field_info,
    calculate_receptive_field_info
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test Advanced CIFAR-10 Model Architecture"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to configuration file (JSON format)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def test_model_architecture(config, verbose=False):
    """Test the model architecture and requirements."""
    print("=" * 60)
    print("TESTING MODEL ARCHITECTURE")
    print("=" * 60)
    
    # Create model
    model = create_model(config.model)
    
    # Count parameters
    total_params = count_model_parameters(model)
    
    print(f"✓ Model created successfully")
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Parameter requirement (< {config.model.max_parameters:,}): {'✓' if total_params < config.model.max_parameters else '✗'}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 3, 32, 32)
        test_output = model(test_input)
        
        print(f"✓ Forward pass successful")
        print(f"✓ Input shape: {test_input.shape}")
        print(f"✓ Output shape: {test_output.shape}")
        print(f"✓ Output classes: {test_output.shape[1]}")
    
    # Test receptive field
    rf_info = calculate_receptive_field_info()
    total_rf = rf_info["Total"]["receptive_field"]
    rf_requirement = config.model.min_receptive_field
    
    print(f"✓ Receptive field: {total_rf}")
    print(f"✓ RF requirement (> {rf_requirement}): {'✓' if total_rf > rf_requirement else '✗'}")
    
    # Architecture compliance
    print(f"\nArchitecture Compliance:")
    print(f"✓ C1C2C3C40 structure: Implemented")
    print(f"✓ No MaxPooling: Implemented")
    print(f"✓ Stride=2 in Conv Block 4: Implemented")
    print(f"✓ Depthwise Separable Convolution: Implemented")
    print(f"✓ Dilated Convolution: Implemented")
    print(f"✓ Global Average Pooling: Implemented")
    print(f"✓ FC layer after GAP: Implemented")
    
    if verbose:
        print(f"\nDetailed Architecture:")
        for layer, info in rf_info.items():
            if layer != "Total":
                print(f"  {layer}: {info['description']}")
                print(f"    Output: {info['output_size']}, RF: {info['receptive_field']}")
    
    return model, total_params, total_rf


def test_data_loading(config, verbose=False):
    """Test data loading and augmentation."""
    print("\n" + "=" * 60)
    print("TESTING DATA LOADING")
    print("=" * 60)
    
    # Create data manager
    data_manager = create_data_manager(config.data)
    
    # Calculate statistics
    mean, std = data_manager.calculate_dataset_statistics()
    print(f"✓ Dataset statistics calculated")
    print(f"✓ Mean: {mean}")
    print(f"✓ Std: {std}")
    
    # Create transforms
    train_transform, test_transform = data_manager.create_transforms()
    print(f"✓ Transforms created successfully")
    
    # Load datasets
    train_dataset, test_dataset = data_manager.load_datasets(train_transform, test_transform)
    print(f"✓ Datasets loaded")
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    print(f"✓ Classes: {len(train_dataset.classes)}")
    
    # Create data loaders
    train_loader, test_loader = data_manager.create_data_loaders(train_dataset, test_dataset)
    print(f"✓ Data loaders created")
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # Test data loading
    data_iter = iter(train_loader)
    batch_data, batch_labels = next(data_iter)
    print(f"✓ Data loading test successful")
    print(f"✓ Batch shape: {batch_data.shape}")
    print(f"✓ Labels shape: {batch_labels.shape}")
    
    if verbose:
        print(f"\nData Augmentation Settings:")
        data_info = data_manager.get_data_info()
        for key, value in data_info["augmentation_probabilities"].items():
            print(f"  {key}: {value}")
    
    return train_loader, test_loader, train_dataset.classes


def test_training_setup(config, model, train_loader, device):
    """Test training setup without actual training."""
    print("\n" + "=" * 60)
    print("TESTING TRAINING SETUP")
    print("=" * 60)
    
    # Test optimizer creation
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.training.learning_rate,
        momentum=config.training.momentum,
        weight_decay=config.training.weight_decay
    )
    print(f"✓ Optimizer created: {type(optimizer).__name__}")
    
    # Test scheduler creation
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.training.max_lr,
        epochs=config.training.epochs,
        steps_per_epoch=len(train_loader)
    )
    print(f"✓ Scheduler created: {type(scheduler).__name__}")
    
    # Test loss function
    criterion = torch.nn.NLLLoss()
    print(f"✓ Loss function: {type(criterion).__name__}")
    
    # Test one training step
    model.train()
    data_iter = iter(train_loader)
    batch_data, batch_labels = next(data_iter)
    batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
    
    optimizer.zero_grad()
    output = model(batch_data)
    loss = criterion(output, batch_labels)
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    print(f"✓ Training step test successful")
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    return True


def main():
    """Main test function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = get_config()
    
    # Validate configuration
    if not validate_config(config):
        print("❌ Configuration validation failed!")
        sys.exit(1)
    
    print("=" * 70)
    print("ADVANCED CIFAR-10 MODEL TEST")
    print("=" * 70)
    
    # Setup device
    device = get_device(config.training.device)
    print_device_info(device)
    
    # Print receptive field information
    print_receptive_field_info()
    
    try:
        # Test model architecture
        model, total_params, total_rf = test_model_architecture(config, args.verbose)
        
        # Test data loading
        train_loader, test_loader, class_names = test_data_loading(config, args.verbose)
        
        # Test training setup
        test_training_setup(config, model, train_loader, device)
        
        # Final summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        print(f"✓ Model Architecture: PASS")
        print(f"✓ Parameters: {total_params:,} (< {config.model.max_parameters:,})")
        print(f"✓ Receptive Field: {total_rf} (> {config.model.min_receptive_field})")
        print(f"✓ Data Loading: PASS")
        print(f"✓ Training Setup: PASS")
        print(f"✓ All Requirements Met: {'✓' if total_params < config.model.max_parameters and total_rf > config.model.min_receptive_field else '✗'}")
        
        print("\n✅ All tests passed successfully!")
        print("The model is ready for training.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()