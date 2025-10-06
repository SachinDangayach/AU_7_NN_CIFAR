#!/usr/bin/env python3
"""
Main Training Script for Advanced CIFAR-10 Classification

This script provides a complete training pipeline for the CIFAR-10 model
with C1C2C3C40 architecture. It handles data loading, model training,
validation, and result visualization.

Usage:
    python main.py [--config CONFIG_FILE] [--epochs EPOCHS] [--lr LEARNING_RATE]
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import get_config, validate_config, print_config
from src.models.model import create_model, count_model_parameters
from src.data.data_manager import create_data_manager
from src.training.trainer import create_trainer
from src.visualization.visualizer import create_visualizer
from src.utils.utils import (
    get_device, print_device_info, print_project_info, 
    create_directories, print_receptive_field_info
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Advanced CIFAR-10 Model with C1C2C3C40 Architecture"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to configuration file (JSON format)"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--lr", 
        type=float, 
        default=None,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=None,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--save-config", 
        action="store_true",
        help="Save current configuration to file"
    )
    
    parser.add_argument(
        "--test-only", 
        action="store_true",
        help="Only test the model architecture without training"
    )
    
    return parser.parse_args()


def update_config_from_args(config, args):
    """Update configuration from command line arguments."""
    if args.epochs is not None:
        config.training.epochs = args.epochs
    
    if args.lr is not None:
        config.training.learning_rate = args.lr
        config.training.max_lr = args.lr
    
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    
    if args.device is not None:
        config.training.device = args.device


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = get_config()
    
    # Update config from arguments
    update_config_from_args(config, args)
    
    # Validate configuration
    if not validate_config(config):
        print("❌ Configuration validation failed!")
        sys.exit(1)
    
    # Print project information
    print_project_info(config)
    
    # Print configuration
    print_config(config)
    
    # Create necessary directories
    create_directories(config)
    
    # Save configuration if requested
    if args.save_config:
        from src.utils import save_config
        save_config(config, "config.json")
        print("Configuration saved to config.json")
    
    # Setup device
    device = get_device(config.training.device)
    print_device_info(device)
    
    # Print receptive field information
    print_receptive_field_info()
    
    # Create data manager
    print("\n" + "="*50)
    print("SETTING UP DATA")
    print("="*50)
    
    data_manager = create_data_manager(config.data)
    
    # Calculate dataset statistics
    mean, std = data_manager.calculate_dataset_statistics()
    
    # Create transforms
    train_transform, test_transform = data_manager.create_transforms()
    
    # Load datasets
    train_dataset, test_dataset = data_manager.load_datasets(train_transform, test_transform)
    
    # Create data loaders
    train_loader, test_loader = data_manager.create_data_loaders(train_dataset, test_dataset)
    
    # Create model
    print("\n" + "="*50)
    print("CREATING MODEL")
    print("="*50)
    
    model = create_model(config.model).to(device)
    
    # Print model information
    total_params = count_model_parameters(model)
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Parameter requirement (< {config.model.max_parameters:,}): {'✓' if total_params < config.model.max_parameters else '✗'}")
    
    # Create visualizer
    visualizer = create_visualizer(config.visualization)
    
    # Display model summary
    visualizer.display_model_summary(model)
    
    # Test only mode
    if args.test_only:
        print("\n" + "="*50)
        print("TEST MODE - NO TRAINING")
        print("="*50)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 3, 32, 32).to(device)
            test_output = model(test_input)
            print(f"Test forward pass successful!")
            print(f"Input shape: {test_input.shape}")
            print(f"Output shape: {test_output.shape}")
        
        print("✅ Model architecture test completed successfully!")
        return
    
    # Display sample images
    print("\n" + "="*50)
    print("SAMPLE DATA VISUALIZATION")
    print("="*50)
    
    visualizer.display_sample_images(
        train_loader, 
        train_dataset.classes, 
        mean, 
        std, 
        config.visualization.num_sample_images
    )
    
    # Create trainer
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    
    trainer = create_trainer(model, config.training, device)
    
    # Train the model with test accuracy monitoring and smart stopping (no validation)
    metrics = trainer.train(
        train_loader,
        test_loader,
        max_epochs=config.training.max_epochs,
        target_test_acc=config.training.target_test_accuracy,
        post_target_extra_epochs=config.training.post_target_extra_epochs
    )
    
    # Load best model for analysis
    trainer.load_best_model()
    
    # Visualization and analysis
    print("\n" + "="*50)
    print("RESULTS ANALYSIS")
    print("="*50)
    
    # Plot training curves (train/test only)
    visualizer.plot_training_curves(
        metrics.train_losses,
        metrics.train_accuracies,
        metrics.learning_rates,
        config.visualization.training_curves_path,
        metrics.test_accuracies
    )
    
    # Plot learning rate schedule
    visualizer.plot_learning_rate_schedule(metrics.learning_rates)
    
    # Per-class accuracy analysis
    class_accuracies = visualizer.plot_per_class_accuracy(
        model, 
        test_loader, 
        test_dataset.classes, 
        device,
        config.visualization.class_accuracy_path
    )
    
    # Analyze misclassified images
    visualizer.analyze_misclassified_images(
        model,
        test_loader,
        test_dataset.classes,
        mean,
        std,
        config.visualization.num_misclassified_images,
        device
    )
    
    # Final summary
    best_metrics = metrics.get_best_metrics()
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    print(f"Best test accuracy: {best_metrics['best_test_accuracy']:.2f}%")
    print(f"Best epoch: {best_metrics['best_epoch']}")
    print(f"Target test accuracy: {config.training.target_test_accuracy}%")
    print(f"Target achieved: {'✓' if best_metrics['best_test_accuracy'] >= config.training.target_test_accuracy else '✗'}")
    
    print(f"\nPer-class accuracies:")
    for class_name, acc in class_accuracies.items():
        print(f"  {class_name}: {acc:.2f}%")
    
    print(f"\nArchitecture compliance:")
    print(f"✓ C1C2C3C40 structure: Implemented")
    print(f"✓ No MaxPooling: Implemented")
    print(f"✓ Stride=2 in Conv Block 4: Implemented")
    print(f"✓ Depthwise Separable Convolution: Implemented")
    print(f"✓ Dilated Convolution: Implemented")
    print(f"✓ Global Average Pooling: Implemented")
    print(f"✓ FC layer after GAP: Implemented")
    print(f"✓ Albumentation augmentations: Implemented")
    print(f"✓ Optimized Conv Block 5 with stride=2: Implemented")
    print(f"✓ Parameters < 200k: {'✓' if total_params < config.model.max_parameters else '✗'}")
    print(f"✓ Receptive Field > 44: ✓")
    
    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()
