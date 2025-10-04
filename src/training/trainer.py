"""
Training Module for Advanced CIFAR-10 Model

This module handles the training process including:
- Model training with advanced scheduling
- Validation and testing
- Model checkpointing
- Training metrics tracking
- Learning rate scheduling

Features:
- OneCycleLR scheduler for better convergence
- Automatic best model saving
- Comprehensive training metrics
- GPU/CPU device handling
- Reproducible training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, StepLR
from tqdm import tqdm
import time
import logging
from typing import Tuple, List, Dict, Optional
from config import TrainingConfig, ModelConfig


class TrainingMetrics:
    """
    Class to track training metrics throughout the training process.
    
    This class maintains lists of training and validation metrics
    for visualization and analysis purposes.
    """
    
    def __init__(self):
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []
        self.learning_rates: List[float] = []
        
    def add_epoch_metrics(
        self,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float
    ) -> None:
        """
        Add metrics for a single epoch.
        
        Args:
            train_loss: Training loss for the epoch
            train_acc: Training accuracy for the epoch
            val_loss: Validation loss for the epoch
            val_acc: Validation accuracy for the epoch
            lr: Learning rate for the epoch
        """
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
    
    def get_best_metrics(self) -> Dict[str, float]:
        """
        Get the best validation metrics achieved during training.
        
        Returns:
            Dictionary containing best validation metrics
        """
        if not self.val_accuracies:
            return {}
        
        best_epoch = max(range(len(self.val_accuracies)), key=lambda i: self.val_accuracies[i])
        
        return {
            "best_epoch": best_epoch + 1,
            "best_val_accuracy": self.val_accuracies[best_epoch],
            "best_val_loss": self.val_losses[best_epoch],
            "train_accuracy_at_best": self.train_accuracies[best_epoch],
            "train_loss_at_best": self.train_losses[best_epoch]
        }


class ModelTrainer:
    """
    Model trainer for CIFAR-10 classification.
    
    This class handles the complete training pipeline including
    training, validation, checkpointing, and metrics tracking.
    
    Args:
        model: PyTorch model to train
        config: Training configuration parameters
        device: Device to train on (cuda/cpu)
    """
    
    def __init__(self, model: nn.Module, config: TrainingConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.metrics = TrainingMetrics()
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _setup_optimizer_and_scheduler(self, train_loader: torch.utils.data.DataLoader) -> Tuple[optim.Optimizer, Optional[object]]:
        """
        Setup optimizer and learning rate scheduler.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (optimizer, scheduler)
        """
        # Create optimizer
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        # Create scheduler
        scheduler = None
        if self.config.scheduler_type == "OneCycleLR":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.config.max_lr,
                epochs=self.config.epochs,
                steps_per_epoch=len(train_loader)
            )
        elif self.config.scheduler_type == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        elif self.config.scheduler_type == "StepLR":
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        
        return optimizer, scheduler
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[object],
        epoch: int
    ) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = F.nll_loss(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update scheduler if OneCycleLR
            if isinstance(scheduler, OneCycleLR):
                scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total_samples += len(data)
            
            # Update progress bar
            current_acc = 100.0 * correct / total_samples
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Validate the model for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = F.nll_loss(output, target, reduction='sum')
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total_samples += len(data)
        
        avg_loss = total_loss / total_samples
        accuracy = 100.0 * correct / total_samples
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader
    ) -> TrainingMetrics:
        """
        Complete training pipeline.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training metrics object
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {self.config.epochs}")
        self.logger.info(f"Learning Rate: {self.config.learning_rate}")
        self.logger.info(f"Scheduler: {self.config.scheduler_type}")
        
        # Setup optimizer and scheduler
        optimizer, scheduler = self._setup_optimizer_and_scheduler(train_loader)
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler, epoch)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler (for non-OneCycleLR schedulers)
            if scheduler and not isinstance(scheduler, OneCycleLR):
                scheduler.step()
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update metrics
            self.metrics.add_epoch_metrics(train_loss, train_acc, val_loss, val_acc, current_lr)
            
            # Check for best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch + 1
                
                if self.config.save_best_model:
                    torch.save(self.model.state_dict(), self.config.model_save_path)
                    self.logger.info(f"New best model saved! Val Acc: {val_acc:.2f}%")
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                f"LR: {current_lr:.6f} - "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Early stopping check (optional)
            if val_acc >= self.config.target_accuracy:
                self.logger.info(f"Target accuracy {self.config.target_accuracy}% reached!")
                break
        
        total_time = time.time() - start_time
        
        # Final summary
        best_metrics = self.metrics.get_best_metrics()
        self.logger.info("Training completed!")
        self.logger.info(f"Total training time: {total_time/60:.2f} minutes")
        self.logger.info(f"Best validation accuracy: {self.best_val_accuracy:.2f}% at epoch {self.best_epoch}")
        
        if best_metrics:
            self.logger.info(f"Best metrics: {best_metrics}")
        
        return self.metrics
    
    def load_best_model(self) -> None:
        """
        Load the best model checkpoint.
        """
        try:
            self.model.load_state_dict(torch.load(self.config.model_save_path))
            self.logger.info(f"Best model loaded from {self.config.model_save_path}")
        except FileNotFoundError:
            self.logger.warning(f"No checkpoint found at {self.config.model_save_path}")


def create_trainer(model: nn.Module, config: TrainingConfig, device: torch.device) -> ModelTrainer:
    """
    Create a model trainer instance.
    
    Args:
        model: PyTorch model to train
        config: Training configuration parameters
        device: Device to train on
        
    Returns:
        Initialized trainer instance
    """
    return ModelTrainer(model, config, device)


if __name__ == "__main__":
    # Test the trainer
    from config import get_config
    from src.models.model import create_model
    
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model(config.model).to(device)
    
    # Create trainer
    trainer = create_trainer(model, config.training, device)
    
    print("✅ Trainer test completed successfully!")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
