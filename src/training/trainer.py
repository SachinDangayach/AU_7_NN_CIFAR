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
    
    This class maintains lists of training and test metrics
    for visualization and analysis purposes.
    """
    
    def __init__(self):
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.test_losses: List[float] = []
        self.test_accuracies: List[float] = []
        self.learning_rates: List[float] = []
        # Backward-compatibility aliases expected by some notebooks
        self.val_losses: List[float] = []  # we don't compute separate val loss before; now mirrors test loss when available
        self.val_accuracies: List[float] = self.test_accuracies  # alias: validation == test in this project
        
    def add_epoch_metrics(
        self,
        train_loss: float,
        train_acc: float,
        test_acc: Optional[float],
        lr: float,
        test_loss: Optional[float] = None
    ) -> None:
        """
        Add metrics for a single epoch.
        
        Args:
            train_loss: Training loss for the epoch
            train_acc: Training accuracy for the epoch
            test_acc: Test accuracy for the epoch
            lr: Learning rate for the epoch
            test_loss: Optional test loss for the epoch
        """
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        # accuracy
        if test_acc is not None:
            self.test_accuracies.append(test_acc)
        else:
            self.test_accuracies.append(float('nan'))
        # loss
        if test_loss is not None:
            self.test_losses.append(test_loss)
            self.val_losses.append(test_loss)
        else:
            self.test_losses.append(float('nan'))
            self.val_losses.append(float('nan'))
        self.learning_rates.append(lr)
    
    def get_best_metrics(self) -> Dict[str, float]:
        """
        Get the best test metrics achieved during training.
        
        Returns:
            Dictionary containing best test metrics
        """
        if not self.test_accuracies:
            return {}
        best_epoch = max(range(len(self.test_accuracies)), key=lambda i: (self.test_accuracies[i] if not (self.test_accuracies[i] != self.test_accuracies[i]) else -1))
        return {
            "best_epoch": best_epoch + 1,
            "best_test_accuracy": self.test_accuracies[best_epoch],
            "best_val_accuracy": self.test_accuracies[best_epoch],  # alias for backward compatibility
            "train_accuracy_at_best": self.train_accuracies[best_epoch],
            "train_loss_at_best": self.train_losses[best_epoch]
        }


class ModelTrainer:
    """
    Model trainer for CIFAR-10 classification.
    
    This class handles the complete training pipeline including
    training, testing, checkpointing, and metrics tracking.
    
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
        self.best_test_accuracy = 0.0
        self.best_val_accuracy = 0.0  # alias for notebooks expecting validation nomenclature
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
                epochs=getattr(self.config, 'scheduler_epochs', 100),
                steps_per_epoch=len(train_loader)
            )
        elif self.config.scheduler_type == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        elif self.config.scheduler_type == "StepLR":
            scheduler = StepLR(
                optimizer,
                step_size=getattr(self.config, 'step_lr_step_size', 20),
                gamma=0.1
            )
        
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
                'Train': f'{current_acc:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total_samples
        
        return avg_loss, accuracy
    

    def test_epoch(self, test_loader: Optional[torch.utils.data.DataLoader]) -> Optional[Tuple[float, float]]:
        """
        Evaluate the model on the test set for one epoch and return (loss, accuracy).
        """
        if test_loader is None:
            return None
        self.model.eval()
        correct = 0
        total_samples = 0
        total_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = F.nll_loss(output, target)
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total_samples += len(data)
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else float('nan')
        acc = 100.0 * correct / total_samples if total_samples > 0 else 0.0
        return (avg_loss, acc)
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        max_epochs: Optional[int] = None,
        target_test_acc: Optional[float] = None,
        post_target_extra_epochs: Optional[int] = None
    ) -> TrainingMetrics:
        """
        Complete training pipeline.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            max_epochs: Maximum number of epochs
            target_test_acc: Target test accuracy for early stopping
            post_target_extra_epochs: Extra epochs after reaching target
            
        Returns:
            Training metrics object
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        effective_max_epochs = max_epochs or getattr(self.config, 'max_epochs', self.config.epochs)
        self.logger.info(f"Epochs: {effective_max_epochs}")
        self.logger.info(f"Learning Rate: {self.config.learning_rate}")
        self.logger.info(f"Scheduler: {self.config.scheduler_type}")
        
        # Setup optimizer and scheduler
        optimizer, scheduler = self._setup_optimizer_and_scheduler(train_loader)
        
        # Training loop
        start_time = time.time()
        
        target_test_acc = target_test_acc if target_test_acc is not None else getattr(self.config, 'target_test_accuracy', 85.0)
        post_target_extra_epochs = post_target_extra_epochs if post_target_extra_epochs is not None else getattr(self.config, 'post_target_extra_epochs', 3)
        target_reached_epoch: Optional[int] = None

        for epoch in range(effective_max_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler, epoch)
            
            # Test
            test_out = self.test_epoch(test_loader)
            if test_out is None:
                test_loss, test_acc = None, None
            else:
                test_loss, test_acc = test_out
            
            # Update scheduler (for non-OneCycleLR schedulers)
            if scheduler and not isinstance(scheduler, OneCycleLR):
                scheduler.step()
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update metrics (includes test loss if available)
            self.metrics.add_epoch_metrics(train_loss, train_acc, test_acc, current_lr, test_loss)
            
            # Check for best model
            if test_acc is not None and test_acc > self.best_test_accuracy:
                self.best_test_accuracy = test_acc
                self.best_val_accuracy = test_acc  # keep alias updated
                self.best_epoch = epoch + 1
                
                if self.config.save_best_model:
                    torch.save(self.model.state_dict(), self.config.model_save_path)
                    self.logger.info(f"New best model saved! Test Acc: {test_acc:.2f}%")
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            if test_acc is not None:
                self.logger.info(
                    f"Epoch {epoch+1}/{effective_max_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                    f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% - "
                    f"LR: {current_lr:.6f} - "
                    f"Time: {epoch_time:.2f}s"
                )
                # Emit a tqdm-like completed bar line
                try:
                    total_batches = len(train_loader)
                    mins = int(epoch_time // 60)
                    secs = int(epoch_time % 60)
                    it_per_s = total_batches / epoch_time if epoch_time > 0 else 0.0
                    from tqdm import tqdm
                    tqdm.write(
                        f"Epoch {epoch+1}/{effective_max_epochs}: 100%|{'█'*10}{' '*(10-10)}| {total_batches}/{total_batches} "
                        f"[{mins:02d}:{secs:02d}<00:00,  {it_per_s:5.2f}it/s, Train={train_acc:.2f}%,  Test={test_acc:.2f}%, LR={current_lr:.6f}]"
                    )
                except Exception:
                    pass
            else:
                self.logger.info(
                    f"Epoch {epoch+1}/{effective_max_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                    f"LR: {current_lr:.6f} - "
                    f"Time: {epoch_time:.2f}s"
                )

            # Smart stopping based on test accuracy
            if test_acc is not None:
                if test_acc >= target_test_acc and target_reached_epoch is None:
                    target_reached_epoch = epoch
                    self.logger.info(f"🎯 Target test accuracy {target_test_acc:.2f}% reached at epoch {epoch+1}.")
                if target_reached_epoch is not None:
                    if epoch - target_reached_epoch + 1 >= post_target_extra_epochs:
                        self.logger.info(f"✅ Stopping after {post_target_extra_epochs} extra epochs post target.")
                        break
            
            # Early stopping check (optional)
            # No validation-based stop anymore
        
        total_time = time.time() - start_time
        
        # Final summary
        best_metrics = self.metrics.get_best_metrics()
        self.logger.info("Training completed!")
        self.logger.info(f"Total training time: {total_time/60:.2f} minutes")
        self.logger.info(f"Best test accuracy: {self.best_test_accuracy:.2f}% at epoch {self.best_epoch}")
        
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
