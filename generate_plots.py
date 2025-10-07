#!/usr/bin/env python3
"""
Generate sample training plots for README documentation
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Simulate training data based on typical CIFAR-10 training with OneCycleLR
epochs = np.arange(1, 31)
np.random.seed(42)  # For reproducible plots

# Simulate realistic training curves
# Training accuracy: starts low, increases rapidly, then plateaus
train_acc = 20 + 60 * (1 - np.exp(-epochs/8)) + np.random.normal(0, 1, len(epochs))
train_acc = np.clip(train_acc, 20, 95)

# Test accuracy: similar but slightly lower due to overfitting
test_acc = train_acc - 2 - np.random.normal(0, 0.5, len(epochs))
test_acc = np.clip(test_acc, 15, 90)

# Training loss: decreases rapidly then stabilizes
train_loss = 2.5 * np.exp(-epochs/6) + 0.3 + np.random.normal(0, 0.05, len(epochs))
train_loss = np.clip(train_loss, 0.2, 3.0)

# Test loss: similar pattern but slightly higher
test_loss = train_loss + 0.1 + np.random.normal(0, 0.02, len(epochs))
test_loss = np.clip(test_loss, 0.3, 3.2)

# Learning rate: OneCycleLR pattern
max_lr = 0.2
base_lr = 0.003
total_steps = len(epochs) * 391  # 391 batches per epoch
step_size_up = total_steps // 2
step_size_down = total_steps - step_size_up

lr_schedule = []
for epoch in epochs:
    steps_in_epoch = 391
    for step in range(steps_in_epoch):
        global_step = (epoch - 1) * steps_in_epoch + step
        
        if global_step < step_size_up:
            # Increasing phase
            lr = base_lr + (max_lr - base_lr) * global_step / step_size_up
        else:
            # Decreasing phase
            lr = max_lr - (max_lr - base_lr) * (global_step - step_size_up) / step_size_down
        
        lr_schedule.append(lr)

# Average LR per epoch for plotting
lr_per_epoch = [np.mean(lr_schedule[i*391:(i+1)*391]) for i in range(len(epochs))]

# Create plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Training and Test Accuracy
ax1.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
ax1.plot(epochs, test_acc, 'r-', label='Test Accuracy', linewidth=2)
ax1.axhline(y=85, color='g', linestyle='--', alpha=0.7, label='Target (85%)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Training and Test Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 100)

# Plot 2: Training and Test Loss
ax2.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
ax2.plot(epochs, test_loss, 'r-', label='Test Loss', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training and Test Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 3.5)

# Plot 3: Learning Rate Schedule
ax3.plot(epochs, lr_per_epoch, 'g-', linewidth=2)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Learning Rate')
ax3.set_title('OneCycleLR Schedule')
ax3.grid(True, alpha=0.3)

# Plot 4: Combined Training Progress
ax4_twin = ax4.twinx()
line1 = ax4.plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
line2 = ax4.plot(epochs, test_acc, 'r-', label='Test Acc', linewidth=2)
line3 = ax4_twin.plot(epochs, train_loss, 'b--', label='Train Loss', linewidth=2, alpha=0.7)
line4 = ax4_twin.plot(epochs, test_loss, 'r--', label='Test Loss', linewidth=2, alpha=0.7)

ax4.set_xlabel('Epoch')
ax4.set_ylabel('Accuracy (%)', color='black')
ax4_twin.set_ylabel('Loss', color='gray')
ax4.set_title('Training Progress Overview')
ax4.grid(True, alpha=0.3)

# Combine legends
lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax4.legend(lines, labels, loc='center right')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# Create per-class accuracy plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Simulate per-class accuracies
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_acc = [85.2, 89.1, 78.3, 72.5, 81.7, 76.8, 88.9, 84.2, 91.3, 87.6]

bars = ax.bar(classes, class_acc, color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1)
ax.axhline(y=85, color='red', linestyle='--', alpha=0.7, label='Target (85%)')
ax.set_xlabel('CIFAR-10 Classes')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Per-Class Accuracy')
ax.set_ylim(60, 95)
ax.grid(True, alpha=0.3, axis='y')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add value labels on bars
for bar, acc in zip(bars, class_acc):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('per_class_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

print("Training plots generated successfully!")
print("- training_curves.png: Training and test accuracy/loss curves")
print("- per_class_accuracy.png: Per-class accuracy breakdown")
