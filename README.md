# Advanced CIFAR-10 Classification Project

## 🎯 Overview

This project implements a **C1C2C3C4 CNN architecture** for CIFAR-10 classification, achieving **86.62% test accuracy** with only **162,458 parameters** (under 200k limit). The model uses advanced techniques including Depthwise Separable Convolutions, Dilated Convolutions, and Global Average Pooling.

## ✅ Key Achievements

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **Test Accuracy** | ≥85% | **86.62%** | ✅ |
| **Parameters** | <200k | **162,458** | ✅ |
| **Receptive Field** | >44 | **45** | ✅ |
| **Training Epochs** | - | **26** | ✅ |

## 🏗️ Model Architecture

### C1C2C3C4 Structure
```
Input Image (32×32×3)
│
├── C1: Initial Features (3→16, RF: 3→5)
│   ├── Conv3×3 + BN + ReLU + Dropout
│   └── Conv3×3 + BN + ReLU + Dropout
│
├── C2: Depthwise Separable (16→32, RF: 5→11)
│   ├── DW Conv3×3 (groups=16) + PW Conv1×1
│   └── Conv3×3 + BN + ReLU + Dropout
│
├── C3: Dilated Convolutions (32→48, RF: 11→19)
│   ├── Conv3×3 + BN + ReLU + Dropout
│   ├── Dilated Conv3×3 (d=2) + BN + ReLU + Dropout
│   └── Conv3×3 + BN + ReLU + Dropout
│
├── C4: High Dilation Block (48→56, RF: 19→45)
│   ├── Dilated Conv3×3 (d=4) + BN + ReLU + Dropout
│   ├── Conv3×3 + BN + ReLU + Dropout
│   ├── Dilated Conv3×3 (d=8) + BN + ReLU + Dropout
│   └── Conv1×1 + BN + ReLU + Dropout
│
├── Global Average Pooling (32×32×56 → 1×1×56)
└── Linear Classifier (56 → 10)
```

### Advanced Features
- **Depthwise Separable Convolution**: Parameter-efficient feature extraction
- **Dilated Convolutions**: Multi-scale receptive field (d=2,4,8)
- **Global Average Pooling**: Spatial dimension reduction
- **No MaxPooling**: Uses dilated convolutions for downsampling

## 📊 Training Results

### Training Curves
![Training Curves](training_curves.png)

### Per-Class Accuracy
![Per-Class Accuracy](per_class_accuracy.png)

### Training Logs (Sample)
```
Epoch 1/30: Train=24.36%, Test=36.25%, LR=0.013793
Epoch 5/30: Train=58.04%, Test=67.17%, LR=0.120717
Epoch 10/30: Train=72.05%, Test=77.03%, LR=0.198877
Epoch 26/30: Train=85.23%, Test=86.62%, LR=0.045123
```

### Final Results
```
============================================================
FINAL RESULTS SUMMARY
============================================================
Best validation accuracy: 86.62%
Best epoch: 26
Target accuracy: 85.0%
Target achieved: ✓

Model Architecture Compliance:
✓ C1C2C3C4 structure: Implemented
✓ No MaxPooling: Implemented
✓ Depthwise Separable Convolution: Implemented
✓ Dilated Convolution: Implemented
✓ Global Average Pooling: Implemented
✓ FC layer after GAP: Implemented
✓ Albumentation augmentations: Implemented

Parameter count: 162,458 (< 200,000 requirement: ✓)
Receptive Field: 45 (> 44 requirement: ✓)

Data Augmentation Applied:
✓ Horizontal Flip: p=0.5
✓ ShiftScaleRotate: p=0.5
✓ CoarseDropout: p=0.3
============================================================
```

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/SachinDangayach/AU_7_NN_CIFAR.git
cd AU_7_NN_CIFAR
pip install -r requirements.txt
```

### Training
```bash
# Command line training
python main.py

# Interactive notebook
jupyter notebook interactive-training.ipynb
```

### Testing
```bash
# Test model architecture
python test_model.py
```

## 📁 Project Structure
```
AU_7_NN_CIFAR/
├── src/
│   ├── models/model.py          # C1C2C3C4 architecture
│   ├── data/data_manager.py     # Data loading & augmentation
│   ├── training/trainer.py      # Training pipeline
│   └── visualization/visualizer.py  # Plots & analysis
├── config.py                    # Configuration management
├── main.py                      # Main training script
├── interactive-training.ipynb   # Interactive training
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🔧 Configuration

### Model Configuration
```python
@dataclass
class ModelConfig:
    # Architecture
    c1_out_channels: int = 16
    c2_out_channels: int = 32
    c3_out_channels: int = 48
    c4_out_channels: int = 56
    
    # Advanced features
    c3_dilation: int = 2
    c4_dilation_1: int = 4
    c4_dilation_2: int = 8
    dropout_rate: float = 0.05
    
    # Constraints
    max_parameters: int = 200000
    min_receptive_field: int = 44
```

### Training Configuration
```python
@dataclass
class TrainingConfig:
    max_epochs: int = 30
    learning_rate: float = 0.003
    max_lr: float = 0.2
    scheduler_type: str = "OneCycleLR"
    target_test_accuracy: float = 85.0
```

## 📈 Performance Analysis

### Training Efficiency
- **Convergence**: Achieved 85% accuracy in 26 epochs
- **Learning Rate**: OneCycleLR (0.003 → 0.2 → 0.003)
- **Overfitting**: Minimal gap (85.23% train vs 86.62% test)
- **Stability**: Consistent improvement throughout training

### Architecture Efficiency
- **Parameter Efficiency**: 162,458 parameters (81% of limit)
- **Receptive Field**: 45 (exceeds 44 requirement)
- **Memory Usage**: Efficient with GAP instead of FC layers
- **Training Speed**: ~11-12 iterations/second on GPU

## 🛠️ Dependencies

### Core
- **PyTorch**: Deep learning framework
- **Torchvision**: Computer vision utilities
- **Albumentation**: Advanced data augmentation
- **NumPy**: Numerical computing

### Visualization
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical visualization
- **Jupyter**: Interactive notebook environment

## 📚 Key Features

### 1. Advanced Architecture
- C1C2C3C4 structure without MaxPooling
- Depthwise Separable Convolutions for parameter efficiency
- Dilated Convolutions for multi-scale feature extraction
- Global Average Pooling for spatial dimension reduction

### 2. Optimized Training
- OneCycleLR scheduler for fast convergence
- Smart early stopping with target-based termination
- Comprehensive metrics tracking and visualization
- Automatic best model checkpointing

### 3. Data Augmentation
- Albumentation library integration
- Horizontal flip, shift/scale/rotate, coarse dropout
- Dataset-specific normalization (CIFAR-10 mean/std)
- Configurable augmentation parameters

### 4. Interactive Development
- Jupyter notebook for end-to-end training
- Real-time training progress visualization
- Per-class accuracy analysis
- Misclassified image visualization

## 🎉 Project Highlights

- **Advanced Architecture**: C1C2C3C4 with modern CNN techniques
- **Parameter Efficiency**: 162,458 parameters (under 200k limit)
- **High Performance**: 86.62% accuracy (exceeds 85% target)
- **Fast Training**: 26 epochs with OneCycleLR scheduler
- **Comprehensive Analysis**: Interactive notebook with visualizations
- **Code Quality**: Modular design with comprehensive documentation

## 📝 License

This project is part of the EVA5 course curriculum and follows the specified requirements for advanced CIFAR-10 classification.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the model architecture
5. Submit a pull request

---

**Note**: This project successfully demonstrates advanced CNN architecture design, achieving high performance while maintaining parameter efficiency and code quality.