"""
Advanced CIFAR-10 Model Architecture 

This module implements the C1C2C3C40 architecture for CIFAR-10 classification
with advanced convolution techniques including Depthwise Separable Convolution
and Dilated Convolution.

Architecture Requirements:
- C1C2C3C40 structure (No MaxPooling, stride=2 in last conv)
- Total Receptive Field > 44
- Depthwise Separable Convolution in one layer
- Dilated Convolution in one layer
- Global Average Pooling (compulsory)
- FC layer after GAP (optional)
- Parameters < 200k

RECEPTIVE FIELD CALCULATIONS:
=============================

Formula: RF_new = RF_old + (kernel_size - 1) * stride_old

Layer-by-Layer RF Calculation:
-----------------------------
Input: 32x32, RF = 1 (single pixel)

Conv Block 1 (C1):
- Conv1: 3x3 kernel, stride=1, padding=1
  RF = 1 + (3-1)*1 = 3
- Conv2: 3x3 kernel, stride=1, padding=1  
  RF = 3 + (3-1)*1 = 5
Output: 32x32, RF = 5

Conv Block 2 (C2) - Depthwise Separable:
- Depthwise: 3x3 kernel, stride=1, padding=1
  RF = 5 + (3-1)*1 = 7
- Pointwise: 1x1 kernel, stride=1, padding=0
  RF = 7 + (1-1)*1 = 7 (no change)
- Conv2: 3x3 kernel, stride=1, padding=1
  RF = 7 + (3-1)*1 = 9
Output: 32x32, RF = 9

Conv Block 3 (C3) - Dilated Convolution:
- Conv1: 3x3 kernel, dilation=2, stride=1, padding=2
  Effective kernel size = 3 + (3-1)*(2-1) = 5
  RF = 9 + (5-1)*1 = 13
- Conv2: 3x3 kernel, stride=1, padding=1
  RF = 13 + (3-1)*1 = 15
Output: 32x32, RF = 15

Conv Block 4 (C40) - Stride=2 instead of MaxPooling:
- Conv1: 3x3 kernel, stride=2, padding=1
  RF = 15 + (3-1)*2 = 19
- Conv2: 3x3 kernel, stride=1, padding=1
  RF = 19 + (3-1)*1 = 21
Output: 16x16, RF = 21

Conv Block 5 - Optimized: 2 stride=2 operations with minimum 2 conv gap:
- Conv1: 3x3 kernel, stride=1, padding=1
  RF = 21 + (3-1)*1 = 23
- Conv2: 3x3 kernel, stride=1, padding=1
  RF = 23 + (3-1)*1 = 25
- Conv3: 3x3 kernel, stride=2, padding=1 (stride=2 #1)
  RF = 25 + (3-1)*2 = 29
- Conv4: 3x3 kernel, stride=1, padding=1
  RF = 29 + (3-1)*1 = 31
- Conv5: 3x3 kernel, stride=1, padding=1
  RF = 31 + (3-1)*1 = 33
- Conv6: 3x3 kernel, stride=1, padding=1
  RF = 33 + (3-1)*1 = 35
- Conv7: 3x3 kernel, stride=2, padding=1 (stride=2 #2)
  RF = 35 + (3-1)*2 = 39
- Conv8: 3x3 kernel, stride=1, padding=1
  RF = 39 + (3-1)*1 = 41
- Conv9: 3x3 kernel, stride=1, padding=1
  RF = 41 + (3-1)*1 = 43
- Conv10: 3x3 kernel, stride=1, padding=1
  RF = 43 + (3-1)*1 = 45
Output: 4x4, RF = 45

Global Average Pooling:
- GAP: RF remains the same as input
  RF = 45 (no change)
Output: 1x1, RF = 45

FINAL RECEPTIVE FIELD: 45
Requirement: > 44
Status: ✅ Meets requirement

Note: The optimized architecture uses exactly 2 stride=2 operations with minimum 2 convolutions
between them, achieving RF=45 with 10 layers total.
""" 

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from config import ModelConfig


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution implementation.
    
    This reduces the number of parameters compared to standard convolution
    by separating the depthwise and pointwise operations.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Stride of the convolution
        padding: Padding added to both sides of the input
        dilation: Spacing between kernel elements
        bias: If True, adds a learnable bias to the output
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = False
    ):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution: each input channel is convolved separately
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # Groups = in_channels for depthwise
            bias=bias
        )
        
        # Pointwise convolution: 1x1 convolution to combine channels
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through depthwise separable convolution.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvBlock(nn.Module):
    """
    Standard convolution block with ReLU, BatchNorm, and Dropout.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Stride of the convolution
        padding: Padding added to both sides of the input
        dilation: Spacing between kernel elements
        dropout_rate: Dropout probability
        use_depthwise_separable: Whether to use depthwise separable convolution
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        dropout_rate: float = 0.1,
        use_depthwise_separable: bool = False
    ):
        super(ConvBlock, self).__init__()
        
        if use_depthwise_separable:
            self.conv = DepthwiseSeparableConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False
            )
        
        self.activation = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolution block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after convolution, activation, batch norm, and dropout
        """
        x = self.conv(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x


class CIFAR10Net(nn.Module):
    """
    Advanced CIFAR-10 Model with C1C2C3C40 Architecture.
    
    This model implements the required architecture with:
    - C1C2C3C40 structure (No MaxPooling, stride=2 in last conv)
    - Depthwise Separable Convolution in Conv Block 2
    - Dilated Convolution in Conv Block 3
    - Global Average Pooling with FC layer
    - Total Receptive Field > 44
    - Parameters < 200k
    
    Args:
        config: Model configuration parameters
    """
    
    def __init__(self, config: ModelConfig):
        super(CIFAR10Net, self).__init__()
        
        self.config = config
        
        # Conv Block 1 (C1) - Standard convolutions
        self.conv_block_1 = nn.Sequential(
            ConvBlock(
                in_channels=config.input_channels,
                out_channels=config.c1_out_channels,
                dropout_rate=config.dropout_rate
            ),
            ConvBlock(
                in_channels=config.c1_out_channels,
                out_channels=config.c1_out_channels,
                dropout_rate=config.dropout_rate
            )
        )
        
        # Conv Block 2 (C2) - Depthwise Separable Convolution
        self.conv_block_2 = nn.Sequential(
            ConvBlock(
                in_channels=config.c1_out_channels,
                out_channels=config.c2_out_channels,
                dropout_rate=config.dropout_rate,
                use_depthwise_separable=True
            ),
            ConvBlock(
                in_channels=config.c2_out_channels,
                out_channels=config.c2_out_channels,
                dropout_rate=config.dropout_rate
            )
        )
        
        # Conv Block 3 (C3) - Dilated Convolution
        self.conv_block_3 = nn.Sequential(
            ConvBlock(
                in_channels=config.c2_out_channels,
                out_channels=config.c3_out_channels,
                dilation=config.c3_dilation,
                padding=config.c3_dilation,  # Adjust padding for dilation
                dropout_rate=config.dropout_rate
            ),
            ConvBlock(
                in_channels=config.c3_out_channels,
                out_channels=config.c3_out_channels,
                dropout_rate=config.dropout_rate
            )
        )
        
        # Conv Block 4 (C40) - Stride=2 instead of MaxPooling
        self.conv_block_4 = nn.Sequential(
            ConvBlock(
                in_channels=config.c3_out_channels,
                out_channels=config.c4_out_channels,
                stride=config.c4_stride,
                dropout_rate=config.dropout_rate
            ),
            ConvBlock(
                in_channels=config.c4_out_channels,
                out_channels=config.c4_out_channels,
                dropout_rate=config.dropout_rate
            )
        )
        
        # Conv Block 5 - Optimized: 2 stride=2 operations with minimum 2 conv gap
        # Uses stride=2 exactly 2 times with minimum 2 convolutions between them
        # Stops when RF = 45 is achieved
        self.conv_block_5 = nn.Sequential(
            # First two convolutions: 16x16 -> 16x16
            ConvBlock(
                in_channels=config.c4_out_channels,
                out_channels=config.c5_out_channels,
                dropout_rate=config.dropout_rate
            ),
            ConvBlock(
                in_channels=config.c5_out_channels,
                out_channels=config.c5_out_channels,
                dropout_rate=config.dropout_rate
            ),
            # First stride=2: 16x16 -> 8x8
            ConvBlock(
                in_channels=config.c5_out_channels,
                out_channels=config.c5_out_channels,
                stride=2,
                dropout_rate=config.dropout_rate
            ),
            # Minimum 2 convolutions between stride=2 operations
            ConvBlock(
                in_channels=config.c5_out_channels,
                out_channels=config.c5_out_channels,
                dropout_rate=config.dropout_rate
            ),
            ConvBlock(
                in_channels=config.c5_out_channels,
                out_channels=config.c5_out_channels,
                dropout_rate=config.dropout_rate
            ),
            ConvBlock(
                in_channels=config.c5_out_channels,
                out_channels=config.c5_out_channels,
                dropout_rate=config.dropout_rate
            ),
            # Second stride=2: 8x8 -> 4x4
            ConvBlock(
                in_channels=config.c5_out_channels,
                out_channels=config.c5_out_channels,
                stride=2,
                dropout_rate=config.dropout_rate
            ),
            # Final convolutions to reach RF=45
            ConvBlock(
                in_channels=config.c5_out_channels,
                out_channels=config.c5_out_channels,
                dropout_rate=config.dropout_rate
            ),
            ConvBlock(
                in_channels=config.c5_out_channels,
                out_channels=config.c5_out_channels,
                dropout_rate=config.dropout_rate
            ),
            ConvBlock(
                in_channels=config.c5_out_channels,
                out_channels=config.c5_out_channels,
                dropout_rate=config.dropout_rate
            )
        )
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully Connected layer after GAP
        self.classifier = nn.Linear(config.c5_out_channels, config.num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the entire model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Output tensor of shape (batch_size, num_classes) with log probabilities
        """
        # INPUT: 32x32, RF = 1 (single pixel)
        
        # Conv Block 1: 32x32 -> 32x32
        # Conv1: 3x3 kernel, RF = 1 + (3-1)*1 = 3
        # Conv2: 3x3 kernel, RF = 3 + (3-1)*1 = 5
        x = self.conv_block_1(x)  # Output: 32x32, RF = 5
        
        # Conv Block 2: 32x32 -> 32x32 (Depthwise Separable)
        # Depthwise: 3x3 kernel, RF = 5 + (3-1)*1 = 7
        # Pointwise: 1x1 kernel, RF = 7 + (1-1)*1 = 7
        # Conv2: 3x3 kernel, RF = 7 + (3-1)*1 = 9
        x = self.conv_block_2(x)  # Output: 32x32, RF = 9
        
        # Conv Block 3: 32x32 -> 32x32 (Dilated Convolution)
        # Conv1: 3x3 kernel with dilation=2, effective kernel=5, RF = 9 + (5-1)*1 = 13
        # Conv2: 3x3 kernel, RF = 13 + (3-1)*1 = 15
        x = self.conv_block_3(x)  # Output: 32x32, RF = 15
        
        # Conv Block 4: 32x32 -> 16x16 (Stride=2 instead of MaxPooling)
        # Conv1: 3x3 kernel with stride=2, RF = 15 + (3-1)*2 = 19
        # Conv2: 3x3 kernel, RF = 19 + (3-1)*1 = 21
        x = self.conv_block_4(x)  # Output: 16x16, RF = 21
        
        # Conv Block 5: 16x16 -> 4x4 (Optimized: 2 stride=2 operations)
        # Conv1: 3x3 kernel, stride=1, RF = 21 + (3-1)*1 = 23
        # Conv2: 3x3 kernel, stride=1, RF = 23 + (3-1)*1 = 25
        # Conv3: 3x3 kernel, stride=2, RF = 25 + (3-1)*2 = 29 (stride=2 #1)
        # Conv4: 3x3 kernel, stride=1, RF = 29 + (3-1)*1 = 31
        # Conv5: 3x3 kernel, stride=1, RF = 31 + (3-1)*1 = 33
        # Conv6: 3x3 kernel, stride=1, RF = 33 + (3-1)*1 = 35
        # Conv7: 3x3 kernel, stride=2, RF = 35 + (3-1)*2 = 39 (stride=2 #2)
        # Conv8: 3x3 kernel, stride=1, RF = 39 + (3-1)*1 = 41
        # Conv9: 3x3 kernel, stride=1, RF = 41 + (3-1)*1 = 43
        # Conv10: 3x3 kernel, stride=1, RF = 43 + (3-1)*1 = 45
        x = self.conv_block_5(x)  # Output: 4x4, RF = 45
        
        # Global Average Pooling: 4x4 -> 1x1
        # GAP: RF remains the same as input, RF = 45
        x = self.global_avg_pool(x)  # Output: 1x1, RF = 45
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.classifier(x)  # FC layer doesn't change RF
        
        # Return log probabilities
        return F.log_softmax(x, dim=-1)
    
    def get_receptive_field_info(self) -> dict:
        """
        Get receptive field information for each layer.
        
        Returns:
            Dictionary containing receptive field information
        """
        return {
            "Conv Block 1": {"output_size": "32x32", "receptive_field": 5},
            "Conv Block 2": {"output_size": "32x32", "receptive_field": 9},
            "Conv Block 3": {"output_size": "32x32", "receptive_field": 15},
            "Conv Block 4": {"output_size": "16x16", "receptive_field": 21},
            "Conv Block 5": {"output_size": "4x4", "receptive_field": 45},
            "Global Avg Pool": {"output_size": "1x1", "receptive_field": 45},
            "Total RF": 45
        }


def count_model_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(config: ModelConfig) -> CIFAR10Net:
    """
    Create and initialize the Advanced CIFAR-10 model.
    
    Args:
        config: Model configuration parameters
        
    Returns:
        Initialized model instance
    """
    model = CIFAR10Net(config)
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)
    
    return model


if __name__ == "__main__":
    # Test the model
    from config import get_config
    
    config = get_config()
    model = create_model(config.model)
    
    total_params = count_model_parameters(model)
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Parameter requirement (< {config.model.max_parameters:,}): {'✓' if total_params < config.model.max_parameters else '✗'}")
    
    # Test forward pass
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Print receptive field info
    rf_info = model.get_receptive_field_info()
    print(f"\nReceptive Field Information:")
    for layer, info in rf_info.items():
        if layer != "Total RF":
            print(f"  {layer}: {info['output_size']}, RF = {info['receptive_field']}")
        else:
            print(f"  {layer}: {info}")
    
    print(f"RF requirement (> {config.model.min_receptive_field}): {'✓' if rf_info['Total RF'] > config.model.min_receptive_field else '✗'}")
