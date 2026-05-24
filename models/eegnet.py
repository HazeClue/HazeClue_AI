"""
EEGNet — Compact CNN for EEG Classification
=============================================
Lawhern et al. (2018) "EEGNet: A Compact Convolutional Neural Network
for EEG-based Brain–Computer Interfaces"

Architecture:
  Block 1: Temporal Conv → Depthwise Spatial Conv → BN → ELU → Pool → Dropout
  Block 2: Separable Conv → BN → ELU → Pool → Dropout
  Classifier: Linear

Input:  (batch, 1, n_channels, n_samples) = (B, 1, 14, 512)
Output: (batch, n_classes) = (B, 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet for 14-channel, 128 Hz, 4-second EEG windows.
    
    Parameters:
        n_channels: Number of EEG channels (14)
        n_samples: Number of time samples per window (512)
        n_classes: Number of output classes (2)
        F1: Number of temporal filters (8)
        D: Depth multiplier for depthwise conv (2)
        F2: Number of pointwise filters (16)
        dropout: Dropout rate (0.5)
    """
    
    def __init__(
        self,
        n_channels: int = 14,
        n_samples: int = 512,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        
        # Block 1: Temporal + Spatial filtering
        # Temporal conv: learn frequency filters
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Depthwise conv: learn spatial filters (per temporal feature)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        
        # Block 2: Separable convolution
        # Depthwise temporal
        self.separable_depth = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False
        )
        # Pointwise
        self.separable_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        
        # Classifier
        # After block1: (B, F1*D, 1, n_samples//4) = (B, 16, 1, 128)
        # After block2: (B, F2, 1, n_samples//32) = (B, 16, 1, 16)
        self.classifier = nn.Linear(F2 * (n_samples // 32), n_classes)
        
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Shape (batch, 1, n_channels, n_samples)
        
        Returns:
            logits: Shape (batch, n_classes)
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Block 2
        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Classify
        x = x.flatten(1)
        x = self.classifier(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ShallowConvNet(nn.Module):
    """
    Shallow ConvNet baseline (Schirrmeister et al., 2017).
    Simpler architecture, sometimes better for small datasets.
    
    Input:  (batch, 1, n_channels, n_samples)
    Output: (batch, n_classes)
    """
    
    def __init__(
        self,
        n_channels: int = 14,
        n_samples: int = 512,
        n_classes: int = 2,
        n_filters: int = 40,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # Temporal convolution
        self.conv_time = nn.Conv2d(1, n_filters, (1, 25), bias=False)
        # Spatial convolution
        self.conv_space = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        
        # Pool
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.drop = nn.Dropout(dropout)
        
        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            dummy = self.conv_time(dummy)
            dummy = self.conv_space(dummy)
            dummy = self.pool(dummy)
            flat_size = dummy.numel()
        
        self.classifier = nn.Linear(flat_size, n_classes)
    
    def forward(self, x):
        x = self.conv_time(x)
        x = self.conv_space(x)
        x = self.bn(x)
        x = x ** 2  # Square activation
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))  # Log activation
        x = self.drop(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
