#!/usr/bin/env python3
"""
Custom Model Architectures for TorchVision Datasets.

Provides custom neural network architectures (SimpleCNN, LeNet5) optimized
for smaller datasets like MNIST, along with code generation utilities.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

import inspect

import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN that adapts to standard preprocessing (3-channel RGB, 224×224).
    
    Compatible with the same preprocessing pipeline as standard TorchVision models.
    """
    
    def __init__(self, num_classes: int = 10, weights=None):
        """
        Args:
            num_classes: Number of output classes
            weights: Ignored (custom model has no pre-trained weights)
        """
        super(SimpleCNN, self).__init__()
        # Accept 3 channels (RGB) after standard preprocessing
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # After 2 pooling layers on 224x224: 224 -> 112 -> 56
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LeNet5(nn.Module):
    """
    LeNet-5 architecture adapted for standard preprocessing (3-channel RGB, 224×224).
    
    Compatible with the same preprocessing pipeline as standard TorchVision models.
    """
    
    def __init__(self, num_classes: int = 10, weights=None):
        """
        Args:
            num_classes: Number of output classes
            weights: Ignored (custom model has no pre-trained weights)
        """
        super(LeNet5, self).__init__()
        # Accept 3 channels (RGB) after standard preprocessing
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # After conv1 (224->220) + pool (220->110) + conv2 (110->106) + pool (106->53)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.pool = nn.AvgPool2d(2, 2)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.pool(self.tanh(self.conv1(x)))
        x = self.pool(self.tanh(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


_CUSTOM_MODELS = {"simple_cnn": SimpleCNN, "lenet5": LeNet5}


def _get_custom_model_definition(model_name: str, num_classes: int) -> str:
    """Return the Python source written to models/<name>.py for a custom model."""
    if model_name not in _CUSTOM_MODELS:
        raise ValueError(f"Unknown custom model: {model_name}")
    model_class = _CUSTOM_MODELS[model_name]
    return (
        f"import torch.nn as nn\n\n{inspect.getsource(model_class)}\n"
        f"# Create model\nmodel = {model_class.__name__}(num_classes={num_classes})\n"
    )
