"""
Weight initialization utilities for better model training
"""

import torch
import torch.nn as nn
import math


def init_weights_xavier(module):
    """
    Initialize weights using Xavier/Glorot initialization
    Better for sigmoid/tanh activations
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


def init_weights_kaiming(module):
    """
    Initialize weights using Kaiming/He initialization
    Better for ReLU activations
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


def init_weights_small(module):
    """
    Initialize with smaller weights to start with lower outputs
    Useful for preventing saturation in MIL models
    """
    if isinstance(module, nn.Linear):
        # Use smaller standard deviation
        std = 0.02
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            # Initialize bias to negative values to start with lower probabilities
            nn.init.constant_(module.bias, -1.0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


def init_final_layer_negative(module, bias_value=-2.0):
    """
    Initialize final layer with negative bias to start with low probabilities
    """
    if isinstance(module, nn.Linear):
        # Small weights
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            # Negative bias to start with low outputs
            nn.init.constant_(module.bias, bias_value)