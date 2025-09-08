"""
Encoder architectures for protein embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import logging


class ProteinEncoder(nn.Module):
    """
    Single tower encoder for protein embeddings
    Maps protein embeddings to a shared representation space
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 use_layer_norm: bool = True,
                 name: str = "protein_encoder"):
        """
        Initialize the protein encoder
        
        Args:
            input_dim: Input dimension (ESM-2 embedding size)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            activation: Activation function ('relu', 'gelu', 'tanh')
            use_layer_norm: Whether to use layer normalization
            name: Name of the encoder (for logging)
        """
        super().__init__()
        
        self.name = name
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = hidden_dims[-1]
        
        # Build encoder layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Layer normalization (before activation if used)
            if use_layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            
            # Activation (except for last layer)
            if i < len(dims) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
                
                # Dropout (except for last layer)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        # Final layer norm (always applied to output)
        if use_layer_norm and not isinstance(layers[-1], nn.LayerNorm):
            layers.append(nn.LayerNorm(dims[-1]))
            
        self.encoder = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        # Log architecture
        logger = logging.getLogger(__name__)
        logger.info(f"{name} architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))}")
        
    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the encoder
        
        Args:
            x: Input tensor of shape (batch_size, n_proteins, input_dim)
            mask: Optional mask tensor of shape (batch_size, n_proteins)
                  1 for real proteins, 0 for padding
        
        Returns:
            Encoded tensor of shape (batch_size, n_proteins, output_dim)
        """
        # Get original shape
        batch_size, n_proteins, _ = x.shape
        
        # Reshape for processing
        x_flat = x.view(-1, self.input_dim)
        
        # Encode
        encoded_flat = self.encoder(x_flat)
        
        # Reshape back
        encoded = encoded_flat.view(batch_size, n_proteins, self.output_dim)
        
        # Apply mask if provided (zero out padded positions)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (B, n_proteins, 1)
            encoded = encoded * mask_expanded
            
        return encoded


class TwoTowerEncoder(nn.Module):
    """
    Two-tower encoder architecture for phage and host proteins
    Each tower has its own encoder but maps to the same embedding space
    """
    
    def __init__(self,
                 input_dim: int,
                 marker_hidden_dims: List[int],
                 rbp_hidden_dims: Optional[List[int]] = None,
                 shared_architecture: bool = True,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 use_layer_norm: bool = True):
        """
        Initialize the two-tower encoder
        
        Args:
            input_dim: Input dimension (ESM-2 embedding size)
            marker_hidden_dims: Hidden dimensions for marker (host) encoder
            rbp_hidden_dims: Hidden dimensions for RBP (phage) encoder
                           If None and shared_architecture=True, uses marker_hidden_dims
            shared_architecture: Whether to use the same architecture for both towers
            dropout: Dropout rate
            activation: Activation function
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        # Set RBP dimensions
        if shared_architecture or rbp_hidden_dims is None:
            rbp_hidden_dims = marker_hidden_dims
            
        # Ensure both encoders output to the same dimension
        if marker_hidden_dims[-1] != rbp_hidden_dims[-1]:
            raise ValueError(f"Both encoders must have the same output dimension. "
                           f"Got marker: {marker_hidden_dims[-1]}, rbp: {rbp_hidden_dims[-1]}")
        
        self.output_dim = marker_hidden_dims[-1]
        
        # Create encoders
        self.marker_encoder = ProteinEncoder(
            input_dim=input_dim,
            hidden_dims=marker_hidden_dims,
            dropout=dropout,
            activation=activation,
            use_layer_norm=use_layer_norm,
            name="marker_encoder"
        )
        
        if shared_architecture:
            # Share weights between encoders
            self.rbp_encoder = self.marker_encoder
            logger = logging.getLogger(__name__)
            logger.info("Using shared architecture for both towers")
        else:
            # Separate encoders
            self.rbp_encoder = ProteinEncoder(
                input_dim=input_dim,
                hidden_dims=rbp_hidden_dims,
                dropout=dropout,
                activation=activation,
                use_layer_norm=use_layer_norm,
                name="rbp_encoder"
            )
            
    def forward(self,
                marker_embeddings: torch.Tensor,
                rbp_embeddings: torch.Tensor,
                marker_mask: Optional[torch.Tensor] = None,
                rbp_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both towers
        
        Args:
            marker_embeddings: Marker embeddings (B, n_markers, input_dim)
            rbp_embeddings: RBP embeddings (B, n_rbps, input_dim)
            marker_mask: Mask for markers (B, n_markers)
            rbp_mask: Mask for RBPs (B, n_rbps)
            
        Returns:
            Tuple of (encoded_markers, encoded_rbps)
            Both of shape (B, n_proteins, output_dim)
        """
        # Encode markers (host proteins)
        encoded_markers = self.marker_encoder(marker_embeddings, marker_mask)
        
        # Encode RBPs (phage proteins)
        encoded_rbps = self.rbp_encoder(rbp_embeddings, rbp_mask)
        
        return encoded_markers, encoded_rbps
    
    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension"""
        return self.output_dim