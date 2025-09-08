"""
Multi-Instance Learning Model with Noisy-OR Aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging
import math

from .encoders import TwoTowerEncoder


class NoisyORLayer(nn.Module):
    """
    Noisy-OR aggregation layer for multi-instance learning
    Computes bag-level probability from instance-level probabilities
    """
    
    def __init__(self, epsilon: float = 1e-7):
        """
        Initialize the Noisy-OR layer
        
        Args:
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, 
                pairwise_probs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute bag-level probability using Noisy-OR
        
        P_bag = 1 - ∏(1 - P_ij) for all valid pairs
        
        Args:
            pairwise_probs: Pairwise probabilities (B, n_markers, n_rbps)
            mask: Binary mask for valid pairs (B, n_markers, n_rbps)
            
        Returns:
            Bag-level probabilities (B,)
        """
        # Apply mask if provided
        if mask is not None:
            # Set masked positions to 0 (so 1-p becomes 1)
            pairwise_probs = pairwise_probs * mask
            
        # Compute 1 - P for each pair
        one_minus_p = 1.0 - pairwise_probs
        
        # Add epsilon for numerical stability
        one_minus_p = torch.clamp(one_minus_p, min=self.epsilon, max=1.0 - self.epsilon)
        
        # Compute log(1 - P) for numerical stability
        log_one_minus_p = torch.log(one_minus_p)
        
        # Sum over all pairs (in log space = product in probability space)
        # Shape: (B, n_markers, n_rbps) -> (B,)
        sum_log = log_one_minus_p.sum(dim=(1, 2))
        
        # Compute bag probability: 1 - exp(sum_log)
        bag_probs = 1.0 - torch.exp(sum_log)
        
        # Clamp for numerical stability
        bag_probs = torch.clamp(bag_probs, min=self.epsilon, max=1.0 - self.epsilon)
        
        return bag_probs


class MILModel(nn.Module):
    """
    Multi-Instance Learning model for phage-host interaction prediction
    Combines two-tower encoders with noisy-OR aggregation
    """
    
    def __init__(self,
                 input_dim: int = 1280,
                 encoder_dims: list = [768, 512, 256],
                 shared_architecture: bool = True,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 use_layer_norm: bool = True,
                 temperature: float = 1.0,
                 epsilon: float = 1e-7):
        """
        Initialize the MIL model
        
        Args:
            input_dim: Input dimension (ESM-2 embedding size)
            encoder_dims: Hidden dimensions for encoders
            shared_architecture: Whether to share weights between towers
            dropout: Dropout rate
            activation: Activation function
            use_layer_norm: Whether to use layer normalization
            temperature: Temperature for sigmoid scaling
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        
        self.temperature = temperature
        self.epsilon = epsilon
        
        # Two-tower encoder
        self.encoder = TwoTowerEncoder(
            input_dim=input_dim,
            marker_hidden_dims=encoder_dims,
            rbp_hidden_dims=encoder_dims if shared_architecture else encoder_dims,
            shared_architecture=shared_architecture,
            dropout=dropout,
            activation=activation,
            use_layer_norm=use_layer_norm
        )
        
        self.embedding_dim = self.encoder.get_embedding_dim()
        
        # Noisy-OR aggregation
        self.noisy_or = NoisyORLayer(epsilon=epsilon)
        
        # Logging
        logger = logging.getLogger(__name__)
        logger.info(f"MILModel initialized with embedding_dim={self.embedding_dim}, "
                   f"temperature={temperature}")
        
    def compute_pairwise_scores(self,
                                encoded_markers: torch.Tensor,
                                encoded_rbps: torch.Tensor,
                                marker_mask: Optional[torch.Tensor] = None,
                                rbp_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute pairwise scores between all marker-RBP pairs
        
        Args:
            encoded_markers: Encoded marker proteins (B, n_markers, d)
            encoded_rbps: Encoded RBP proteins (B, n_rbps, d)
            marker_mask: Mask for markers (B, n_markers)
            rbp_mask: Mask for RBPs (B, n_rbps)
            
        Returns:
            Tuple of (scores, mask)
            - scores: Pairwise scores (B, n_markers, n_rbps)
            - mask: Combined mask (B, n_markers, n_rbps)
        """
        # Compute scaled dot product
        # (B, n_markers, d) @ (B, d, n_rbps) -> (B, n_markers, n_rbps)
        scores = torch.bmm(encoded_markers, encoded_rbps.transpose(1, 2))
        
        # Scale by sqrt(d) for stability
        scores = scores / math.sqrt(self.embedding_dim)
        
        # Apply temperature scaling
        scores = scores * self.temperature
        
        # Create combined mask
        if marker_mask is not None and rbp_mask is not None:
            # Expand masks and multiply
            # (B, n_markers, 1) * (B, 1, n_rbps) -> (B, n_markers, n_rbps)
            mask = marker_mask.unsqueeze(2) * rbp_mask.unsqueeze(1)
        elif marker_mask is not None:
            mask = marker_mask.unsqueeze(2)
        elif rbp_mask is not None:
            mask = rbp_mask.unsqueeze(1)
        else:
            mask = None
            
        return scores, mask
    
    def forward(self,
                marker_embeddings: torch.Tensor,
                rbp_embeddings: torch.Tensor,
                marker_mask: Optional[torch.Tensor] = None,
                rbp_mask: Optional[torch.Tensor] = None,
                return_pairwise: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            marker_embeddings: Marker protein embeddings (B, n_markers, input_dim)
            rbp_embeddings: RBP embeddings (B, n_rbps, input_dim)
            marker_mask: Mask for markers (B, n_markers)
            rbp_mask: Mask for RBPs (B, n_rbps)
            return_pairwise: Whether to return pairwise probabilities
            
        Returns:
            Dictionary containing:
            - 'bag_probs': Bag-level probabilities (B,)
            - 'pairwise_probs': Pairwise probabilities (B, n_markers, n_rbps) if requested
            - 'encoded_markers': Encoded markers (B, n_markers, d) if requested
            - 'encoded_rbps': Encoded RBPs (B, n_rbps, d) if requested
        """
        # Encode proteins
        encoded_markers, encoded_rbps = self.encoder(
            marker_embeddings, 
            rbp_embeddings,
            marker_mask,
            rbp_mask
        )
        
        # Compute pairwise scores
        scores, combined_mask = self.compute_pairwise_scores(
            encoded_markers,
            encoded_rbps,
            marker_mask,
            rbp_mask
        )
        
        # Convert scores to probabilities
        pairwise_probs = torch.sigmoid(scores)
        
        # Apply mask to probabilities
        if combined_mask is not None:
            pairwise_probs = pairwise_probs * combined_mask
            
        # Aggregate to bag level using Noisy-OR
        bag_probs = self.noisy_or(pairwise_probs, combined_mask)
        
        # Prepare output
        output = {'bag_probs': bag_probs}
        
        if return_pairwise:
            output['pairwise_probs'] = pairwise_probs
            output['encoded_markers'] = encoded_markers
            output['encoded_rbps'] = encoded_rbps
            
        return output
    
    def predict(self,
                marker_embeddings: torch.Tensor,
                rbp_embeddings: torch.Tensor,
                marker_mask: Optional[torch.Tensor] = None,
                rbp_mask: Optional[torch.Tensor] = None,
                threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Make predictions with the model
        
        Args:
            marker_embeddings: Marker protein embeddings
            rbp_embeddings: RBP embeddings
            marker_mask: Mask for markers
            rbp_mask: Mask for RBPs
            threshold: Classification threshold
            
        Returns:
            Dictionary containing:
            - 'predictions': Binary predictions (B,)
            - 'probabilities': Bag-level probabilities (B,)
            - 'pairwise_probs': Pairwise probabilities (B, n_markers, n_rbps)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(
                marker_embeddings,
                rbp_embeddings,
                marker_mask,
                rbp_mask,
                return_pairwise=True
            )
            
            probabilities = output['bag_probs']
            predictions = (probabilities > threshold).float()
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'pairwise_probs': output['pairwise_probs']
            }