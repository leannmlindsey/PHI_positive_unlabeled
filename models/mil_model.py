"""
Multi-Instance Learning Model with Noisy-OR Aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging
import math
from functools import wraps

from .encoders import TwoTowerEncoder
from .init_utils import init_weights_small
from .scorer import CosineScorer, StableNoisyOR, init_pair_bias


def validate_outputs(func):
    """
    Decorator to validate output tensors from model forward pass
    
    Validates:
    - Output dictionary structure
    - NaN/Inf values in outputs
    - Probability ranges [0, 1]
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        outputs = func(self, *args, **kwargs)
        
        # Check that outputs is a dictionary
        assert isinstance(outputs, dict), \
            f"Expected outputs to be a dictionary, got {type(outputs)}"
        
        # Validate bag probabilities
        if 'bag_probs' in outputs:
            bag_probs = outputs['bag_probs']
            assert torch.is_tensor(bag_probs), \
                "bag_probs must be a tensor"
            assert not torch.isnan(bag_probs).any(), \
                "bag_probs contains NaN values"
            assert not torch.isinf(bag_probs).any(), \
                "bag_probs contains Inf values"
            assert (bag_probs >= 0).all() and (bag_probs <= 1).all(), \
                f"bag_probs must be in [0, 1], got range [{bag_probs.min():.4f}, {bag_probs.max():.4f}]"
        
        # Validate pairwise scores if present
        if 'pairwise_scores' in outputs:
            pairwise_scores = outputs['pairwise_scores']
            assert torch.is_tensor(pairwise_scores), \
                "pairwise_scores must be a tensor"
            assert not torch.isnan(pairwise_scores).any(), \
                "pairwise_scores contains NaN values"
            assert not torch.isinf(pairwise_scores).any(), \
                "pairwise_scores contains Inf values"
        
        # Validate pairwise probabilities if present
        if 'pairwise_probs' in outputs:
            pairwise_probs = outputs['pairwise_probs']
            assert torch.is_tensor(pairwise_probs), \
                "pairwise_probs must be a tensor"
            assert not torch.isnan(pairwise_probs).any(), \
                "pairwise_probs contains NaN values"
            assert not torch.isinf(pairwise_probs).any(), \
                "pairwise_probs contains Inf values"
            assert (pairwise_probs >= 0).all() and (pairwise_probs <= 1).all(), \
                f"pairwise_probs must be in [0, 1], got range [{pairwise_probs.min():.4f}, {pairwise_probs.max():.4f}]"
        
        # Validate embeddings if present
        for embed_key in ['marker_encoded', 'rbp_encoded']:
            if embed_key in outputs:
                embeddings = outputs[embed_key]
                assert torch.is_tensor(embeddings), \
                    f"{embed_key} must be a tensor"
                assert not torch.isnan(embeddings).any(), \
                    f"{embed_key} contains NaN values"
                assert not torch.isinf(embeddings).any(), \
                    f"{embed_key} contains Inf values"
        
        return outputs
    
    return wrapper


def validate_inputs(func):
    """
    Decorator to validate input tensors for model forward pass
    
    Validates:
    - Tensor dimensions
    - NaN/Inf values
    - Shape consistency
    - Mask validity
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Handle both positional and keyword arguments
        if len(args) >= 2:
            marker_embeddings = args[0]
            rbp_embeddings = args[1]
            marker_mask = args[2] if len(args) > 2 else None
            rbp_mask = args[3] if len(args) > 3 else None
        else:
            marker_embeddings = kwargs.get('marker_embeddings')
            rbp_embeddings = kwargs.get('rbp_embeddings')
            marker_mask = kwargs.get('marker_mask')
            rbp_mask = kwargs.get('rbp_mask')
        
        # Validate marker embeddings
        if marker_embeddings is not None:
            assert marker_embeddings.dim() == 3, \
                f"Expected marker_embeddings to be 3D (batch, n_markers, embed_dim), got {marker_embeddings.dim()}D"
            assert not torch.isnan(marker_embeddings).any(), \
                "marker_embeddings contains NaN values"
            assert not torch.isinf(marker_embeddings).any(), \
                "marker_embeddings contains Inf values"
            assert marker_embeddings.size(0) > 0, \
                "Batch size must be greater than 0"
        
        # Validate RBP embeddings
        if rbp_embeddings is not None:
            assert rbp_embeddings.dim() == 3, \
                f"Expected rbp_embeddings to be 3D (batch, n_rbps, embed_dim), got {rbp_embeddings.dim()}D"
            assert not torch.isnan(rbp_embeddings).any(), \
                "rbp_embeddings contains NaN values"
            assert not torch.isinf(rbp_embeddings).any(), \
                "rbp_embeddings contains Inf values"
            
            # Check batch size consistency
            if marker_embeddings is not None:
                assert marker_embeddings.size(0) == rbp_embeddings.size(0), \
                    f"Batch size mismatch: markers={marker_embeddings.size(0)}, rbps={rbp_embeddings.size(0)}"
        
        # Validate marker mask
        if marker_mask is not None:
            assert marker_mask.dim() in [1, 2], \
                f"Expected marker_mask to be 1D or 2D, got {marker_mask.dim()}D"
            if marker_mask.dim() == 1:
                # Convert to 2D for consistency
                marker_mask = marker_mask.unsqueeze(0)
            assert not torch.isnan(marker_mask).any(), \
                "marker_mask contains NaN values"
            assert ((marker_mask == 0) | (marker_mask == 1)).all(), \
                "marker_mask must be binary (0 or 1)"
            
            # Check shape consistency
            if marker_embeddings is not None:
                expected_shape = (marker_embeddings.size(0), marker_embeddings.size(1))
                assert marker_mask.shape == expected_shape, \
                    f"marker_mask shape {marker_mask.shape} doesn't match expected {expected_shape}"
        
        # Validate RBP mask
        if rbp_mask is not None:
            assert rbp_mask.dim() in [1, 2], \
                f"Expected rbp_mask to be 1D or 2D, got {rbp_mask.dim()}D"
            if rbp_mask.dim() == 1:
                # Convert to 2D for consistency
                rbp_mask = rbp_mask.unsqueeze(0)
            assert not torch.isnan(rbp_mask).any(), \
                "rbp_mask contains NaN values"
            assert ((rbp_mask == 0) | (rbp_mask == 1)).all(), \
                "rbp_mask must be binary (0 or 1)"
            
            # Check shape consistency
            if rbp_embeddings is not None:
                expected_shape = (rbp_embeddings.size(0), rbp_embeddings.size(1))
                assert rbp_mask.shape == expected_shape, \
                    f"rbp_mask shape {rbp_mask.shape} doesn't match expected {expected_shape}"
        
        return func(self, *args, **kwargs)
    
    return wrapper


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
            # Ensure mask has same shape as pairwise_probs
            assert mask.shape == pairwise_probs.shape, \
                f"Mask shape {mask.shape} doesn't match pairwise_probs shape {pairwise_probs.shape}"
            # Convert to float if needed
            if mask.dtype != torch.float32:
                mask = mask.float()
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
        
        # Cosine similarity scorer with learnable scale and bias
        # Initialize bias based on expected bag prior (0.5 for balanced training)
        init_bias = init_pair_bias(bag_prior=0.5, mean_num_pairs=20)  # Assuming ~20 valid pairs per bag
        self.scorer = CosineScorer(init_scale=1.0, init_bias=init_bias)
        
        # Stable Noisy-OR aggregation (works in log space)
        self.noisy_or = StableNoisyOR(epsilon=epsilon)
        
        # Initialize weights with smaller values to prevent saturation
        self.apply(init_weights_small)
        
        # Logging
        logger = logging.getLogger(__name__)
        logger.info(f"MILModel initialized with embedding_dim={self.embedding_dim}, "
                   f"temperature={temperature}, using small weight initialization")
        
    def compute_pairwise_scores(self,
                                encoded_markers: torch.Tensor,
                                encoded_rbps: torch.Tensor,
                                marker_mask: Optional[torch.Tensor] = None,
                                rbp_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute pairwise scores between all marker-RBP pairs using cosine similarity
        
        Args:
            encoded_markers: Encoded marker proteins (B, n_markers, d)
            encoded_rbps: Encoded RBP proteins (B, n_rbps, d)
            marker_mask: Mask for markers (B, n_markers) or (n_markers,)
            rbp_mask: Mask for RBPs (B, n_rbps) or (n_rbps,)
            
        Returns:
            Tuple of (scores, mask)
            - scores: Pairwise scores (B, n_markers, n_rbps)
            - mask: Combined mask (B, n_markers, n_rbps)
        """
        # Ensure proper dimensions for masks
        if marker_mask is not None:
            # Convert to float for multiplication
            marker_mask = marker_mask.float()
            # Add batch dimension if missing
            if marker_mask.dim() == 1:
                marker_mask = marker_mask.unsqueeze(0)
            # Ensure batch size matches
            if marker_mask.size(0) == 1 and encoded_markers.size(0) > 1:
                marker_mask = marker_mask.expand(encoded_markers.size(0), -1)
        
        if rbp_mask is not None:
            # Convert to float for multiplication
            rbp_mask = rbp_mask.float()
            # Add batch dimension if missing
            if rbp_mask.dim() == 1:
                rbp_mask = rbp_mask.unsqueeze(0)
            # Ensure batch size matches
            if rbp_mask.size(0) == 1 and encoded_rbps.size(0) > 1:
                rbp_mask = rbp_mask.expand(encoded_rbps.size(0), -1)
        
        # Use CosineScorer to compute logits
        # This normalizes embeddings and applies learnable scale and bias
        scores = self.scorer(encoded_markers, encoded_rbps)
        
        # Create combined mask with proper shape checking
        if marker_mask is not None and rbp_mask is not None:
            # Ensure masks have compatible shapes
            assert marker_mask.size(0) == rbp_mask.size(0), \
                f"Batch size mismatch: marker_mask {marker_mask.shape} vs rbp_mask {rbp_mask.shape}"
            # Expand masks and multiply
            # (B, n_markers, 1) * (B, 1, n_rbps) -> (B, n_markers, n_rbps)
            mask = marker_mask.unsqueeze(2) * rbp_mask.unsqueeze(1)
        elif marker_mask is not None:
            # Only marker mask available
            mask = marker_mask.unsqueeze(2).expand(-1, -1, scores.size(2))
        elif rbp_mask is not None:
            # Only RBP mask available
            mask = rbp_mask.unsqueeze(1).expand(-1, scores.size(1), -1)
        else:
            # No masks provided
            mask = None
            
        return scores, mask
    
    @validate_inputs
    @validate_outputs
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