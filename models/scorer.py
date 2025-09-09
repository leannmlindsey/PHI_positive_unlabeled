"""
Improved scoring functions for MIL model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CosineScorer(nn.Module):
    """
    Cosine similarity scorer with learnable scale and bias
    Prevents dot product explosion by using normalized vectors
    """
    
    def __init__(self, init_scale=1.0, init_bias=-3.0):
        """
        Initialize the cosine scorer
        
        Args:
            init_scale: Initial scale for logits (kept small)
            init_bias: Initial bias (set based on bag prior)
        """
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))
    
    def forward(self, marker_embeddings, rbp_embeddings):
        """
        Compute cosine similarity scores
        
        Args:
            marker_embeddings: [B, M, D] marker embeddings
            rbp_embeddings: [B, R, D] RBP embeddings
            
        Returns:
            logits: [B, M, R] pairwise logits
        """
        # L2 normalize to unit vectors
        m = F.normalize(marker_embeddings, p=2, dim=-1)  # [B, M, D]
        r = F.normalize(rbp_embeddings, p=2, dim=-1)     # [B, R, D]
        
        # Compute cosine similarity in [-1, 1]
        cos = torch.einsum('bmd,brd->bmr', m, r)  # [B, M, R]
        
        # Apply learnable scale and bias
        # Clamp scale to prevent runaway during early training
        scale = self.logit_scale.clamp(0.1, 10.0)
        logits = scale * cos + self.bias
        
        return logits


def init_pair_bias(bag_prior, mean_num_pairs):
    """
    Initialize bias to match desired bag prior
    
    With Noisy-OR aggregation, if each pair has probability q and there are K pairs:
    P_bag = 1 - (1 - q)^K
    Therefore: q = 1 - (1 - P_bag)^(1/K)
    
    Args:
        bag_prior: Desired prior probability for positive bags
        mean_num_pairs: Average number of valid pairs per bag
        
    Returns:
        bias: Initial bias for logits
    """
    # Compute per-pair probability needed for desired bag prior
    q = 1.0 - (1.0 - bag_prior) ** (1.0 / max(mean_num_pairs, 1.0))
    # Clamp for numerical stability
    q = min(max(q, 1e-4), 1 - 1e-4)
    # Convert to logit space
    bias = math.log(q / (1 - q))
    return bias


class StableNoisyOR(nn.Module):
    """
    Numerically stable Noisy-OR aggregation working in log space
    """
    
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, logits, mask=None):
        """
        Compute Noisy-OR aggregation in log space for numerical stability
        
        Args:
            logits: [B, M, R] pairwise logits
            mask: [B, M, R] binary mask (1 for valid pairs, 0 for invalid)
            
        Returns:
            bag_probs: [B] bag-level probabilities
        """
        # Use identity: 1 - σ(ℓ) = σ(-ℓ)
        # log σ(-ℓ) is stable for large |ℓ|
        log_one_minus_p = F.logsigmoid(-logits)  # [B, M, R]
        
        # Apply mask: invalid pairs contribute log(1) = 0 to the sum
        if mask is not None:
            log_one_minus_p = log_one_minus_p * mask
        
        # Sum over all pairs
        sum_log = log_one_minus_p.sum(dim=(1, 2))  # [B]
        
        # Compute bag probability: P = 1 - exp(sum_log)
        # Clamp to avoid under/overflow
        p_neg = torch.exp(sum_log.clamp(min=-50.0, max=0.0))
        p_pos = (1.0 - p_neg).clamp(self.epsilon, 1 - self.epsilon)
        
        return p_pos


class MaxAggregator(nn.Module):
    """
    Simple max aggregation as an alternative to Noisy-OR
    Often more stable for initial training
    """
    
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, probs, mask=None):
        """
        Take maximum probability across all valid pairs
        
        Args:
            probs: [B, M, R] pairwise probabilities
            mask: [B, M, R] binary mask
            
        Returns:
            bag_probs: [B] bag-level probabilities
        """
        if mask is not None:
            # Set invalid pairs to -inf before max
            probs = probs.masked_fill(mask == 0, -float('inf'))
        
        # Max over all pairs
        bag_probs, _ = probs.view(probs.size(0), -1).max(dim=1)
        
        # Handle case where all pairs are masked
        bag_probs = torch.where(
            torch.isfinite(bag_probs),
            bag_probs,
            torch.zeros_like(bag_probs)
        )
        
        return bag_probs.clamp(self.epsilon, 1 - self.epsilon)