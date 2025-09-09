"""
Debug model initialization to understand extreme outputs
"""

import torch
import numpy as np
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.mil_model import MILModel
from training.dataset import PhageHostDataModule

def debug_model_layers():
    """Check what's happening in each layer"""
    
    # Load config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create simple model
    model = MILModel(
        input_dim=config['model']['input_dim'],
        encoder_dims=[768, 512, 256],  # balanced
        temperature=1.0,
        dropout=0.0  # No dropout for debugging
    )
    
    # Get a simple batch
    batch_size = 4
    n_markers = 2
    n_rbps = 2
    embedding_dim = 1280
    
    # Create dummy inputs with reasonable values
    marker_embeddings = torch.randn(batch_size, n_markers, embedding_dim) * 0.1
    rbp_embeddings = torch.randn(batch_size, n_rbps, embedding_dim) * 0.1
    marker_mask = torch.ones(batch_size, n_markers)
    rbp_mask = torch.ones(batch_size, n_rbps)
    
    model.eval()
    
    with torch.no_grad():
        # Get encodings
        encoded_markers, encoded_rbps = model.encoder(
            marker_embeddings, 
            rbp_embeddings,
            marker_mask,
            rbp_mask
        )
        
        print("After encoding:")
        print(f"  Encoded markers: shape={encoded_markers.shape}")
        print(f"    Min={encoded_markers.min():.4f}, Max={encoded_markers.max():.4f}, Mean={encoded_markers.mean():.4f}, Std={encoded_markers.std():.4f}")
        print(f"  Encoded RBPs: shape={encoded_rbps.shape}")
        print(f"    Min={encoded_rbps.min():.4f}, Max={encoded_rbps.max():.4f}, Mean={encoded_rbps.mean():.4f}, Std={encoded_rbps.std():.4f}")
        
        # Compute pairwise scores
        scores = torch.bmm(encoded_markers, encoded_rbps.transpose(1, 2))
        print(f"\nRaw dot product scores: shape={scores.shape}")
        print(f"  Min={scores.min():.4f}, Max={scores.max():.4f}, Mean={scores.mean():.4f}, Std={scores.std():.4f}")
        
        # Scale by sqrt(d)
        import math
        scores_scaled = scores / math.sqrt(256)  # embedding_dim=256
        print(f"\nAfter scaling by sqrt(d):")
        print(f"  Min={scores_scaled.min():.4f}, Max={scores_scaled.max():.4f}, Mean={scores_scaled.mean():.4f}, Std={scores_scaled.std():.4f}")
        
        # Apply temperature (now 1.0)
        scores_temp = scores_scaled / 1.0
        print(f"\nAfter temperature (1.0):")
        print(f"  Min={scores_temp.min():.4f}, Max={scores_temp.max():.4f}, Mean={scores_temp.mean():.4f}, Std={scores_temp.std():.4f}")
        
        # Apply sigmoid
        probs = torch.sigmoid(scores_temp)
        print(f"\nAfter sigmoid:")
        print(f"  Min={probs.min():.4f}, Max={probs.max():.4f}, Mean={probs.mean():.4f}, Std={probs.std():.4f}")
        
        # Check how many are extreme
        very_low = (probs < 0.01).sum().item()
        very_high = (probs > 0.99).sum().item()
        total = probs.numel()
        print(f"\nExtreme values:")
        print(f"  < 0.01: {very_low}/{total} ({100*very_low/total:.1f}%)")
        print(f"  > 0.99: {very_high}/{total} ({100*very_high/total:.1f}%)")
        
        # Apply noisy-OR aggregation
        from models.mil_model import NoisyORLayer
        aggregator = NoisyORLayer()
        mask = torch.ones_like(probs)
        bag_probs = aggregator(probs, mask)
        print(f"\nAfter Noisy-OR aggregation:")
        print(f"  Bag probs: {bag_probs.numpy()}")

if __name__ == "__main__":
    debug_model_layers()