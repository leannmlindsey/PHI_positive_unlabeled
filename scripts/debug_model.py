#!/usr/bin/env python3
"""
Debug script to check model behavior
"""

import torch
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.mil_model import MILModel
from training.dataset import PhageHostDataModule
import yaml

def debug_model():
    # Load config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize data module
    print("Loading data...")
    data_module = PhageHostDataModule(
        data_path=config['data']['data_path'],
        splits_path=config['data']['splits_path'],
        embeddings_path=config['data']['embeddings_path'],
        batch_size=2,  # Small batch for debugging
        negative_ratio=config['dataset']['negative_ratio'],
        max_markers=config['dataset']['max_markers'],
        max_rbps=config['dataset']['max_rbps'],
        num_workers=0,
        pin_memory=False,
        augment_train=False,
        seed=42
    )
    
    # Get a batch
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    print("\n=== Batch Info ===")
    print(f"Marker embeddings shape: {batch['marker_embeddings'].shape}")
    print(f"RBP embeddings shape: {batch['rbp_embeddings'].shape}")
    print(f"Marker mask shape: {batch['marker_mask'].shape}")
    print(f"RBP mask shape: {batch['rbp_mask'].shape}")
    print(f"Labels: {batch['label']}")
    
    # Check if embeddings are non-zero
    print(f"\nMarker embeddings stats:")
    print(f"  Min: {batch['marker_embeddings'].min():.4f}")
    print(f"  Max: {batch['marker_embeddings'].max():.4f}")
    print(f"  Mean: {batch['marker_embeddings'].mean():.4f}")
    print(f"  Std: {batch['marker_embeddings'].std():.4f}")
    print(f"  % zeros: {(batch['marker_embeddings'] == 0).float().mean():.2%}")
    
    print(f"\nRBP embeddings stats:")
    print(f"  Min: {batch['rbp_embeddings'].min():.4f}")
    print(f"  Max: {batch['rbp_embeddings'].max():.4f}")
    print(f"  Mean: {batch['rbp_embeddings'].mean():.4f}")
    print(f"  Std: {batch['rbp_embeddings'].std():.4f}")
    print(f"  % zeros: {(batch['rbp_embeddings'] == 0).float().mean():.2%}")
    
    # Initialize model
    print("\n=== Model Test ===")
    model = MILModel(
        input_dim=config['model']['input_dim'],
        encoder_dims=config['model'][f"encoder_dims_{config['model']['encoder_type']}"],
        shared_architecture=True,
        dropout=config['model']['dropout'],
        activation=config['model']['activation'],
        use_layer_norm=config['model']['use_layer_norm'],
        temperature=config['model']['temperature']
    )
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            batch['marker_embeddings'],
            batch['rbp_embeddings'],
            batch['marker_mask'],
            batch['rbp_mask'],
            return_pairwise=True
        )
    
    print(f"Bag probabilities: {outputs['bag_probs']}")
    print(f"Pairwise probs shape: {outputs['pairwise_probs'].shape}")
    print(f"Pairwise probs stats:")
    print(f"  Min: {outputs['pairwise_probs'].min():.4f}")
    print(f"  Max: {outputs['pairwise_probs'].max():.4f}")
    print(f"  Mean: {outputs['pairwise_probs'].mean():.4f}")
    
    # Check encoded embeddings
    print(f"\nEncoded markers shape: {outputs['encoded_markers'].shape}")
    print(f"Encoded markers stats:")
    print(f"  Min: {outputs['encoded_markers'].min():.4f}")
    print(f"  Max: {outputs['encoded_markers'].max():.4f}")
    print(f"  Mean: {outputs['encoded_markers'].mean():.4f}")
    print(f"  Std: {outputs['encoded_markers'].std():.4f}")
    
    print(f"\nEncoded RBPs shape: {outputs['encoded_rbps'].shape}")
    print(f"Encoded RBPs stats:")
    print(f"  Min: {outputs['encoded_rbps'].min():.4f}")
    print(f"  Max: {outputs['encoded_rbps'].max():.4f}")
    print(f"  Mean: {outputs['encoded_rbps'].mean():.4f}")
    print(f"  Std: {outputs['encoded_rbps'].std():.4f}")

if __name__ == "__main__":
    debug_model()