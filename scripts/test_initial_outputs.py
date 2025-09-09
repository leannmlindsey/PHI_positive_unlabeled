"""
Test initial model outputs with fixed temperature
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

def test_initial_model():
    """Test model outputs at initialization"""
    
    # Load config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Temperature setting: {config['model']['temperature']}")
    
    # Create data module
    data_module = PhageHostDataModule(
        data_path=config['data']['data_path'],
        splits_path=config['data']['splits_path'],
        embeddings_dir=config['data']['embeddings_path'],
        batch_size=32,
        max_hosts=config['dataset']['max_markers'],
        max_phages=config['dataset']['max_rbps'],
        negative_ratio_train=1.0,
        negative_ratio_val=1.0,
        negative_ratio_test=1.0,
        num_workers=0,
        pin_memory=False,
        augment_train=False
    )
    
    # Create model
    encoder_type = config['model']['encoder_type']
    if encoder_type == 'conservative':
        encoder_dims = config['model']['encoder_dims_conservative']
    elif encoder_type == 'balanced':
        encoder_dims = config['model']['encoder_dims_balanced']
    else:
        encoder_dims = config['model']['encoder_dims_aggressive']
    
    model = MILModel(
        input_dim=config['model']['input_dim'],
        encoder_dims=encoder_dims,
        temperature=config['model']['temperature'],
        dropout=config['model']['dropout']
    )
    
    model.eval()
    
    # Get a batch
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    with torch.no_grad():
        outputs = model(
            batch['marker_embeddings'],
            batch['rbp_embeddings'],
            batch['marker_mask'],
            batch['rbp_mask'],
            return_pairwise=True
        )
        
        bag_probs = outputs['bag_probs'].numpy()
        labels = batch['label'].numpy()
        
        print(f"\nInitial model outputs (batch_size={len(labels)}):")
        print(f"  Labels: {labels.sum()}/{len(labels)} positive")
        print(f"  Bag probabilities:")
        print(f"    Min: {bag_probs.min():.4f}")
        print(f"    Max: {bag_probs.max():.4f}")
        print(f"    Mean: {bag_probs.mean():.4f}")
        print(f"    Std: {bag_probs.std():.4f}")
        
        # Separate by class
        pos_probs = bag_probs[labels == 1]
        neg_probs = bag_probs[labels == 0]
        
        if len(pos_probs) > 0:
            print(f"  Positive class:")
            print(f"    Mean: {pos_probs.mean():.4f}, Std: {pos_probs.std():.4f}")
        if len(neg_probs) > 0:
            print(f"  Negative class:")
            print(f"    Mean: {neg_probs.mean():.4f}, Std: {neg_probs.std():.4f}")
        
        # Check pairwise probabilities
        pairwise_probs = outputs['pairwise_probs']
        print(f"\n  Pairwise probabilities (before aggregation):")
        print(f"    Shape: {pairwise_probs.shape}")
        print(f"    Min: {pairwise_probs.min():.4f}")
        print(f"    Max: {pairwise_probs.max():.4f}")
        print(f"    Mean: {pairwise_probs.mean():.4f}")

if __name__ == "__main__":
    test_initial_model()