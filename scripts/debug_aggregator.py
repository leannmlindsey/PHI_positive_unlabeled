"""
Debug script to verify label polarity and aggregator behavior
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.mil_model import MILModel, NoisyORLayer
from training.dataset import PhageHostDataModule

def test_aggregator():
    """Test the aggregator behavior with known inputs"""
    print("=" * 50)
    print("Testing Aggregator Behavior")
    print("=" * 50)
    
    # Create a simple noisy-OR layer
    aggregator = NoisyORLayer()
    
    # Test case 1: All pairs have probability 0
    print("\nTest 1: All pairs p=0 (should output ~0)")
    pairwise_probs = torch.zeros((1, 2, 2))  # 1 batch, 2x2 pairs
    mask = torch.ones((1, 2, 2))
    result = aggregator(pairwise_probs, mask)
    print(f"Input: all zeros")
    print(f"Output: {result.item():.4f}")
    
    # Test case 2: All pairs have probability 1
    print("\nTest 2: All pairs p=1 (should output ~1)")
    pairwise_probs = torch.ones((1, 2, 2))
    result = aggregator(pairwise_probs, mask)
    print(f"Input: all ones")
    print(f"Output: {result.item():.4f}")
    
    # Test case 3: One pair has p=0.5, others are 0
    print("\nTest 3: One pair p=0.5, others p=0 (should output ~0.5)")
    pairwise_probs = torch.zeros((1, 2, 2))
    pairwise_probs[0, 0, 0] = 0.5
    result = aggregator(pairwise_probs, mask)
    print(f"Input: one 0.5, rest zeros")
    print(f"Output: {result.item():.4f}")
    
    # Test case 4: Testing masking
    print("\nTest 4: Masking test - masked pairs should not contribute")
    pairwise_probs = torch.ones((1, 2, 2)) * 0.9  # All high probability
    mask = torch.zeros((1, 2, 2))
    mask[0, 0, 0] = 1  # Only one pair is valid
    result = aggregator(pairwise_probs, mask)
    print(f"Input: all 0.9, but only (0,0) unmasked")
    print(f"Output: {result.item():.4f} (should be ~0.9)")

def check_data_labels():
    """Check actual data labels and model outputs"""
    print("\n" + "=" * 50)
    print("Checking Data Labels and Model Outputs")
    print("=" * 50)
    
    # Load a small batch of data
    import yaml
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data module with small batch
    data_module = PhageHostDataModule(
        data_path=config['data']['data_path'],
        splits_path=config['data']['splits_path'],
        embeddings_dir=config['data']['embeddings_path'],
        batch_size=4,  # Small batch for debugging
        max_hosts=config['dataset']['max_markers'],
        max_phages=config['dataset']['max_rbps'],
        negative_ratio_train=1.0,
        negative_ratio_val=1.0,
        negative_ratio_test=1.0,
        num_workers=0,
        pin_memory=False,
        augment_train=False
    )
    
    # Get one batch
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"\nBatch info:")
    print(f"  Batch size: {batch['label'].shape[0]}")
    print(f"  Labels: {batch['label'].numpy()}")
    print(f"  Label distribution: {batch['label'].sum().item()}/{len(batch['label'])} positive")
    
    # Create model
    model = MILModel(
        input_dim=config['model']['input_dim'],
        encoder_dims=config['model']['encoder_dims_balanced'],
        temperature=config['model']['temperature']
    )
    model.eval()
    
    with torch.no_grad():
        # Get model outputs
        outputs = model(
            batch['marker_embeddings'],
            batch['rbp_embeddings'],
            batch['marker_mask'],
            batch['rbp_mask'],
            return_pairwise=True
        )
        
        print(f"\nModel outputs:")
        print(f"  Bag probabilities: {outputs['bag_probs'].numpy()}")
        print(f"  Pairwise probabilities shape: {outputs['pairwise_probs'].shape}")
        
        # Check pairwise probabilities for each sample
        for i in range(min(2, batch['label'].shape[0])):  # Check first 2 samples
            label = batch['label'][i].item()
            bag_prob = outputs['bag_probs'][i].item()
            pairwise = outputs['pairwise_probs'][i]
            
            # Get valid pairs from mask
            marker_mask = batch['marker_mask'][i]
            rbp_mask = batch['rbp_mask'][i]
            n_markers = marker_mask.sum().item()
            n_rbps = rbp_mask.sum().item()
            
            print(f"\n  Sample {i}:")
            print(f"    Label: {label}")
            print(f"    Valid markers: {int(n_markers)}, Valid RBPs: {int(n_rbps)}")
            print(f"    Active pairs: {int(n_markers * n_rbps)}")
            print(f"    Pairwise probs (valid only):")
            for m in range(int(n_markers)):
                for r in range(int(n_rbps)):
                    print(f"      Pair ({m},{r}): {pairwise[m,r].item():.4f}")
            print(f"    Final bag probability: {bag_prob:.4f}")

def main():
    """Run all debugging checks"""
    test_aggregator()
    check_data_labels()
    
    print("\n" + "=" * 50)
    print("Debugging Complete")
    print("=" * 50)
    print("\nKey things to verify:")
    print("1. Labels: 1 should mean positive interaction, 0 negative")
    print("2. Aggregator: Should output high prob when ANY pair has high prob (Noisy-OR)")
    print("3. Masking: Masked pairs should contribute 0 to aggregation")
    print("4. The bag probability should increase with more high pairwise probs")

if __name__ == "__main__":
    main()