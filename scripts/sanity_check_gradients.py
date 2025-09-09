"""
Sanity check script to verify gradients are flowing properly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.mil_model import MILModel
from training.dataset import PhageHostDataModule

def check_gradients():
    """Check that gradients are non-zero for key components"""
    
    print("=" * 60)
    print("GRADIENT SANITY CHECK")
    print("=" * 60)
    
    # Load config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data module
    data_module = PhageHostDataModule(
        data_path=config['data']['data_path'],
        splits_path=config['data']['splits_path'],
        embeddings_dir=config['data']['embeddings_path'],
        batch_size=8,  # Small batch for testing
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
    encoder_dims = config['model']['encoder_dims_balanced']
    model = MILModel(
        input_dim=config['model']['input_dim'],
        encoder_dims=encoder_dims,
        temperature=1.0,
        dropout=0.0  # No dropout for testing
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get a batch
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    # Move to device
    marker_embeddings = batch['marker_embeddings'].to(device)
    rbp_embeddings = batch['rbp_embeddings'].to(device)
    marker_mask = batch['marker_mask'].to(device)
    rbp_mask = batch['rbp_mask'].to(device)
    labels = batch['label'].to(device)
    
    print(f"\nBatch info:")
    print(f"  Batch size: {marker_embeddings.size(0)}")
    print(f"  Markers shape: {marker_embeddings.shape}")
    print(f"  RBPs shape: {rbp_embeddings.shape}")
    print(f"  Labels: {labels.sum().item()}/{len(labels)} positive")
    
    # Forward pass
    outputs = model(
        marker_embeddings,
        rbp_embeddings,
        marker_mask,
        rbp_mask,
        return_pairwise=True
    )
    
    bag_probs = outputs['bag_probs']
    pairwise_probs = outputs['pairwise_probs']
    
    print(f"\nModel outputs:")
    print(f"  Bag probs: min={bag_probs.min():.4f}, max={bag_probs.max():.4f}, mean={bag_probs.mean():.4f}")
    print(f"  Pairwise probs: min={pairwise_probs.min():.4f}, max={pairwise_probs.max():.4f}, mean={pairwise_probs.mean():.4f}")
    
    # Compute loss (simple BCE for now)
    loss = F.binary_cross_entropy(bag_probs, labels.float())
    print(f"\nLoss: {loss.item():.4f}")
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Check gradients for key components
    print(f"\n{'='*60}")
    print("GRADIENT NORMS (should be non-zero):")
    print(f"{'='*60}")
    
    # 1. Scorer parameters
    if hasattr(model, 'scorer'):
        if hasattr(model.scorer, 'logit_scale'):
            scale_grad = model.scorer.logit_scale.grad
            if scale_grad is not None:
                print(f"Scorer logit_scale grad: {scale_grad.abs().item():.6f}")
            else:
                print(f"Scorer logit_scale grad: None (ERROR!)")
                
        if hasattr(model.scorer, 'bias'):
            bias_grad = model.scorer.bias.grad
            if bias_grad is not None:
                print(f"Scorer bias grad: {bias_grad.abs().item():.6f}")
            else:
                print(f"Scorer bias grad: None (ERROR!)")
    
    # 2. Encoder parameters
    # Check last layer of marker encoder
    last_layer = None
    for name, module in model.encoder.marker_encoder.named_modules():
        if isinstance(module, nn.Linear):
            last_layer = module
    
    if last_layer is not None and last_layer.weight.grad is not None:
        print(f"Last encoder layer weight grad norm: {last_layer.weight.grad.abs().mean().item():.6f}")
        print(f"Last encoder layer bias grad norm: {last_layer.bias.grad.abs().mean().item():.6f}")
    else:
        print("Last encoder layer grad: None (ERROR!)")
    
    # 3. Check first layer gradients
    first_layer = None
    for name, module in model.encoder.marker_encoder.named_modules():
        if isinstance(module, nn.Linear):
            first_layer = module
            break
    
    if first_layer is not None and first_layer.weight.grad is not None:
        print(f"First encoder layer weight grad norm: {first_layer.weight.grad.abs().mean().item():.6f}")
    
    # Check total number of parameters with gradients
    total_params = 0
    params_with_grad = 0
    zero_grad_params = []
    
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            if param.grad.abs().sum() > 0:
                params_with_grad += 1
            else:
                zero_grad_params.append(name)
    
    print(f"\n{'='*60}")
    print(f"GRADIENT SUMMARY:")
    print(f"  Total parameters: {total_params}")
    print(f"  Parameters with non-zero gradients: {params_with_grad}")
    print(f"  Parameters with zero gradients: {total_params - params_with_grad}")
    
    if zero_grad_params:
        print(f"\nParameters with zero gradients:")
        for name in zero_grad_params[:10]:  # Show first 10
            print(f"  - {name}")
        if len(zero_grad_params) > 10:
            print(f"  ... and {len(zero_grad_params) - 10} more")
    
    print(f"\n{'='*60}")
    if params_with_grad > 0:
        print("✓ GRADIENTS ARE FLOWING")
    else:
        print("✗ NO GRADIENTS - CHECK MODEL/LOSS")
    print(f"{'='*60}")

def check_initialization():
    """Check that model initialization matches expected priors"""
    
    print("\n" + "=" * 60)
    print("INITIALIZATION SANITY CHECK")
    print("=" * 60)
    
    # Load config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    encoder_dims = config['model']['encoder_dims_balanced']
    model = MILModel(
        input_dim=config['model']['input_dim'],
        encoder_dims=encoder_dims,
        temperature=1.0,
        dropout=0.0
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create synthetic data with known statistics
    batch_size = 100
    n_markers = 2
    n_rbps = 2
    
    # Random embeddings
    marker_embeddings = torch.randn(batch_size, n_markers, 1280) * 0.1
    rbp_embeddings = torch.randn(batch_size, n_rbps, 1280) * 0.1
    marker_mask = torch.ones(batch_size, n_markers)
    rbp_mask = torch.ones(batch_size, n_rbps)
    
    marker_embeddings = marker_embeddings.to(device)
    rbp_embeddings = rbp_embeddings.to(device)
    marker_mask = marker_mask.to(device)
    rbp_mask = rbp_mask.to(device)
    
    with torch.no_grad():
        outputs = model(
            marker_embeddings,
            rbp_embeddings,
            marker_mask,
            rbp_mask,
            return_pairwise=True
        )
    
    bag_probs = outputs['bag_probs'].cpu().numpy()
    pairwise_probs = outputs['pairwise_probs'].cpu().numpy()
    
    # Calculate statistics
    mean_pair_prob = pairwise_probs.mean()
    mean_bag_prob = bag_probs.mean()
    K = n_markers * n_rbps  # Number of pairs
    
    # Expected values based on initialization
    # From init_pair_bias with bag_prior=0.2, mean_num_pairs=4
    expected_bag_prior = 0.2
    expected_q = 1.0 - (1.0 - expected_bag_prior) ** (1.0 / 4)
    
    print(f"\nInitialization statistics (batch_size={batch_size}):")
    print(f"  Number of pairs per bag (K): {K}")
    print(f"  Mean pair probability: {mean_pair_prob:.4f}")
    print(f"  Expected pair prob (q): {expected_q:.4f}")
    print(f"  Mean bag probability: {mean_bag_prob:.4f}")
    print(f"  Expected bag prior: {expected_bag_prior:.4f}")
    
    # Check if close to expected
    pair_prob_ok = abs(mean_pair_prob - expected_q) < 0.1
    bag_prob_ok = abs(mean_bag_prob - expected_bag_prior) < 0.1
    
    print(f"\n{'='*60}")
    if pair_prob_ok and bag_prob_ok:
        print("✓ INITIALIZATION LOOKS REASONABLE")
    else:
        print("✗ INITIALIZATION MAY BE OFF")
        if not pair_prob_ok:
            print(f"  - Pair prob {mean_pair_prob:.4f} far from expected {expected_q:.4f}")
        if not bag_prob_ok:
            print(f"  - Bag prob {mean_bag_prob:.4f} far from expected {expected_bag_prior:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    check_gradients()
    check_initialization()