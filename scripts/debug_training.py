"""
Debug script to test data loading and model training on a small sample
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.mil_model import MILModel
from models.losses import nnPULoss
from training.dataset import PhageHostDataModule
from utils.logging_utils import setup_logger


def check_data_loading(config, logger):
    """Test data loading and verify embeddings"""
    logger.info("=" * 50)
    logger.info("Testing Data Loading")
    logger.info("=" * 50)
    
    # Initialize data module with small batch size
    data_module = PhageHostDataModule(
        data_path=config['data']['data_path'],
        splits_path=config['data']['splits_path'],
        embeddings_dir=config['data']['embeddings_path'],
        batch_size=4,  # Small batch for debugging
        max_hosts=config['dataset']['max_markers'],
        max_phages=config['dataset']['max_rbps'],
        negative_ratio_train=1.0,  # Balanced for debugging
        negative_ratio_val=1.0,
        negative_ratio_test=1.0,
        num_workers=0,  # Single-threaded for debugging
        pin_memory=False,
        augment_train=False,  # No augmentation for debugging
        cache_size=100,
        preload_all=False,
        logger=logger
    )
    
    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Check first batch
    logger.info("\n" + "=" * 50)
    logger.info("Checking First Training Batch")
    logger.info("=" * 50)
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 3:  # Check first 3 batches
            break
            
        logger.info(f"\nBatch {batch_idx + 1}:")
        
        # Extract batch data
        marker_embeddings = batch['marker_embeddings']
        rbp_embeddings = batch['rbp_embeddings']
        marker_mask = batch['marker_mask']
        rbp_mask = batch['rbp_mask']
        labels = batch['label']
        
        # Log shapes
        logger.info(f"  Shapes:")
        logger.info(f"    Marker embeddings: {marker_embeddings.shape}")
        logger.info(f"    RBP embeddings: {rbp_embeddings.shape}")
        logger.info(f"    Marker mask: {marker_mask.shape}")
        logger.info(f"    RBP mask: {rbp_mask.shape}")
        logger.info(f"    Labels: {labels.shape}")
        
        # Check for zeros
        marker_zeros = (marker_embeddings == 0).all(dim=-1)  # Check if entire embedding is zero
        rbp_zeros = (rbp_embeddings == 0).all(dim=-1)
        
        logger.info(f"  Data validity:")
        logger.info(f"    Marker embeddings non-zero: {(~marker_zeros).any().item()}")
        logger.info(f"    RBP embeddings non-zero: {(~rbp_zeros).any().item()}")
        
        # Check mask validity
        for i in range(len(labels)):
            n_markers = marker_mask[i].sum().item()
            n_rbps = rbp_mask[i].sum().item()
            
            # Check if masked positions have zero embeddings
            marker_valid = True
            rbp_valid = True
            
            # Check that masked positions have non-zero embeddings
            for j in range(int(n_markers)):
                if (marker_embeddings[i, j] == 0).all():
                    marker_valid = False
                    logger.warning(f"    Sample {i}: Marker {j} is masked but has zero embedding!")
                    
            for j in range(int(n_rbps)):
                if (rbp_embeddings[i, j] == 0).all():
                    rbp_valid = False
                    logger.warning(f"    Sample {i}: RBP {j} is masked but has zero embedding!")
            
            logger.info(f"    Sample {i}: Label={labels[i].item():.0f}, Markers={n_markers:.0f}/{marker_mask.shape[1]}, RBPs={n_rbps:.0f}/{rbp_mask.shape[1]}, Valid={marker_valid and rbp_valid}")
        
        # Check embedding statistics
        logger.info(f"  Embedding statistics:")
        logger.info(f"    Marker embeddings: min={marker_embeddings.min():.4f}, max={marker_embeddings.max():.4f}, mean={marker_embeddings.mean():.4f}, std={marker_embeddings.std():.4f}")
        logger.info(f"    RBP embeddings: min={rbp_embeddings.min():.4f}, max={rbp_embeddings.max():.4f}, mean={rbp_embeddings.mean():.4f}, std={rbp_embeddings.std():.4f}")
        
        # Check label distribution
        n_positive = (labels == 1).sum().item()
        n_negative = (labels == 0).sum().item()
        logger.info(f"  Label distribution: {n_positive} positive, {n_negative} negative")
    
    return data_module


def test_model_forward(config, data_module, logger):
    """Test model forward pass"""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Model Forward Pass")
    logger.info("=" * 50)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Get encoder dimensions
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
        shared_architecture=True,
        dropout=0.0,  # No dropout for debugging
        activation=config['model']['activation'],
        use_layer_norm=config['model']['use_layer_norm'],
        temperature=config['model']['temperature']
    ).to(device)
    
    # Log model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Test forward pass
    train_loader = data_module.train_dataloader()
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 1:  # Test one batch
            break
        
        # Move to device
        marker_embeddings = batch['marker_embeddings'].to(device)
        rbp_embeddings = batch['rbp_embeddings'].to(device)
        marker_mask = batch['marker_mask'].to(device)
        rbp_mask = batch['rbp_mask'].to(device)
        labels = batch['label'].to(device)
        
        logger.info(f"\nForward pass test:")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(marker_embeddings, rbp_embeddings, marker_mask, rbp_mask)
        
        # Check outputs
        logger.info(f"  Output keys: {outputs.keys()}")
        logger.info(f"  Bag probabilities shape: {outputs['bag_probs'].shape}")
        logger.info(f"  Bag probabilities: min={outputs['bag_probs'].min():.4f}, max={outputs['bag_probs'].max():.4f}, mean={outputs['bag_probs'].mean():.4f}")
        
        if 'pairwise_scores' in outputs:
            logger.info(f"  Pairwise scores shape: {outputs['pairwise_scores'].shape}")
            logger.info(f"  Pairwise scores: min={outputs['pairwise_scores'].min():.4f}, max={outputs['pairwise_scores'].max():.4f}")
        
        # Test loss computation
        criterion = nnPULoss(
            prior=0.5,  # Balanced for debugging
            beta=config['loss']['beta'],
            gamma=config['loss']['gamma']
        )
        
        loss_dict = criterion(outputs['bag_probs'], labels)
        logger.info(f"\nLoss computation:")
        logger.info(f"  Total loss: {loss_dict['loss'].item():.4f}")
        logger.info(f"  Positive risk: {loss_dict['positive_risk'].item():.4f}")
        logger.info(f"  Negative risk: {loss_dict['negative_risk'].item():.4f}")
        
        # Check gradients
        loss_dict['loss'].backward()
        
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm
        
        logger.info(f"\nGradient norms (top 5):")
        sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:5]
        for name, norm in sorted_grads:
            logger.info(f"  {name}: {norm:.6f}")
        
        # Check for zero gradients
        zero_grad_params = [name for name, norm in grad_norms.items() if norm < 1e-8]
        if zero_grad_params:
            logger.warning(f"\nParameters with zero gradients: {zero_grad_params[:5]}")
    
    return model


def test_mini_training(config, data_module, model, logger):
    """Test a few training steps"""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Mini Training Loop")
    logger.info("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Initialize loss
    criterion = nnPULoss(
        prior=0.5,
        beta=config['loss']['beta'],
        gamma=config['loss']['gamma']
    )
    
    # Train for a few steps
    model.train()
    train_loader = data_module.train_dataloader()
    
    initial_predictions = []
    final_predictions = []
    
    for step in range(10):  # 10 training steps
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 1:  # One batch per step
                break
            
            # Move to device
            marker_embeddings = batch['marker_embeddings'].to(device)
            rbp_embeddings = batch['rbp_embeddings'].to(device)
            marker_mask = batch['marker_mask'].to(device)
            rbp_mask = batch['rbp_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(marker_embeddings, rbp_embeddings, marker_mask, rbp_mask)
            
            # Store predictions
            if step == 0:
                initial_predictions = outputs['bag_probs'].detach().cpu().numpy()
            if step == 9:
                final_predictions = outputs['bag_probs'].detach().cpu().numpy()
            
            # Compute loss
            loss_dict = criterion(outputs['bag_probs'], labels)
            loss = loss_dict['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradient norm
            total_grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            # Optimizer step
            optimizer.step()
            
            # Log progress
            if step % 2 == 0:
                logger.info(f"Step {step + 1}: Loss={loss.item():.4f}, Grad norm={total_grad_norm:.4f}, "
                           f"Pred mean={outputs['bag_probs'].mean().item():.4f}, "
                           f"Pred std={outputs['bag_probs'].std().item():.4f}")
    
    # Check if predictions changed
    if len(initial_predictions) > 0 and len(final_predictions) > 0:
        pred_change = np.abs(final_predictions - initial_predictions).mean()
        logger.info(f"\nPrediction change after 10 steps: {pred_change:.6f}")
        
        if pred_change < 1e-6:
            logger.warning("WARNING: Predictions did not change during training!")
        else:
            logger.info("SUCCESS: Model is learning (predictions are changing)")


def main():
    """Main debugging function"""
    # Load configuration
    config_path = "configs/default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up logging
    logger = setup_logger(
        name='debug_training',
        log_dir='logs',
        log_file='debug_training.log',
        level='DEBUG'
    )
    
    logger.info("Starting debugging script...")
    logger.info(f"Configuration loaded from: {config_path}")
    
    try:
        # Test data loading
        data_module = check_data_loading(config, logger)
        
        # Test model forward pass
        model = test_model_forward(config, data_module, logger)
        
        # Test mini training loop
        test_mini_training(config, data_module, model, logger)
        
        logger.info("\n" + "=" * 50)
        logger.info("Debugging Complete!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error during debugging: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()