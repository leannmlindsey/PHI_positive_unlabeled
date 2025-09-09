"""
Supervised sanity test - train on small clean subset to verify model can learn
This should achieve high AUROC quickly if the model is wired correctly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.mil_model import MILModel
from models.pu_loss import SupervisedLoss
from training.dataset import PhageHostDataModule


def create_simple_dataset(n_samples=200, device='cpu'):
    """
    Create a simple supervised dataset with 1 marker and 1 RBP per bag
    This removes complexity to test if the model can learn at all
    """
    # Create embeddings that are somewhat separable
    np.random.seed(42)
    
    # Positive samples - embeddings cluster in one region
    pos_marker = torch.randn(n_samples // 2, 1, 1280) * 0.5 + 1.0
    pos_rbp = torch.randn(n_samples // 2, 1, 1280) * 0.5 + 1.0
    
    # Negative samples - embeddings cluster in different region
    neg_marker = torch.randn(n_samples // 2, 1, 1280) * 0.5 - 1.0
    neg_rbp = torch.randn(n_samples // 2, 1, 1280) * 0.5 - 1.0
    
    # Combine
    marker_embeddings = torch.cat([pos_marker, neg_marker], dim=0)
    rbp_embeddings = torch.cat([pos_rbp, neg_rbp], dim=0)
    
    # Labels
    labels = torch.cat([
        torch.ones(n_samples // 2),
        torch.zeros(n_samples // 2)
    ])
    
    # Masks (all valid)
    marker_mask = torch.ones(n_samples, 1)
    rbp_mask = torch.ones(n_samples, 1)
    
    # Shuffle
    perm = torch.randperm(n_samples)
    marker_embeddings = marker_embeddings[perm].to(device)
    rbp_embeddings = rbp_embeddings[perm].to(device)
    labels = labels[perm].to(device)
    marker_mask = marker_mask[perm].to(device)
    rbp_mask = rbp_mask[perm].to(device)
    
    return marker_embeddings, rbp_embeddings, labels, marker_mask, rbp_mask


def supervised_sanity_test():
    """
    Test if model can learn on simple supervised data
    """
    print("=" * 60)
    print("SUPERVISED SANITY TEST")
    print("Training on simple 1x1 bags with clear separation")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model
    model = MILModel(
        input_dim=1280,
        encoder_dims=[512, 256],  # Simpler architecture
        temperature=1.0,
        dropout=0.0,  # No dropout for sanity test
        aggregation='noisy_or'
    ).to(device)
    
    # Create simple dataset
    marker_emb, rbp_emb, labels, marker_mask, rbp_mask = create_simple_dataset(
        n_samples=200, 
        device=device
    )
    
    print(f"\nDataset created:")
    print(f"  Samples: {len(labels)}")
    print(f"  Positive: {labels.sum().item()}")
    print(f"  Negative: {(1-labels).sum().item()}")
    
    # Loss and optimizer
    criterion = SupervisedLoss(pos_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    n_epochs = 20
    batch_size = 32
    n_batches = len(labels) // batch_size
    
    train_losses = []
    train_aurocs = []
    
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 40)
    
    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        all_logits = []
        all_labels = []
        
        # Mini-batch training
        indices = torch.randperm(len(labels))
        
        for i in range(0, len(labels), batch_size):
            batch_idx = indices[i:i+batch_size]
            
            # Get batch
            batch_marker = marker_emb[batch_idx]
            batch_rbp = rbp_emb[batch_idx]
            batch_labels = labels[batch_idx]
            batch_marker_mask = marker_mask[batch_idx]
            batch_rbp_mask = rbp_mask[batch_idx]
            
            # Forward pass
            outputs = model(
                batch_marker,
                batch_rbp,
                batch_marker_mask,
                batch_rbp_mask
            )
            
            # Get logits (not probabilities!)
            bag_logits = outputs['bag_logits']
            
            # Compute loss
            loss_dict = criterion(bag_logits, batch_labels)
            loss = loss_dict['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track
            epoch_losses.append(loss.item())
            all_logits.append(bag_logits.detach().cpu())
            all_labels.append(batch_labels.cpu())
        
        # Compute epoch metrics
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        all_probs = torch.sigmoid(all_logits)
        
        train_loss = np.mean(epoch_losses)
        train_auroc = roc_auc_score(all_labels.numpy(), all_probs.numpy())
        
        train_losses.append(train_loss)
        train_aurocs.append(train_auroc)
        
        print(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f}, AUROC={train_auroc:.4f}")
    
    print("-" * 40)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(marker_emb, rbp_emb, marker_mask, rbp_mask)
        final_logits = outputs['bag_logits']
        final_probs = torch.sigmoid(final_logits)
        final_auroc = roc_auc_score(labels.cpu().numpy(), final_probs.cpu().numpy())
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"  Final AUROC: {final_auroc:.4f}")
    print(f"  Best AUROC:  {max(train_aurocs):.4f}")
    
    if final_auroc > 0.9:
        print("\n✓ SUCCESS: Model can learn on simple supervised data!")
        print("  The model architecture and loss are wired correctly.")
    elif final_auroc > 0.7:
        print("\n⚠ PARTIAL SUCCESS: Model learns but not perfectly.")
        print("  Check initialization and hyperparameters.")
    else:
        print("\n✗ FAILURE: Model cannot learn even simple patterns.")
        print("  Check for bugs in aggregator, masking, or loss computation.")
    print(f"{'='*60}")
    
    # Plot learning curve
    if len(train_losses) > 1:
        try:
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(train_losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(train_aurocs)
            plt.xlabel('Epoch')
            plt.ylabel('AUROC')
            plt.title('Training AUROC')
            plt.grid(True)
            plt.ylim([0.4, 1.0])
            
            plt.tight_layout()
            plt.savefig('supervised_sanity_test.png')
            print(f"\nPlot saved to supervised_sanity_test.png")
        except:
            pass


def test_with_real_data():
    """
    Test on real data but with supervised loss
    """
    print("\n" + "=" * 60)
    print("TESTING WITH REAL DATA (SUPERVISED MODE)")
    print("=" * 60)
    
    # Load config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data module
    data_module = PhageHostDataModule(
        data_path=config['data']['data_path'],
        splits_path=config['data']['splits_path'],
        embeddings_dir=config['data']['embeddings_path'],
        batch_size=32,
        max_hosts=config['dataset']['max_markers'],
        max_phages=config['dataset']['max_rbps'],
        negative_ratio_train=1.0,  # Balanced for supervised
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
        dropout=0.1
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = SupervisedLoss(pos_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # Get one epoch of data
    train_loader = data_module.train_dataloader()
    
    print(f"\nTraining for 1 epoch on real data...")
    print("-" * 40)
    
    model.train()
    epoch_losses = []
    all_logits = []
    all_labels = []
    
    for i, batch in enumerate(tqdm(train_loader, desc="Training")):
        if i >= 50:  # Just test on 50 batches
            break
        
        # Move to device
        marker_embeddings = batch['marker_embeddings'].to(device)
        rbp_embeddings = batch['rbp_embeddings'].to(device)
        marker_mask = batch['marker_mask'].to(device)
        rbp_mask = batch['rbp_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(
            marker_embeddings,
            rbp_embeddings,
            marker_mask,
            rbp_mask
        )
        
        # Get logits
        bag_logits = outputs['bag_logits']
        
        # Compute loss
        loss_dict = criterion(bag_logits, labels)
        loss = loss_dict['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track
        epoch_losses.append(loss.item())
        all_logits.append(bag_logits.detach().cpu())
        all_labels.append(labels.cpu())
        
        if i % 10 == 0:
            print(f"  Batch {i}: Loss={loss.item():.4f}, Grad norm={grad_norm:.4f}")
    
    # Compute metrics
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    all_probs = torch.sigmoid(all_logits)
    
    auroc = roc_auc_score(all_labels.numpy(), all_probs.numpy())
    
    print("-" * 40)
    print(f"Results after 50 batches:")
    print(f"  Mean loss: {np.mean(epoch_losses):.4f}")
    print(f"  AUROC: {auroc:.4f}")
    
    if auroc > 0.55:
        print("\n✓ Model shows learning signal on real data")
    else:
        print("\n⚠ Model may need more training or tuning")


if __name__ == "__main__":
    # First test on simple synthetic data
    supervised_sanity_test()
    
    # Then test on real data
    print("\n" + "="*60)
    response = input("Test on real data? (y/n): ")
    if response.lower() == 'y':
        test_with_real_data()