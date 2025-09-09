"""
True PU Learning training script with separate P and U streams
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.mil_model import MILModel
from models.pu_loss import TruePULoss
from training.dataset import PhageHostDataset, PhageHostDataModule
from utils.logging_utils import setup_logger, MetricLogger
from training.evaluation import evaluate_model


class PUDataLoaders:
    """
    Creates separate data loaders for positive and unlabeled samples
    """
    
    def __init__(self,
                 data_module: PhageHostDataModule,
                 batch_size_pos: int = 32,
                 batch_size_unlab: int = 160,
                 unlabeled_ratio: int = 5):
        """
        Args:
            data_module: PhageHostDataModule instance
            batch_size_pos: Batch size for positive samples
            batch_size_unlab: Batch size for unlabeled samples
            unlabeled_ratio: Ratio of unlabeled to positive samples per step
        """
        self.data_module = data_module
        self.batch_size_pos = batch_size_pos
        self.batch_size_unlab = batch_size_unlab
        self.unlabeled_ratio = unlabeled_ratio
        
        # Get the full training dataset
        self.full_dataset = data_module.train_dataset
        
        # Separate positive and unlabeled indices
        self.pos_indices = []
        self.unlab_indices = []
        
        for idx in range(len(self.full_dataset)):
            sample = self.full_dataset[idx]
            if sample['label'] == 1:
                self.pos_indices.append(idx)
            else:
                self.unlab_indices.append(idx)
        
        print(f"Dataset split: {len(self.pos_indices)} positive, {len(self.unlab_indices)} unlabeled")
        
        # Create separate datasets
        self.pos_dataset = Subset(self.full_dataset, self.pos_indices)
        self.unlab_dataset = Subset(self.full_dataset, self.unlab_indices)
        
        # Create data loaders
        self.pos_loader = DataLoader(
            self.pos_dataset,
            batch_size=batch_size_pos,
            shuffle=True,
            num_workers=data_module.num_workers,
            pin_memory=data_module.pin_memory,
            drop_last=True  # Important for consistent batch sizes
        )
        
        self.unlab_loader = DataLoader(
            self.unlab_dataset,
            batch_size=batch_size_unlab,
            shuffle=True,
            num_workers=data_module.num_workers,
            pin_memory=data_module.pin_memory,
            drop_last=True
        )
        
        # Create cycling iterators
        self.reset_iterators()
    
    def reset_iterators(self):
        """Reset the cycling iterators"""
        self.pos_iter = iter(self.pos_loader)
        self.unlab_iter = iter(self.unlab_loader)
    
    def get_batch(self) -> Tuple[Dict, Dict]:
        """
        Get one batch of positive and one batch of unlabeled samples
        
        Returns:
            pos_batch, unlab_batch
        """
        # Get positive batch
        try:
            pos_batch = next(self.pos_iter)
        except StopIteration:
            self.pos_iter = iter(self.pos_loader)
            pos_batch = next(self.pos_iter)
        
        # Get unlabeled batch
        try:
            unlab_batch = next(self.unlab_iter)
        except StopIteration:
            self.unlab_iter = iter(self.unlab_loader)
            unlab_batch = next(self.unlab_iter)
        
        return pos_batch, unlab_batch
    
    def __len__(self):
        """Number of iterations per epoch"""
        # Use the smaller of the two to ensure we see all positive samples
        return min(len(self.pos_loader), len(self.unlab_loader))


class PUTrainer:
    """
    Trainer for true PU learning with separate P and U streams
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(config['device']['device'])
        
        # Initialize model
        self.model = self._build_model()
        
        # Initialize loss (with true biological prior, not training ratio!)
        # This should be the estimated prevalence in the unlabeled pool
        self.prior = config['loss'].get('true_prior', 0.02)  # 2% default based on biology
        self.criterion = TruePULoss(prior=self.prior, reduction='mean')
        
        # Initialize optimizer
        self.optimizer = self._build_optimizer()
        
        # Metrics
        self.metric_logger = MetricLogger(log_dir='logs')
        
        self.logger.info(f"PU Trainer initialized with prior={self.prior}")
    
    def _build_model(self) -> nn.Module:
        """Build and initialize model"""
        config = self.config['model']
        
        encoder_type = config['encoder_type']
        if encoder_type == 'conservative':
            encoder_dims = config['encoder_dims_conservative']
        elif encoder_type == 'balanced':
            encoder_dims = config['encoder_dims_balanced']
        else:
            encoder_dims = config['encoder_dims_aggressive']
        
        model = MILModel(
            input_dim=config['input_dim'],
            encoder_dims=encoder_dims,
            temperature=config.get('temperature', 1.0),
            dropout=config.get('dropout', 0.1),
            aggregation=config.get('aggregation', 'noisy_or')
        )
        
        return model.to(self.device)
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer"""
        config = self.config['training']
        
        if config['optimizer'] == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config.get('weight_decay', 0)
            )
        elif config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config.get('weight_decay', 1e-5)
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")
        
        return optimizer
    
    def train_epoch(self, pu_loaders: PUDataLoaders, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with separate P and U streams
        """
        self.model.train()
        
        epoch_metrics = {
            'loss': [],
            'pos_risk': [],
            'neg_risk': [],
            'grad_norm': []
        }
        
        pbar = tqdm(range(len(pu_loaders)), desc=f"Epoch {epoch}")
        
        for step in pbar:
            # Get separate P and U batches
            pos_batch, unlab_batch = pu_loaders.get_batch()
            
            # Move positive batch to device
            pos_marker = pos_batch['marker_embeddings'].to(self.device)
            pos_rbp = pos_batch['rbp_embeddings'].to(self.device)
            pos_marker_mask = pos_batch['marker_mask'].to(self.device)
            pos_rbp_mask = pos_batch['rbp_mask'].to(self.device)
            
            # Move unlabeled batch to device
            unlab_marker = unlab_batch['marker_embeddings'].to(self.device)
            unlab_rbp = unlab_batch['rbp_embeddings'].to(self.device)
            unlab_marker_mask = unlab_batch['marker_mask'].to(self.device)
            unlab_rbp_mask = unlab_batch['rbp_mask'].to(self.device)
            
            # Forward pass for positive samples
            pos_outputs = self.model(
                pos_marker, pos_rbp,
                pos_marker_mask, pos_rbp_mask
            )
            pos_logits = pos_outputs['bag_logits']
            
            # Forward pass for unlabeled samples  
            unlab_outputs = self.model(
                unlab_marker, unlab_rbp,
                unlab_marker_mask, unlab_rbp_mask
            )
            unlab_logits = unlab_outputs['bag_logits']
            
            # Compute PU loss with separate streams
            loss_dict = self.criterion(pos_logits, unlab_logits)
            loss = loss_dict['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training'].get('gradient_clip', 1.0)
            )
            
            self.optimizer.step()
            
            # Track metrics
            epoch_metrics['loss'].append(loss.item())
            epoch_metrics['pos_risk'].append(loss_dict['pos_risk'].item())
            epoch_metrics['neg_risk'].append(loss_dict['neg_risk'].item())
            epoch_metrics['grad_norm'].append(grad_norm.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'pos_risk': f"{loss_dict['pos_risk']:.4f}",
                'neg_risk': f"{loss_dict['neg_risk']:.4f}",
                'grad': f"{grad_norm:.4f}"
            })
            
            # Log first batch details
            if step == 0:
                self.logger.info(f"First batch - Pos batch size: {len(pos_logits)}, "
                               f"Unlab batch size: {len(unlab_logits)}")
                self.logger.info(f"Pos logits: min={pos_logits.min():.3f}, "
                               f"max={pos_logits.max():.3f}, mean={pos_logits.mean():.3f}")
                self.logger.info(f"Unlab logits: min={unlab_logits.min():.3f}, "
                               f"max={unlab_logits.max():.3f}, mean={unlab_logits.mean():.3f}")
        
        # Compute epoch averages
        avg_metrics = {
            'loss': np.mean(epoch_metrics['loss']),
            'pos_risk': np.mean(epoch_metrics['pos_risk']),
            'neg_risk': np.mean(epoch_metrics['neg_risk']),
            'grad_norm': np.mean(epoch_metrics['grad_norm'])
        }
        
        return avg_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate on labeled validation set
        """
        self.model.eval()
        
        all_probs = []
        all_labels = []
        val_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move to device
                marker_embeddings = batch['marker_embeddings'].to(self.device)
                rbp_embeddings = batch['rbp_embeddings'].to(self.device)
                marker_mask = batch['marker_mask'].to(self.device)
                rbp_mask = batch['rbp_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    marker_embeddings, rbp_embeddings,
                    marker_mask, rbp_mask
                )
                
                bag_probs = outputs['bag_probs']
                
                # Track predictions
                all_probs.extend(bag_probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Compute validation loss (using BCE for validation)
                loss = F.binary_cross_entropy(bag_probs, labels.float())
                val_loss += loss.item()
                n_batches += 1
        
        # Compute metrics
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_preds = (all_probs > 0.5).astype(int)
        
        metrics = {
            'val_loss': val_loss / n_batches,
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds),
            'auroc': roc_auc_score(all_labels, all_probs)
        }
        
        return metrics
    
    def train(self, 
              data_module: PhageHostDataModule,
              n_epochs: int,
              batch_size_pos: int = 32,
              batch_size_unlab: int = 160):
        """
        Main training loop
        """
        self.logger.info("="*60)
        self.logger.info("Starting True PU Training")
        self.logger.info(f"Prior (estimated positive prevalence): {self.prior}")
        self.logger.info(f"Batch sizes - Positive: {batch_size_pos}, Unlabeled: {batch_size_unlab}")
        self.logger.info("="*60)
        
        # Create PU data loaders
        pu_loaders = PUDataLoaders(
            data_module,
            batch_size_pos=batch_size_pos,
            batch_size_unlab=batch_size_unlab
        )
        
        # Get validation loader
        val_loader = data_module.val_dataloader()
        
        best_auroc = 0
        
        for epoch in range(1, n_epochs + 1):
            self.logger.info(f"\nEpoch {epoch}/{n_epochs}")
            self.logger.info("-" * 40)
            
            # Train
            train_metrics = self.train_epoch(pu_loaders, epoch)
            self.logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                           f"Pos risk: {train_metrics['pos_risk']:.4f}, "
                           f"Neg risk: {train_metrics['neg_risk']:.4f}")
            
            # Validate
            if epoch % self.config['training'].get('validate_every_n_epochs', 1) == 0:
                val_metrics = self.validate(val_loader)
                self.logger.info(f"Val - Loss: {val_metrics['val_loss']:.4f}, "
                               f"Acc: {val_metrics['accuracy']:.4f}, "
                               f"AUROC: {val_metrics['auroc']:.4f}")
                
                # Save best model
                if val_metrics['auroc'] > best_auroc:
                    best_auroc = val_metrics['auroc']
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'metrics': val_metrics,
                        'config': self.config
                    }, 'checkpoints/best_pu_model.pt')
                    self.logger.info(f"Saved best model with AUROC: {best_auroc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size_pos', type=int, default=32)
    parser.add_argument('--batch_size_unlab', type=int, default=160)
    parser.add_argument('--prior', type=float, default=0.02,
                       help='Estimated positive prevalence in unlabeled data')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override prior
    config['loss']['true_prior'] = args.prior
    
    # Setup logging
    logger = setup_logger('pu_trainer', 'logs/')
    
    # Create data module
    data_module = PhageHostDataModule(
        data_path=config['data']['data_path'],
        splits_path=config['data']['splits_path'],
        embeddings_dir=config['data']['embeddings_path'],
        batch_size=32,  # Not used, we set custom batch sizes
        max_hosts=config['dataset']['max_markers'],
        max_phages=config['dataset']['max_rbps'],
        negative_ratio_train=1.0,  # Ignored, we separate P and U
        negative_ratio_val=4.0,
        negative_ratio_test=49.0,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory']
    )
    
    # Create trainer
    trainer = PUTrainer(config, logger)
    
    # Train
    trainer.train(
        data_module,
        n_epochs=args.epochs,
        batch_size_pos=args.batch_size_pos,
        batch_size_unlab=args.batch_size_unlab
    )


if __name__ == "__main__":
    main()