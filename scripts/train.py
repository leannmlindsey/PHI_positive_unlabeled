"""
Main training script for phage-host interaction prediction model
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, LambdaLR, OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import math

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.mil_model import MILModel
from models.losses import nnPULoss
from models.calibration import TemperatureScaling, PlattScaling
from training.dataset import PhageHostDataModule
from utils.logging_utils import setup_logger, MetricLogger, log_model_info, log_training_config
from training.evaluation import evaluate_model, compute_metrics


class Trainer:
    """
    Main trainer class for the MIL model
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the trainer
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Set random seeds
        self._set_seeds()
        
        # Set device
        self.device = self._get_device()
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize data module
        self.data_module = self._init_data_module()
        
        # Initialize model
        self.model = self._init_model()
        
        # Initialize loss function
        self.criterion = self._init_loss()
        
        # Initialize optimizer
        self.optimizer = self._init_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._init_scheduler()
        
        # Initialize metric logger
        self.metric_logger = MetricLogger(
            log_dir=config['logging']['log_dir'],
            experiment_name=config.get('experiment_name', 'experiment')
        )
        
        # Initialize wandb if enabled
        if config['logging']['use_wandb']:
            self._init_wandb()
            
        # Training state
        self.current_epoch = 0
        self.best_val_metric = -float('inf') if self._is_metric_higher_better() else float('inf')
        self.patience_counter = 0
        
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        seed = self.config['seeds']['random_seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
    def _get_device(self):
        """Get the device to use for training"""
        device_config = self.config['device']['device']
        
        if device_config == 'cuda' and torch.cuda.is_available():
            return torch.device(f"cuda:{self.config['device']['cuda_device_id']}")
        elif device_config == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
            
    def _init_data_module(self) -> PhageHostDataModule:
        """Initialize the data module"""
        self.logger.info("Initializing data module...")
        
        return PhageHostDataModule(
            data_path=self.config['data']['data_path'],
            splits_path=self.config['data']['splits_path'],
            embeddings_path=self.config['data']['embeddings_path'],
            batch_size=self.config['training']['batch_size'],
            negative_ratio=self.config['dataset']['negative_ratio'],
            max_markers=self.config['dataset']['max_markers'],
            max_rbps=self.config['dataset']['max_rbps'],
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=self.config['dataset']['pin_memory'],
            augment_train=True,
            seed=self.config['seeds']['random_seed'],
            logger=self.logger
        )
        
    def _init_model(self) -> MILModel:
        """Initialize the model"""
        self.logger.info("Initializing model...")
        
        # Get encoder dimensions based on type
        encoder_type = self.config['model']['encoder_type']
        if encoder_type == 'conservative':
            encoder_dims = self.config['model']['encoder_dims_conservative']
        elif encoder_type == 'balanced':
            encoder_dims = self.config['model']['encoder_dims_balanced']
        elif encoder_type == 'aggressive':
            encoder_dims = self.config['model']['encoder_dims_aggressive']
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
            
        model = MILModel(
            input_dim=self.config['model']['input_dim'],
            encoder_dims=encoder_dims,
            shared_architecture=True,
            dropout=self.config['model']['dropout'],
            activation=self.config['model']['activation'],
            use_layer_norm=self.config['model']['use_layer_norm'],
            temperature=self.config['model']['temperature']
        ).to(self.device)
        
        # Log model info
        log_model_info(model, self.logger)
        
        return model
        
    def _init_loss(self) -> nnPULoss:
        """Initialize the loss function"""
        return nnPULoss(
            prior=self.config['loss']['class_prior'],
            beta=self.config['loss']['beta'],
            gamma=self.config['loss']['gamma']
        )
        
    def _init_optimizer(self) -> optim.Optimizer:
        """Initialize the optimizer"""
        opt_name = self.config['training']['optimizer']
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if opt_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
            
    def _init_scheduler(self) -> Optional[object]:
        """Initialize the learning rate scheduler"""
        scheduler_name = self.config['training']['scheduler']
        
        if scheduler_name == 'none':
            return None
            
        elif scheduler_name == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['scheduler_params']['T_max'],
                eta_min=self.config['training']['scheduler_params']['eta_min']
            )
            
        elif scheduler_name == 'warmup_cosine':
            # Warmup + Cosine Annealing (recommended for PU learning)
            warmup_steps = self.config['training']['scheduler_params'].get('warmup_steps', 1000)
            warmup_epochs = self.config['training']['scheduler_params'].get('warmup_epochs', 5)
            
            # Calculate total steps accounting for gradient accumulation
            accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
            steps_per_epoch = len(self.data_module.train_dataloader()) // accumulation_steps
            total_steps = self.config['training']['num_epochs'] * steps_per_epoch
            
            # Use warmup_epochs if warmup_steps not specified
            if 'warmup_steps' not in self.config['training']['scheduler_params']:
                warmup_steps = warmup_epochs * steps_per_epoch
            
            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup
                    return step / warmup_steps
                # Cosine annealing after warmup
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
            
            scheduler = LambdaLR(self.optimizer, lr_lambda)
            self.logger.info(f"Using warmup_cosine scheduler: {warmup_steps} warmup steps, {total_steps} total steps")
            return scheduler
            
        elif scheduler_name == 'onecycle':
            # One Cycle scheduler (good for finding optimal LR)
            accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
            steps_per_epoch = len(self.data_module.train_dataloader()) // accumulation_steps
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config['training']['learning_rate'] * 10,  # Peak LR
                epochs=self.config['training']['num_epochs'],
                steps_per_epoch=steps_per_epoch,
                pct_start=self.config['training']['scheduler_params'].get('pct_start', 0.3),
                anneal_strategy=self.config['training']['scheduler_params'].get('anneal_strategy', 'cos'),
                div_factor=self.config['training']['scheduler_params'].get('div_factor', 25.0),
                final_div_factor=self.config['training']['scheduler_params'].get('final_div_factor', 10000.0)
            )
            
        elif scheduler_name == 'exponential':
            # Exponential decay (smooth alternative to step)
            from torch.optim.lr_scheduler import ExponentialLR
            return ExponentialLR(
                self.optimizer,
                gamma=self.config['training']['scheduler_params'].get('gamma', 0.95)
            )
            
        elif scheduler_name == 'polynomial':
            # Polynomial decay (used in BERT)
            total_steps = self.config['training']['num_epochs'] * len(self.data_module.train_dataloader())
            power = self.config['training']['scheduler_params'].get('power', 1.0)
            
            def lr_lambda(step):
                return (1 - step / total_steps) ** power
                
            return LambdaLR(self.optimizer, lr_lambda)
            
        elif scheduler_name == 'step':
            return StepLR(
                self.optimizer,
                step_size=self.config['training']['scheduler_params']['step_size'],
                gamma=self.config['training']['scheduler_params']['gamma']
            )
            
        elif scheduler_name == 'plateau':
            # Not recommended for noisy PU learning, but kept for compatibility
            self.logger.warning("Plateau scheduler may not work well with noisy PU learning. Consider 'warmup_cosine' instead.")
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max' if self._is_metric_higher_better() else 'min',
                patience=self.config['training']['scheduler_params']['patience'],
                factor=self.config['training']['scheduler_params']['factor']
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
            
    def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        wandb.init(
            project=self.config['logging']['wandb_project'],
            entity=self.config['logging']['wandb_entity'],
            config=self.config,
            name=self.config.get('experiment_name', None)
        )
        wandb.watch(self.model)
        
    def _is_metric_higher_better(self) -> bool:
        """Check if the validation metric should be maximized"""
        # For most metrics (AUROC, accuracy, F1), higher is better
        return True
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0
        total_positive_risk = 0
        total_negative_risk = 0
        total_samples = 0
        
        train_loader = self.data_module.train_dataloader()
        
        # Gradient accumulation settings
        accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        effective_batch_size = self.config['training']['batch_size'] * accumulation_steps
        
        # Zero gradients at start
        self.optimizer.zero_grad()
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1} Training") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                marker_embeddings = batch['marker_embeddings'].to(self.device)
                rbp_embeddings = batch['rbp_embeddings'].to(self.device)
                marker_mask = batch['marker_mask'].to(self.device)
                rbp_mask = batch['rbp_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    marker_embeddings,
                    rbp_embeddings,
                    marker_mask,
                    rbp_mask
                )
                
                # Compute loss
                loss_dict = self.criterion(outputs['bag_probs'], labels)
                
                # Scale loss by accumulation steps to maintain effective learning rate
                loss = loss_dict['loss'] / accumulation_steps
                
                # Backward pass (accumulate gradients)
                loss.backward()
                
                # Perform optimizer step every accumulation_steps batches
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # Gradient clipping (applied to accumulated gradients)
                    if self.config['training']['gradient_clip'] > 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['training']['gradient_clip']
                        )
                    
                    # Optimizer step
                    self.optimizer.step()
                    
                    # Zero gradients for next accumulation
                    self.optimizer.zero_grad()
                
                # Update per-batch schedulers (only after optimizer step)
                if self.scheduler and ((batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader)):
                    if isinstance(self.scheduler, OneCycleLR):
                        # OneCycle scheduler steps per optimizer step
                        self.scheduler.step()
                    elif isinstance(self.scheduler, LambdaLR) and self.config['training']['scheduler'] == 'warmup_cosine':
                        # Warmup cosine also steps per optimizer step for smooth warmup
                        self.scheduler.step()
                
                # Update metrics (use unscaled loss for accurate tracking)
                batch_size = labels.size(0)
                unscaled_loss = loss_dict['loss'].item()
                total_loss += unscaled_loss * batch_size
                total_positive_risk += loss_dict['positive_risk'].item() * batch_size
                total_negative_risk += loss_dict['negative_risk'].item() * batch_size
                total_samples += batch_size
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Calculate effective batch info for display
                accumulated_batches = min((batch_idx % accumulation_steps) + 1, accumulation_steps)
                effective_samples = accumulated_batches * batch_size
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': unscaled_loss,
                    'pos_risk': loss_dict['positive_risk'].item(),
                    'neg_risk': loss_dict['negative_risk'].item(),
                    'lr': current_lr,
                    'acc_steps': f"{accumulated_batches}/{accumulation_steps}"
                })
                
                # Log to wandb
                if self.config['logging']['use_wandb'] and batch_idx % self.config['logging']['log_every_n_batches'] == 0:
                    wandb.log({
                        'train/batch_loss': unscaled_loss,
                        'train/batch_positive_risk': loss_dict['positive_risk'].item(),
                        'train/batch_negative_risk': loss_dict['negative_risk'].item(),
                        'train/learning_rate': current_lr,
                        'train/effective_batch_size': effective_samples,
                        'train/step': epoch * len(train_loader) + batch_idx
                    })
        
        # Compute epoch metrics
        metrics = {
            'loss': total_loss / total_samples,
            'positive_risk': total_positive_risk / total_samples,
            'negative_risk': total_negative_risk / total_samples
        }
        
        return metrics
        
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        val_loader = self.data_module.val_dataloader()
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1} Validation") as pbar:
                for batch in pbar:
                    # Move batch to device
                    marker_embeddings = batch['marker_embeddings'].to(self.device)
                    rbp_embeddings = batch['rbp_embeddings'].to(self.device)
                    marker_mask = batch['marker_mask'].to(self.device)
                    rbp_mask = batch['rbp_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        marker_embeddings,
                        rbp_embeddings,
                        marker_mask,
                        rbp_mask
                    )
                    
                    # Compute loss
                    loss_dict = self.criterion(outputs['bag_probs'], labels)
                    loss = loss_dict['loss']
                    
                    # Store predictions
                    probabilities = outputs['bag_probs'].cpu().numpy()
                    predictions = (probabilities > self.config['evaluation']['classification_threshold']).astype(float)
                    
                    all_probabilities.extend(probabilities)
                    all_predictions.extend(predictions)
                    all_labels.extend(labels.cpu().numpy())
                    
                    # Update metrics
                    batch_size = labels.size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                    
                    pbar.set_postfix({'loss': loss.item()})
        
        # Compute metrics
        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities),
            self.config['evaluation']['metrics'],
            self.config['evaluation']['k_values']
        )
        
        metrics['loss'] = total_loss / total_samples
        
        return metrics
        
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = Path(self.config['data']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_metric': self.best_val_metric,
            'config': self.config
        }
        
        # Save regular checkpoint
        if epoch % self.config['training']['save_every_n_epochs'] == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
            
        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")
            
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.current_epoch = checkpoint['epoch']
        self.best_val_metric = checkpoint['best_val_metric']
        
        self.logger.info(f"Resumed from epoch {self.current_epoch}")
        
    def train(self):
        """
        Main training loop
        """
        self.logger.info("Starting training...")
        log_training_config(self.config, self.logger)
        
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(self.current_epoch, num_epochs):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            self.logger.info('='*50)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.logger.info(f"Train metrics: {train_metrics}")
            
            # Validate
            if (epoch + 1) % self.config['training']['validate_every_n_epochs'] == 0:
                val_metrics = self.validate(epoch)
                self.logger.info(f"Validation metrics: {val_metrics}")
                
                # Check if best model
                val_metric = val_metrics.get('auroc', val_metrics.get('accuracy', 0))
                is_best = False
                
                if self._is_metric_higher_better():
                    if val_metric > self.best_val_metric:
                        self.best_val_metric = val_metric
                        is_best = True
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                else:
                    if val_metric < self.best_val_metric:
                        self.best_val_metric = val_metric
                        is_best = True
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                        
                # Save checkpoint
                if self.config['training']['save_best_only']:
                    if is_best:
                        self.save_checkpoint(epoch, is_best=True)
                else:
                    self.save_checkpoint(epoch, is_best=is_best)
                    
                # Log metrics
                self.metric_logger.log_metrics(val_metrics, phase='val', epoch=epoch)
                
                if self.config['logging']['use_wandb']:
                    wandb.log({f'val/{k}': v for k, v in val_metrics.items()})
                    
            else:
                val_metrics = {}
                
            # Log training metrics
            self.metric_logger.log_metrics(train_metrics, phase='train', epoch=epoch)
            
            if self.config['logging']['use_wandb']:
                wandb.log({f'train/{k}': v for k, v in train_metrics.items()})
                
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    # Plateau scheduler needs a metric to track
                    metric_value = val_metrics.get('auroc', train_metrics['loss']) if val_metrics else train_metrics['loss']
                    self.scheduler.step(metric_value)
                elif isinstance(self.scheduler, OneCycleLR):
                    # OneCycle scheduler is stepped per batch, not per epoch
                    # (handled in train_epoch method)
                    pass
                elif isinstance(self.scheduler, LambdaLR):
                    # LambdaLR schedulers (warmup_cosine, polynomial) track total steps
                    # Need to update them per batch or per epoch depending on configuration
                    if self.config['training']['scheduler'] == 'warmup_cosine':
                        # Warmup cosine is stepped per batch (handled in train_epoch)
                        pass
                    else:
                        self.scheduler.step()
                else:
                    # Standard schedulers (cosine, step, exponential) - step per epoch
                    self.scheduler.step()
                    
            # Early stopping
            if self.config['training']['early_stopping']:
                if self.patience_counter >= self.config['training']['early_stopping_patience']:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
                    
        self.logger.info("Training completed!")
        
        # Calibrate model if enabled
        if self.config['training'].get('calibrate_model', False):
            self.calibrate_model()
        
    def calibrate_model(self):
        """
        Calibrate the trained model using temperature scaling
        """
        self.logger.info("Calibrating model...")
        
        # Load best checkpoint
        best_checkpoint_path = Path(self.config['data']['checkpoint_dir']) / 'best_model.pt'
        if best_checkpoint_path.exists():
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded best model from {best_checkpoint_path}")
        
        # Get validation loader
        val_loader = self.data_module.val_dataloader()
        
        # Apply temperature scaling
        calibration_method = self.config['training'].get('calibration_method', 'temperature')
        
        if calibration_method == 'temperature':
            # Create temperature-scaled model
            self.calibrated_model = TemperatureScaling(
                self.model,
                device=self.device,
                logger=self.logger
            )
            
            # Optimize temperature on validation set
            optimal_temp = self.calibrated_model.optimize_temperature(
                val_loader,
                criterion=nn.BCELoss(),
                max_iter=self.config['training'].get('calibration_max_iter', 50),
                lr=self.config['training'].get('calibration_lr', 0.01)
            )
            
            self.logger.info(f"Optimal temperature: {optimal_temp:.4f}")
            
        elif calibration_method == 'platt':
            # Create Platt-scaled model
            self.calibrated_model = PlattScaling(
                self.model,
                device=self.device,
                logger=self.logger
            )
            
            # Optimize Platt scaling parameters
            scale, bias = self.calibrated_model.optimize_parameters(
                val_loader,
                max_iter=self.config['training'].get('calibration_max_iter', 100),
                lr=self.config['training'].get('calibration_lr', 0.01)
            )
            
            self.logger.info(f"Platt scaling - Scale: {scale:.4f}, Bias: {bias:.4f}")
        else:
            self.logger.warning(f"Unknown calibration method: {calibration_method}")
            self.calibrated_model = self.model
        
        # Save calibrated model
        self.save_calibrated_model()
        
        # Evaluate calibrated model
        self.logger.info("Evaluating calibrated model...")
        cal_metrics = self.evaluate_calibrated_model(val_loader)
        self.logger.info(f"Calibrated model metrics: {cal_metrics}")
        
    def save_calibrated_model(self):
        """Save the calibrated model"""
        checkpoint_dir = Path(self.config['data']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        calibrated_path = checkpoint_dir / 'calibrated_model.pt'
        
        # Save calibrated model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'calibration_type': self.config['training'].get('calibration_method', 'temperature'),
            'calibration_params': {
                'temperature': self.calibrated_model.get_temperature() if hasattr(self.calibrated_model, 'get_temperature') else 1.0
            },
            'config': self.config
        }, calibrated_path)
        
        self.logger.info(f"Saved calibrated model to {calibrated_path}")
    
    def evaluate_calibrated_model(self, loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate calibrated model and return calibration metrics
        """
        all_probs = []
        all_labels = []
        
        self.calibrated_model.eval()
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating calibration"):
                # Move batch to device
                marker_embeddings = batch['marker_embeddings'].to(self.device)
                rbp_embeddings = batch['rbp_embeddings'].to(self.device)
                marker_mask = batch['marker_mask'].to(self.device)
                rbp_mask = batch['rbp_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get calibrated predictions
                outputs = self.calibrated_model(
                    marker_embeddings,
                    rbp_embeddings,
                    marker_mask,
                    rbp_mask
                )
                
                all_probs.extend(outputs['bag_probs'].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate calibration metrics
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # ECE and MCE
        calibration_metrics = self.calibrated_model.calculate_calibration_metrics(
            torch.tensor(all_probs),
            torch.tensor(all_labels)
        )
        
        return calibration_metrics
        
    def test(self):
        """
        Test the model on the test set
        """
        self.logger.info("Testing model...")
        
        test_loader = self.data_module.test_dataloader()
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        self.model.eval()
        
        with torch.no_grad():
            with tqdm(test_loader, desc="Testing") as pbar:
                for batch in pbar:
                    # Move batch to device
                    marker_embeddings = batch['marker_embeddings'].to(self.device)
                    rbp_embeddings = batch['rbp_embeddings'].to(self.device)
                    marker_mask = batch['marker_mask'].to(self.device)
                    rbp_mask = batch['rbp_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        marker_embeddings,
                        rbp_embeddings,
                        marker_mask,
                        rbp_mask
                    )
                    
                    # Store predictions
                    probabilities = outputs['bag_probs'].cpu().numpy()
                    predictions = (probabilities > self.config['evaluation']['classification_threshold']).astype(float)
                    
                    all_probabilities.extend(probabilities)
                    all_predictions.extend(predictions)
                    all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        test_metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities),
            self.config['evaluation']['metrics'],
            self.config['evaluation']['k_values']
        )
        
        self.logger.info("Test metrics:")
        for key, value in test_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
            
        # Save metrics
        self.metric_logger.log_metrics(test_metrics, phase='test', epoch=self.current_epoch)
        
        if self.config['logging']['use_wandb']:
            wandb.log({f'test/{k}': v for k, v in test_metrics.items()})
            
        # Save predictions if requested
        if self.config['evaluation']['save_predictions']:
            output_dir = Path(self.config['data']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            predictions_dict = {
                'labels': all_labels,
                'predictions': all_predictions,
                'probabilities': all_probabilities
            }
            
            with open(output_dir / 'test_predictions.json', 'w') as f:
                json.dump(predictions_dict, f)
                
            self.logger.info(f"Saved predictions to {output_dir / 'test_predictions.json'}")
            
        return test_metrics


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train phage-host interaction model")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--test_only", action="store_true",
                       help="Only run testing (requires checkpoint)")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Name for this experiment")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Override with command line arguments
    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    else:
        config['experiment_name'] = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    # Set up logging
    logger = setup_logger(
        name='trainer',
        log_dir=config['logging']['log_dir'],
        log_file=f"{config['experiment_name']}.log",
        level=config['logging']['verbosity'].upper()
    )
    
    # Initialize trainer
    trainer = Trainer(config, logger)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        
    # Run training or testing
    if args.test_only:
        if not args.checkpoint:
            raise ValueError("--checkpoint required for test_only mode")
        trainer.test()
    else:
        trainer.train()
        trainer.test()
        
    logger.info("Done!")


if __name__ == "__main__":
    main()