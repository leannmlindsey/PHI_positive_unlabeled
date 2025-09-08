"""
Model calibration techniques for improving probability estimates
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Dict, Any
import logging
from tqdm import tqdm


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for model calibration
    
    Learns a single temperature parameter to calibrate model predictions
    Based on "On Calibration of Modern Neural Networks" (Guo et al., 2017)
    """
    
    def __init__(self, 
                 model: nn.Module,
                 init_temperature: float = 1.0,
                 device: torch.device = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize temperature scaling wrapper
        
        Args:
            model: Base model to calibrate
            init_temperature: Initial temperature value
            device: Device for computation
            logger: Optional logger
        """
        super().__init__()
        self.model = model
        self.device = device or torch.device('cpu')
        self.logger = logger or logging.getLogger(__name__)
        
        # Temperature parameter (learned)
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)
        
        # Move to device
        self.to(self.device)
        
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with temperature scaling
        
        Returns:
            Model outputs with calibrated probabilities
        """
        # Get original model outputs
        with torch.no_grad():
            outputs = self.model(*args, **kwargs)
        
        # Apply temperature scaling to probabilities
        if 'bag_probs' in outputs:
            # Convert probabilities to logits, scale, then back to probabilities
            probs = outputs['bag_probs']
            
            # Clamp to avoid numerical issues
            probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
            
            # Convert to logits
            logits = torch.logit(probs)
            
            # Apply temperature scaling
            scaled_logits = logits / self.temperature
            
            # Convert back to probabilities
            outputs['bag_probs'] = torch.sigmoid(scaled_logits)
            
            # Store temperature for logging
            outputs['temperature'] = self.temperature.item()
        
        return outputs
    
    def optimize_temperature(self, 
                           valid_loader: DataLoader,
                           criterion: nn.Module = None,
                           max_iter: int = 50,
                           lr: float = 0.01,
                           patience: int = 5) -> float:
        """
        Optimize temperature parameter on validation set
        
        Args:
            valid_loader: Validation data loader
            criterion: Loss criterion (default: BCE)
            max_iter: Maximum optimization iterations
            lr: Learning rate for temperature optimization
            patience: Early stopping patience
            
        Returns:
            Optimal temperature value
        """
        if criterion is None:
            criterion = nn.BCELoss()
        
        self.logger.info("Optimizing temperature scaling...")
        
        # Set model to eval mode
        self.model.eval()
        
        # Collect predictions and labels
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Collecting predictions"):
                # Move batch to device
                marker_embeddings = batch['marker_embeddings'].to(self.device)
                rbp_embeddings = batch['rbp_embeddings'].to(self.device)
                marker_mask = batch.get('marker_mask', None)
                rbp_mask = batch.get('rbp_mask', None)
                labels = batch['label'].to(self.device)
                
                if marker_mask is not None:
                    marker_mask = marker_mask.to(self.device)
                if rbp_mask is not None:
                    rbp_mask = rbp_mask.to(self.device)
                
                # Get model predictions
                outputs = self.model(
                    marker_embeddings,
                    rbp_embeddings,
                    marker_mask,
                    rbp_mask
                )
                
                all_probs.append(outputs['bag_probs'])
                all_labels.append(labels)
        
        # Concatenate all predictions
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        
        # Convert probabilities to logits for optimization
        all_logits = torch.logit(torch.clamp(all_probs, 1e-7, 1 - 1e-7))
        
        # Optimize temperature
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        best_loss = float('inf')
        best_temp = self.temperature.item()
        patience_counter = 0
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = all_logits / self.temperature
            scaled_probs = torch.sigmoid(scaled_logits)
            loss = criterion(scaled_probs, all_labels)
            loss.backward()
            return loss
        
        # Optimization loop
        for i in range(max_iter):
            loss = optimizer.step(eval_loss)
            current_loss = loss.item()
            
            self.logger.info(f"Iteration {i+1}: Temperature = {self.temperature.item():.4f}, Loss = {current_loss:.6f}")
            
            # Early stopping
            if current_loss < best_loss:
                best_loss = current_loss
                best_temp = self.temperature.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at iteration {i+1}")
                break
        
        # Set best temperature
        self.temperature.data = torch.tensor([best_temp])
        
        self.logger.info(f"Optimal temperature: {best_temp:.4f}")
        
        # Calculate calibration metrics
        metrics = self.calculate_calibration_metrics(all_probs, all_labels)
        self.logger.info(f"Calibration metrics before: {metrics}")
        
        # Calculate metrics after calibration
        with torch.no_grad():
            scaled_logits = all_logits / self.temperature
            calibrated_probs = torch.sigmoid(scaled_logits)
            
        metrics_after = self.calculate_calibration_metrics(calibrated_probs, all_labels)
        self.logger.info(f"Calibration metrics after: {metrics_after}")
        
        return best_temp
    
    def calculate_calibration_metrics(self, 
                                     probs: torch.Tensor, 
                                     labels: torch.Tensor,
                                     n_bins: int = 10) -> Dict[str, float]:
        """
        Calculate calibration metrics (ECE, MCE)
        
        Args:
            probs: Predicted probabilities
            labels: True labels
            n_bins: Number of bins for calibration
            
        Returns:
            Dictionary of calibration metrics
        """
        probs = probs.cpu().numpy()
        labels = labels.cpu().numpy()
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(probs, labels, n_bins)
        
        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(probs, labels, n_bins)
        
        # Brier Score
        brier_score = np.mean((probs - labels) ** 2)
        
        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier_score
        }
    
    def _calculate_ece(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error
        
        ECE = Σ (|Bm| / n) * |acc(Bm) - conf(Bm)|
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece
    
    def _calculate_mce(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Maximum Calibration Error
        
        MCE = max |acc(Bm) - conf(Bm)|
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
                
        return mce
    
    def set_temperature(self, temperature: float):
        """Manually set temperature value"""
        self.temperature.data = torch.tensor([temperature])
        
    def get_temperature(self) -> float:
        """Get current temperature value"""
        return self.temperature.item()


class PlattScaling(nn.Module):
    """
    Platt scaling for binary calibration
    
    Learns parameters a, b such that calibrated_prob = sigmoid(a * logit + b)
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Platt scaling
        
        Args:
            model: Base model to calibrate
            device: Device for computation
            logger: Optional logger
        """
        super().__init__()
        self.model = model
        self.device = device or torch.device('cpu')
        self.logger = logger or logging.getLogger(__name__)
        
        # Platt scaling parameters
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Move to device
        self.to(self.device)
        
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Platt scaling
        """
        # Get original model outputs
        with torch.no_grad():
            outputs = self.model(*args, **kwargs)
        
        # Apply Platt scaling
        if 'bag_probs' in outputs:
            probs = outputs['bag_probs']
            
            # Convert to logits
            probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
            logits = torch.logit(probs)
            
            # Apply scaling
            scaled_logits = self.scale * logits + self.bias
            
            # Convert back to probabilities
            outputs['bag_probs'] = torch.sigmoid(scaled_logits)
            
            # Store parameters for logging
            outputs['platt_scale'] = self.scale.item()
            outputs['platt_bias'] = self.bias.item()
        
        return outputs
    
    def optimize_parameters(self,
                           valid_loader: DataLoader,
                           max_iter: int = 100,
                           lr: float = 0.01) -> tuple:
        """
        Optimize Platt scaling parameters
        
        Returns:
            Tuple of (scale, bias) parameters
        """
        self.logger.info("Optimizing Platt scaling parameters...")
        
        # Similar optimization as temperature scaling but with two parameters
        optimizer = optim.LBFGS([self.scale, self.bias], lr=lr, max_iter=max_iter)
        
        # ... (similar implementation to temperature scaling)
        
        return self.scale.item(), self.bias.item()