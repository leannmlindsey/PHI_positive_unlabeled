"""
Loss functions for positive-unlabeled learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import logging


class nnPULoss(nn.Module):
    """
    Non-negative PU Loss for positive-unlabeled learning
    
    Based on:
    "Positive-Unlabeled Learning with Non-Negative Risk Estimator"
    Kiryo et al., NeurIPS 2017
    """
    
    def __init__(self,
                 prior: float = 0.3,
                 beta: float = 0.0,
                 gamma: float = 1.0,
                 loss_type: str = 'sigmoid',
                 epsilon: float = 1e-7):
        """
        Initialize nnPU loss
        
        Args:
            prior: Class prior π (estimated proportion of positives in unlabeled data)
            beta: Beta parameter for gradient penalty (0 = no penalty)
            gamma: Gamma parameter for negative risk correction
            loss_type: Type of base loss ('sigmoid' or 'logistic')
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        
        if not 0 < prior < 1:
            raise ValueError(f"Prior must be in (0, 1), got {prior}")
            
        self.prior = prior
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.epsilon = epsilon
        
        # Logging
        logger = logging.getLogger(__name__)
        logger.info(f"nnPULoss initialized with prior={prior}, beta={beta}, "
                   f"gamma={gamma}, loss_type={loss_type}")
        
    def _sigmoid_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Sigmoid loss: -log(sigmoid(y * f(x)))
        
        Args:
            outputs: Model outputs (logits or probabilities)
            targets: Target labels (1 for positive, -1 for negative)
            
        Returns:
            Loss values
        """
        # Convert probabilities to logits if needed
        if outputs.min() >= 0 and outputs.max() <= 1:
            # Outputs are probabilities, convert to logits
            outputs = torch.logit(outputs, eps=self.epsilon)
            
        # Sigmoid loss
        return F.softplus(-targets * outputs)
    
    def _logistic_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Logistic loss: log(1 + exp(-y * f(x)))
        
        Args:
            outputs: Model outputs (logits)
            targets: Target labels (1 for positive, -1 for negative)
            
        Returns:
            Loss values
        """
        return self._sigmoid_loss(outputs, targets)
    
    def forward(self,
                outputs: torch.Tensor,
                labels: torch.Tensor,
                is_unlabeled: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute nnPU loss
        
        Args:
            outputs: Model outputs (bag probabilities) of shape (B,)
            labels: Binary labels (1 for positive, 0 for negative/unlabeled) of shape (B,)
            is_unlabeled: Optional mask indicating unlabeled samples (B,)
                         If None, treats all labels==0 as unlabeled
            
        Returns:
            Dictionary containing:
            - 'loss': Total loss (scalar)
            - 'positive_risk': Risk on positive samples
            - 'negative_risk': Risk on negative samples  
            - 'unlabeled_risk': Risk on unlabeled samples
        """
        # Ensure outputs are 1D
        outputs = outputs.view(-1)
        labels = labels.view(-1)
        
        # Determine positive and unlabeled samples
        if is_unlabeled is None:
            positive_mask = labels == 1
            unlabeled_mask = labels == 0
        else:
            positive_mask = (labels == 1) & ~is_unlabeled
            unlabeled_mask = is_unlabeled
            
        n_positive = positive_mask.sum().item()
        n_unlabeled = unlabeled_mask.sum().item()
        
        # Select appropriate loss function
        if self.loss_type == 'sigmoid':
            loss_func = self._sigmoid_loss
        elif self.loss_type == 'logistic':
            loss_func = self._logistic_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Initialize risks
        positive_risk = torch.tensor(0.0, device=outputs.device)
        negative_risk = torch.tensor(0.0, device=outputs.device)
        unlabeled_risk = torch.tensor(0.0, device=outputs.device)
        
        # Compute positive risk
        if n_positive > 0:
            positive_outputs = outputs[positive_mask]
            # Positive samples have target = 1
            positive_losses = loss_func(positive_outputs, torch.ones_like(positive_outputs))
            positive_risk = positive_losses.mean()
            
        # Compute negative risk on positive samples (for correction)
        if n_positive > 0:
            # Treat positive samples as if they were negative
            positive_outputs = outputs[positive_mask]
            positive_as_negative_losses = loss_func(positive_outputs, -torch.ones_like(positive_outputs))
            positive_negative_risk = positive_as_negative_losses.mean()
        else:
            positive_negative_risk = torch.tensor(0.0, device=outputs.device)
            
        # Compute unlabeled risk
        if n_unlabeled > 0:
            unlabeled_outputs = outputs[unlabeled_mask]
            # Treat unlabeled as negative
            unlabeled_losses = loss_func(unlabeled_outputs, -torch.ones_like(unlabeled_outputs))
            unlabeled_risk = unlabeled_losses.mean()
            
        # Compute negative risk with correction
        # R_n = R_u - π * R_n^+
        negative_risk = unlabeled_risk - self.prior * positive_negative_risk
        
        # Apply non-negative correction
        if negative_risk < 0:
            # Use γ to control the strength of correction
            negative_risk = self.gamma * negative_risk
            # Alternative: negative_risk = -self.beta * negative_risk
            
        # Ensure non-negative
        negative_risk = torch.max(negative_risk, torch.tensor(0.0, device=outputs.device))
        
        # Total risk
        # R = π * R_p^+ + R_n
        total_loss = self.prior * positive_risk + negative_risk
        
        # Add gradient penalty if beta > 0
        if self.beta > 0 and outputs.requires_grad:
            # Gradient penalty to stabilize training
            gradients = torch.autograd.grad(
                outputs=outputs.sum(),
                inputs=outputs,
                create_graph=True,
                retain_graph=True
            )[0]
            gradient_penalty = self.beta * (gradients ** 2).mean()
            total_loss = total_loss + gradient_penalty
            
        return {
            'loss': total_loss,
            'positive_risk': positive_risk.detach(),
            'negative_risk': negative_risk.detach(),
            'unlabeled_risk': unlabeled_risk.detach()
        }


class PULoss(nn.Module):
    """
    Standard PU Loss (without non-negative correction)
    Provided as an alternative/baseline
    """
    
    def __init__(self, prior: float = 0.3, epsilon: float = 1e-7):
        """
        Initialize standard PU loss
        
        Args:
            prior: Class prior π
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.prior = prior
        self.epsilon = epsilon
        
    def forward(self,
                outputs: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute standard PU loss
        
        Args:
            outputs: Model outputs (probabilities)
            labels: Binary labels (1 for positive, 0 for unlabeled)
            
        Returns:
            Loss value
        """
        outputs = outputs.view(-1)
        labels = labels.view(-1)
        
        # Clamp outputs for stability
        outputs = torch.clamp(outputs, self.epsilon, 1 - self.epsilon)
        
        # Positive samples
        positive_mask = labels == 1
        if positive_mask.sum() > 0:
            positive_loss = -torch.log(outputs[positive_mask]).mean()
        else:
            positive_loss = torch.tensor(0.0, device=outputs.device)
            
        # Unlabeled samples (treated as negative)
        unlabeled_mask = labels == 0
        if unlabeled_mask.sum() > 0:
            unlabeled_loss = -torch.log(1 - outputs[unlabeled_mask]).mean()
        else:
            unlabeled_loss = torch.tensor(0.0, device=outputs.device)
            
        # Total loss
        loss = self.prior * positive_loss + unlabeled_loss
        
        return loss