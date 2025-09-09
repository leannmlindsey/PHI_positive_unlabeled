"""
Correct nnPU loss implementation for true PU learning with logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import logging


def logistic_pos(f):
    """Logistic loss for positive samples: log(1 + exp(-f))"""
    return F.softplus(-f)


def logistic_neg(f):
    """Logistic loss for negative samples: log(1 + exp(f))"""
    return F.softplus(f)


class TruePULoss(nn.Module):
    """
    True nnPU loss for positive-unlabeled learning
    Works with bag logits (not probabilities)
    """
    
    def __init__(self, prior: float, reduction: str = 'mean'):
        """
        Initialize nnPU loss
        
        Args:
            prior: True positive prevalence in the unlabeled pool (not artificial ratio!)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        
        if not 0 < prior < 1:
            raise ValueError(f"Prior must be in (0, 1), got {prior}")
        
        self.prior = prior
        self.reduction = reduction
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"TruePULoss initialized with prior={prior}")
        
        # Track correction statistics
        self.correction_count = 0
        self.total_count = 0
    
    def forward(self, 
                bag_logits_pos: torch.Tensor,
                bag_logits_unlab: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute nnPU loss on separate positive and unlabeled batches
        
        Args:
            bag_logits_pos: Logits for positive bags [B_p]
            bag_logits_unlab: Logits for unlabeled bags [B_u]
            
        Returns:
            Dictionary with 'loss' and diagnostics
        """
        # Compute losses using logistic surrogate
        Lp = logistic_pos(bag_logits_pos)  # Loss for treating pos as pos
        Lpn = logistic_neg(bag_logits_pos)  # Loss for treating pos as neg (for correction)
        Lu = logistic_neg(bag_logits_unlab)  # Loss for treating unlab as neg
        
        # Compute expectations
        if self.reduction == 'mean':
            Ep = Lp.mean()
            Ep_neg = Lpn.mean()
            Eu_neg = Lu.mean()
        else:
            Ep = Lp.sum()
            Ep_neg = Lpn.sum()
            Eu_neg = Lu.sum()
        
        # Unbiased negative risk with non-negative correction
        neg_risk = Eu_neg - self.prior * Ep_neg
        
        # Track corrections
        self.total_count += 1
        if neg_risk < 0:
            self.correction_count += 1
            if self.total_count % 100 == 0:
                self.logger.info(f"Negative risk corrections: {self.correction_count}/{self.total_count} "
                               f"({100*self.correction_count/self.total_count:.1f}%)")
        
        neg_risk_corrected = torch.clamp(neg_risk, min=0.0)
        
        # Total risk
        risk = self.prior * Ep + neg_risk_corrected
        
        return {
            'loss': risk,
            'pos_risk': Ep.detach(),
            'neg_risk': neg_risk_corrected.detach(),
            'neg_risk_uncorrected': neg_risk.detach(),
            'correction_applied': (neg_risk < 0).float()
        }


class MixedBatchPULoss(nn.Module):
    """
    nnPU loss for mixed batches (positive and unlabeled in same batch)
    This is less ideal than separate streams but sometimes necessary
    """
    
    def __init__(self, prior: float, reduction: str = 'mean'):
        """
        Initialize nnPU loss for mixed batches
        
        Args:
            prior: True positive prevalence in the unlabeled pool
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        
        if not 0 < prior < 1:
            raise ValueError(f"Prior must be in (0, 1), got {prior}")
        
        self.prior = prior
        self.reduction = reduction
        self.logger = logging.getLogger(__name__)
    
    def forward(self,
                bag_logits: torch.Tensor,
                labels: torch.Tensor,
                is_unlabeled: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute nnPU loss on mixed batch
        
        Args:
            bag_logits: Bag logits [B]
            labels: Binary labels (1 for positive, 0 for unlabeled) [B]
            is_unlabeled: Optional mask for unlabeled (if None, uses labels==0)
            
        Returns:
            Dictionary with loss and diagnostics
        """
        # Determine positive and unlabeled samples
        if is_unlabeled is None:
            pos_mask = labels == 1
            unlab_mask = labels == 0
        else:
            pos_mask = (labels == 1) & ~is_unlabeled
            unlab_mask = is_unlabeled
        
        n_pos = pos_mask.sum().item()
        n_unlab = unlab_mask.sum().item()
        
        # Initialize risks
        device = bag_logits.device
        Ep = torch.tensor(0.0, device=device)
        Ep_neg = torch.tensor(0.0, device=device)
        Eu_neg = torch.tensor(0.0, device=device)
        
        # Compute positive risks
        if n_pos > 0:
            logits_pos = bag_logits[pos_mask]
            Lp = logistic_pos(logits_pos)
            Lpn = logistic_neg(logits_pos)
            
            if self.reduction == 'mean':
                Ep = Lp.mean()
                Ep_neg = Lpn.mean()
            else:
                Ep = Lp.sum()
                Ep_neg = Lpn.sum()
        
        # Compute unlabeled risk
        if n_unlab > 0:
            logits_unlab = bag_logits[unlab_mask]
            Lu = logistic_neg(logits_unlab)
            
            if self.reduction == 'mean':
                Eu_neg = Lu.mean()
            else:
                Eu_neg = Lu.sum()
        
        # Compute negative risk with correction
        neg_risk = Eu_neg - self.prior * Ep_neg
        neg_risk_corrected = torch.clamp(neg_risk, min=0.0)
        
        # Total risk
        risk = self.prior * Ep + neg_risk_corrected
        
        return {
            'loss': risk,
            'pos_risk': Ep.detach(),
            'neg_risk': neg_risk_corrected.detach(),
            'n_pos': n_pos,
            'n_unlab': n_unlab
        }


class SupervisedLoss(nn.Module):
    """
    Standard supervised loss for comparison/debugging
    Uses BCEWithLogitsLoss with optional class weighting
    """
    
    def __init__(self, pos_weight: Optional[float] = None):
        """
        Initialize supervised loss
        
        Args:
            pos_weight: Weight for positive class (useful for imbalanced data)
        """
        super().__init__()
        self.pos_weight = torch.tensor(pos_weight) if pos_weight else None
    
    def forward(self,
                bag_logits: torch.Tensor,
                labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute supervised BCE loss
        
        Args:
            bag_logits: Bag logits [B]
            labels: Binary labels (1 for positive, 0 for negative) [B]
            
        Returns:
            Dictionary with loss
        """
        loss = F.binary_cross_entropy_with_logits(
            bag_logits,
            labels.float(),
            pos_weight=self.pos_weight
        )
        
        # Compute per-class losses for monitoring
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        pos_loss = 0.0
        neg_loss = 0.0
        
        if pos_mask.sum() > 0:
            pos_loss = F.binary_cross_entropy_with_logits(
                bag_logits[pos_mask],
                labels[pos_mask].float(),
                reduction='mean'
            ).item()
        
        if neg_mask.sum() > 0:
            neg_loss = F.binary_cross_entropy_with_logits(
                bag_logits[neg_mask],
                labels[neg_mask].float(),
                reduction='mean'
            ).item()
        
        return {
            'loss': loss,
            'pos_loss': pos_loss,
            'neg_loss': neg_loss
        }