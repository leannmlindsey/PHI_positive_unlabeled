"""
Evaluation metrics for phage-host interaction prediction
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
import logging


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    metric_names: List[str],
    k_values: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics
    
    Args:
        labels: True labels (binary)
        predictions: Binary predictions
        probabilities: Prediction probabilities
        metric_names: List of metrics to compute
        k_values: K values for Hit@K and Recall@K metrics
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Binary classification metrics
    if 'accuracy' in metric_names:
        metrics['accuracy'] = accuracy_score(labels, predictions)
        
    if 'mcc' in metric_names:
        metrics['mcc'] = matthews_corrcoef(labels, predictions)
        
    if 'f1' in metric_names:
        metrics['f1'] = f1_score(labels, predictions, average='binary')
        
    if 'precision' in metric_names:
        metrics['precision'] = precision_score(labels, predictions, average='binary', zero_division=0)
        
    if 'recall' in metric_names:
        metrics['recall'] = recall_score(labels, predictions, average='binary')
        
    # Ranking metrics (need probabilities)
    if 'auroc' in metric_names:
        try:
            metrics['auroc'] = roc_auc_score(labels, probabilities)
        except ValueError:
            # Handle case where only one class is present
            metrics['auroc'] = 0.5
            
    if 'auprc' in metric_names:
        try:
            metrics['auprc'] = average_precision_score(labels, probabilities)
        except ValueError:
            metrics['auprc'] = np.mean(labels)
            
    # Top-K metrics
    if k_values is not None:
        for k in k_values:
            hit_at_k = compute_hit_at_k(labels, probabilities, k)
            recall_at_k = compute_recall_at_k(labels, probabilities, k)
            
            metrics[f'hit@{k}'] = hit_at_k
            metrics[f'recall@{k}'] = recall_at_k
            
    return metrics


def compute_hit_at_k(labels: np.ndarray, probabilities: np.ndarray, k: int) -> float:
    """
    Compute Hit Rate @ K
    Proportion of test cases where at least one true positive appears in top-K predictions
    
    Args:
        labels: True labels
        probabilities: Prediction probabilities
        k: Number of top predictions to consider
        
    Returns:
        Hit rate @ K
    """
    if len(labels) <= k:
        # If we have fewer samples than k, adjust
        k = len(labels)
        
    # Get indices of top-k predictions
    top_k_indices = np.argpartition(probabilities, -k)[-k:]
    
    # Check if any of the top-k predictions are positive
    top_k_labels = labels[top_k_indices]
    
    # Hit if at least one positive in top-k
    hit = np.any(top_k_labels == 1)
    
    return float(hit)


def compute_recall_at_k(labels: np.ndarray, probabilities: np.ndarray, k: int) -> float:
    """
    Compute Recall @ K
    Proportion of true positives that appear in top-K predictions
    
    Args:
        labels: True labels
        probabilities: Prediction probabilities
        k: Number of top predictions to consider
        
    Returns:
        Recall @ K
    """
    n_positives = np.sum(labels == 1)
    
    if n_positives == 0:
        return 0.0
        
    if len(labels) <= k:
        k = len(labels)
        
    # Get indices of top-k predictions
    top_k_indices = np.argpartition(probabilities, -k)[-k:]
    
    # Count true positives in top-k
    top_k_labels = labels[top_k_indices]
    n_true_positives_in_top_k = np.sum(top_k_labels == 1)
    
    recall_at_k = n_true_positives_in_top_k / n_positives
    
    return float(recall_at_k)


def compute_hit_recall_at_k_batch(
    labels: np.ndarray,
    probabilities: np.ndarray,
    k_values: List[int]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute Hit@K and Recall@K for multiple K values efficiently
    
    Args:
        labels: True labels
        probabilities: Prediction probabilities
        k_values: List of K values
        
    Returns:
        Tuple of (hit@k dict, recall@k dict)
    """
    hit_at_k = {}
    recall_at_k = {}
    
    # Sort indices by probability (descending)
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_labels = labels[sorted_indices]
    
    n_positives = np.sum(labels == 1)
    
    for k in k_values:
        if k > len(labels):
            k_actual = len(labels)
        else:
            k_actual = k
            
        # Get top-k labels
        top_k_labels = sorted_labels[:k_actual]
        
        # Hit@K: at least one positive in top-k
        hit_at_k[f'hit@{k}'] = float(np.any(top_k_labels == 1))
        
        # Recall@K: proportion of positives in top-k
        if n_positives > 0:
            n_positives_in_top_k = np.sum(top_k_labels == 1)
            recall_at_k[f'recall@{k}'] = float(n_positives_in_top_k / n_positives)
        else:
            recall_at_k[f'recall@{k}'] = 0.0
            
    return hit_at_k, recall_at_k


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    config: Dict,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Evaluate model on a dataset
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to use
        config: Configuration dictionary
        logger: Optional logger
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
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
            
            # Compute loss
            loss_dict = criterion(outputs['bag_probs'], labels)
            loss = loss_dict['loss']
            
            # Store predictions
            probabilities = outputs['bag_probs'].cpu().numpy()
            threshold = config['evaluation']['classification_threshold']
            predictions = (probabilities > threshold).astype(float)
            
            all_probabilities.extend(probabilities)
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            
            # Update metrics
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Compute metrics
    metrics = compute_metrics(
        all_labels,
        all_predictions,
        all_probabilities,
        config['evaluation']['metrics'],
        config['evaluation'].get('k_values', None)
    )
    
    # Add loss
    metrics['loss'] = total_loss / total_samples
    
    # Log if logger provided
    if logger:
        logger.info("Evaluation metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
            
    return metrics


def generate_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray
) -> np.ndarray:
    """
    Generate confusion matrix
    
    Args:
        labels: True labels
        predictions: Predicted labels
        
    Returns:
        Confusion matrix
    """
    return confusion_matrix(labels, predictions)


def compute_optimal_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metric: str = 'f1'
) -> float:
    """
    Find optimal classification threshold
    
    Args:
        labels: True labels
        probabilities: Prediction probabilities
        metric: Metric to optimize ('f1', 'mcc', 'accuracy')
        
    Returns:
        Optimal threshold
    """
    thresholds = np.linspace(0, 1, 101)
    best_score = -float('inf')
    best_threshold = 0.5
    
    for threshold in thresholds:
        predictions = (probabilities > threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(labels, predictions, average='binary')
        elif metric == 'mcc':
            score = matthews_corrcoef(labels, predictions)
        elif metric == 'accuracy':
            score = accuracy_score(labels, predictions)
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        if score > best_score:
            best_score = score
            best_threshold = threshold
            
    return best_threshold


def compute_calibration_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute calibration metrics (ECE, MCE)
    
    Args:
        labels: True labels
        probabilities: Prediction probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary with ECE and MCE
    """
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0  # Expected Calibration Error
    mce = 0  # Maximum Calibration Error
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Accuracy in this bin
            accuracy_in_bin = labels[in_bin].mean()
            # Average confidence in this bin
            avg_confidence_in_bin = probabilities[in_bin].mean()
            # Calibration error for this bin
            calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
            
            ece += prop_in_bin * calibration_error
            mce = max(mce, calibration_error)
            
    return {
        'ece': float(ece),
        'mce': float(mce)
    }