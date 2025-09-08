"""
Standalone evaluation script for trained models
Provides detailed evaluation and analysis of model performance
"""

import os
import sys
import argparse
import yaml
import json
import pickle
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.mil_model import MILModel
from models.losses import nnPULoss
from training.dataset import PhageHostDataModule
from training.evaluation import (
    compute_metrics,
    compute_optimal_threshold,
    compute_calibration_metrics,
    generate_confusion_matrix
)
from utils.logging_utils import setup_logger


class ModelEvaluator:
    """
    Comprehensive model evaluation class
    """
    
    def __init__(self, 
                 config_path: str,
                 checkpoint_path: str,
                 output_dir: str,
                 logger: logging.Logger):
        """
        Initialize the evaluator
        
        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to model checkpoint
            output_dir: Directory for output files
            logger: Logger instance
        """
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set device
        self.device = self._get_device()
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Initialize data module
        self.data_module = self._init_data_module()
        
        # Initialize loss function
        self.criterion = nnPULoss(
            prior=self.config['loss']['class_prior'],
            beta=self.config['loss']['beta'],
            gamma=self.config['loss']['gamma']
        )
        
    def _get_device(self):
        """Get the device to use"""
        device_config = self.config['device']['device']
        
        if device_config == 'cuda' and torch.cuda.is_available():
            return torch.device(f"cuda:{self.config['device']['cuda_device_id']}")
        elif device_config == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
            
    def _load_model(self, checkpoint_path: str) -> MILModel:
        """Load model from checkpoint"""
        self.logger.info(f"Loading model from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get encoder dimensions
        encoder_type = self.config['model']['encoder_type']
        if encoder_type == 'conservative':
            encoder_dims = self.config['model']['encoder_dims_conservative']
        elif encoder_type == 'balanced':
            encoder_dims = self.config['model']['encoder_dims_balanced']
        elif encoder_type == 'aggressive':
            encoder_dims = self.config['model']['encoder_dims_aggressive']
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
            
        # Initialize model
        model = MILModel(
            input_dim=self.config['model']['input_dim'],
            encoder_dims=encoder_dims,
            shared_architecture=True,
            dropout=self.config['model']['dropout'],
            activation=self.config['model']['activation'],
            use_layer_norm=self.config['model']['use_layer_norm'],
            temperature=self.config['model']['temperature']
        ).to(self.device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        self.logger.info(f"Model loaded from epoch {checkpoint['epoch']}")
        
        return model
        
    def _init_data_module(self) -> PhageHostDataModule:
        """Initialize data module"""
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
            augment_train=False,  # No augmentation for evaluation
            seed=self.config['seeds']['random_seed'],
            logger=self.logger
        )
        
    def evaluate_split(self, split: str = 'test') -> Dict[str, Any]:
        """
        Evaluate model on a specific split
        
        Args:
            split: Which split to evaluate ('train', 'val', 'test')
            
        Returns:
            Dictionary of evaluation results
        """
        self.logger.info(f"Evaluating on {split} split...")
        
        # Get appropriate dataloader
        if split == 'train':
            dataloader = self.data_module.train_dataloader()
        elif split == 'val':
            dataloader = self.data_module.val_dataloader()
        elif split == 'test':
            dataloader = self.data_module.test_dataloader()
        else:
            raise ValueError(f"Unknown split: {split}")
            
        # Collect predictions
        all_labels = []
        all_probabilities = []
        all_pairwise_probs = []
        total_loss = 0
        total_samples = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
                # Move batch to device
                marker_embeddings = batch['marker_embeddings'].to(self.device)
                rbp_embeddings = batch['rbp_embeddings'].to(self.device)
                marker_mask = batch['marker_mask'].to(self.device)
                rbp_mask = batch['rbp_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass with pairwise probabilities
                outputs = self.model(
                    marker_embeddings,
                    rbp_embeddings,
                    marker_mask,
                    rbp_mask,
                    return_pairwise=True
                )
                
                # Compute loss
                loss_dict = self.criterion(outputs['bag_probs'], labels)
                
                # Store results
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(outputs['bag_probs'].cpu().numpy())
                all_pairwise_probs.append(outputs['pairwise_probs'].cpu().numpy())
                
                total_loss += loss_dict['loss'].item() * len(labels)
                total_samples += len(labels)
                
        # Convert to arrays
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Compute predictions with default threshold
        threshold = self.config['evaluation']['classification_threshold']
        all_predictions = (all_probabilities > threshold).astype(int)
        
        # Compute metrics
        metrics = compute_metrics(
            all_labels,
            all_predictions,
            all_probabilities,
            self.config['evaluation']['metrics'],
            self.config['evaluation']['k_values']
        )
        
        # Add loss
        metrics['loss'] = total_loss / total_samples
        
        # Find optimal threshold
        optimal_threshold = compute_optimal_threshold(
            all_labels,
            all_probabilities,
            metric='f1'
        )
        
        # Compute metrics with optimal threshold
        optimal_predictions = (all_probabilities > optimal_threshold).astype(int)
        optimal_metrics = compute_metrics(
            all_labels,
            optimal_predictions,
            all_probabilities,
            self.config['evaluation']['metrics'],
            None  # Skip k metrics for optimal
        )
        
        # Compute calibration metrics
        calibration = compute_calibration_metrics(all_labels, all_probabilities)
        
        # Generate confusion matrix
        cm = generate_confusion_matrix(all_labels, all_predictions)
        cm_optimal = generate_confusion_matrix(all_labels, optimal_predictions)
        
        # Compile results
        results = {
            'split': split,
            'n_samples': len(all_labels),
            'n_positive': int(np.sum(all_labels == 1)),
            'n_negative': int(np.sum(all_labels == 0)),
            'metrics': metrics,
            'optimal_threshold': float(optimal_threshold),
            'optimal_metrics': optimal_metrics,
            'calibration': calibration,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_optimal': cm_optimal.tolist(),
            'labels': all_labels.tolist(),
            'probabilities': all_probabilities.tolist(),
            'predictions': all_predictions.tolist()
        }
        
        return results
        
    def analyze_errors(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Analyze prediction errors
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            DataFrame with error analysis
        """
        labels = np.array(results['labels'])
        probabilities = np.array(results['probabilities'])
        predictions = np.array(results['predictions'])
        
        # Find errors
        errors = predictions != labels
        
        # False positives
        false_positives = (predictions == 1) & (labels == 0)
        false_negatives = (predictions == 0) & (labels == 1)
        
        # Create error analysis
        error_data = []
        
        for i in range(len(labels)):
            if errors[i]:
                error_type = 'FP' if false_positives[i] else 'FN'
                error_data.append({
                    'index': i,
                    'true_label': labels[i],
                    'prediction': predictions[i],
                    'probability': probabilities[i],
                    'error_type': error_type,
                    'confidence': probabilities[i] if error_type == 'FP' else 1 - probabilities[i]
                })
                
        error_df = pd.DataFrame(error_data)
        
        if len(error_df) > 0:
            # Sort by confidence (most confident errors first)
            error_df = error_df.sort_values('confidence', ascending=False)
            
        return error_df
        
    def plot_metrics(self, results: Dict[str, Any]):
        """
        Create visualization plots
        
        Args:
            results: Evaluation results dictionary
        """
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Confusion Matrix
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. Probability Distribution
        labels = np.array(results['labels'])
        probabilities = np.array(results['probabilities'])
        
        axes[0, 1].hist(probabilities[labels == 0], bins=30, alpha=0.5, 
                       label='Negative', color='red', density=True)
        axes[0, 1].hist(probabilities[labels == 1], bins=30, alpha=0.5,
                       label='Positive', color='blue', density=True)
        axes[0, 1].axvline(x=results['optimal_threshold'], color='green', 
                          linestyle='--', label=f'Optimal Threshold: {results["optimal_threshold"]:.3f}')
        axes[0, 1].set_xlabel('Prediction Probability')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Probability Distribution by Class')
        axes[0, 1].legend()
        
        # 3. ROC Curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(labels, probabilities)
        roc_auc = auc(fpr, tpr)
        
        axes[0, 2].plot(fpr, tpr, color='darkorange', lw=2,
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 2].set_xlim([0.0, 1.0])
        axes[0, 2].set_ylim([0.0, 1.05])
        axes[0, 2].set_xlabel('False Positive Rate')
        axes[0, 2].set_ylabel('True Positive Rate')
        axes[0, 2].set_title('ROC Curve')
        axes[0, 2].legend(loc="lower right")
        
        # 4. Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        avg_precision = average_precision_score(labels, probabilities)
        
        axes[1, 0].plot(recall, precision, color='darkgreen', lw=2,
                       label=f'PR curve (AP = {avg_precision:.3f})')
        axes[1, 0].set_xlim([0.0, 1.0])
        axes[1, 0].set_ylim([0.0, 1.05])
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].legend(loc="lower left")
        
        # 5. Hit@K and Recall@K
        k_values = self.config['evaluation']['k_values']
        hit_at_k = [results['metrics'][f'hit@{k}'] for k in k_values]
        recall_at_k = [results['metrics'][f'recall@{k}'] for k in k_values]
        
        axes[1, 1].plot(k_values, hit_at_k, marker='o', label='Hit@K')
        axes[1, 1].plot(k_values, recall_at_k, marker='s', label='Recall@K')
        axes[1, 1].set_xlabel('K')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Top-K Metrics')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Calibration Plot
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(n_bins):
            bin_mask = (probabilities > bin_boundaries[i]) & (probabilities <= bin_boundaries[i+1])
            if bin_mask.sum() > 0:
                bin_accuracies.append(labels[bin_mask].mean())
                bin_confidences.append(probabilities[bin_mask].mean())
                bin_counts.append(bin_mask.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(bin_centers[i])
                bin_counts.append(0)
                
        axes[1, 2].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        axes[1, 2].scatter(bin_confidences, bin_accuracies, s=np.array(bin_counts)*10,
                          alpha=0.7, label='Model calibration')
        axes[1, 2].set_xlabel('Mean Predicted Probability')
        axes[1, 2].set_ylabel('Fraction of Positives')
        axes[1, 2].set_title(f'Calibration Plot (ECE={results["calibration"]["ece"]:.3f})')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / f"{results['split']}_evaluation_plots.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved plots to {output_path}")
        
    def generate_report(self, results: Dict[str, Any]):
        """
        Generate a comprehensive evaluation report
        
        Args:
            results: Evaluation results dictionary
        """
        report = []
        report.append("="*60)
        report.append(f"EVALUATION REPORT - {results['split'].upper()} SET")
        report.append("="*60)
        report.append("")
        
        # Dataset statistics
        report.append("Dataset Statistics:")
        report.append(f"  Total samples: {results['n_samples']}")
        report.append(f"  Positive samples: {results['n_positive']} ({results['n_positive']/results['n_samples']:.1%})")
        report.append(f"  Negative samples: {results['n_negative']} ({results['n_negative']/results['n_samples']:.1%})")
        report.append("")
        
        # Metrics with default threshold
        report.append(f"Metrics (threshold={self.config['evaluation']['classification_threshold']}):")
        for key, value in results['metrics'].items():
            if not key.startswith('hit@') and not key.startswith('recall@'):
                report.append(f"  {key:15s}: {value:.4f}")
        report.append("")
        
        # Metrics with optimal threshold
        report.append(f"Metrics (optimal threshold={results['optimal_threshold']:.3f}):")
        for key, value in results['optimal_metrics'].items():
            if not key.startswith('hit@') and not key.startswith('recall@'):
                report.append(f"  {key:15s}: {value:.4f}")
        report.append("")
        
        # Calibration metrics
        report.append("Calibration Metrics:")
        report.append(f"  ECE: {results['calibration']['ece']:.4f}")
        report.append(f"  MCE: {results['calibration']['mce']:.4f}")
        report.append("")
        
        # Top-K metrics
        report.append("Top-K Metrics:")
        for k in [1, 5, 10, 20]:
            if f'hit@{k}' in results['metrics']:
                report.append(f"  Hit@{k:2d}: {results['metrics'][f'hit@{k}']:.4f}  "
                            f"Recall@{k:2d}: {results['metrics'][f'recall@{k}']:.4f}")
        report.append("")
        
        # Confusion Matrix
        cm = np.array(results['confusion_matrix'])
        report.append("Confusion Matrix:")
        report.append("            Predicted")
        report.append("            Neg    Pos")
        report.append(f"Actual Neg  {cm[0,0]:5d}  {cm[0,1]:5d}")
        report.append(f"       Pos  {cm[1,0]:5d}  {cm[1,1]:5d}")
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.output_dir / f"{results['split']}_evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write(report_text)
            
        # Also print to console
        print(report_text)
        
        self.logger.info(f"Saved report to {report_path}")
        
    def run_full_evaluation(self):
        """
        Run complete evaluation on all splits
        """
        all_results = {}
        
        for split in ['train', 'val', 'test']:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Evaluating {split} split")
            self.logger.info('='*60)
            
            # Evaluate
            results = self.evaluate_split(split)
            all_results[split] = results
            
            # Generate plots
            self.plot_metrics(results)
            
            # Generate report
            self.generate_report(results)
            
            # Analyze errors
            error_df = self.analyze_errors(results)
            if len(error_df) > 0:
                error_path = self.output_dir / f"{split}_errors.csv"
                error_df.to_csv(error_path, index=False)
                self.logger.info(f"Saved error analysis to {error_path}")
                
        # Save all results to JSON
        results_path = self.output_dir / "evaluation_results.json"
        
        # Remove non-serializable data for JSON
        json_results = {}
        for split, res in all_results.items():
            json_results[split] = {
                'metrics': res['metrics'],
                'optimal_threshold': res['optimal_threshold'],
                'optimal_metrics': res['optimal_metrics'],
                'calibration': res['calibration'],
                'n_samples': res['n_samples'],
                'n_positive': res['n_positive'],
                'n_negative': res['n_negative']
            }
            
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
            
        self.logger.info(f"\nSaved all results to {results_path}")
        
        # Create comparison table
        self._create_comparison_table(json_results)
        
    def _create_comparison_table(self, results: Dict[str, Any]):
        """
        Create a comparison table across splits
        
        Args:
            results: Results dictionary for all splits
        """
        # Create DataFrame
        comparison_data = []
        
        for split in ['train', 'val', 'test']:
            if split in results:
                row = {'split': split}
                row.update(results[split]['metrics'])
                comparison_data.append(row)
                
        df = pd.DataFrame(comparison_data)
        
        # Select important columns
        important_cols = ['split', 'accuracy', 'mcc', 'f1', 'precision', 'recall', 
                         'auroc', 'auprc', 'hit@5', 'recall@5']
        
        # Filter to available columns
        available_cols = [col for col in important_cols if col in df.columns]
        df = df[available_cols]
        
        # Format numeric columns
        for col in df.columns:
            if col != 'split':
                df[col] = df[col].apply(lambda x: f"{x:.4f}")
                
        # Save to CSV
        table_path = self.output_dir / "comparison_table.csv"
        df.to_csv(table_path, index=False)
        
        # Print table
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        print(df.to_string(index=False))
        
        self.logger.info(f"Saved comparison table to {table_path}")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Directory for output files")
    parser.add_argument("--split", type=str, default=None,
                       help="Specific split to evaluate (train/val/test). If not specified, evaluates all.")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger(
        name='evaluator',
        log_dir=args.output_dir,
        log_file='evaluation.log'
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        logger=logger
    )
    
    # Run evaluation
    if args.split:
        # Evaluate specific split
        results = evaluator.evaluate_split(args.split)
        evaluator.plot_metrics(results)
        evaluator.generate_report(results)
        
        # Analyze errors
        error_df = evaluator.analyze_errors(results)
        if len(error_df) > 0:
            error_path = Path(args.output_dir) / f"{args.split}_errors.csv"
            error_df.to_csv(error_path, index=False)
            logger.info(f"Saved error analysis to {error_path}")
    else:
        # Run full evaluation on all splits
        evaluator.run_full_evaluation()
        
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()