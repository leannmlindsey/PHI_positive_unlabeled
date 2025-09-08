"""
Inference script for making predictions on new phage-host pairs
"""

import os
import sys
import argparse
import yaml
import json
import hashlib
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import h5py
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.mil_model import MILModel
from utils.data_utils import EmbeddingLoader
from utils.logging_utils import setup_logger
from scripts.generate_embeddings import ESM2EmbeddingGenerator


class PhageHostPredictor:
    """
    Predictor for new phage-host interactions
    """
    
    def __init__(self,
                 config_path: str,
                 checkpoint_path: str,
                 embeddings_path: str,
                 logger: logging.Logger):
        """
        Initialize the predictor
        
        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to model checkpoint
            embeddings_path: Path to pre-computed embeddings
            logger: Logger instance
        """
        self.logger = logger
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set device
        self.device = self._get_device()
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Load embeddings
        self.embedding_loader = EmbeddingLoader(embeddings_path, logger)
        
        # Cache for new embeddings
        self.new_embeddings = {}
        
        # ESM-2 generator (initialized lazily)
        self.esm_generator = None
        
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
            dropout=0.0,  # No dropout for inference
            activation=self.config['model']['activation'],
            use_layer_norm=self.config['model']['use_layer_norm'],
            temperature=self.config['model']['temperature']
        ).to(self.device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        self.logger.info(f"Model loaded successfully")
        
        return model
        
    def _get_embedding(self, sequence: str, sequence_hash: Optional[str] = None) -> np.ndarray:
        """
        Get embedding for a protein sequence
        
        Args:
            sequence: Protein sequence
            sequence_hash: Optional pre-computed MD5 hash
            
        Returns:
            Embedding array
        """
        # Compute hash if not provided
        if sequence_hash is None:
            sequence_hash = hashlib.md5(sequence.encode()).hexdigest()
            
        # Check if embedding exists in loader
        try:
            return self.embedding_loader.get_embedding(sequence_hash)
        except KeyError:
            # Check if in new embeddings cache
            if sequence_hash in self.new_embeddings:
                return self.new_embeddings[sequence_hash]
                
            # Generate new embedding
            self.logger.info(f"Generating embedding for new sequence (hash: {sequence_hash})")
            
            if self.esm_generator is None:
                self.logger.info("Initializing ESM-2 generator...")
                self.esm_generator = ESM2EmbeddingGenerator(
                    model_name=self.config['model']['input_dim'] == 1280 and 
                              "facebook/esm2_t33_650M_UR50D" or "facebook/esm2_t30_150M_UR50D",
                    device=str(self.device),
                    logger=self.logger
                )
                self.esm_generator.load_model()
                
            # Generate embedding
            embedding = self.esm_generator.generate_embedding(sequence)
            
            # Cache it
            self.new_embeddings[sequence_hash] = embedding
            
            return embedding
            
    def predict_single(self,
                      marker_sequences: List[str],
                      rbp_sequences: List[str],
                      return_pairwise: bool = False) -> Dict[str, Any]:
        """
        Predict interaction for a single phage-host pair
        
        Args:
            marker_sequences: List of host marker protein sequences
            rbp_sequences: List of phage RBP sequences
            return_pairwise: Whether to return pairwise probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Get embeddings
        marker_embeddings = []
        for seq in marker_sequences:
            embedding = self._get_embedding(seq)
            marker_embeddings.append(embedding)
            
        rbp_embeddings = []
        for seq in rbp_sequences:
            embedding = self._get_embedding(seq)
            rbp_embeddings.append(embedding)
            
        # Convert to tensors and add batch dimension
        marker_embeddings = torch.tensor(np.stack(marker_embeddings), dtype=torch.float32).unsqueeze(0)
        rbp_embeddings = torch.tensor(np.stack(rbp_embeddings), dtype=torch.float32).unsqueeze(0)
        
        # Create masks
        marker_mask = torch.ones(1, len(marker_sequences), dtype=torch.float32)
        rbp_mask = torch.ones(1, len(rbp_sequences), dtype=torch.float32)
        
        # Pad if necessary
        max_markers = self.config['dataset']['max_markers']
        max_rbps = self.config['dataset']['max_rbps']
        
        if marker_embeddings.shape[1] < max_markers:
            padding = torch.zeros(1, max_markers - marker_embeddings.shape[1], 
                                 marker_embeddings.shape[2])
            marker_embeddings = torch.cat([marker_embeddings, padding], dim=1)
            mask_padding = torch.zeros(1, max_markers - marker_mask.shape[1])
            marker_mask = torch.cat([marker_mask, mask_padding], dim=1)
            
        if rbp_embeddings.shape[1] < max_rbps:
            padding = torch.zeros(1, max_rbps - rbp_embeddings.shape[1],
                                 rbp_embeddings.shape[2])
            rbp_embeddings = torch.cat([rbp_embeddings, padding], dim=1)
            mask_padding = torch.zeros(1, max_rbps - rbp_mask.shape[1])
            rbp_mask = torch.cat([rbp_mask, mask_padding], dim=1)
            
        # Move to device
        marker_embeddings = marker_embeddings.to(self.device)
        rbp_embeddings = rbp_embeddings.to(self.device)
        marker_mask = marker_mask.to(self.device)
        rbp_mask = rbp_mask.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(
                marker_embeddings,
                rbp_embeddings,
                marker_mask,
                rbp_mask,
                return_pairwise=return_pairwise
            )
            
        # Extract results
        probability = outputs['bag_probs'].cpu().numpy()[0]
        prediction = int(probability > self.config['evaluation']['classification_threshold'])
        
        result = {
            'probability': float(probability),
            'prediction': prediction,
            'predicted_class': 'positive' if prediction == 1 else 'negative'
        }
        
        if return_pairwise:
            pairwise_probs = outputs['pairwise_probs'].cpu().numpy()[0]
            # Only return non-padded values
            pairwise_probs = pairwise_probs[:len(marker_sequences), :len(rbp_sequences)]
            result['pairwise_probabilities'] = pairwise_probs.tolist()
            
        return result
        
    def predict_batch(self,
                     data_path: str,
                     output_path: str,
                     has_labels: bool = False) -> pd.DataFrame:
        """
        Predict interactions for a batch of phage-host pairs
        
        Args:
            data_path: Path to input TSV file
            output_path: Path to save predictions
            has_labels: Whether the input file has labels
            
        Returns:
            DataFrame with predictions
        """
        self.logger.info(f"Loading data from {data_path}")
        
        # Load data
        df = pd.read_csv(data_path, sep='\t')
        
        # Check required columns
        required_cols = ['marker_gene_seq', 'rbp_seq']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
                
        predictions = []
        probabilities = []
        
        # Process each row
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Making predictions"):
            # Parse sequences
            marker_seqs = row['marker_gene_seq'].split(',')
            rbp_seqs = row['rbp_seq'].split(',')
            
            # Make prediction
            result = self.predict_single(marker_seqs, rbp_seqs)
            
            predictions.append(result['prediction'])
            probabilities.append(result['probability'])
            
        # Add predictions to dataframe
        df['predicted_probability'] = probabilities
        df['predicted_class'] = predictions
        
        # If labels are available, compute metrics
        if has_labels and 'label' in df.columns:
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(df['label'], df['predicted_class'])
            f1 = f1_score(df['label'], df['predicted_class'])
            
            try:
                auroc = roc_auc_score(df['label'], df['predicted_probability'])
            except:
                auroc = 0.5
                
            self.logger.info(f"Accuracy: {accuracy:.4f}")
            self.logger.info(f"F1 Score: {f1:.4f}")
            self.logger.info(f"AUROC: {auroc:.4f}")
            
        # Save predictions
        df.to_csv(output_path, sep='\t', index=False)
        self.logger.info(f"Saved predictions to {output_path}")
        
        return df
        
    def predict_cross_matrix(self,
                           hosts_path: str,
                           phages_path: str,
                           output_path: str) -> np.ndarray:
        """
        Predict all pairwise interactions between hosts and phages
        
        Args:
            hosts_path: Path to file with host proteins
            phages_path: Path to file with phage proteins
            output_path: Path to save interaction matrix
            
        Returns:
            Interaction probability matrix
        """
        self.logger.info("Loading host and phage data...")
        
        # Load hosts
        hosts_df = pd.read_csv(hosts_path, sep='\t')
        if 'marker_gene_seq' not in hosts_df.columns:
            raise ValueError("hosts file must have 'marker_gene_seq' column")
            
        # Load phages
        phages_df = pd.read_csv(phages_path, sep='\t')
        if 'rbp_seq' not in phages_df.columns:
            raise ValueError("phages file must have 'rbp_seq' column")
            
        n_hosts = len(hosts_df)
        n_phages = len(phages_df)
        
        self.logger.info(f"Predicting {n_hosts} x {n_phages} = {n_hosts * n_phages} interactions")
        
        # Initialize matrix
        interaction_matrix = np.zeros((n_hosts, n_phages))
        
        # Predict all pairs
        with tqdm(total=n_hosts * n_phages, desc="Predicting interactions") as pbar:
            for i, host_row in hosts_df.iterrows():
                marker_seqs = host_row['marker_gene_seq'].split(',')
                
                for j, phage_row in phages_df.iterrows():
                    rbp_seqs = phage_row['rbp_seq'].split(',')
                    
                    # Make prediction
                    result = self.predict_single(marker_seqs, rbp_seqs)
                    interaction_matrix[i, j] = result['probability']
                    
                    pbar.update(1)
                    
        # Save matrix
        np.save(output_path, interaction_matrix)
        self.logger.info(f"Saved interaction matrix to {output_path}")
        
        # Also save as CSV with labels
        csv_path = output_path.replace('.npy', '.csv')
        
        # Create DataFrame with labels
        host_labels = hosts_df.get('host_id', [f'host_{i}' for i in range(n_hosts)])
        phage_labels = phages_df.get('phage_id', [f'phage_{j}' for j in range(n_phages)])
        
        matrix_df = pd.DataFrame(interaction_matrix, 
                                index=host_labels,
                                columns=phage_labels)
        matrix_df.to_csv(csv_path)
        self.logger.info(f"Saved labeled matrix to {csv_path}")
        
        return interaction_matrix
        
    def explain_prediction(self,
                          marker_sequences: List[str],
                          rbp_sequences: List[str]) -> Dict[str, Any]:
        """
        Explain a prediction by showing pairwise contributions
        
        Args:
            marker_sequences: List of host marker protein sequences
            rbp_sequences: List of phage RBP sequences
            
        Returns:
            Dictionary with detailed explanation
        """
        # Get prediction with pairwise probabilities
        result = self.predict_single(marker_sequences, rbp_sequences, return_pairwise=True)
        
        # Analyze pairwise contributions
        pairwise_probs = np.array(result['pairwise_probabilities'])
        
        # Find top contributing pairs
        top_pairs = []
        for i in range(len(marker_sequences)):
            for j in range(len(rbp_sequences)):
                top_pairs.append({
                    'marker_idx': i,
                    'rbp_idx': j,
                    'probability': pairwise_probs[i, j],
                    'contribution': pairwise_probs[i, j]  # Simplified contribution
                })
                
        # Sort by contribution
        top_pairs = sorted(top_pairs, key=lambda x: x['probability'], reverse=True)
        
        # Create explanation
        explanation = {
            'overall_probability': result['probability'],
            'prediction': result['predicted_class'],
            'n_markers': len(marker_sequences),
            'n_rbps': len(rbp_sequences),
            'top_contributing_pairs': top_pairs[:5],  # Top 5 pairs
            'pairwise_matrix': result['pairwise_probabilities'],
            'max_pairwise_prob': float(np.max(pairwise_probs)),
            'mean_pairwise_prob': float(np.mean(pairwise_probs)),
            'interpretation': self._generate_interpretation(result, pairwise_probs)
        }
        
        return explanation
        
    def _generate_interpretation(self, result: Dict, pairwise_probs: np.ndarray) -> str:
        """
        Generate human-readable interpretation
        
        Args:
            result: Prediction result
            pairwise_probs: Pairwise probability matrix
            
        Returns:
            Interpretation string
        """
        prob = result['probability']
        pred_class = result['predicted_class']
        max_pair = np.max(pairwise_probs)
        
        if pred_class == 'positive':
            if prob > 0.8:
                interpretation = f"Strong evidence of interaction (p={prob:.3f}). "
            elif prob > 0.6:
                interpretation = f"Moderate evidence of interaction (p={prob:.3f}). "
            else:
                interpretation = f"Weak evidence of interaction (p={prob:.3f}). "
                
            if max_pair > 0.7:
                interpretation += f"At least one protein pair shows strong binding potential (p={max_pair:.3f})."
            else:
                interpretation += f"Multiple weak interactions may contribute to overall binding."
        else:
            interpretation = f"No significant interaction predicted (p={prob:.3f}). "
            if max_pair < 0.3:
                interpretation += "No individual protein pairs show binding potential."
            else:
                interpretation += f"Some weak pairwise signals detected but insufficient for interaction."
                
        return interpretation


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    
    # Model paths
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--embeddings", type=str, required=True,
                       help="Path to pre-computed embeddings")
    
    # Prediction modes
    subparsers = parser.add_subparsers(dest='mode', help='Prediction mode')
    
    # Single prediction
    single_parser = subparsers.add_parser('single', help='Predict single interaction')
    single_parser.add_argument("--markers", type=str, required=True,
                             help="Comma-separated marker sequences or file path")
    single_parser.add_argument("--rbps", type=str, required=True,
                             help="Comma-separated RBP sequences or file path")
    single_parser.add_argument("--explain", action='store_true',
                             help="Provide detailed explanation")
    
    # Batch prediction
    batch_parser = subparsers.add_parser('batch', help='Predict batch of interactions')
    batch_parser.add_argument("--input", type=str, required=True,
                            help="Input TSV file")
    batch_parser.add_argument("--output", type=str, required=True,
                            help="Output file path")
    batch_parser.add_argument("--has_labels", action='store_true',
                            help="Input file has labels for evaluation")
    
    # Matrix prediction
    matrix_parser = subparsers.add_parser('matrix', help='Predict all pairwise interactions')
    matrix_parser.add_argument("--hosts", type=str, required=True,
                             help="File with host proteins")
    matrix_parser.add_argument("--phages", type=str, required=True,
                             help="File with phage proteins")
    matrix_parser.add_argument("--output", type=str, required=True,
                             help="Output matrix file")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger(name='predictor', log_dir='logs', log_file='predictions.log')
    
    # Initialize predictor
    predictor = PhageHostPredictor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        embeddings_path=args.embeddings,
        logger=logger
    )
    
    # Run prediction based on mode
    if args.mode == 'single':
        # Parse sequences
        if Path(args.markers).exists():
            with open(args.markers, 'r') as f:
                marker_sequences = [line.strip() for line in f if line.strip()]
        else:
            marker_sequences = args.markers.split(',')
            
        if Path(args.rbps).exists():
            with open(args.rbps, 'r') as f:
                rbp_sequences = [line.strip() for line in f if line.strip()]
        else:
            rbp_sequences = args.rbps.split(',')
            
        # Make prediction
        if args.explain:
            result = predictor.explain_prediction(marker_sequences, rbp_sequences)
            print("\nPREDICTION EXPLANATION")
            print("="*50)
            print(f"Overall Probability: {result['overall_probability']:.4f}")
            print(f"Prediction: {result['prediction']}")
            print(f"Interpretation: {result['interpretation']}")
            print(f"\nTop Contributing Pairs:")
            for pair in result['top_contributing_pairs']:
                print(f"  Marker {pair['marker_idx']} - RBP {pair['rbp_idx']}: {pair['probability']:.4f}")
        else:
            result = predictor.predict_single(marker_sequences, rbp_sequences)
            print(f"\nPrediction: {result['predicted_class']}")
            print(f"Probability: {result['probability']:.4f}")
            
    elif args.mode == 'batch':
        predictor.predict_batch(args.input, args.output, args.has_labels)
        
    elif args.mode == 'matrix':
        predictor.predict_cross_matrix(args.hosts, args.phages, args.output)
        
    else:
        parser.print_help()
        
    logger.info("Prediction completed!")


if __name__ == "__main__":
    main()