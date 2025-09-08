#!/usr/bin/env python3
"""
Simple baseline model using XGBoost
- Only uses single marker + single RBP pairs
- Concatenates embeddings as features
- Binary classification (interaction vs no interaction)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import xgboost as xgb
import argparse
from tqdm import tqdm
import logging

def setup_logger():
    """Set up logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_embeddings(embeddings_dir):
    """Load host and phage embeddings"""
    logger = logging.getLogger(__name__)
    
    host_path = Path(embeddings_dir) / 'host_embeddings.h5'
    phage_path = Path(embeddings_dir) / 'phage_embeddings.h5'
    
    logger.info(f"Loading embeddings from {embeddings_dir}")
    
    # Load host embeddings
    with h5py.File(host_path, 'r') as f:
        host_embeddings = f['embeddings'][:]
        host_hashes = f['hashes'][:].astype(str)
    
    host_embed_dict = {hash_val: emb for hash_val, emb in zip(host_hashes, host_embeddings)}
    logger.info(f"Loaded {len(host_embed_dict)} host embeddings")
    
    # Load phage embeddings
    with h5py.File(phage_path, 'r') as f:
        phage_embeddings = f['embeddings'][:]
        phage_hashes = f['hashes'][:].astype(str)
    
    phage_embed_dict = {hash_val: emb for hash_val, emb in zip(phage_hashes, phage_embeddings)}
    logger.info(f"Loaded {len(phage_embed_dict)} phage embeddings")
    
    return host_embed_dict, phage_embed_dict

def prepare_simple_dataset(data_path, splits_path, host_embeddings, phage_embeddings):
    """
    Prepare simple dataset with single marker-RBP pairs
    """
    logger = logging.getLogger(__name__)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, sep='\t')
    
    # Load splits
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    
    # Process each split
    datasets = {}
    
    for split_name in ['train', 'val', 'test']:
        logger.info(f"Processing {split_name} split...")
        
        split_indices = splits[f'{split_name}_idx']
        split_df = df.iloc[split_indices]
        
        positive_pairs = []
        positive_features = []
        
        # Extract positive pairs (only single marker + single RBP)
        skipped = 0
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name}"):
            # Parse markers
            marker_field = str(row['marker_md5'])
            if ',' in marker_field:
                # Skip multi-marker cases for simplicity
                markers = marker_field.split(',')
            else:
                markers = [marker_field.strip()]
            
            # Parse RBPs
            rbp_field = str(row['rbp_md5'])
            if ',' in rbp_field:
                # Skip multi-RBP cases for simplicity
                skipped += 1
                continue
            else:
                rbps = [rbp_field.strip()]
            
            # Only use single marker + single RBP
            if len(markers) == 1 and len(rbps) == 1:
                marker_hash = markers[0]
                rbp_hash = rbps[0]
                
                # Check if embeddings exist
                if marker_hash in host_embeddings and rbp_hash in phage_embeddings:
                    # Concatenate embeddings
                    marker_emb = host_embeddings[marker_hash]
                    rbp_emb = phage_embeddings[rbp_hash]
                    feature = np.concatenate([marker_emb, rbp_emb])
                    
                    positive_pairs.append((marker_hash, rbp_hash))
                    positive_features.append(feature)
        
        logger.info(f"  Collected {len(positive_pairs)} positive pairs (skipped {skipped} multi-RBP)")
        
        # Generate negative pairs (same number as positive)
        negative_pairs = []
        negative_features = []
        
        # Get all unique markers and RBPs in this split
        all_markers = set()
        all_rbps = set()
        
        for marker_hash, rbp_hash in positive_pairs:
            all_markers.add(marker_hash)
            all_rbps.add(rbp_hash)
        
        all_markers = list(all_markers)
        all_rbps = list(all_rbps)
        
        # Create set of positive pairs for checking
        positive_set = set(positive_pairs)
        
        # Generate negative pairs
        np.random.seed(42 + ['train', 'val', 'test'].index(split_name))
        attempts = 0
        max_attempts = len(positive_pairs) * 100
        
        while len(negative_pairs) < len(positive_pairs) and attempts < max_attempts:
            attempts += 1
            
            # Random pairing
            marker_hash = np.random.choice(all_markers)
            rbp_hash = np.random.choice(all_rbps)
            
            # Check if this is not a positive pair
            if (marker_hash, rbp_hash) not in positive_set:
                # Create feature
                marker_emb = host_embeddings[marker_hash]
                rbp_emb = phage_embeddings[rbp_hash]
                feature = np.concatenate([marker_emb, rbp_emb])
                
                negative_pairs.append((marker_hash, rbp_hash))
                negative_features.append(feature)
        
        logger.info(f"  Generated {len(negative_pairs)} negative pairs")
        
        # Combine positive and negative
        X = np.vstack(positive_features + negative_features)
        y = np.array([1] * len(positive_features) + [0] * len(negative_features))
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        datasets[split_name] = {
            'X': X,
            'y': y,
            'positive_pairs': positive_pairs,
            'negative_pairs': negative_pairs
        }
        
        logger.info(f"  {split_name}: {len(X)} samples ({sum(y)} positive, {len(y)-sum(y)} negative)")
    
    return datasets

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model"""
    logger = logging.getLogger(__name__)
    
    logger.info("Training XGBoost model...")
    
    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'use_label_encoder': False
    }
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train with early stopping
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evallist,
        early_stopping_rounds=50,
        verbose_eval=10
    )
    
    return model

def evaluate_model(model, X, y, name="Test"):
    """Evaluate model performance"""
    logger = logging.getLogger(__name__)
    
    # Convert to DMatrix
    dtest = xgb.DMatrix(X)
    
    # Get predictions
    y_prob = model.predict(dtest)
    y_pred = (y_prob > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'auroc': roc_auc_score(y, y_prob),
        'auprc': average_precision_score(y, y_prob)
    }
    
    logger.info(f"\n{name} Set Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
    logger.info(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # Classification report
    logger.info(f"\nClassification Report:")
    logger.info(classification_report(y, y_pred))
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Simple XGBoost baseline for phage-host interaction")
    parser.add_argument("--data_path", type=str, 
                       default="data/dedup.phage_marker_rbp_with_phage_entropy.tsv",
                       help="Path to data file")
    parser.add_argument("--splits_path", type=str,
                       default="data/processed/splits.pkl",
                       help="Path to splits file")
    parser.add_argument("--embeddings_dir", type=str,
                       default="data/embeddings",
                       help="Directory containing embedding files")
    parser.add_argument("--output_dir", type=str,
                       default="outputs/baseline",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logger()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load embeddings
    host_embeddings, phage_embeddings = load_embeddings(args.embeddings_dir)
    
    # Prepare dataset
    datasets = prepare_simple_dataset(
        args.data_path,
        args.splits_path,
        host_embeddings,
        phage_embeddings
    )
    
    # Train model
    model = train_xgboost(
        datasets['train']['X'],
        datasets['train']['y'],
        datasets['val']['X'],
        datasets['val']['y']
    )
    
    # Save model
    model_path = output_dir / 'xgboost_model.json'
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Evaluate on all sets
    for split_name in ['train', 'val', 'test']:
        evaluate_model(
            model,
            datasets[split_name]['X'],
            datasets[split_name]['y'],
            name=split_name.capitalize()
        )
    
    # Save predictions
    for split_name in ['test']:
        dtest = xgb.DMatrix(datasets[split_name]['X'])
        y_prob = model.predict(dtest)
        
        results = pd.DataFrame({
            'marker_hash': [p[0] for p in datasets[split_name]['positive_pairs']] + 
                          [p[0] for p in datasets[split_name]['negative_pairs']],
            'rbp_hash': [p[1] for p in datasets[split_name]['positive_pairs']] + 
                       [p[1] for p in datasets[split_name]['negative_pairs']],
            'true_label': datasets[split_name]['y'],
            'predicted_prob': y_prob
        })
        
        results_path = output_dir / f'{split_name}_predictions.csv'
        results.to_csv(results_path, index=False)
        logger.info(f"Predictions saved to {results_path}")

if __name__ == "__main__":
    main()