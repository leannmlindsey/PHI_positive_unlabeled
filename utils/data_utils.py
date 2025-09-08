"""
Data utility functions for preprocessing and handling multi-instance bags
"""

import numpy as np
import pandas as pd
import h5py
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import hashlib


class EmbeddingLoader:
    """
    Loads and manages protein embeddings from HDF5 file
    """
    
    def __init__(self, embedding_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the embedding loader
        
        Args:
            embedding_path: Path to HDF5 file with embeddings
            logger: Optional logger instance
        """
        self.embedding_path = Path(embedding_path)
        self.logger = logger or logging.getLogger(__name__)
        self.embeddings = {}
        self.embedding_dim = None
        self._load_embeddings()
        
    def _load_embeddings(self) -> None:
        """Load embeddings from HDF5 file into memory"""
        if not self.embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {self.embedding_path}")
            
        self.logger.info(f"Loading embeddings from {self.embedding_path}")
        
        with h5py.File(self.embedding_path, 'r') as h5f:
            embeddings_array = h5f['embeddings'][:]
            hashes_array = h5f['hashes'][:]
            
            # Create dictionary for fast lookup
            for i, hash_val in enumerate(hashes_array):
                if isinstance(hash_val, bytes):
                    hash_val = hash_val.decode('utf-8')
                self.embeddings[hash_val] = embeddings_array[i]
                
            self.embedding_dim = embeddings_array.shape[1]
            
        self.logger.info(f"Loaded {len(self.embeddings)} embeddings with dimension {self.embedding_dim}")
        
    def get_embedding(self, protein_hash: str) -> np.ndarray:
        """
        Get embedding for a protein by its MD5 hash
        
        Args:
            protein_hash: MD5 hash of the protein sequence
            
        Returns:
            Embedding array
        """
        if protein_hash not in self.embeddings:
            raise KeyError(f"Embedding not found for hash: {protein_hash}")
        return self.embeddings[protein_hash]
    
    def get_embeddings_batch(self, protein_hashes: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple proteins
        
        Args:
            protein_hashes: List of MD5 hashes
            
        Returns:
            Array of embeddings with shape (n_proteins, embedding_dim)
        """
        embeddings = []
        for hash_val in protein_hashes:
            embeddings.append(self.get_embedding(hash_val))
        return np.stack(embeddings)


class MultiInstanceBag:
    """
    Represents a multi-instance bag for a single interaction
    """
    
    def __init__(self, 
                 marker_hashes: List[str],
                 rbp_hashes: List[str],
                 phage_id: str,
                 label: int = 1):
        """
        Initialize a multi-instance bag
        
        Args:
            marker_hashes: List of marker protein MD5 hashes (host)
            rbp_hashes: List of RBP MD5 hashes (phage)
            phage_id: Phage identifier
            label: Interaction label (1 for positive, 0 for negative/unlabeled)
        """
        self.marker_hashes = marker_hashes
        self.rbp_hashes = rbp_hashes
        self.phage_id = phage_id
        self.label = label
        
    def get_embeddings(self, embedding_loader: EmbeddingLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get embeddings for all proteins in the bag
        
        Args:
            embedding_loader: EmbeddingLoader instance
            
        Returns:
            Tuple of (marker_embeddings, rbp_embeddings)
        """
        marker_embeddings = embedding_loader.get_embeddings_batch(self.marker_hashes)
        rbp_embeddings = embedding_loader.get_embeddings_batch(self.rbp_hashes)
        return marker_embeddings, rbp_embeddings
    
    def __repr__(self) -> str:
        return (f"MultiInstanceBag(markers={len(self.marker_hashes)}, "
                f"rbps={len(self.rbp_hashes)}, label={self.label})")


class DataProcessor:
    """
    Processes raw data into multi-instance bags
    """
    
    def __init__(self, 
                 data_path: str,
                 split_path: str,
                 embedding_loader: EmbeddingLoader,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the data processor
        
        Args:
            data_path: Path to the original TSV data
            split_path: Path to the splits pickle file
            embedding_loader: EmbeddingLoader instance
            logger: Optional logger instance
        """
        self.data_path = Path(data_path)
        self.split_path = Path(split_path)
        self.embedding_loader = embedding_loader
        self.logger = logger or logging.getLogger(__name__)
        
        # Load data and splits
        self.df = pd.read_csv(self.data_path, sep='\t')
        with open(self.split_path, 'rb') as f:
            self.splits = pickle.load(f)
            
        self.logger.info(f"Loaded data with {len(self.df)} samples")
        self.logger.info(f"Train: {len(self.splits['train_idx'])}, "
                        f"Val: {len(self.splits['val_idx'])}, "
                        f"Test: {len(self.splits['test_idx'])}")
        
    def _parse_row_to_bag(self, row: pd.Series, label: int = 1) -> MultiInstanceBag:
        """
        Convert a dataframe row to a MultiInstanceBag
        
        Args:
            row: Pandas Series representing a data row
            label: Label for the bag
            
        Returns:
            MultiInstanceBag instance
        """
        # Parse marker hashes
        if ',' in str(row['marker_md5']):
            marker_hashes = row['marker_md5'].split(',')
        else:
            marker_hashes = [row['marker_md5']]
            
        # Parse RBP hashes
        if ',' in str(row['rbp_md5']):
            rbp_hashes = row['rbp_md5'].split(',')
        else:
            rbp_hashes = [row['rbp_md5']]
            
        return MultiInstanceBag(
            marker_hashes=marker_hashes,
            rbp_hashes=rbp_hashes,
            phage_id=row['phage_id'],
            label=label
        )
    
    def get_split_data(self, split: str) -> List[MultiInstanceBag]:
        """
        Get multi-instance bags for a specific split
        
        Args:
            split: One of 'train', 'val', or 'test'
            
        Returns:
            List of MultiInstanceBag instances
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}")
            
        split_indices = self.splits[f'{split}_idx']
        bags = []
        
        for idx in split_indices:
            row = self.df.iloc[idx]
            bag = self._parse_row_to_bag(row, label=1)  # All positive
            bags.append(bag)
            
        self.logger.info(f"Created {len(bags)} bags for {split} split")
        return bags
    
    def generate_negative_samples(self, 
                                 positive_bags: List[MultiInstanceBag],
                                 negative_ratio: float = 1.0,
                                 seed: int = 42) -> List[MultiInstanceBag]:
        """
        Generate negative samples by random pairing
        
        Args:
            positive_bags: List of positive MultiInstanceBag instances
            negative_ratio: Ratio of negative to positive samples
            seed: Random seed
            
        Returns:
            List of negative MultiInstanceBag instances
        """
        np.random.seed(seed)
        
        # Collect all unique markers and RBPs
        all_markers = set()
        all_rbps = set()
        
        for bag in positive_bags:
            all_markers.update(bag.marker_hashes)
            all_rbps.update(bag.rbp_hashes)
            
        all_markers = list(all_markers)
        all_rbps = list(all_rbps)
        
        # Generate negative samples
        n_negative = int(len(positive_bags) * negative_ratio)
        negative_bags = []
        
        # Keep track of positive pairs to avoid
        positive_pairs = set()
        for bag in positive_bags:
            for m in bag.marker_hashes:
                for r in bag.rbp_hashes:
                    positive_pairs.add((m, r))
        
        attempts = 0
        max_attempts = n_negative * 10
        
        while len(negative_bags) < n_negative and attempts < max_attempts:
            attempts += 1
            
            # Random sample markers and RBPs
            n_markers = np.random.choice([1, 2], p=[0.2, 0.8])  # Match distribution
            n_rbps = np.random.choice([1, 2, 3], p=[0.7, 0.25, 0.05])
            
            marker_hashes = list(np.random.choice(all_markers, n_markers, replace=False))
            rbp_hashes = list(np.random.choice(all_rbps, n_rbps, replace=False))
            
            # Check if any pair is positive
            is_positive = False
            for m in marker_hashes:
                for r in rbp_hashes:
                    if (m, r) in positive_pairs:
                        is_positive = True
                        break
                if is_positive:
                    break
                    
            if not is_positive:
                negative_bags.append(MultiInstanceBag(
                    marker_hashes=marker_hashes,
                    rbp_hashes=rbp_hashes,
                    phage_id=f"negative_{len(negative_bags)}",
                    label=0
                ))
                
        self.logger.info(f"Generated {len(negative_bags)} negative samples")
        return negative_bags


def collate_bags(bags: List[MultiInstanceBag], 
                 embedding_loader: EmbeddingLoader,
                 pad_to_max: bool = True) -> Dict[str, np.ndarray]:
    """
    Collate a list of bags into batch tensors
    
    Args:
        bags: List of MultiInstanceBag instances
        embedding_loader: EmbeddingLoader instance
        pad_to_max: Whether to pad to maximum length in batch
        
    Returns:
        Dictionary with batch tensors
    """
    batch_size = len(bags)
    
    # Get maximum lengths
    max_markers = max(len(bag.marker_hashes) for bag in bags)
    max_rbps = max(len(bag.rbp_hashes) for bag in bags)
    
    embedding_dim = embedding_loader.embedding_dim
    
    # Initialize arrays
    marker_embeddings = np.zeros((batch_size, max_markers, embedding_dim), dtype=np.float32)
    rbp_embeddings = np.zeros((batch_size, max_rbps, embedding_dim), dtype=np.float32)
    marker_mask = np.zeros((batch_size, max_markers), dtype=np.float32)
    rbp_mask = np.zeros((batch_size, max_rbps), dtype=np.float32)
    labels = np.array([bag.label for bag in bags], dtype=np.float32)
    
    # Fill arrays
    for i, bag in enumerate(bags):
        # Get embeddings
        m_emb, r_emb = bag.get_embeddings(embedding_loader)
        
        # Fill marker embeddings and mask
        n_markers = len(bag.marker_hashes)
        marker_embeddings[i, :n_markers] = m_emb
        marker_mask[i, :n_markers] = 1.0
        
        # Fill RBP embeddings and mask
        n_rbps = len(bag.rbp_hashes)
        rbp_embeddings[i, :n_rbps] = r_emb
        rbp_mask[i, :n_rbps] = 1.0
        
    return {
        'marker_embeddings': marker_embeddings,
        'rbp_embeddings': rbp_embeddings,
        'marker_mask': marker_mask,
        'rbp_mask': rbp_mask,
        'labels': labels
    }