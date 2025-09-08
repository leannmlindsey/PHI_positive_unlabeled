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
from collections import OrderedDict
import atexit


class LazyEmbeddingLoader:
    """
    Lazily loads protein embeddings from HDF5 file with LRU caching
    """
    
    def __init__(self, 
                 embedding_path: str, 
                 cache_size: int = 10000,
                 preload_all: bool = False,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the lazy embedding loader
        
        Args:
            embedding_path: Path to HDF5 file with embeddings
            cache_size: Maximum number of embeddings to cache in memory
            preload_all: If True, loads all embeddings into memory (like original behavior)
            logger: Optional logger instance
        """
        self.embedding_path = Path(embedding_path)
        self.cache_size = cache_size
        self.preload_all = preload_all
        self.logger = logger or logging.getLogger(__name__)
        
        # LRU cache using OrderedDict
        self._cache = OrderedDict()
        self._file = None
        self._hash_to_idx = {}
        self.embedding_dim = None
        self.total_embeddings = 0
        
        # Initialize the loader
        self._initialize()
        
        # Register cleanup on exit
        atexit.register(self._cleanup)
        
    def _initialize(self) -> None:
        """Initialize the loader and build hash-to-index mapping"""
        if not self.embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {self.embedding_path}")
        
        self.logger.info(f"Initializing lazy loader for {self.embedding_path}")
        
        # Open file and build index
        with h5py.File(self.embedding_path, 'r') as h5f:
            hashes_array = h5f['hashes'][:]
            embeddings_shape = h5f['embeddings'].shape
            
            self.embedding_dim = embeddings_shape[1]
            self.total_embeddings = embeddings_shape[0]
            
            # Build hash to index mapping
            for i, hash_val in enumerate(hashes_array):
                if isinstance(hash_val, bytes):
                    hash_val = hash_val.decode('utf-8')
                self._hash_to_idx[hash_val] = i
                
        self.logger.info(f"Initialized with {self.total_embeddings} embeddings, "
                        f"dimension {self.embedding_dim}, cache size {self.cache_size}")
        
        # Preload all if requested
        if self.preload_all:
            self._preload_all_embeddings()
    
    def _preload_all_embeddings(self) -> None:
        """Preload all embeddings into cache (for compatibility with original behavior)"""
        self.logger.info("Preloading all embeddings into memory...")
        
        with h5py.File(self.embedding_path, 'r') as h5f:
            embeddings_array = h5f['embeddings'][:]
            
            for hash_val, idx in self._hash_to_idx.items():
                self._cache[hash_val] = embeddings_array[idx]
        
        self.logger.info(f"Preloaded {len(self._cache)} embeddings")
    
    def _open_file(self) -> None:
        """Open the HDF5 file if not already open"""
        if self._file is None:
            self._file = h5py.File(self.embedding_path, 'r')
    
    def _cleanup(self) -> None:
        """Clean up resources"""
        if self._file is not None:
            self._file.close()
            self._file = None
    
    def get_embedding(self, protein_hash: str) -> np.ndarray:
        """
        Get embedding for a protein by its MD5 hash
        
        Args:
            protein_hash: MD5 hash of the protein sequence
            
        Returns:
            Embedding array
        """
        # Check cache first
        if protein_hash in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(protein_hash)
            return self._cache[protein_hash]
        
        # Check if hash exists
        if protein_hash not in self._hash_to_idx:
            raise KeyError(f"Embedding not found for hash: {protein_hash}")
        
        # Load from file
        self._open_file()
        idx = self._hash_to_idx[protein_hash]
        embedding = self._file['embeddings'][idx][:]
        
        # Add to cache
        self._add_to_cache(protein_hash, embedding)
        
        return embedding
    
    def _add_to_cache(self, hash_val: str, embedding: np.ndarray) -> None:
        """Add embedding to cache with LRU eviction"""
        # Check cache size
        if len(self._cache) >= self.cache_size:
            # Remove least recently used (first item)
            self._cache.popitem(last=False)
        
        # Add new item (becomes most recently used)
        self._cache[hash_val] = embedding
    
    def get_embeddings_batch(self, protein_hashes: List[str], use_zero_for_missing: bool = True) -> np.ndarray:
        """
        Get embeddings for multiple proteins with error handling
        
        Args:
            protein_hashes: List of MD5 hashes
            use_zero_for_missing: If True, use zero vectors for missing embeddings
            
        Returns:
            Array of embeddings with shape (n_proteins, embedding_dim)
        """
        embeddings = []
        missing_count = 0
        
        for hash_val in protein_hashes:
            try:
                embeddings.append(self.get_embedding(hash_val))
            except KeyError:
                if use_zero_for_missing:
                    self.logger.warning(f"Missing embedding for hash {hash_val}, using zero vector")
                    embeddings.append(np.zeros(self.embedding_dim))
                    missing_count += 1
                else:
                    raise
        
        if missing_count > 0:
            self.logger.warning(f"Total missing embeddings: {missing_count}/{len(protein_hashes)}")
            
        return np.stack(embeddings) if embeddings else np.zeros((0, self.embedding_dim))
    
    def cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'cache_size': len(self._cache),
            'max_cache_size': self.cache_size,
            'total_embeddings': self.total_embeddings,
            'cache_hit_rate': len(self._cache) / self.total_embeddings if self.total_embeddings > 0 else 0
        }
    
    def __del__(self):
        """Cleanup on deletion"""
        self._cleanup()


# Keep original EmbeddingLoader for backward compatibility
class EmbeddingLoader(LazyEmbeddingLoader):
    """
    Legacy embedding loader that loads all embeddings into memory
    Kept for backward compatibility
    """
    
    def __init__(self, embedding_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the embedding loader (loads all into memory)
        
        Args:
            embedding_path: Path to HDF5 file with embeddings
            logger: Optional logger instance
        """
        # Use lazy loader with preload_all=True and unlimited cache
        super().__init__(
            embedding_path=embedding_path,
            cache_size=float('inf'),  # Unlimited cache
            preload_all=True,  # Load everything at init
            logger=logger
        )
        
        # For backward compatibility, expose embeddings dict
        self.embeddings = self._cache
        
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
    
    def get_embeddings_batch(self, protein_hashes: List[str], use_zero_for_missing: bool = True) -> np.ndarray:
        """
        Get embeddings for multiple proteins with error handling
        
        Args:
            protein_hashes: List of MD5 hashes
            use_zero_for_missing: If True, use zero vectors for missing embeddings; if False, raise error
            
        Returns:
            Array of embeddings with shape (n_proteins, embedding_dim)
        """
        embeddings = []
        missing_count = 0
        
        for hash_val in protein_hashes:
            try:
                embeddings.append(self.get_embedding(hash_val))
            except KeyError:
                if use_zero_for_missing:
                    self.logger.warning(f"Missing embedding for hash {hash_val}, using zero vector")
                    embeddings.append(np.zeros(self.embedding_dim))
                    missing_count += 1
                else:
                    raise
        
        if missing_count > 0:
            self.logger.warning(f"Total missing embeddings: {missing_count}/{len(protein_hashes)}")
            
        return np.stack(embeddings) if embeddings else np.zeros((0, self.embedding_dim))


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
        
    def get_embeddings(self, embedding_loader, use_zero_for_missing: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get embeddings for all proteins in the bag with error handling
        
        Args:
            embedding_loader: EmbeddingLoader or DualEmbeddingLoader instance
            use_zero_for_missing: If True, use zero vectors for missing embeddings
            
        Returns:
            Tuple of (marker_embeddings, rbp_embeddings)
            
        Raises:
            ValueError: If no valid embeddings found for either markers or RBPs
        """
        # Check if we have a DualEmbeddingLoader
        from utils.dual_embedding_loader import DualEmbeddingLoader
        if isinstance(embedding_loader, DualEmbeddingLoader):
            # Get host embeddings for markers
            marker_embeddings = []
            for hash_val in self.marker_hashes:
                try:
                    marker_embeddings.append(embedding_loader.get_host_embedding(hash_val))
                except KeyError:
                    if use_zero_for_missing:
                        marker_embeddings.append(np.zeros(embedding_loader.embedding_dim))
                    else:
                        raise ValueError(f"Missing embedding for marker hash: {hash_val}")
            
            # Get phage embeddings for RBPs
            rbp_embeddings = []
            for hash_val in self.rbp_hashes:
                try:
                    rbp_embeddings.append(embedding_loader.get_phage_embedding(hash_val))
                except KeyError:
                    if use_zero_for_missing:
                        rbp_embeddings.append(np.zeros(embedding_loader.embedding_dim))
                    else:
                        raise ValueError(f"Missing embedding for RBP hash: {hash_val}")
            
            marker_embeddings = np.array(marker_embeddings)
            rbp_embeddings = np.array(rbp_embeddings)
        else:
            # Use legacy single loader
            marker_embeddings = embedding_loader.get_embeddings_batch(self.marker_hashes, use_zero_for_missing)
            rbp_embeddings = embedding_loader.get_embeddings_batch(self.rbp_hashes, use_zero_for_missing)
        
        # Validate we have at least some non-zero embeddings
        if marker_embeddings.shape[0] == 0 or np.all(marker_embeddings == 0):
            raise ValueError(f"No valid marker embeddings found for phage {self.phage_id}")
        if rbp_embeddings.shape[0] == 0 or np.all(rbp_embeddings == 0):
            raise ValueError(f"No valid RBP embeddings found for phage {self.phage_id}")
            
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