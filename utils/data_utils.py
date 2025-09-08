"""
Data utility functions for MD5 hash-based multi-instance bag loading
"""

import numpy as np
import pandas as pd
import h5py
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set
import logging
from collections import OrderedDict
import atexit


class HashBasedEmbeddingLoader:
    """
    Loads protein embeddings using MD5 hashes as identifiers
    Supports lazy loading with LRU caching for memory efficiency
    """
    
    def __init__(self, 
                 host_embedding_path: str,
                 phage_embedding_path: str,
                 cache_size: int = 10000,
                 preload_all: bool = False,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the hash-based embedding loader
        
        Args:
            host_embedding_path: Path to HDF5 file with host embeddings
            phage_embedding_path: Path to HDF5 file with phage embeddings
            cache_size: Maximum number of embeddings to cache in memory
            preload_all: If True, loads all embeddings into memory
            logger: Optional logger instance
        """
        self.host_path = Path(host_embedding_path)
        self.phage_path = Path(phage_embedding_path)
        self.cache_size = cache_size
        self.preload_all = preload_all
        self.logger = logger or logging.getLogger(__name__)
        
        # Separate caches for host and phage
        self._host_cache = OrderedDict()
        self._phage_cache = OrderedDict()
        
        # File handles
        self._host_file = None
        self._phage_file = None
        
        # Hash to index mappings
        self._host_hash_to_idx = {}
        self._phage_hash_to_idx = {}
        
        # Metadata
        self.host_embedding_dim = None
        self.phage_embedding_dim = None
        self.total_host_embeddings = 0
        self.total_phage_embeddings = 0
        
        # Initialize the loaders
        self._initialize()
        
        # Register cleanup on exit
        atexit.register(self._cleanup)
    
    def _initialize(self) -> None:
        """Initialize both host and phage loaders"""
        # Initialize host embeddings
        if not self.host_path.exists():
            raise FileNotFoundError(f"Host embedding file not found: {self.host_path}")
        
        with h5py.File(self.host_path, 'r') as h5f:
            hashes_array = h5f['hashes'][:]
            embeddings_shape = h5f['embeddings'].shape
            
            self.host_embedding_dim = embeddings_shape[1]
            self.total_host_embeddings = embeddings_shape[0]
            
            # Build hash to index mapping
            for i, hash_val in enumerate(hashes_array):
                if isinstance(hash_val, bytes):
                    hash_val = hash_val.decode('utf-8')
                self._host_hash_to_idx[hash_val] = i
        
        self.logger.info(f"Loaded {self.total_host_embeddings} host embeddings, "
                        f"dimension {self.host_embedding_dim}")
        
        # Initialize phage embeddings
        if not self.phage_path.exists():
            raise FileNotFoundError(f"Phage embedding file not found: {self.phage_path}")
        
        with h5py.File(self.phage_path, 'r') as h5f:
            hashes_array = h5f['hashes'][:]
            embeddings_shape = h5f['embeddings'].shape
            
            self.phage_embedding_dim = embeddings_shape[1]
            self.total_phage_embeddings = embeddings_shape[0]
            
            # Build hash to index mapping
            for i, hash_val in enumerate(hashes_array):
                if isinstance(hash_val, bytes):
                    hash_val = hash_val.decode('utf-8')
                self._phage_hash_to_idx[hash_val] = i
        
        self.logger.info(f"Loaded {self.total_phage_embeddings} phage embeddings, "
                        f"dimension {self.phage_embedding_dim}")
        
        # Check dimensions match
        if self.host_embedding_dim != self.phage_embedding_dim:
            raise ValueError(f"Embedding dimensions don't match: "
                           f"host={self.host_embedding_dim}, phage={self.phage_embedding_dim}")
        
        self.embedding_dim = self.host_embedding_dim
        
        # Preload if requested
        if self.preload_all:
            self._preload_all_embeddings()
    
    def _preload_all_embeddings(self) -> None:
        """Preload all embeddings into memory"""
        self.logger.info("Preloading all embeddings into memory...")
        
        # Preload host embeddings
        with h5py.File(self.host_path, 'r') as h5f:
            embeddings_array = h5f['embeddings'][:]
            for hash_val, idx in self._host_hash_to_idx.items():
                self._host_cache[hash_val] = embeddings_array[idx]
        
        # Preload phage embeddings
        with h5py.File(self.phage_path, 'r') as h5f:
            embeddings_array = h5f['embeddings'][:]
            for hash_val, idx in self._phage_hash_to_idx.items():
                self._phage_cache[hash_val] = embeddings_array[idx]
        
        self.logger.info(f"Preloaded {len(self._host_cache)} host and "
                        f"{len(self._phage_cache)} phage embeddings")
    
    def _get_from_cache_or_load(self, hash_val: str, is_host: bool) -> Optional[np.ndarray]:
        """Get embedding from cache or load from file"""
        if is_host:
            cache = self._host_cache
            hash_to_idx = self._host_hash_to_idx
            file_path = self.host_path
        else:
            cache = self._phage_cache
            hash_to_idx = self._phage_hash_to_idx
            file_path = self.phage_path
        
        # Check cache first
        if hash_val in cache:
            # Move to end (LRU)
            cache.move_to_end(hash_val)
            return cache[hash_val]
        
        # Check if hash exists
        if hash_val not in hash_to_idx:
            return None
        
        # Load from file
        idx = hash_to_idx[hash_val]
        with h5py.File(file_path, 'r') as h5f:
            embedding = h5f['embeddings'][idx]
        
        # Add to cache with LRU eviction
        if len(cache) >= self.cache_size:
            # Remove oldest
            cache.popitem(last=False)
        cache[hash_val] = embedding
        
        return embedding
    
    def get_host_embedding(self, hash_val: str) -> Optional[np.ndarray]:
        """Get host embedding by MD5 hash"""
        return self._get_from_cache_or_load(hash_val, is_host=True)
    
    def get_phage_embedding(self, hash_val: str) -> Optional[np.ndarray]:
        """Get phage embedding by MD5 hash"""
        return self._get_from_cache_or_load(hash_val, is_host=False)
    
    def get_host_embeddings(self, hash_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get multiple host embeddings
        
        Returns:
            Tuple of (embeddings, mask) where mask indicates valid embeddings
        """
        embeddings = []
        mask = []
        
        for hash_val in hash_list:
            emb = self.get_host_embedding(hash_val)
            if emb is not None:
                embeddings.append(emb)
                mask.append(1)
            else:
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
                mask.append(0)
        
        return np.array(embeddings), np.array(mask)
    
    def get_phage_embeddings(self, hash_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get multiple phage embeddings
        
        Returns:
            Tuple of (embeddings, mask) where mask indicates valid embeddings
        """
        embeddings = []
        mask = []
        
        for hash_val in hash_list:
            emb = self.get_phage_embedding(hash_val)
            if emb is not None:
                embeddings.append(emb)
                mask.append(1)
            else:
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
                mask.append(0)
        
        return np.array(embeddings), np.array(mask)
    
    def _cleanup(self) -> None:
        """Clean up resources"""
        if self._host_file is not None:
            self._host_file.close()
        if self._phage_file is not None:
            self._phage_file.close()


class MultiInstanceBag:
    """
    Represents a multi-instance bag for phage-host interaction
    Uses MD5 hashes to identify proteins
    """
    
    def __init__(self,
                 host_hashes: List[str],
                 phage_hashes: List[str],
                 label: int,
                 phage_id: str,
                 metadata: Optional[Dict] = None):
        """
        Initialize a multi-instance bag
        
        Args:
            host_hashes: List of MD5 hashes for host proteins (wzx, wzm)
            phage_hashes: List of MD5 hashes for phage RBPs
            label: Binary label (1 for positive, 0 for negative)
            phage_id: Phage identifier
            metadata: Optional additional metadata
        """
        self.host_hashes = host_hashes
        self.phage_hashes = phage_hashes
        self.label = label
        self.phage_id = phage_id
        self.metadata = metadata or {}
        
        # Validate
        if not host_hashes:
            raise ValueError("Bag must have at least one host protein")
        if not phage_hashes:
            raise ValueError("Bag must have at least one phage protein")
    
    def get_embeddings(self, 
                      loader: HashBasedEmbeddingLoader,
                      max_host: Optional[int] = None,
                      max_phage: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get embeddings for this bag
        
        Args:
            loader: HashBasedEmbeddingLoader instance
            max_host: Maximum number of host proteins (for padding)
            max_phage: Maximum number of phage proteins (for padding)
            
        Returns:
            Tuple of (host_embeddings, phage_embeddings, host_mask, phage_mask)
        """
        # Get host embeddings
        host_emb, host_mask = loader.get_host_embeddings(self.host_hashes)
        
        # Get phage embeddings
        phage_emb, phage_mask = loader.get_phage_embeddings(self.phage_hashes)
        
        # Pad if needed
        if max_host and len(host_emb) < max_host:
            pad_size = max_host - len(host_emb)
            host_emb = np.pad(host_emb, ((0, pad_size), (0, 0)), mode='constant')
            host_mask = np.pad(host_mask, (0, pad_size), mode='constant')
        elif max_host and len(host_emb) > max_host:
            # Truncate
            host_emb = host_emb[:max_host]
            host_mask = host_mask[:max_host]
        
        if max_phage and len(phage_emb) < max_phage:
            pad_size = max_phage - len(phage_emb)
            phage_emb = np.pad(phage_emb, ((0, pad_size), (0, 0)), mode='constant')
            phage_mask = np.pad(phage_mask, (0, pad_size), mode='constant')
        elif max_phage and len(phage_emb) > max_phage:
            # Truncate
            phage_emb = phage_emb[:max_phage]
            phage_mask = phage_mask[:max_phage]
        
        return host_emb, phage_emb, host_mask, phage_mask
    
    def __repr__(self):
        return (f"MultiInstanceBag(hosts={len(self.host_hashes)}, "
                f"phages={len(self.phage_hashes)}, label={self.label}, "
                f"phage_id={self.phage_id})")


class DataProcessor:
    """
    Processes data files with MD5 hash columns into MultiInstanceBags
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the data processor
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def load_data_with_hashes(self, data_path: str) -> pd.DataFrame:
        """
        Load data file with hash columns
        
        Args:
            data_path: Path to TSV file with host_md5_set and phage_md5_set columns
            
        Returns:
            DataFrame with parsed hash columns
        """
        self.logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path, sep='\t')
        
        # Check required columns
        required = ['host_md5_set', 'phage_md5_set', 'phage_id']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.logger.info(f"Loaded {len(df)} samples")
        return df
    
    def create_bags_from_dataframe(self,
                                  df: pd.DataFrame,
                                  negative_ratio: float = 1.0,
                                  seed: int = 42) -> List[MultiInstanceBag]:
        """
        Create MultiInstanceBags from dataframe
        
        Args:
            df: DataFrame with hash columns
            negative_ratio: Ratio of negative to positive samples
            seed: Random seed for negative sampling
            
        Returns:
            List of MultiInstanceBag objects
        """
        np.random.seed(seed)
        bags = []
        
        # Count skipped rows
        skipped_count = 0
        
        # Process positive samples
        for idx, row in df.iterrows():
            # Check if host_md5_set is NaN or 'nan' string
            if pd.isna(row['host_md5_set']) or str(row['host_md5_set']).strip() == 'nan':
                skipped_count += 1
                continue
                
            # Check if phage_md5_set is NaN or empty
            if pd.isna(row['phage_md5_set']) or str(row['phage_md5_set']).strip() == '':
                skipped_count += 1
                continue
            
            # Parse host hashes
            host_hashes = []
            host_field = str(row['host_md5_set']).strip()
            if host_field and host_field != '':
                host_hashes = [h.strip() for h in host_field.split(',') 
                             if h.strip() and h.strip() != 'nan' and h.strip() != '']
            
            # Parse phage hashes
            phage_hashes = []
            phage_field = str(row['phage_md5_set']).strip()
            if phage_field and phage_field != '':
                phage_hashes = [h.strip() for h in phage_field.split(',')
                              if h.strip() and h.strip() != 'nan' and h.strip() != '']
            
            # Skip if empty after parsing
            if not host_hashes or not phage_hashes:
                skipped_count += 1
                continue
            
            # Create positive bag
            bag = MultiInstanceBag(
                host_hashes=host_hashes,
                phage_hashes=phage_hashes,
                label=1,
                phage_id=str(row['phage_id'])
            )
            bags.append(bag)
        
        n_positive = len(bags)
        self.logger.info(f"Created {n_positive} positive bags (skipped {skipped_count} rows with missing data)")
        
        # Generate negative samples if requested
        if negative_ratio > 0:
            n_negative = int(n_positive * negative_ratio)
            negative_bags = self._generate_negative_samples(df, n_negative, seed)
            bags.extend(negative_bags)
            self.logger.info(f"Generated {len(negative_bags)} negative bags")
        
        return bags
    
    def _generate_negative_samples(self,
                                  df: pd.DataFrame,
                                  n_samples: int,
                                  seed: int) -> List[MultiInstanceBag]:
        """
        Generate negative samples by random pairing
        
        Args:
            df: DataFrame with positive samples
            n_samples: Number of negative samples to generate
            seed: Random seed
            
        Returns:
            List of negative MultiInstanceBag objects
        """
        np.random.seed(seed)
        negative_bags = []
        
        # Collect all unique host and phage hash sets
        all_host_sets = []
        all_phage_sets = []
        
        for idx, row in df.iterrows():
            # Host hashes
            host_field = str(row['host_md5_set']).strip()
            if pd.notna(row['host_md5_set']) and host_field and host_field != 'nan' and host_field != '':
                host_hashes = [h.strip() for h in host_field.split(',')
                             if h.strip() and h.strip() != 'nan' and h.strip() != '']
                if host_hashes:
                    all_host_sets.append(host_hashes)
            
            # Phage hashes
            phage_field = str(row['phage_md5_set']).strip()
            if pd.notna(row['phage_md5_set']) and phage_field and phage_field != 'nan' and phage_field != '':
                phage_hashes = [h.strip() for h in phage_field.split(',')
                              if h.strip() and h.strip() != 'nan' and h.strip() != '']
                if phage_hashes:
                    all_phage_sets.append(phage_hashes)
        
        # Create set of positive pairs for checking
        positive_pairs = set()
        for idx, row in df.iterrows():
            host_str = str(row['host_md5_set'])
            phage_str = str(row['phage_md5_set'])
            positive_pairs.add((host_str, phage_str))
        
        # Generate negative samples
        attempts = 0
        max_attempts = n_samples * 10
        
        while len(negative_bags) < n_samples and attempts < max_attempts:
            attempts += 1
            
            # Random selection
            host_idx = np.random.randint(len(all_host_sets))
            phage_idx = np.random.randint(len(all_phage_sets))
            
            host_hashes = all_host_sets[host_idx]
            phage_hashes = all_phage_sets[phage_idx]
            
            # Check if this is a positive pair
            host_str = ','.join(host_hashes)
            phage_str = ','.join(phage_hashes)
            
            if (host_str, phage_str) not in positive_pairs:
                # Create negative bag
                bag = MultiInstanceBag(
                    host_hashes=host_hashes,
                    phage_hashes=phage_hashes,
                    label=0,
                    phage_id=f"neg_{len(negative_bags)}"
                )
                negative_bags.append(bag)
        
        if len(negative_bags) < n_samples:
            self.logger.warning(f"Could only generate {len(negative_bags)}/{n_samples} negative samples")
        
        return negative_bags

