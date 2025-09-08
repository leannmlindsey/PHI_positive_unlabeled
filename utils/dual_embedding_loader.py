"""
Dual embedding loader for separate host and phage embeddings
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from collections import OrderedDict
import atexit


class DualEmbeddingLoader:
    """
    Loads embeddings from separate host and phage HDF5 files with LRU caching
    """
    
    def __init__(self,
                 host_embeddings_path: str,
                 phage_embeddings_path: str,
                 cache_size: int = 10000,
                 preload_all: bool = False,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the dual embedding loader
        
        Args:
            host_embeddings_path: Path to host embeddings HDF5 file
            phage_embeddings_path: Path to phage embeddings HDF5 file
            cache_size: Maximum number of embeddings to cache per file
            preload_all: If True, loads all embeddings into memory
            logger: Optional logger instance
        """
        self.host_path = Path(host_embeddings_path)
        self.phage_path = Path(phage_embeddings_path)
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
        
        # Embedding dimensions
        self.embedding_dim = None
        self.total_host_embeddings = 0
        self.total_phage_embeddings = 0
        
        # Cache statistics
        self._host_hits = 0
        self._host_misses = 0
        self._phage_hits = 0
        self._phage_misses = 0
        
        # Initialize the loader
        self._initialize()
        
        # Register cleanup
        atexit.register(self.close)
    
    def _initialize(self):
        """Initialize the embedding loader"""
        # Open host embeddings file
        if not self.host_path.exists():
            raise FileNotFoundError(f"Host embeddings file not found: {self.host_path}")
        
        self._host_file = h5py.File(self.host_path, 'r')
        host_embeddings = self._host_file['embeddings']
        host_hashes = self._host_file['hashes'][:].astype(str)
        
        self.total_host_embeddings = len(host_hashes)
        self.embedding_dim = host_embeddings.shape[1]
        
        # Build host hash to index mapping
        for idx, hash_val in enumerate(host_hashes):
            self._host_hash_to_idx[hash_val] = idx
        
        self.logger.info(f"Loaded host embeddings: {self.total_host_embeddings} proteins, "
                        f"{self.embedding_dim}-dim embeddings")
        
        # Open phage embeddings file
        if not self.phage_path.exists():
            raise FileNotFoundError(f"Phage embeddings file not found: {self.phage_path}")
        
        self._phage_file = h5py.File(self.phage_path, 'r')
        phage_embeddings = self._phage_file['embeddings']
        phage_hashes = self._phage_file['hashes'][:].astype(str)
        
        self.total_phage_embeddings = len(phage_hashes)
        
        # Verify embedding dimensions match
        if phage_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: host={self.embedding_dim}, "
                           f"phage={phage_embeddings.shape[1]}")
        
        # Build phage hash to index mapping
        for idx, hash_val in enumerate(phage_hashes):
            self._phage_hash_to_idx[hash_val] = idx
        
        self.logger.info(f"Loaded phage embeddings: {self.total_phage_embeddings} proteins, "
                        f"{self.embedding_dim}-dim embeddings")
        
        # Preload if requested
        if self.preload_all:
            self.logger.info("Preloading all embeddings into memory...")
            
            # Load all host embeddings
            for hash_val, idx in self._host_hash_to_idx.items():
                self._host_cache[hash_val] = host_embeddings[idx]
            
            # Load all phage embeddings
            for hash_val, idx in self._phage_hash_to_idx.items():
                self._phage_cache[hash_val] = phage_embeddings[idx]
            
            self.logger.info("All embeddings loaded into memory")
    
    def get_host_embedding(self, protein_hash: str) -> np.ndarray:
        """
        Get embedding for a host protein by its MD5 hash
        
        Args:
            protein_hash: MD5 hash of the protein sequence
            
        Returns:
            Embedding vector as numpy array
        """
        # Check cache first
        if protein_hash in self._host_cache:
            # Move to end (most recently used)
            self._host_cache.move_to_end(protein_hash)
            self._host_hits += 1
            return self._host_cache[protein_hash]
        
        # Cache miss - load from file
        self._host_misses += 1
        
        if protein_hash not in self._host_hash_to_idx:
            raise KeyError(f"Host protein hash not found: {protein_hash}")
        
        idx = self._host_hash_to_idx[protein_hash]
        embedding = self._host_file['embeddings'][idx]
        
        # Add to cache
        self._add_to_host_cache(protein_hash, embedding)
        
        return embedding
    
    def get_phage_embedding(self, protein_hash: str) -> np.ndarray:
        """
        Get embedding for a phage protein by its MD5 hash
        
        Args:
            protein_hash: MD5 hash of the protein sequence
            
        Returns:
            Embedding vector as numpy array
        """
        # Check cache first
        if protein_hash in self._phage_cache:
            # Move to end (most recently used)
            self._phage_cache.move_to_end(protein_hash)
            self._phage_hits += 1
            return self._phage_cache[protein_hash]
        
        # Cache miss - load from file
        self._phage_misses += 1
        
        if protein_hash not in self._phage_hash_to_idx:
            raise KeyError(f"Phage protein hash not found: {protein_hash}")
        
        idx = self._phage_hash_to_idx[protein_hash]
        embedding = self._phage_file['embeddings'][idx]
        
        # Add to cache
        self._add_to_phage_cache(protein_hash, embedding)
        
        return embedding
    
    def get_embedding(self, protein_hash: str, protein_type: str = 'auto') -> np.ndarray:
        """
        Get embedding for a protein by its MD5 hash
        
        Args:
            protein_hash: MD5 hash of the protein sequence
            protein_type: 'host', 'phage', or 'auto' (tries both)
            
        Returns:
            Embedding vector as numpy array
        """
        if protein_type == 'host':
            return self.get_host_embedding(protein_hash)
        elif protein_type == 'phage':
            return self.get_phage_embedding(protein_hash)
        elif protein_type == 'auto':
            # Try host first, then phage
            if protein_hash in self._host_hash_to_idx:
                return self.get_host_embedding(protein_hash)
            elif protein_hash in self._phage_hash_to_idx:
                return self.get_phage_embedding(protein_hash)
            else:
                raise KeyError(f"Protein hash not found in either host or phage: {protein_hash}")
        else:
            raise ValueError(f"Invalid protein_type: {protein_type}")
    
    def _add_to_host_cache(self, protein_hash: str, embedding: np.ndarray):
        """Add embedding to host cache with LRU eviction"""
        if len(self._host_cache) >= self.cache_size and protein_hash not in self._host_cache:
            # Evict least recently used
            self._host_cache.popitem(last=False)
        
        self._host_cache[protein_hash] = embedding
    
    def _add_to_phage_cache(self, protein_hash: str, embedding: np.ndarray):
        """Add embedding to phage cache with LRU eviction"""
        if len(self._phage_cache) >= self.cache_size and protein_hash not in self._phage_cache:
            # Evict least recently used
            self._phage_cache.popitem(last=False)
        
        self._phage_cache[protein_hash] = embedding
    
    def has_embedding(self, protein_hash: str, protein_type: str = 'auto') -> bool:
        """
        Check if embedding exists for a protein hash
        
        Args:
            protein_hash: MD5 hash of the protein sequence
            protein_type: 'host', 'phage', or 'auto'
            
        Returns:
            True if embedding exists
        """
        if protein_type == 'host':
            return protein_hash in self._host_hash_to_idx
        elif protein_type == 'phage':
            return protein_hash in self._phage_hash_to_idx
        elif protein_type == 'auto':
            return (protein_hash in self._host_hash_to_idx or 
                   protein_hash in self._phage_hash_to_idx)
        else:
            raise ValueError(f"Invalid protein_type: {protein_type}")
    
    def cache_stats(self) -> Dict[str, any]:
        """Get cache statistics"""
        host_total = self._host_hits + self._host_misses
        phage_total = self._phage_hits + self._phage_misses
        
        return {
            'host_cache_size': len(self._host_cache),
            'host_cache_hits': self._host_hits,
            'host_cache_misses': self._host_misses,
            'host_hit_rate': self._host_hits / host_total if host_total > 0 else 0.0,
            'phage_cache_size': len(self._phage_cache),
            'phage_cache_hits': self._phage_hits,
            'phage_cache_misses': self._phage_misses,
            'phage_hit_rate': self._phage_hits / phage_total if phage_total > 0 else 0.0,
            'total_host_embeddings': self.total_host_embeddings,
            'total_phage_embeddings': self.total_phage_embeddings
        }
    
    def close(self):
        """Close the HDF5 files"""
        if self._host_file is not None:
            self._host_file.close()
            self._host_file = None
        
        if self._phage_file is not None:
            self._phage_file.close()
            self._phage_file = None
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()