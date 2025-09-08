"""
PyTorch Dataset classes for phage-host interaction data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Any
import h5py

from utils.data_utils import EmbeddingLoader, MultiInstanceBag, DataProcessor


class PhageHostDataset(Dataset):
    """
    PyTorch Dataset for phage-host interaction data
    Handles multi-instance bags with variable numbers of proteins
    """
    
    def __init__(self,
                 bags: List[MultiInstanceBag],
                 embedding_loader: EmbeddingLoader,
                 max_markers: int = 2,
                 max_rbps: int = 20,
                 augment: bool = False,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the dataset
        
        Args:
            bags: List of MultiInstanceBag objects
            embedding_loader: EmbeddingLoader instance
            max_markers: Maximum number of markers (for padding)
            max_rbps: Maximum number of RBPs (for padding)
            augment: Whether to apply data augmentation
            logger: Optional logger
        """
        self.bags = bags
        self.embedding_loader = embedding_loader
        self.max_markers = max_markers
        self.max_rbps = max_rbps
        self.augment = augment
        self.logger = logger or logging.getLogger(__name__)
        
        # Get embedding dimension
        self.embedding_dim = embedding_loader.embedding_dim
        
        # Log dataset statistics
        self._log_statistics()
        
    def _log_statistics(self):
        """Log dataset statistics"""
        n_positive = sum(1 for bag in self.bags if bag.label == 1)
        n_negative = len(self.bags) - n_positive
        
        marker_counts = [len(bag.marker_hashes) for bag in self.bags]
        rbp_counts = [len(bag.rbp_hashes) for bag in self.bags]
        
        self.logger.info(f"Dataset statistics:")
        self.logger.info(f"  Total samples: {len(self.bags)}")
        self.logger.info(f"  Positive: {n_positive} ({n_positive/len(self.bags):.1%})")
        self.logger.info(f"  Negative: {n_negative} ({n_negative/len(self.bags):.1%})")
        self.logger.info(f"  Markers per bag: min={min(marker_counts)}, "
                        f"max={max(marker_counts)}, avg={np.mean(marker_counts):.1f}")
        self.logger.info(f"  RBPs per bag: min={min(rbp_counts)}, "
                        f"max={max(rbp_counts)}, avg={np.mean(rbp_counts):.1f}")
        
    def __len__(self) -> int:
        """Get dataset size"""
        return len(self.bags)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
            - marker_embeddings: Padded marker embeddings (max_markers, embedding_dim)
            - rbp_embeddings: Padded RBP embeddings (max_rbps, embedding_dim)
            - marker_mask: Binary mask for markers (max_markers,)
            - rbp_mask: Binary mask for RBPs (max_rbps,)
            - label: Binary label (scalar)
        """
        bag = self.bags[idx]
        
        # Get embeddings with error handling
        try:
            marker_embeddings, rbp_embeddings = bag.get_embeddings(self.embedding_loader, use_zero_for_missing=True)
        except ValueError as e:
            # Log error and return a sample with zero embeddings (will be masked out)
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Failed to get embeddings for index {idx}: {e}")
            # Return minimal valid sample that will be ignored due to masks
            return {
                'marker_embeddings': torch.zeros((self.max_markers, self.embedding_dim)),
                'rbp_embeddings': torch.zeros((self.max_rbps, self.embedding_dim)),
                'marker_mask': torch.zeros(self.max_markers),
                'rbp_mask': torch.zeros(self.max_rbps),
                'label': torch.tensor(0.0, dtype=torch.float32)
            }
        
        # Get actual counts
        n_markers = len(bag.marker_hashes)
        n_rbps = len(bag.rbp_hashes)
        
        # Initialize padded tensors
        padded_markers = np.zeros((self.max_markers, self.embedding_dim), dtype=np.float32)
        padded_rbps = np.zeros((self.max_rbps, self.embedding_dim), dtype=np.float32)
        marker_mask = np.zeros(self.max_markers, dtype=np.float32)
        rbp_mask = np.zeros(self.max_rbps, dtype=np.float32)
        
        # Fill with actual data
        padded_markers[:n_markers] = marker_embeddings
        padded_rbps[:n_rbps] = rbp_embeddings
        marker_mask[:n_markers] = 1.0
        rbp_mask[:n_rbps] = 1.0
        
        # Apply augmentation if enabled
        if self.augment and bag.label == 1:  # Only augment positive samples
            # Random dropout of proteins (simulate missing proteins)
            if np.random.random() < 0.1:  # 10% chance
                if n_markers > 1 and np.random.random() < 0.5:
                    # Drop one marker
                    drop_idx = np.random.randint(n_markers)
                    marker_mask[drop_idx] = 0.0
                    
                if n_rbps > 1 and np.random.random() < 0.5:
                    # Drop one RBP
                    drop_idx = np.random.randint(n_rbps)
                    rbp_mask[drop_idx] = 0.0
                    
            # Add small noise to embeddings
            if np.random.random() < 0.2:  # 20% chance
                noise_scale = 0.01
                padded_markers += np.random.randn(*padded_markers.shape).astype(np.float32) * noise_scale
                padded_rbps += np.random.randn(*padded_rbps.shape).astype(np.float32) * noise_scale
        
        # Convert to tensors
        return {
            'marker_embeddings': torch.from_numpy(padded_markers),
            'rbp_embeddings': torch.from_numpy(padded_rbps),
            'marker_mask': torch.from_numpy(marker_mask),
            'rbp_mask': torch.from_numpy(rbp_mask),
            'label': torch.tensor(bag.label, dtype=torch.float32)
        }


class PhageHostDataModule:
    """
    Data module that manages all data loading for training
    """
    
    def __init__(self,
                 data_path: str,
                 splits_path: str,
                 embeddings_path: str,
                 batch_size: int = 32,
                 negative_ratio: float = 1.0,
                 max_markers: int = 2,
                 max_rbps: int = 20,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 augment_train: bool = True,
                 seed: int = 42,
                 lazy_loading: bool = True,
                 cache_size: int = 10000,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the data module
        
        Args:
            data_path: Path to original TSV data
            splits_path: Path to splits pickle file
            embeddings_path: Path to HDF5 embeddings file
            batch_size: Batch size for data loaders
            negative_ratio: Ratio of negative to positive samples
            max_markers: Maximum number of markers
            max_rbps: Maximum number of RBPs
            num_workers: Number of data loader workers
            pin_memory: Whether to pin memory for GPU
            augment_train: Whether to augment training data
            seed: Random seed
            lazy_loading: Whether to use lazy loading for embeddings
            cache_size: Size of LRU cache for lazy loading
            logger: Optional logger
        """
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.max_markers = max_markers
        self.max_rbps = max_rbps
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augment_train = augment_train
        self.seed = seed
        self.lazy_loading = lazy_loading
        self.cache_size = cache_size
        self.logger = logger or logging.getLogger(__name__)
        
        # Load embeddings - check if we have separate host/phage files
        embeddings_dir = Path(embeddings_path)
        host_embeddings_file = embeddings_dir / 'host_embeddings.h5'
        phage_embeddings_file = embeddings_dir / 'phage_embeddings.h5'
        
        if host_embeddings_file.exists() and phage_embeddings_file.exists():
            # Use dual embedding loader for separate files
            self.logger.info(f"Loading separate host and phage embeddings from {embeddings_path}")
            from utils.dual_embedding_loader import DualEmbeddingLoader
            self.embedding_loader = DualEmbeddingLoader(
                host_embeddings_path=str(host_embeddings_file),
                phage_embeddings_path=str(phage_embeddings_file),
                cache_size=cache_size,
                preload_all=not lazy_loading,
                logger=self.logger
            )
        else:
            # Fall back to single file loader (legacy)
            self.logger.info(f"Loading embeddings from {embeddings_path} (lazy={lazy_loading})")
            if lazy_loading:
                from utils.data_utils import LazyEmbeddingLoader
                self.embedding_loader = LazyEmbeddingLoader(
                    embeddings_path, 
                    cache_size=cache_size,
                    logger=self.logger
                )
            else:
                self.embedding_loader = EmbeddingLoader(embeddings_path, self.logger)
        
        # Load data processor
        self.logger.info(f"Loading data from {data_path}")
        self.data_processor = DataProcessor(
            data_path=data_path,
            split_path=splits_path,
            embedding_loader=self.embedding_loader,
            logger=self.logger
        )
        
        # Prepare datasets
        self._prepare_datasets()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics if using lazy loading"""
        if self.lazy_loading and hasattr(self.embedding_loader, 'cache_stats'):
            return self.embedding_loader.cache_stats()
        return {}
        
    def _prepare_datasets(self):
        """Prepare train, validation, and test datasets"""
        # Get positive bags for each split
        train_positive = self.data_processor.get_split_data('train')
        val_positive = self.data_processor.get_split_data('val')
        test_positive = self.data_processor.get_split_data('test')
        
        # Generate negative samples for training
        train_negative = self.data_processor.generate_negative_samples(
            train_positive,
            negative_ratio=self.negative_ratio,
            seed=self.seed
        )
        
        # Generate negative samples for validation (less than training)
        val_negative = self.data_processor.generate_negative_samples(
            val_positive,
            negative_ratio=self.negative_ratio * 0.5,  # Less negatives for validation
            seed=self.seed + 1
        )
        
        # Generate negative samples for test
        test_negative = self.data_processor.generate_negative_samples(
            test_positive,
            negative_ratio=self.negative_ratio,
            seed=self.seed + 2
        )
        
        # Combine positive and negative
        train_bags = train_positive + train_negative
        val_bags = val_positive + val_negative
        test_bags = test_positive + test_negative
        
        # Shuffle training data
        np.random.seed(self.seed)
        np.random.shuffle(train_bags)
        
        # Create datasets
        self.train_dataset = PhageHostDataset(
            bags=train_bags,
            embedding_loader=self.embedding_loader,
            max_markers=self.max_markers,
            max_rbps=self.max_rbps,
            augment=self.augment_train,
            logger=self.logger
        )
        
        self.val_dataset = PhageHostDataset(
            bags=val_bags,
            embedding_loader=self.embedding_loader,
            max_markers=self.max_markers,
            max_rbps=self.max_rbps,
            augment=False,
            logger=self.logger
        )
        
        self.test_dataset = PhageHostDataset(
            bags=test_bags,
            embedding_loader=self.embedding_loader,
            max_markers=self.max_markers,
            max_rbps=self.max_rbps,
            augment=False,
            logger=self.logger
        )
        
        self.logger.info(f"Dataset sizes - Train: {len(self.train_dataset)}, "
                        f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        
    def train_dataloader(self) -> DataLoader:
        """Get training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True  # Drop last incomplete batch for stable training
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test data loader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )