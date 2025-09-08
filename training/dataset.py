"""
PyTorch Dataset classes for phage-host interaction data using MD5 hash-based loading
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

from utils.data_utils import HashBasedEmbeddingLoader, MultiInstanceBag, DataProcessor


class PhageHostDataset(Dataset):
    """
    PyTorch Dataset for phage-host interaction data
    Uses MD5 hashes to load embeddings for multi-instance bags
    """
    
    def __init__(self,
                 bags: List[MultiInstanceBag],
                 embedding_loader: HashBasedEmbeddingLoader,
                 max_hosts: int = 2,
                 max_phages: int = 20,
                 augment: bool = False,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the dataset
        
        Args:
            bags: List of MultiInstanceBag objects
            embedding_loader: HashBasedEmbeddingLoader instance
            max_hosts: Maximum number of host proteins (for padding)
            max_phages: Maximum number of phage proteins (for padding)
            augment: Whether to apply data augmentation
            logger: Optional logger
        """
        self.bags = bags
        self.embedding_loader = embedding_loader
        self.max_hosts = max_hosts
        self.max_phages = max_phages
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
        
        host_counts = [len(bag.host_hashes) for bag in self.bags]
        phage_counts = [len(bag.phage_hashes) for bag in self.bags]
        
        self.logger.info(f"Dataset statistics:")
        self.logger.info(f"  Total samples: {len(self.bags)}")
        self.logger.info(f"  Positive: {n_positive} ({n_positive/len(self.bags):.1%})")
        self.logger.info(f"  Negative: {n_negative} ({n_negative/len(self.bags):.1%})")
        self.logger.info(f"  Host proteins per bag: min={min(host_counts)}, "
                        f"max={max(host_counts)}, avg={np.mean(host_counts):.1f}")
        self.logger.info(f"  Phage proteins per bag: min={min(phage_counts)}, "
                        f"max={max(phage_counts)}, avg={np.mean(phage_counts):.1f}")
        
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
            - host_embeddings: Padded host embeddings (max_hosts, embedding_dim)
            - phage_embeddings: Padded phage embeddings (max_phages, embedding_dim)
            - host_mask: Binary mask for hosts (max_hosts,)
            - phage_mask: Binary mask for phages (max_phages,)
            - label: Binary label (scalar)
        """
        bag = self.bags[idx]
        
        # Get embeddings using the new method
        try:
            host_embeddings, phage_embeddings, host_mask, phage_mask = bag.get_embeddings(
                self.embedding_loader,
                max_host=self.max_hosts,
                max_phage=self.max_phages
            )
        except Exception as e:
            # Log error and return a zero sample
            if self.logger:
                self.logger.error(f"Failed to get embeddings for index {idx}: {e}")
            return {
                'host_embeddings': torch.zeros((self.max_hosts, self.embedding_dim)),
                'phage_embeddings': torch.zeros((self.max_phages, self.embedding_dim)),
                'host_mask': torch.zeros(self.max_hosts),
                'phage_mask': torch.zeros(self.max_phages),
                'label': torch.tensor(0.0, dtype=torch.float32)
            }
        
        # Apply augmentation if enabled
        if self.augment and bag.label == 1:  # Only augment positive samples
            # Random dropout of proteins (simulate missing proteins)
            if np.random.random() < 0.1:  # 10% chance
                n_hosts = np.sum(host_mask > 0)
                n_phages = np.sum(phage_mask > 0)
                
                if n_hosts > 1 and np.random.random() < 0.5:
                    # Drop one host protein
                    drop_idx = np.random.randint(n_hosts)
                    host_mask[drop_idx] = 0.0
                    
                if n_phages > 1 and np.random.random() < 0.5:
                    # Drop one phage protein
                    drop_idx = np.random.randint(n_phages)
                    phage_mask[drop_idx] = 0.0
                    
            # Add small noise to embeddings
            if np.random.random() < 0.2:  # 20% chance
                noise_scale = 0.01
                host_embeddings = host_embeddings + np.random.randn(*host_embeddings.shape).astype(np.float32) * noise_scale
                phage_embeddings = phage_embeddings + np.random.randn(*phage_embeddings.shape).astype(np.float32) * noise_scale
        
        # Convert to tensors
        return {
            'host_embeddings': torch.from_numpy(host_embeddings.astype(np.float32)),
            'phage_embeddings': torch.from_numpy(phage_embeddings.astype(np.float32)),
            'host_mask': torch.from_numpy(host_mask.astype(np.float32)),
            'phage_mask': torch.from_numpy(phage_mask.astype(np.float32)),
            'label': torch.tensor(bag.label, dtype=torch.float32)
        }


class PhageHostDataModule:
    """
    Data module for managing train/val/test datasets
    """
    
    def __init__(self,
                 data_path: str,
                 splits_path: str,
                 embeddings_dir: str,
                 batch_size: int = 32,
                 max_hosts: int = 2,
                 max_phages: int = 20,
                 negative_ratio_train: float = 1.0,
                 negative_ratio_val: float = 4.0,
                 negative_ratio_test: float = 49.0,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 augment_train: bool = True,
                 cache_size: int = 10000,
                 preload_all: bool = False,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the data module
        
        Args:
            data_path: Path to data file with hash columns
            splits_path: Path to splits pickle file
            embeddings_dir: Directory containing embedding files
            batch_size: Batch size for data loaders
            max_hosts: Maximum number of host proteins
            max_phages: Maximum number of phage proteins
            negative_ratio_train: Negative to positive ratio for training
            negative_ratio_val: Negative to positive ratio for validation
            negative_ratio_test: Negative to positive ratio for testing
            num_workers: Number of data loader workers
            pin_memory: Whether to pin memory for GPU transfer
            augment_train: Whether to augment training data
            cache_size: Size of embedding cache
            preload_all: Whether to preload all embeddings
            logger: Optional logger
        """
        self.data_path = Path(data_path)
        self.splits_path = Path(splits_path)
        self.embeddings_dir = Path(embeddings_dir)
        self.batch_size = batch_size
        self.max_hosts = max_hosts
        self.max_phages = max_phages
        self.negative_ratios = {
            'train': negative_ratio_train,
            'val': negative_ratio_val,
            'test': negative_ratio_test
        }
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augment_train = augment_train
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize embedding loader
        self.embedding_loader = HashBasedEmbeddingLoader(
            host_embedding_path=str(self.embeddings_dir / 'host_embeddings.h5'),
            phage_embedding_path=str(self.embeddings_dir / 'phage_embeddings.h5'),
            cache_size=cache_size,
            preload_all=preload_all,
            logger=self.logger
        )
        
        # Initialize data processor
        self.data_processor = DataProcessor(logger=self.logger)
        
        # Load and prepare datasets
        self._prepare_datasets()
        
    def _prepare_datasets(self):
        """Prepare train, validation, and test datasets"""
        # Load splits
        if self.splits_path.exists():
            self.logger.info(f"Loading splits from {self.splits_path}")
            with open(self.splits_path, 'rb') as f:
                splits = pickle.load(f)
            
            train_df = splits['train']
            val_df = splits['val']
            test_df = splits['test']
        else:
            # Load full data and create default split
            self.logger.warning(f"Splits file not found at {self.splits_path}")
            self.logger.info("Creating default 60-20-20 split")
            
            df = self.data_processor.load_data_with_hashes(str(self.data_path))
            
            # Shuffle and split
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            n_total = len(df)
            n_train = int(n_total * 0.6)
            n_val = int(n_total * 0.2)
            
            train_df = df.iloc[:n_train]
            val_df = df.iloc[n_train:n_train+n_val]
            test_df = df.iloc[n_train+n_val:]
        
        # Create bags for each split
        self.train_bags = self.data_processor.create_bags_from_dataframe(
            train_df, 
            negative_ratio=self.negative_ratios['train'],
            seed=42
        )
        
        self.val_bags = self.data_processor.create_bags_from_dataframe(
            val_df,
            negative_ratio=self.negative_ratios['val'],
            seed=43
        )
        
        self.test_bags = self.data_processor.create_bags_from_dataframe(
            test_df,
            negative_ratio=self.negative_ratios['test'],
            seed=44
        )
        
        self.logger.info(f"Dataset sizes:")
        self.logger.info(f"  Train: {len(self.train_bags)} bags")
        self.logger.info(f"  Val: {len(self.val_bags)} bags")
        self.logger.info(f"  Test: {len(self.test_bags)} bags")
        
        # Create datasets
        self.train_dataset = PhageHostDataset(
            bags=self.train_bags,
            embedding_loader=self.embedding_loader,
            max_hosts=self.max_hosts,
            max_phages=self.max_phages,
            augment=self.augment_train,
            logger=self.logger
        )
        
        self.val_dataset = PhageHostDataset(
            bags=self.val_bags,
            embedding_loader=self.embedding_loader,
            max_hosts=self.max_hosts,
            max_phages=self.max_phages,
            augment=False,
            logger=self.logger
        )
        
        self.test_dataset = PhageHostDataset(
            bags=self.test_bags,
            embedding_loader=self.embedding_loader,
            max_hosts=self.max_hosts,
            max_phages=self.max_phages,
            augment=False,
            logger=self.logger
        )
    
    def train_dataloader(self) -> DataLoader:
        """Get training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
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