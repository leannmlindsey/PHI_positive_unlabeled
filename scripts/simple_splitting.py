"""
Data splitting with RBP deduplication using MD5 hashes
Ensures val/test don't contain RBPs that were in train
Works with the new data format that includes host_md5_set and phage_md5_set columns
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
from typing import Tuple, List, Set
import os
import argparse


class DataSplitter:
    """
    Splits phage-host interaction data with proper RBP deduplication
    Uses MD5 hashes to track unique proteins
    """
    
    def __init__(self, data_path: str, seed: int = 42):
        self.data_path = data_path
        self.seed = seed
        np.random.seed(seed)
        
        # Load data with hash columns
        print(f"Loading data from {data_path}")
        self.df = pd.read_csv(data_path, sep='\t')
        print(f"Loaded {len(self.df)} interactions")
        
        # Check for required columns
        required_cols = ['host_md5_set', 'phage_md5_set', 'phage_id']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate data integrity
        self.validate_data()
    
    def validate_data(self):
        """Validate data integrity and report any issues"""
        print("\nValidating data integrity...")
        
        # Check for missing hash sets
        missing_host = self.df['host_md5_set'].isna().sum()
        missing_phage = self.df['phage_md5_set'].isna().sum()
        empty_host = (self.df['host_md5_set'] == '').sum()
        empty_phage = (self.df['phage_md5_set'] == '').sum()
        
        print(f"  Missing host MD5 sets: {missing_host}")
        print(f"  Missing phage MD5 sets: {missing_phage}")
        print(f"  Empty host MD5 sets: {empty_host}")
        print(f"  Empty phage MD5 sets: {empty_phage}")
        
        # Count multi-protein instances
        multi_host = (self.df['host_md5_set'].str.count(',') >= 1).sum()
        multi_phage = (self.df['phage_md5_set'].str.count(',') >= 1).sum()
        
        print(f"  Rows with multiple host proteins: {multi_host}")
        print(f"  Rows with multiple phage proteins: {multi_phage}")
        
        # Get unique protein counts
        all_host_hashes = set()
        all_phage_hashes = set()
        
        for idx, row in self.df.iterrows():
            host_hashes = str(row['host_md5_set']).split(',') if pd.notna(row['host_md5_set']) else []
            phage_hashes = str(row['phage_md5_set']).split(',') if pd.notna(row['phage_md5_set']) else []
            
            all_host_hashes.update([h for h in host_hashes if h and h != 'nan'])
            all_phage_hashes.update([h for h in phage_hashes if h and h != 'nan'])
        
        print(f"  Total unique host proteins: {len(all_host_hashes)}")
        print(f"  Total unique phage proteins: {len(all_phage_hashes)}")
    
    def get_rbp_hashes(self, row) -> Set[str]:
        """Extract RBP MD5 hashes from a row"""
        phage_field = str(row['phage_md5_set'])
        
        if pd.isna(row['phage_md5_set']) or phage_field == 'nan' or not phage_field:
            return set()
        
        # Split by comma and clean
        hashes = [h.strip() for h in phage_field.split(',')]
        return set(h for h in hashes if h and h != 'nan')
    
    def split_data(self, train_ratio: float = 0.6, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test with RBP deduplication
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print("\n=== Splitting Data ===")
        
        # Shuffle data
        shuffled_df = self.df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        # Calculate split points
        n_total = len(shuffled_df)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Initial random split
        train_df = shuffled_df.iloc[:n_train].copy()
        val_df = shuffled_df.iloc[n_train:n_train + n_val].copy()
        test_df = shuffled_df.iloc[n_train + n_val:].copy()
        
        print(f"\nInitial split:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        
        # Get all RBP hashes from training set
        train_rbp_hashes = set()
        for idx, row in train_df.iterrows():
            train_rbp_hashes.update(self.get_rbp_hashes(row))
        
        print(f"\nUnique RBPs in training: {len(train_rbp_hashes)}")
        
        # Remove validation samples with RBPs seen in training
        val_mask = []
        val_removed = 0
        for idx, row in val_df.iterrows():
            rbp_hashes = self.get_rbp_hashes(row)
            if rbp_hashes and rbp_hashes.intersection(train_rbp_hashes):
                val_mask.append(False)
                val_removed += 1
            else:
                val_mask.append(True)
        
        val_df = val_df[val_mask].reset_index(drop=True)
        print(f"Removed {val_removed} validation samples with training RBPs")
        
        # Remove test samples with RBPs seen in training
        test_mask = []
        test_removed = 0
        for idx, row in test_df.iterrows():
            rbp_hashes = self.get_rbp_hashes(row)
            if rbp_hashes and rbp_hashes.intersection(train_rbp_hashes):
                test_mask.append(False)
                test_removed += 1
            else:
                test_mask.append(True)
        
        test_df = test_df[test_mask].reset_index(drop=True)
        print(f"Removed {test_removed} test samples with training RBPs")
        
        # Final statistics
        print(f"\n=== Final Split ===")
        print(f"Train: {len(train_df)} samples ({len(train_df)/n_total:.1%})")
        print(f"Val: {len(val_df)} samples ({len(val_df)/n_total:.1%})")
        print(f"Test: {len(test_df)} samples ({len(test_df)/n_total:.1%})")
        print(f"Total retained: {len(train_df) + len(val_df) + len(test_df)} / {n_total}")
        
        # Verify no RBP overlap
        val_rbps = set()
        for idx, row in val_df.iterrows():
            val_rbps.update(self.get_rbp_hashes(row))
        
        test_rbps = set()
        for idx, row in test_df.iterrows():
            test_rbps.update(self.get_rbp_hashes(row))
        
        val_overlap = val_rbps.intersection(train_rbp_hashes)
        test_overlap = test_rbps.intersection(train_rbp_hashes)
        
        if val_overlap:
            print(f"WARNING: {len(val_overlap)} RBPs still overlap between train and val!")
        if test_overlap:
            print(f"WARNING: {len(test_overlap)} RBPs still overlap between train and test!")
        
        if not val_overlap and not test_overlap:
            print("\n✓ Successfully verified: No RBP overlap between train and val/test")
        
        return train_df, val_df, test_df
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str):
        """Save the splits to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as TSV files
        train_path = os.path.join(output_dir, 'train.tsv')
        val_path = os.path.join(output_dir, 'val.tsv')
        test_path = os.path.join(output_dir, 'test.tsv')
        
        train_df.to_csv(train_path, sep='\t', index=False)
        val_df.to_csv(val_path, sep='\t', index=False)
        test_df.to_csv(test_path, sep='\t', index=False)
        
        print(f"\nSaved splits to {output_dir}/")
        print(f"  train.tsv: {len(train_df)} samples")
        print(f"  val.tsv: {len(val_df)} samples")
        print(f"  test.tsv: {len(test_df)} samples")
        
        # Save as pickle for easy loading
        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        pickle_path = os.path.join(output_dir, 'splits.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(splits, f)
        print(f"  splits.pkl: Combined pickle file")
        
        # Save statistics
        stats_path = os.path.join(output_dir, 'split_stats.txt')
        with open(stats_path, 'w') as f:
            f.write(f"Train: {len(train_df)} samples\n")
            f.write(f"Val: {len(val_df)} samples\n")
            f.write(f"Test: {len(test_df)} samples\n")
            f.write(f"Total: {len(train_df) + len(val_df) + len(test_df)} samples\n")
        
        print(f"  split_stats.txt: Statistics file")


def main():
    parser = argparse.ArgumentParser(description='Split data with RBP deduplication')
    parser.add_argument('--data_path', type=str,
                       default='data/dedup.labeled_marker_rbp_phageID_with_hashes.tsv',
                       help='Path to data file with hash columns')
    parser.add_argument('--output_dir', type=str,
                       default='data/processed',
                       help='Directory to save splits')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                       help='Proportion of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Proportion of data for validation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create splitter
    splitter = DataSplitter(args.data_path, args.seed)
    
    # Split data
    train_df, val_df, test_df = splitter.split_data(args.train_ratio, args.val_ratio)
    
    # Save splits
    splitter.save_splits(train_df, val_df, test_df, args.output_dir)
    
    print("\n=== Splitting Complete ===")


if __name__ == "__main__":
    main()