"""
Simple random splitting with RBP deduplication
Ensures val/test don't contain RBPs that were in train
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
from typing import Tuple, List, Set
import os

class SimpleDataSplitter:
    """
    Splits phage-host interaction data randomly, then removes samples
    from val/test that contain RBPs already seen in training
    """
    
    def __init__(self, data_path: str, seed: int = 42):
        self.data_path = data_path
        self.seed = seed
        np.random.seed(seed)
        
        # Load data
        self.df = pd.read_csv(data_path, sep='\t')
        print(f"Loaded {len(self.df)} interactions")
        
    def parse_rbps(self, indices: List[int]) -> Set[str]:
        """Extract all unique RBP hashes from given sample indices"""
        rbp_hashes = set()
        for idx in indices:
            row = self.df.iloc[idx]
            # Handle both single and multiple RBPs
            if ',' in str(row['rbp_md5']):
                rbp_hashes.update(row['rbp_md5'].split(','))
            else:
                rbp_hashes.add(row['rbp_md5'])
        return rbp_hashes
    
    def parse_markers(self, indices: List[int]) -> Set[str]:
        """Extract all unique marker hashes from given sample indices"""
        marker_hashes = set()
        for idx in indices:
            row = self.df.iloc[idx]
            # Handle both single and multiple markers
            if ',' in str(row['marker_md5']):
                marker_hashes.update(row['marker_md5'].split(','))
            else:
                marker_hashes.add(row['marker_md5'])
        return marker_hashes
    
    def has_rbp_overlap(self, sample_idx: int, train_rbps: Set[str]) -> bool:
        """Check if a sample contains any RBPs that are in the training set"""
        row = self.df.iloc[sample_idx]
        sample_rbps = row['rbp_md5'].split(',') if ',' in str(row['rbp_md5']) else [row['rbp_md5']]
        return any(rbp in train_rbps for rbp in sample_rbps)
    
    def split_data(self, train_ratio: float = 0.6, val_ratio: float = 0.2) -> Tuple[List[int], List[int], List[int]]:
        """
        Create train/val/test splits ensuring no RBP overlap
        """
        print(f"\nCreating splits (train={train_ratio}, val={val_ratio}, test={1-train_ratio-val_ratio})...")
        
        # Initial random split
        n_samples = len(self.df)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Calculate split points
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        # Initial splits
        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train:n_train + n_val].tolist()
        test_idx = indices[n_train + n_val:].tolist()
        
        print(f"Initial split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        # Get all RBPs in training set
        train_rbps = self.parse_rbps(train_idx)
        print(f"Training set contains {len(train_rbps)} unique RBPs")
        
        # Remove samples from val that contain training RBPs
        val_clean = []
        val_removed = []
        for idx in val_idx:
            if self.has_rbp_overlap(idx, train_rbps):
                val_removed.append(idx)
            else:
                val_clean.append(idx)
        
        # Remove samples from test that contain training RBPs
        test_clean = []
        test_removed = []
        for idx in test_idx:
            if self.has_rbp_overlap(idx, train_rbps):
                test_removed.append(idx)
            else:
                test_clean.append(idx)
        
        print(f"\nRemoved {len(val_removed)} samples from validation (had training RBPs)")
        print(f"Removed {len(test_removed)} samples from test (had training RBPs)")
        
        # Update splits
        val_idx = val_clean
        test_idx = test_clean
        
        print(f"\nFinal split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        total_used = len(train_idx) + len(val_idx) + len(test_idx)
        print(f"Total samples used: {total_used}/{n_samples} ({total_used/n_samples:.1%})")
        
        return train_idx, val_idx, test_idx
    
    def verify_splits(self, train_idx: List[int], val_idx: List[int], test_idx: List[int]):
        """
        Verify the quality of splits and check for any remaining overlap
        """
        print("\n" + "="*50)
        print("VERIFICATION REPORT")
        print("="*50)
        
        # Get proteins in each split
        train_markers = self.parse_markers(train_idx)
        train_rbps = self.parse_rbps(train_idx)
        
        val_markers = self.parse_markers(val_idx)
        val_rbps = self.parse_rbps(val_idx)
        
        test_markers = self.parse_markers(test_idx)
        test_rbps = self.parse_rbps(test_idx)
        
        # Check RBP overlaps (should be zero for val/test with train)
        rbp_train_val = train_rbps & val_rbps
        rbp_train_test = train_rbps & test_rbps
        rbp_val_test = val_rbps & test_rbps
        
        print(f"\nRBP overlaps:")
        print(f"  Train-Val: {len(rbp_train_val)} shared RBPs")
        print(f"  Train-Test: {len(rbp_train_test)} shared RBPs")
        print(f"  Val-Test: {len(rbp_val_test)} shared RBPs")
        
        if len(rbp_train_val) > 0 or len(rbp_train_test) > 0:
            print("  ⚠️ WARNING: RBPs leaked from training set!")
        else:
            print("  ✅ No RBP leakage from training set")
        
        # Check marker overlaps (these are allowed but good to know)
        marker_train_val = train_markers & val_markers
        marker_train_test = train_markers & test_markers
        marker_val_test = val_markers & test_markers
        
        print(f"\nMarker overlaps (allowed):")
        print(f"  Train-Val: {len(marker_train_val)} shared markers")
        print(f"  Train-Test: {len(marker_train_test)} shared markers")
        print(f"  Val-Test: {len(marker_val_test)} shared markers")
        
        # Coverage statistics
        total_unique_markers = len(train_markers | val_markers | test_markers)
        total_unique_rbps = len(train_rbps | val_rbps | test_rbps)
        
        print(f"\nProtein coverage:")
        print(f"  Train: {len(train_markers)} markers ({len(train_markers)/total_unique_markers:.1%}), "
              f"{len(train_rbps)} RBPs ({len(train_rbps)/total_unique_rbps:.1%})")
        print(f"  Val: {len(val_markers)} markers ({len(val_markers)/total_unique_markers:.1%}), "
              f"{len(val_rbps)} RBPs ({len(val_rbps)/total_unique_rbps:.1%})")
        print(f"  Test: {len(test_markers)} markers ({len(test_markers)/total_unique_markers:.1%}), "
              f"{len(test_rbps)} RBPs ({len(test_rbps)/total_unique_rbps:.1%})")
        
        # Distribution of multi-instance samples
        print(f"\nMulti-instance distribution:")
        for split_name, split_idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
            multi_marker = 0
            multi_rbp = 0
            for idx in split_idx:
                row = self.df.iloc[idx]
                if ',' in str(row['marker_md5']):
                    multi_marker += 1
                if ',' in str(row['rbp_md5']):
                    multi_rbp += 1
            print(f"  {split_name}: {multi_marker} samples with multiple markers, "
                  f"{multi_rbp} samples with multiple RBPs")
        
        return len(rbp_train_val) == 0 and len(rbp_train_test) == 0
    
    def save_splits(self, train_idx: List[int], val_idx: List[int], test_idx: List[int], 
                    output_dir: str = 'data/processed'):
        """Save the split indices and data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save indices
        splits = {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'method': 'simple_random_with_rbp_dedup',
            'seed': self.seed
        }
        
        with open(f'{output_dir}/splits.pkl', 'wb') as f:
            pickle.dump(splits, f)
        
        # Save actual data splits
        train_df = self.df.iloc[train_idx].copy()
        val_df = self.df.iloc[val_idx].copy()
        test_df = self.df.iloc[test_idx].copy()
        
        train_df.to_csv(f'{output_dir}/train.tsv', sep='\t', index=False)
        val_df.to_csv(f'{output_dir}/val.tsv', sep='\t', index=False)
        test_df.to_csv(f'{output_dir}/test.tsv', sep='\t', index=False)
        
        # Save split statistics
        stats = {
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx),
            'total_samples': len(self.df),
            'samples_used': len(train_idx) + len(val_idx) + len(test_idx)
        }
        
        with open(f'{output_dir}/split_stats.txt', 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\nSplits saved to {output_dir}/")
    
    def run(self):
        """Execute the splitting pipeline"""
        # Create splits
        train_idx, val_idx, test_idx = self.split_data()
        
        # Verify splits
        is_valid = self.verify_splits(train_idx, val_idx, test_idx)
        
        if is_valid:
            # Save splits
            self.save_splits(train_idx, val_idx, test_idx)
            print("\n✅ Splitting completed successfully!")
        else:
            print("\n⚠️ Warning: Splits have RBP leakage but were saved anyway")
            self.save_splits(train_idx, val_idx, test_idx)
        
        return train_idx, val_idx, test_idx


if __name__ == "__main__":
    splitter = SimpleDataSplitter('data/dedup.phage_marker_rbp_with_phage_entropy.tsv')
    train_idx, val_idx, test_idx = splitter.run()