#!/usr/bin/env python3
"""
Check how data is being loaded and if multi-protein bags are handled correctly
"""

import pandas as pd
import pickle
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def check_data_format():
    """Check the actual format of the data"""
    
    # Load data
    data_path = "data/dedup.phage_marker_rbp_with_phage_entropy.tsv"
    df = pd.read_csv(data_path, sep='\t')
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print()
    
    # Check marker_md5 field
    print("Checking marker_md5 field:")
    print(f"  Sample values (first 5):")
    for i in range(min(5, len(df))):
        val = df.iloc[i]['marker_md5']
        print(f"    Row {i}: '{val}' (length: {len(str(val))}, has comma: {(',' in str(val))})")
    
    # Count commas in marker_md5
    marker_with_comma = df['marker_md5'].astype(str).str.contains(',').sum()
    print(f"  Rows with comma in marker_md5: {marker_with_comma}")
    
    # Check for other separators
    marker_with_semicolon = df['marker_md5'].astype(str).str.contains(';').sum()
    print(f"  Rows with semicolon in marker_md5: {marker_with_semicolon}")
    
    marker_with_space = df['marker_md5'].astype(str).str.contains(' ').sum()
    print(f"  Rows with space in marker_md5: {marker_with_space}")
    
    print()
    
    # Check rbp_md5 field
    print("Checking rbp_md5 field:")
    print(f"  Sample values (first 5):")
    for i in range(min(5, len(df))):
        val = df.iloc[i]['rbp_md5']
        print(f"    Row {i}: '{val}' (length: {len(str(val))}, has comma: {(',' in str(val))})")
    
    # Count commas in rbp_md5
    rbp_with_comma = df['rbp_md5'].astype(str).str.contains(',').sum()
    print(f"  Rows with comma in rbp_md5: {rbp_with_comma}")
    
    # Check for other separators
    rbp_with_semicolon = df['rbp_md5'].astype(str).str.contains(';').sum()
    print(f"  Rows with semicolon in rbp_md5: {rbp_with_semicolon}")
    
    rbp_with_space = df['rbp_md5'].astype(str).str.contains(' ').sum()
    print(f"  Rows with space in rbp_md5: {rbp_with_space}")
    
    print()
    
    # Check the sequence fields for comparison
    print("Checking marker_gene_seq field (for comparison):")
    marker_seq_with_comma = df['marker_gene_seq'].astype(str).str.contains(',').sum()
    print(f"  Rows with comma in marker_gene_seq: {marker_seq_with_comma}")
    
    print()
    print("Checking rbp_seq field (for comparison):")
    rbp_seq_with_comma = df['rbp_seq'].astype(str).str.contains(',').sum()
    print(f"  Rows with comma in rbp_seq: {rbp_seq_with_comma}")
    
    print()
    
    # Check phage_id field for multi-phage entries
    print("Checking phage_id field:")
    print(f"  Sample values (first 5):")
    for i in range(min(5, len(df))):
        val = df.iloc[i]['phage_id']
        print(f"    Row {i}: '{val}' (has comma: {(',' in str(val))})")
    
    phage_with_comma = df['phage_id'].astype(str).str.contains(',').sum()
    print(f"  Rows with comma in phage_id: {phage_with_comma}")
    
    print()
    
    # Load splits and check distribution
    splits_path = "data/processed/splits.pkl"
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    
    print("Split sizes:")
    print(f"  Train: {len(splits['train_idx'])} samples")
    print(f"  Val: {len(splits['val_idx'])} samples")
    print(f"  Test: {len(splits['test_idx'])} samples")
    
    # Check train split for multi-protein entries
    print("\nChecking training split for multi-protein entries:")
    train_df = df.iloc[splits['train_idx']]
    
    train_marker_comma = train_df['marker_md5'].astype(str).str.contains(',').sum()
    train_rbp_comma = train_df['rbp_md5'].astype(str).str.contains(',').sum()
    train_phage_comma = train_df['phage_id'].astype(str).str.contains(',').sum()
    
    print(f"  Rows with comma in marker_md5: {train_marker_comma}")
    print(f"  Rows with comma in rbp_md5: {train_rbp_comma}")
    print(f"  Rows with comma in phage_id: {train_phage_comma}")
    
    # Check if we're looking at the right columns
    print("\nActual column indices:")
    for i, col in enumerate(df.columns):
        print(f"  Column {i}: {col}")

if __name__ == "__main__":
    check_data_format()