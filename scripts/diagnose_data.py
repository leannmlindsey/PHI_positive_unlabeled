"""
Diagnose data issues with hash columns
"""

import pandas as pd
import pickle
import numpy as np
from pathlib import Path

def diagnose_data():
    """Check the data for issues with hash columns"""
    
    # Load data
    data_path = "data/dedup.labeled_marker_rbp_phageID_with_hashes.tsv"
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, sep='\t')
    
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values
    print("\n=== Missing Values Analysis ===")
    for col in ['host_md5_set', 'phage_md5_set']:
        if col in df.columns:
            null_count = df[col].isna().sum()
            empty_count = (df[col] == '').sum()
            nan_str_count = (df[col] == 'nan').sum()
            print(f"{col}:")
            print(f"  - NULL/NaN: {null_count}")
            print(f"  - Empty string: {empty_count}")
            print(f"  - String 'nan': {nan_str_count}")
    
    # Load splits
    splits_path = "data/processed/splits.pkl"
    print(f"\nLoading splits from {splits_path}")
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    
    # Check problematic rows mentioned in the error
    problem_rows = [51, 64, 97, 113, 139, 190, 202, 220, 359, 399, 422, 448]
    
    print("\n=== Checking Specific Problem Rows ===")
    
    # Get train split
    if 'train' in splits:
        train_df = splits['train']
    else:
        train_indices = splits.get('train_idx', [])
        train_df = df.iloc[train_indices]
    
    print(f"Train split has {len(train_df)} rows")
    
    # Check specific rows in train split
    for row_num in problem_rows[:5]:  # Check first 5
        if row_num < len(train_df):
            row = train_df.iloc[row_num]
            print(f"\nRow {row_num}:")
            print(f"  host_md5_set: '{row.get('host_md5_set', 'MISSING')}'")
            print(f"  phage_md5_set: '{row.get('phage_md5_set', 'MISSING')}'")
            print(f"  phage_id: '{row.get('phage_id', 'MISSING')}'")
            
            # Parse hashes
            host_field = str(row.get('host_md5_set', '')).strip()
            phage_field = str(row.get('phage_md5_set', '')).strip()
            
            # Check various conditions
            print(f"  Checks:")
            print(f"    host_field == '': {host_field == ''}")
            print(f"    host_field == 'nan': {host_field == 'nan'}")
            print(f"    pd.isna(row.get('host_md5_set')): {pd.isna(row.get('host_md5_set', None))}")
            
            # Try parsing
            if host_field and host_field != 'nan' and host_field != '':
                host_hashes = [h.strip() for h in host_field.split(',') 
                             if h.strip() and h.strip() != 'nan' and h.strip() != '']
                print(f"    Parsed host hashes: {len(host_hashes)} hashes")
                if host_hashes:
                    print(f"      First hash: {host_hashes[0][:10]}...")
            else:
                print(f"    Host field failed checks")
            
            if phage_field and phage_field != 'nan' and phage_field != '':
                phage_hashes = [h.strip() for h in phage_field.split(',')
                              if h.strip() and h.strip() != 'nan' and h.strip() != '']
                print(f"    Parsed phage hashes: {len(phage_hashes)} hashes")
                if phage_hashes:
                    print(f"      First hash: {phage_hashes[0][:10]}...")
            else:
                print(f"    Phage field failed checks")
    
    # Check overall statistics
    print("\n=== Overall Statistics ===")
    
    total_rows = len(train_df)
    valid_rows = 0
    empty_host_rows = []
    empty_phage_rows = []
    
    for idx, row in train_df.iterrows():
        host_field = str(row.get('host_md5_set', '')).strip()
        phage_field = str(row.get('phage_md5_set', '')).strip()
        
        host_valid = False
        phage_valid = False
        
        if pd.notna(row.get('host_md5_set')) and host_field and host_field != 'nan' and host_field != '':
            host_hashes = [h.strip() for h in host_field.split(',') 
                         if h.strip() and h.strip() != 'nan' and h.strip() != '']
            if host_hashes:
                host_valid = True
        
        if pd.notna(row.get('phage_md5_set')) and phage_field and phage_field != 'nan' and phage_field != '':
            phage_hashes = [h.strip() for h in phage_field.split(',')
                          if h.strip() and h.strip() != 'nan' and h.strip() != '']
            if phage_hashes:
                phage_valid = True
        
        if host_valid and phage_valid:
            valid_rows += 1
        else:
            if not host_valid:
                empty_host_rows.append(idx)
            if not phage_valid:
                empty_phage_rows.append(idx)
    
    print(f"Total rows: {total_rows}")
    print(f"Valid rows: {valid_rows}")
    print(f"Rows with empty hosts: {len(empty_host_rows)}")
    print(f"Rows with empty phages: {len(empty_phage_rows)}")
    
    if empty_host_rows:
        print(f"  First few empty host rows: {empty_host_rows[:10]}")
    if empty_phage_rows:
        print(f"  First few empty phage rows: {empty_phage_rows[:10]}")

if __name__ == "__main__":
    diagnose_data()