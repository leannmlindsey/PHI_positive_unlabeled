"""
Check why some rows have empty host_md5_set
"""

import pandas as pd
import numpy as np

# Load the original data
print("Loading original data...")
df_orig = pd.read_csv('data/dedup.labeled_marker_rbp_phageID.tsv', sep='\t', header=None, 
                      names=['wzx_seq', 'wzm_seq', 'rbp_seq', 'phage_id'])

# Load the data with hashes
print("Loading data with hashes...")
df_hash = pd.read_csv('data/dedup.labeled_marker_rbp_phageID_with_hashes.tsv', sep='\t')

print(f"\nOriginal data shape: {df_orig.shape}")
print(f"Hashed data shape: {df_hash.shape}")

# Check for NaN/empty in host_md5_set
print("\n=== Checking host_md5_set ===")
host_nan = df_hash['host_md5_set'].isna()
host_empty = df_hash['host_md5_set'] == ''
print(f"NaN in host_md5_set: {host_nan.sum()}")
print(f"Empty string in host_md5_set: {host_empty.sum()}")

# Check those rows in original data
problem_indices = df_hash[host_nan].index.tolist()
print(f"\nChecking first 5 problematic rows (indices: {problem_indices[:5]})")

for idx in problem_indices[:5]:
    print(f"\n--- Row {idx} ---")
    orig_row = df_orig.iloc[idx]
    hash_row = df_hash.iloc[idx]
    
    # Check wzx
    wzx = str(orig_row['wzx_seq'])
    print(f"wzx_seq length: {len(wzx)}")
    print(f"wzx_seq first 50 chars: '{wzx[:50]}'")
    print(f"wzx contains comma: {(',' in wzx)}")
    
    # Check wzm  
    wzm = str(orig_row['wzm_seq'])
    print(f"wzm_seq length: {len(wzm)}")
    print(f"wzm_seq first 50 chars: '{wzm[:50]}'")
    print(f"wzm contains comma: {(',' in wzm)}")
    
    # Check what extract_sequences would do
    print(f"\nWhat extract_sequences checks:")
    wzx_field = str(orig_row['wzx_seq']).strip()
    wzm_field = str(orig_row['wzm_seq']).strip()
    
    print(f"wzx_field == 'nan': {wzx_field == 'nan'}")
    print(f"wzx_field == 'NaN': {wzx_field == 'NaN'}")
    print(f"wzx_field == '': {wzx_field == ''}")
    
    print(f"wzm_field == 'nan': {wzm_field == 'nan'}")
    print(f"wzm_field == 'NaN': {wzm_field == 'NaN'}")
    print(f"wzm_field == '': {wzm_field == ''}")
    
    # Check if these would pass the checks
    wzx_would_pass = wzx_field and wzx_field != 'nan' and wzx_field != 'NaN' and wzx_field != ''
    wzm_would_pass = wzm_field and wzm_field != 'nan' and wzm_field != 'NaN' and wzm_field != ''
    
    print(f"\nwzx would pass checks: {wzx_would_pass}")
    print(f"wzm would pass checks: {wzm_would_pass}")
    
    # Check the hash value
    print(f"\nhost_md5_set in hashed file: '{hash_row['host_md5_set']}'")
    print(f"phage_md5_set in hashed file: '{hash_row['phage_md5_set'][:50]}...'")

# Check if there's a pattern with the sequences
print("\n=== Checking for patterns ===")
for idx in problem_indices[:20]:
    wzx = str(df_orig.iloc[idx]['wzx_seq'])
    wzm = str(df_orig.iloc[idx]['wzm_seq'])
    
    # Check if they contain only commas
    wzx_only_commas = wzx.replace(',', '').strip() == ''
    wzm_only_commas = wzm.replace(',', '').strip() == ''
    
    if wzx_only_commas or wzm_only_commas:
        print(f"Row {idx}: wzx_only_commas={wzx_only_commas}, wzm_only_commas={wzm_only_commas}")
        print(f"  wzx: '{wzx[:100]}'")
        print(f"  wzm: '{wzm[:100]}'")