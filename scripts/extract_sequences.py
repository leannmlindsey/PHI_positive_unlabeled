"""
Extract, hash, and deduplicate protein sequences from the new data format
Creates separate files for host (wzx, wzm) and phage (rbp) proteins
Adds MD5 hash columns to the original data file
"""

import os
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Set, List, Tuple
import pandas as pd
from tqdm import tqdm
import json
import numpy as np


def compute_md5(sequence: str) -> str:
    """Compute MD5 hash for a protein sequence"""
    return hashlib.md5(sequence.strip().upper().encode()).hexdigest()


def extract_all_sequences(data_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Extract all unique protein sequences from the dataset
    
    Args:
        data_path: Path to the TSV data file (no header)
        
    Returns:
        Tuple of (host_sequences, phage_sequences) dictionaries
        Each dict maps MD5 hash -> sequence
    """
    print(f"Loading data from {data_path}")
    
    # Load data without header
    columns = ['wzx_seq', 'wzm_seq', 'rbp_seq', 'phage_id']
    df = pd.read_csv(data_path, sep='\t', header=None, names=columns)
    print(f"Loaded {len(df)} interactions")
    
    host_sequences = {}  # MD5 hash -> sequence
    phage_sequences = {}  # MD5 hash -> sequence
    
    # Statistics
    stats = {
        'total_wzx': 0,
        'unique_wzx': 0,
        'total_wzm': 0,
        'unique_wzm': 0,
        'total_rbp': 0,
        'unique_rbp': 0,
        'empty_wzx': 0,
        'empty_wzm': 0,
        'multi_rbp_rows': 0
    }
    
    print("\nExtracting and hashing sequences...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Process wzx sequences (can be comma-separated)
        wzx_field = str(row['wzx_seq']).strip()
        if wzx_field and wzx_field != 'nan' and wzx_field != 'NaN' and wzx_field != '':
            # Split by comma if multiple sequences
            if ',' in wzx_field:
                wzx_seqs = [seq.strip() for seq in wzx_field.split(',') if seq.strip()]
            else:
                wzx_seqs = [wzx_field]
            
            for wzx_seq in wzx_seqs:
                if wzx_seq and wzx_seq != 'nan':
                    wzx_hash = compute_md5(wzx_seq)
                    host_sequences[wzx_hash] = wzx_seq
                    stats['total_wzx'] += 1
        else:
            stats['empty_wzx'] += 1
            
        # Process wzm sequences (can be comma-separated)
        wzm_field = str(row['wzm_seq']).strip()
        if wzm_field and wzm_field != 'nan' and wzm_field != 'NaN' and wzm_field != '':
            # Split by comma if multiple sequences
            if ',' in wzm_field:
                wzm_seqs = [seq.strip() for seq in wzm_field.split(',') if seq.strip()]
            else:
                wzm_seqs = [wzm_field]
            
            for wzm_seq in wzm_seqs:
                if wzm_seq and wzm_seq != 'nan':
                    wzm_hash = compute_md5(wzm_seq)
                    host_sequences[wzm_hash] = wzm_seq
                    stats['total_wzm'] += 1
        else:
            stats['empty_wzm'] += 1
            
        # Process RBP sequences (can be comma-separated)
        rbp_field = str(row['rbp_seq']).strip()
        if ',' in rbp_field:
            stats['multi_rbp_rows'] += 1
            rbp_seqs = [seq.strip() for seq in rbp_field.split(',') if seq.strip()]
        else:
            rbp_seqs = [rbp_field] if rbp_field and rbp_field != 'nan' else []
            
        for rbp_seq in rbp_seqs:
            if rbp_seq and rbp_seq != 'nan' and rbp_seq != 'NaN' and rbp_seq != '':
                rbp_hash = compute_md5(rbp_seq)
                phage_sequences[rbp_hash] = rbp_seq
                stats['total_rbp'] += 1
    
    # Count unique sequences
    unique_wzx = set()
    unique_wzm = set()
    for hash_val, seq in host_sequences.items():
        # Rough heuristic: wzx sequences are typically longer than wzm
        if len(seq) > 300:  # This threshold may need adjustment
            unique_wzx.add(hash_val)
        else:
            unique_wzm.add(hash_val)
    
    stats['unique_wzx'] = len(unique_wzx)
    stats['unique_wzm'] = len(unique_wzm)
    stats['unique_rbp'] = len(phage_sequences)
    
    # Print statistics
    print("\n=== Extraction Statistics ===")
    print(f"Total wzx sequences: {stats['total_wzx']}")
    print(f"Total wzm sequences: {stats['total_wzm']}")
    print(f"Total RBP sequences: {stats['total_rbp']}")
    print(f"Unique host sequences: {len(host_sequences)}")
    print(f"  Estimated unique wzx: {stats['unique_wzx']}")
    print(f"  Estimated unique wzm: {stats['unique_wzm']}")
    print(f"Unique phage sequences: {stats['unique_rbp']}")
    print(f"Rows with multiple RBPs: {stats['multi_rbp_rows']}")
    print(f"Empty wzx fields: {stats['empty_wzx']}")
    print(f"Empty wzm fields: {stats['empty_wzm']}")
    
    return host_sequences, phage_sequences


def add_hash_columns_to_data(data_path: str, output_path: str) -> pd.DataFrame:
    """
    Add MD5 hash columns to the original data file
    Creates two new columns: host_md5_set and phage_md5_set
    
    Args:
        data_path: Path to original TSV file
        output_path: Path to save enhanced TSV file
        
    Returns:
        DataFrame with added hash columns
    """
    print(f"\nAdding hash columns to data file...")
    
    # Load data
    columns = ['wzx_seq', 'wzm_seq', 'rbp_seq', 'phage_id']
    df = pd.read_csv(data_path, sep='\t', header=None, names=columns)
    
    # Create hash columns
    host_hashes = []
    phage_hashes = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing hashes"):
        # Host hashes (wzx and wzm - can be comma-separated)
        host_set = []
        
        # Process wzx sequences
        wzx_field = str(row['wzx_seq']).strip()
        if wzx_field and wzx_field != 'nan' and wzx_field != 'NaN' and wzx_field != '':
            if ',' in wzx_field:
                wzx_seqs = [seq.strip() for seq in wzx_field.split(',') if seq.strip()]
            else:
                wzx_seqs = [wzx_field]
            
            for wzx_seq in wzx_seqs:
                if wzx_seq and wzx_seq != 'nan':
                    host_set.append(compute_md5(wzx_seq))
        
        # Process wzm sequences
        wzm_field = str(row['wzm_seq']).strip()
        if wzm_field and wzm_field != 'nan' and wzm_field != 'NaN' and wzm_field != '':
            if ',' in wzm_field:
                wzm_seqs = [seq.strip() for seq in wzm_field.split(',') if seq.strip()]
            else:
                wzm_seqs = [wzm_field]
            
            for wzm_seq in wzm_seqs:
                if wzm_seq and wzm_seq != 'nan':
                    host_set.append(compute_md5(wzm_seq))
            
        host_hashes.append(','.join(host_set))
        
        # Phage hashes (RBPs)
        phage_set = []
        rbp_field = str(row['rbp_seq']).strip()
        
        if ',' in rbp_field:
            rbp_seqs = [seq.strip() for seq in rbp_field.split(',') if seq.strip()]
        else:
            rbp_seqs = [rbp_field] if rbp_field and rbp_field != 'nan' else []
            
        for rbp_seq in rbp_seqs:
            if rbp_seq and rbp_seq != 'nan' and rbp_seq != 'NaN' and rbp_seq != '':
                phage_set.append(compute_md5(rbp_seq))
                
        phage_hashes.append(','.join(phage_set))
    
    # Add columns to dataframe
    df['host_md5_set'] = host_hashes
    df['phage_md5_set'] = phage_hashes
    
    # Save enhanced dataframe
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved enhanced data to {output_path}")
    
    # Print sample
    print("\nSample of enhanced data:")
    print(df[['phage_id', 'host_md5_set', 'phage_md5_set']].head())
    
    return df


def save_sequences(sequences: Dict[str, str], output_path: str, file_type: str):
    """
    Save sequences to JSON and FASTA files
    
    Args:
        sequences: Dictionary mapping MD5 hash to sequence
        output_path: Base path for output files (without extension)
        file_type: Type of sequences ('host' or 'phage')
    """
    # Save as JSON
    json_path = f"{output_path}_{file_type}_sequences.json"
    with open(json_path, 'w') as f:
        json.dump(sequences, f, indent=2)
    print(f"Saved {len(sequences)} {file_type} sequences to {json_path}")
    
    # Save as FASTA
    fasta_path = f"{output_path}_{file_type}_sequences.fasta"
    with open(fasta_path, 'w') as f:
        for hash_id, sequence in sequences.items():
            f.write(f">{hash_id}\n{sequence}\n")
    print(f"Saved {len(sequences)} {file_type} sequences to {fasta_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract and process sequences with MD5 hashing')
    parser.add_argument('--data_path', type=str, 
                       default='data/dedup.labeled_marker_rbp_phageID.tsv',
                       help='Path to input TSV file')
    parser.add_argument('--output_dir', type=str, 
                       default='data/sequences',
                       help='Directory to save output files')
    parser.add_argument('--enhanced_data_path', type=str,
                       default='data/dedup.labeled_marker_rbp_phageID_with_hashes.tsv',
                       help='Path to save data file with hash columns')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract and deduplicate sequences
    host_sequences, phage_sequences = extract_all_sequences(args.data_path)
    
    # Save sequences
    output_base = os.path.join(args.output_dir, 'dedup')
    save_sequences(host_sequences, output_base, 'host')
    save_sequences(phage_sequences, output_base, 'phage')
    
    # Add hash columns to data file
    enhanced_df = add_hash_columns_to_data(args.data_path, args.enhanced_data_path)
    
    # Save a mapping file for reference
    mapping_path = os.path.join(args.output_dir, 'sequence_stats.txt')
    with open(mapping_path, 'w') as f:
        f.write(f"Total unique host sequences: {len(host_sequences)}\n")
        f.write(f"Total unique phage sequences: {len(phage_sequences)}\n")
        f.write(f"Total interactions: {len(enhanced_df)}\n")
        f.write(f"\nHost sequences with empty MD5 sets: {(enhanced_df['host_md5_set'] == '').sum()}\n")
        f.write(f"Phage sequences with empty MD5 sets: {(enhanced_df['phage_md5_set'] == '').sum()}\n")
    
    print(f"\nSaved statistics to {mapping_path}")
    print("\n=== Processing Complete ===")
    print(f"Enhanced data file: {args.enhanced_data_path}")
    print(f"Sequence files: {args.output_dir}")


if __name__ == "__main__":
    main()