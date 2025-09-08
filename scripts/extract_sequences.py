"""
Extract and deduplicate protein sequences from the dataset
Creates separate FASTA files for host and phage proteins
"""

import os
import sys
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Set, Tuple
import pandas as pd
from tqdm import tqdm
import json


def extract_and_deduplicate_sequences(data_path: str, output_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Extract all unique protein sequences from the dataset
    
    Args:
        data_path: Path to the TSV data file
        output_dir: Directory to save output files
        
    Returns:
        Tuple of (host_sequences, phage_sequences) dictionaries
    """
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, sep='\t')
    print(f"Loaded {len(df)} interactions")
    
    host_sequences = {}  # hash -> sequence
    phage_sequences = {}  # hash -> sequence
    
    # Track problematic rows
    problematic_rows = []
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing sequences"):
        # Process marker (host) sequences
        marker_seqs_raw = str(row['marker_gene_seq'])
        marker_hashes_raw = str(row['marker_md5'])
        
        # Handle multiple sequences (some entries have comma-separated sequences)
        if ',' in marker_seqs_raw:
            # This means multiple sequences are concatenated with commas
            # But we only have one hash - this is the problematic case
            # For now, treat the entire string as one sequence
            seq = marker_seqs_raw.replace(',', '')  # Remove commas
            computed_hash = hashlib.md5(seq.encode()).hexdigest()
            
            # Use the provided hash if it matches, otherwise compute new one
            if ',' not in marker_hashes_raw:
                provided_hash = marker_hashes_raw.strip()
                if hashlib.md5(marker_seqs_raw.encode()).hexdigest() == provided_hash:
                    # The hash was computed on the comma-separated string
                    host_sequences[provided_hash] = marker_seqs_raw
                else:
                    # Use the concatenated sequence
                    host_sequences[computed_hash] = seq
                    problematic_rows.append((idx, 'marker', 'comma-separated sequences'))
            else:
                # Multiple hashes provided
                marker_seqs = marker_seqs_raw.split(',')
                marker_hashes = marker_hashes_raw.split(',')
                
                if len(marker_seqs) == len(marker_hashes):
                    for seq, hash_val in zip(marker_seqs, marker_hashes):
                        seq = seq.strip()
                        hash_val = hash_val.strip()
                        if seq and hash_val:
                            host_sequences[hash_val] = seq
                else:
                    # Mismatch in counts - compute hash for concatenated sequence
                    host_sequences[computed_hash] = seq
                    problematic_rows.append((idx, 'marker', f'{len(marker_seqs)} seqs, {len(marker_hashes)} hashes'))
        else:
            # Single sequence
            seq = marker_seqs_raw.strip()
            hash_val = marker_hashes_raw.strip()
            if seq and hash_val:
                host_sequences[hash_val] = seq
        
        # Process RBP (phage) sequences
        rbp_seqs_raw = str(row['rbp_seq'])
        rbp_hashes_raw = str(row['rbp_md5'])
        
        # Handle multiple sequences
        if ',' in rbp_seqs_raw:
            # Multiple sequences concatenated with commas
            if ',' not in rbp_hashes_raw:
                # Only one hash for multiple sequences - problematic
                seq = rbp_seqs_raw.replace(',', '')  # Remove commas
                computed_hash = hashlib.md5(seq.encode()).hexdigest()
                
                provided_hash = rbp_hashes_raw.strip()
                if hashlib.md5(rbp_seqs_raw.encode()).hexdigest() == provided_hash:
                    # The hash was computed on the comma-separated string
                    phage_sequences[provided_hash] = rbp_seqs_raw
                else:
                    # Use the concatenated sequence
                    phage_sequences[computed_hash] = seq
                    problematic_rows.append((idx, 'rbp', 'comma-separated sequences'))
            else:
                # Multiple hashes provided
                rbp_seqs = rbp_seqs_raw.split(',')
                rbp_hashes = rbp_hashes_raw.split(',')
                
                if len(rbp_seqs) == len(rbp_hashes):
                    for seq, hash_val in zip(rbp_seqs, rbp_hashes):
                        seq = seq.strip()
                        hash_val = hash_val.strip()
                        if seq and hash_val:
                            phage_sequences[hash_val] = seq
                else:
                    # Mismatch in counts - compute hash for concatenated sequence
                    seq = rbp_seqs_raw.replace(',', '')
                    computed_hash = hashlib.md5(seq.encode()).hexdigest()
                    phage_sequences[computed_hash] = seq
                    problematic_rows.append((idx, 'rbp', f'{len(rbp_seqs)} seqs, {len(rbp_hashes)} hashes'))
        else:
            # Single sequence
            seq = rbp_seqs_raw.strip()
            hash_val = rbp_hashes_raw.strip()
            if seq and hash_val:
                phage_sequences[hash_val] = seq
    
    print(f"\nFound {len(host_sequences)} unique host sequences")
    print(f"Found {len(phage_sequences)} unique phage sequences")
    print(f"Found {len(problematic_rows)} problematic rows")
    
    if problematic_rows:
        print("\nFirst 10 problematic rows:")
        for row_idx, seq_type, issue in problematic_rows[:10]:
            print(f"  Row {row_idx} ({seq_type}): {issue}")
    
    # Verify hashes
    print("\nVerifying host sequence hashes...")
    host_mismatches = verify_hashes(host_sequences)
    
    print("Verifying phage sequence hashes...")
    phage_mismatches = verify_hashes(phage_sequences)
    
    return host_sequences, phage_sequences


def verify_hashes(sequences: Dict[str, str]) -> int:
    """
    Verify MD5 hashes match the sequences
    
    Args:
        sequences: Dictionary mapping hash to sequence
        
    Returns:
        Number of mismatches
    """
    mismatches = 0
    for hash_val, seq in sequences.items():
        # Check if sequence contains commas (concatenated sequences)
        if ',' in seq:
            # For comma-separated sequences, check if hash matches the full string
            computed_hash = hashlib.md5(seq.encode()).hexdigest()
            if computed_hash != hash_val:
                # Also try without commas
                seq_no_comma = seq.replace(',', '')
                computed_hash_no_comma = hashlib.md5(seq_no_comma.encode()).hexdigest()
                if computed_hash_no_comma != hash_val:
                    mismatches += 1
        else:
            computed_hash = hashlib.md5(seq.encode()).hexdigest()
            if computed_hash != hash_val:
                mismatches += 1
    
    if mismatches > 0:
        print(f"  Warning: {mismatches} hash mismatches found")
    else:
        print(f"  All {len(sequences)} hashes verified successfully")
    
    return mismatches


def save_sequences_fasta(sequences: Dict[str, str], output_path: str, seq_type: str):
    """
    Save sequences to FASTA file
    
    Args:
        sequences: Dictionary mapping hash to sequence
        output_path: Path to save FASTA file
        seq_type: Type of sequences (host/phage)
    """
    with open(output_path, 'w') as f:
        for hash_val, seq in sequences.items():
            # Clean sequence - remove commas if present
            clean_seq = seq.replace(',', '') if ',' in seq else seq
            f.write(f">{seq_type}_{hash_val}\n")
            # Write sequence in 80-character lines
            for i in range(0, len(clean_seq), 80):
                f.write(clean_seq[i:i+80] + '\n')
    
    print(f"Saved {len(sequences)} {seq_type} sequences to {output_path}")


def save_sequences_json(sequences: Dict[str, str], output_path: str):
    """
    Save sequences to JSON file
    
    Args:
        sequences: Dictionary mapping hash to sequence
        output_path: Path to save JSON file
    """
    # Clean sequences - remove commas if present
    clean_sequences = {}
    for hash_val, seq in sequences.items():
        clean_seq = seq.replace(',', '') if ',' in seq else seq
        clean_sequences[hash_val] = clean_seq
    
    with open(output_path, 'w') as f:
        json.dump(clean_sequences, f, indent=2)
    
    print(f"Saved {len(sequences)} sequences to {output_path}")


def save_mapping_file(host_sequences: Dict[str, str], phage_sequences: Dict[str, str], 
                     data_path: str, output_path: str):
    """
    Save a mapping file that tracks which hashes belong to each interaction
    
    Args:
        host_sequences: Dictionary of host sequences
        phage_sequences: Dictionary of phage sequences
        data_path: Path to original data file
        output_path: Path to save mapping file
    """
    df = pd.read_csv(data_path, sep='\t')
    
    mappings = []
    for idx, row in df.iterrows():
        # Get original hashes
        marker_hashes = str(row['marker_md5']).strip()
        rbp_hashes = str(row['rbp_md5']).strip()
        
        # Handle cases where we might have recomputed hashes
        if ',' in str(row['marker_gene_seq']) and ',' not in marker_hashes:
            # Recompute hash for concatenated sequence
            seq = str(row['marker_gene_seq']).replace(',', '')
            marker_hashes = hashlib.md5(seq.encode()).hexdigest()
        
        if ',' in str(row['rbp_seq']) and ',' not in rbp_hashes:
            # Recompute hash for concatenated sequence
            seq = str(row['rbp_seq']).replace(',', '')
            rbp_hashes = hashlib.md5(seq.encode()).hexdigest()
        
        mappings.append({
            'index': idx,
            'host_genome': row['host_genome'],
            'phage_genome': row['phage_genome'],
            'marker_hashes': marker_hashes,
            'rbp_hashes': rbp_hashes,
            'label': row['label']
        })
    
    mapping_df = pd.DataFrame(mappings)
    mapping_df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved interaction mappings to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract and deduplicate protein sequences")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to the TSV data file")
    parser.add_argument("--output_dir", type=str, default="data/sequences",
                       help="Output directory for sequence files")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract sequences
    host_sequences, phage_sequences = extract_and_deduplicate_sequences(
        args.data_path, args.output_dir
    )
    
    # Save sequences in multiple formats
    print("\nSaving sequence files...")
    
    # FASTA format
    save_sequences_fasta(host_sequences, output_dir / "host_sequences.fasta", "host")
    save_sequences_fasta(phage_sequences, output_dir / "phage_sequences.fasta", "phage")
    
    # JSON format (for easy loading)
    save_sequences_json(host_sequences, output_dir / "host_sequences.json")
    save_sequences_json(phage_sequences, output_dir / "phage_sequences.json")
    
    # Save mapping file
    save_mapping_file(host_sequences, phage_sequences, args.data_path, 
                     output_dir / "sequence_mappings.tsv")
    
    # Save statistics
    stats = {
        'n_host_sequences': len(host_sequences),
        'n_phage_sequences': len(phage_sequences),
        'total_sequences': len(host_sequences) + len(phage_sequences)
    }
    
    with open(output_dir / "sequence_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\nSequence extraction completed!")
    print(f"Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()