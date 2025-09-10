"""
ESM-2 Embedding Generation Script for multiple input formats
Generates embeddings from TSV, FASTA, or JSON files
Adds MD5 hash column/field and saves embeddings in NPZ format
"""

import os
import sys
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import numpy as np
import torch
import hashlib
from tqdm import tqdm
import logging


def setup_logging(name: str = __name__) -> logging.Logger:
    """Set up basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(name)


def calculate_md5(sequence: str) -> str:
    """Calculate MD5 hash of a sequence"""
    return hashlib.md5(sequence.encode()).hexdigest()


def detect_file_format(input_path: str) -> str:
    """
    Detect the format of the input file
    
    Args:
        input_path: Path to input file
        
    Returns:
        File format: 'tsv', 'fasta', or 'json'
    """
    # Check extension first
    ext = Path(input_path).suffix.lower()
    if ext in ['.tsv', '.txt']:
        return 'tsv'
    elif ext in ['.fasta', '.fa', '.faa']:
        return 'fasta'
    elif ext == '.json':
        return 'json'
    
    # If extension is ambiguous, check content
    with open(input_path, 'r') as f:
        first_line = f.readline().strip()
        if first_line.startswith('>'):
            return 'fasta'
        elif first_line.startswith('{') or first_line.startswith('['):
            return 'json'
        else:
            # Assume TSV if has tabs
            if '\t' in first_line:
                return 'tsv'
    
    # Default to TSV
    return 'tsv'


def process_fasta_file(input_path: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Process FASTA file and extract sequences
    
    Args:
        input_path: Path to input FASTA file
        
    Returns:
        Tuple of:
        - sequences_by_id: Dict mapping original ID to sequence
        - sequences_by_md5: Dict mapping MD5 hash to sequence  
        - id_to_md5: Dict mapping original ID to MD5 hash
    """
    logger = setup_logging('process_fasta')
    logger.info(f"Processing FASTA file: {input_path}")
    
    sequences_by_id = {}
    sequences_by_md5 = {}
    id_to_md5 = {}
    
    current_id = None
    current_seq = []
    
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_id and current_seq:
                    sequence = ''.join(current_seq).upper()
                    md5_hash = calculate_md5(sequence)
                    sequences_by_id[current_id] = sequence
                    sequences_by_md5[md5_hash] = sequence
                    id_to_md5[current_id] = md5_hash
                
                # Start new sequence
                current_id = line[1:].split()[0]  # Take first part after '>'
                current_seq = []
            else:
                if line:  # Skip empty lines
                    current_seq.append(line)
        
        # Save last sequence
        if current_id and current_seq:
            sequence = ''.join(current_seq).upper()
            md5_hash = calculate_md5(sequence)
            sequences_by_id[current_id] = sequence
            sequences_by_md5[md5_hash] = sequence
            id_to_md5[current_id] = md5_hash
    
    logger.info(f"Processed {len(sequences_by_id)} sequences")
    logger.info(f"Found {len(sequences_by_md5)} unique sequences (by MD5)")
    
    return sequences_by_id, sequences_by_md5, id_to_md5


def process_json_file(input_path: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Process JSON file with MD5 hash keys mapping to sequences
    
    Args:
        input_path: Path to input JSON file
        
    Returns:
        Tuple of:
        - sequences_by_id: Dict mapping MD5 hash to sequence (same as sequences_by_md5 for JSON)
        - sequences_by_md5: Dict mapping MD5 hash to sequence  
        - id_to_md5: Dict mapping MD5 hash to itself (for consistency)
    """
    logger = setup_logging('process_json')
    logger.info(f"Processing JSON file: {input_path}")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    sequences_by_md5 = {}
    sequences_by_id = {}
    id_to_md5 = {}
    
    # JSON format has MD5 as keys already
    for md5_hash, sequence in data.items():
        sequence = sequence.strip().upper()
        sequences_by_md5[md5_hash] = sequence
        sequences_by_id[md5_hash] = sequence  # Use MD5 as ID for JSON format
        id_to_md5[md5_hash] = md5_hash
    
    logger.info(f"Processed {len(sequences_by_md5)} sequences")
    
    return sequences_by_id, sequences_by_md5, id_to_md5


def process_tsv_file(input_path: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Process TSV file and extract sequences
    
    Args:
        input_path: Path to input TSV file with columns: id, sequence
        
    Returns:
        Tuple of:
        - sequences_by_id: Dict mapping original ID to sequence
        - sequences_by_md5: Dict mapping MD5 hash to sequence  
        - id_to_md5: Dict mapping original ID to MD5 hash
    """
    logger = setup_logging('process_tsv')
    logger.info(f"Processing TSV file: {input_path}")
    
    sequences_by_id = {}
    sequences_by_md5 = {}
    id_to_md5 = {}
    
    with open(input_path, 'r') as infile:
        reader = csv.DictReader(infile, delimiter='\t')
        
        # Validate columns
        if 'id' not in reader.fieldnames or 'sequence' not in reader.fieldnames:
            raise ValueError("TSV must have 'id' and 'sequence' columns")
        
        # Process each row
        for row in reader:
            seq_id = row['id']
            sequence = row['sequence'].strip().upper()
            
            # Calculate MD5
            md5_hash = calculate_md5(sequence)
            
            # Store mappings
            sequences_by_id[seq_id] = sequence
            sequences_by_md5[md5_hash] = sequence
            id_to_md5[seq_id] = md5_hash
    
    logger.info(f"Processed {len(sequences_by_id)} sequences")
    logger.info(f"Found {len(sequences_by_md5)} unique sequences (by MD5)")
    
    return sequences_by_id, sequences_by_md5, id_to_md5


def save_augmented_tsv(sequences_by_id: Dict[str, str], id_to_md5: Dict[str, str], 
                       output_path: str) -> None:
    """
    Save sequences with MD5 hashes to TSV file
    
    Args:
        sequences_by_id: Dict mapping ID to sequence
        id_to_md5: Dict mapping ID to MD5 hash
        output_path: Path to save augmented TSV
    """
    with open(output_path, 'w', newline='') as outfile:
        fieldnames = ['id', 'sequence', 'md5']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        
        for seq_id, sequence in sequences_by_id.items():
            writer.writerow({
                'id': seq_id,
                'sequence': sequence,
                'md5': id_to_md5[seq_id]
            })


def process_input_file(input_path: str, file_format: Optional[str] = None) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Process input file of any supported format
    
    Args:
        input_path: Path to input file
        file_format: Format of file ('tsv', 'fasta', 'json'), auto-detected if None
        
    Returns:
        Tuple of:
        - sequences_by_id: Dict mapping original ID to sequence
        - sequences_by_md5: Dict mapping MD5 hash to sequence  
        - id_to_md5: Dict mapping original ID to MD5 hash
    """
    # Detect format if not specified
    if file_format is None:
        file_format = detect_file_format(input_path)
        logger = setup_logging('process_input')
        logger.info(f"Detected file format: {file_format}")
    
    # Process based on format
    if file_format == 'tsv':
        return process_tsv_file(input_path)
    elif file_format == 'fasta':
        return process_fasta_file(input_path)
    elif file_format == 'json':
        return process_json_file(input_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


class ESM2Embedder:
    """ESM-2 embedder for sequences"""
    
    def __init__(self, model_name: str = 'esm2_t33_650M_UR50D', device: str = 'cuda', batch_size: int = 8):
        """
        Initialize embedder
        
        Args:
            model_name: ESM model name (default: esm2_t33_650M_UR50D)
            device: Device to use
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.logger = setup_logging(self.__class__.__name__)
        
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        
    def load_model(self):
        """Load ESM-2 model"""
        self.logger.info(f"Loading ESM-2 model: {self.model_name}")
        
        # Import ESM
        import esm
        
        # Load model based on name
        if 't33_650M' in self.model_name or self.model_name == 'esm2_t33_650M_UR50D':
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.repr_layer = 33
        elif 't30_150M' in self.model_name:
            self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            self.repr_layer = 30
        elif 't36_3B' in self.model_name:
            self.model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            self.repr_layer = 36
        elif 't48_15B' in self.model_name:
            self.model, self.alphabet = esm.pretrained.esm2_t48_15B_UR50D()
            self.repr_layer = 48
        elif 't12_35M' in self.model_name:
            self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            self.repr_layer = 12
        elif 't6_8M' in self.model_name:
            self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.repr_layer = 6
        else:
            # Default
            self.logger.warning(f"Unknown model name {self.model_name}, using t33_650M")
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.repr_layer = 33
        
        # Set up batch converter
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.args.embed_dim if hasattr(self.model, 'args') else 1280
        
        self.logger.info(f"Model loaded. Device: {self.device}")
        self.logger.info(f"Embedding dimension: {self.embedding_dim}, Using layer: {self.repr_layer}")
    
    def process_single_sequence(self, seq_id: str, sequence: str) -> Optional[np.ndarray]:
        """
        Process a single sequence (for handling OOM on batches)
        
        Args:
            seq_id: Sequence identifier
            sequence: Protein sequence
            
        Returns:
            Embedding array or None if failed
        """
        try:
            # Process as single item batch
            batch_labels, batch_strs, batch_tokens = self.batch_converter([(seq_id, sequence)])
            batch_tokens = batch_tokens.to(self.device)
            
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[self.repr_layer], return_contacts=False)
                batch_embeddings = results["representations"][self.repr_layer]
                
                seq_len = len(sequence)
                # Take mean over sequence length (excluding special tokens)
                seq_embedding = batch_embeddings[0, 1:seq_len+1].mean(0).cpu().numpy()
                
            return seq_embedding
            
        except torch.cuda.OutOfMemoryError:
            self.logger.error(f"OOM for sequence {seq_id} (length: {len(sequence)})")
            torch.cuda.empty_cache()
            return None
        except Exception as e:
            self.logger.error(f"Error processing sequence {seq_id}: {e}")
            return None
    
    def generate_embeddings(self, sequences: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for sequences with OOM handling
        
        Args:
            sequences: Dictionary mapping ID to sequence
            
        Returns:
            Dictionary mapping ID to embedding
        """
        if self.model is None:
            self.load_model()
        
        embeddings = {}
        
        # Sort by sequence length for better batching
        sorted_items = sorted(sequences.items(), key=lambda x: len(x[1]))
        id_list = [item[0] for item in sorted_items]
        
        # Process in batches with adaptive batch size
        current_batch_size = self.batch_size
        i = 0
        pbar = tqdm(total=len(id_list), desc="Generating embeddings")
        
        while i < len(id_list):
            # Get batch
            batch_end = min(i + current_batch_size, len(id_list))
            batch_ids = id_list[i:batch_end]
            batch_data = [(seq_id, sequences[seq_id]) for seq_id in batch_ids]
            
            # Check batch sequence lengths
            max_len = max(len(sequences[sid]) for sid in batch_ids)
            
            # Reduce batch size for very long sequences
            if max_len > 1000 and current_batch_size > 1:
                self.logger.info(f"Long sequences detected (max {max_len}), reducing batch size")
                current_batch_size = max(1, current_batch_size // 2)
                continue
            
            try:
                # Try to process batch
                with torch.no_grad():
                    batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_data)
                    batch_tokens = batch_tokens.to(self.device)
                    
                    results = self.model(batch_tokens, repr_layers=[self.repr_layer], return_contacts=False)
                    batch_embeddings = results["representations"][self.repr_layer]
                    
                    # Extract embeddings
                    for j, seq_id in enumerate(batch_ids):
                        seq_len = len(sequences[seq_id])
                        seq_embedding = batch_embeddings[j, 1:seq_len+1].mean(0).cpu().numpy()
                        embeddings[seq_id] = seq_embedding
                
                # Success - try increasing batch size
                if current_batch_size < self.batch_size:
                    current_batch_size = min(current_batch_size + 1, self.batch_size)
                
                # Update progress
                pbar.update(len(batch_ids))
                i = batch_end
                        
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.logger.warning(f"OOM with batch size {current_batch_size}, reducing...")
                
                if current_batch_size > 1:
                    # Reduce batch size and retry
                    current_batch_size = max(1, current_batch_size // 2)
                else:
                    # Process sequences individually
                    self.logger.info(f"Processing {len(batch_ids)} sequences individually...")
                    for seq_id in batch_ids:
                        seq = sequences[seq_id]
                        embedding = self.process_single_sequence(seq_id, seq)
                        if embedding is not None:
                            embeddings[seq_id] = embedding
                        else:
                            # Create zero embedding for failed sequences
                            embeddings[seq_id] = np.zeros(self.embedding_dim, dtype=np.float32)
                        pbar.update(1)
                    i = batch_end
                    
            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")
                # Process failed batch individually
                for seq_id in batch_ids:
                    if seq_id not in embeddings:
                        embedding = self.process_single_sequence(seq_id, sequences[seq_id])
                        if embedding is not None:
                            embeddings[seq_id] = embedding
                        else:
                            embeddings[seq_id] = np.zeros(self.embedding_dim, dtype=np.float32)
                        pbar.update(1)
                i = batch_end
        
        pbar.close()
        self.logger.info(f"Generated {len(embeddings)} embeddings")
        
        return embeddings
    
    def save_embeddings_npz(self, embeddings: Dict[str, np.ndarray], output_path: str, 
                            use_ids_as_keys: bool = True):
        """
        Save embeddings to NPZ file
        
        Args:
            embeddings: Dictionary mapping ID to embedding
            output_path: Path to save NPZ file
            use_ids_as_keys: If True, save with IDs as keys; if False, save as arrays
        """
        self.logger.info(f"Saving {len(embeddings)} embeddings to {output_path}")
        
        if use_ids_as_keys:
            # Save with IDs as keys in the NPZ
            np.savez_compressed(output_path, **embeddings)
        else:
            # Save as arrays with separate ID array
            ids = list(embeddings.keys())
            embedding_matrix = np.stack([embeddings[id_] for id_ in ids])
            np.savez_compressed(
                output_path,
                ids=np.array(ids),
                embeddings=embedding_matrix,
                embedding_dim=self.embedding_dim
            )
        
        self.logger.info(f"Embeddings saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate ESM-2 embeddings from TSV, FASTA, or JSON files')
    parser.add_argument('input_file', type=str,
                       help='Path to input file (TSV with id/sequence columns, FASTA, or JSON with MD5 keys)')
    parser.add_argument('--output_dir', type=str, default='embeddings_output',
                       help='Directory to save outputs')
    parser.add_argument('--format', type=str, choices=['tsv', 'fasta', 'json'], default=None,
                       help='Input file format (auto-detected if not specified)')
    parser.add_argument('--model_name', type=str, default='esm2_t33_650M_UR50D',
                       help='ESM model name')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--augmented_tsv_name', type=str, default='sequences_with_md5.tsv',
                       help='Name for augmented TSV file with MD5 column')
    parser.add_argument('--embeddings_by_id_name', type=str, default='embeddings_by_id.npz',
                       help='Name for embeddings file with original IDs')
    parser.add_argument('--embeddings_by_md5_name', type=str, default='embeddings_by_md5.npz',
                       help='Name for embeddings file with MD5 hashes')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process input file based on format
    sequences_by_id, sequences_by_md5, id_to_md5 = process_input_file(
        args.input_file, 
        file_format=args.format
    )
    
    # Save augmented TSV with MD5 column
    augmented_tsv_path = os.path.join(args.output_dir, args.augmented_tsv_name)
    save_augmented_tsv(sequences_by_id, id_to_md5, augmented_tsv_path)
    
    print(f"\n=== File Processing Complete ===")
    print(f"Total sequences: {len(sequences_by_id)}")
    print(f"Unique sequences (by MD5): {len(sequences_by_md5)}")
    print(f"Augmented TSV saved to: {augmented_tsv_path}")
    
    # Initialize embedder
    embedder = ESM2Embedder(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Generate embeddings for unique sequences (by MD5)
    print(f"\n=== Generating Embeddings for {len(sequences_by_md5)} Unique Sequences ===")
    embeddings_by_md5 = embedder.generate_embeddings(sequences_by_md5)
    
    # Create embeddings by original ID
    print("\n=== Creating ID-based embeddings mapping ===")
    embeddings_by_id = {}
    for orig_id, md5_hash in id_to_md5.items():
        if md5_hash in embeddings_by_md5:
            embeddings_by_id[orig_id] = embeddings_by_md5[md5_hash]
        else:
            print(f"Warning: No embedding found for {orig_id} (MD5: {md5_hash})")
    
    # Save embeddings with original IDs as keys
    id_output_path = os.path.join(args.output_dir, args.embeddings_by_id_name)
    embedder.save_embeddings_npz(embeddings_by_id, id_output_path, use_ids_as_keys=True)
    print(f"Saved embeddings by ID to: {id_output_path}")
    
    # Save embeddings with MD5 hashes as keys
    md5_output_path = os.path.join(args.output_dir, args.embeddings_by_md5_name)
    embedder.save_embeddings_npz(embeddings_by_md5, md5_output_path, use_ids_as_keys=True)
    print(f"Saved embeddings by MD5 to: {md5_output_path}")
    
    print("\n=== Embedding Generation Complete ===")
    print(f"Output directory: {args.output_dir}")
    print(f"Files created:")
    print(f"  - {args.augmented_tsv_name} (TSV with MD5 column)")
    print(f"  - {args.embeddings_by_id_name} (embeddings keyed by original ID)")
    print(f"  - {args.embeddings_by_md5_name} (embeddings keyed by MD5 hash)")
    
    # Print sample of how to load the files
    print("\n=== How to Load the Embeddings ===")
    print("```python")
    print("import numpy as np")
    print("")
    print("# Load embeddings by original ID")
    print(f"data_by_id = np.load('{args.embeddings_by_id_name}')")
    print("embedding_for_id = data_by_id['your_sequence_id']  # Get embedding for specific ID")
    print("")
    print("# Load embeddings by MD5")
    print(f"data_by_md5 = np.load('{args.embeddings_by_md5_name}')")
    print("embedding_for_md5 = data_by_md5['md5_hash_here']  # Get embedding for specific MD5")
    print("")
    print("# List all available IDs/MD5s")
    print("all_ids = list(data_by_id.files)")
    print("all_md5s = list(data_by_md5.files)")
    print("```")


if __name__ == "__main__":
    main()