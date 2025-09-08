"""
Simple ESM-2 Embedding Generation Script
Generates embeddings from cleaned sequence files
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import h5py
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


class ESM2Embedder:
    """Simple ESM-2 embedder for cleaned sequences"""
    
    def __init__(self, model_path: str, device: str = 'cuda', batch_size: int = 8):
        """
        Initialize embedder
        
        Args:
            model_path: Path to ESM-2 checkpoint file
            device: Device to use
            batch_size: Batch size for processing
        """
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.logger = setup_logging(self.__class__.__name__)
        
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        
    def load_model(self):
        """Load the ESM-2 model"""
        self.logger.info(f"Loading ESM-2 model from {self.model_path}")
        
        try:
            import esm
            
            # Monkey-patch torch.load for PyTorch 2.6+ compatibility
            import torch
            original_load = torch.load
            
            def patched_load(f, *args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(f, *args, **kwargs)
            
            torch.load = patched_load
            
            try:
                # Load model
                self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(self.model_path)
                self.batch_converter = self.alphabet.get_batch_converter()
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # Get embedding dimension
                if hasattr(self.model, 'embed_dim'):
                    self.embed_dim = self.model.embed_dim
                elif hasattr(self.model, 'args'):
                    self.embed_dim = self.model.args.embed_dim
                else:
                    self.embed_dim = 1280  # Default for t33_650M
                    
                self.logger.info(f"Model loaded successfully. Embedding dimension: {self.embed_dim}")
                
            finally:
                # Restore original torch.load
                torch.load = original_load
                
        except ImportError:
            self.logger.error("fair-esm not installed. Install with: pip install fair-esm")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def embed_batch(self, sequences: list) -> np.ndarray:
        """
        Generate embeddings for a batch of sequences with adaptive batching
        
        Args:
            sequences: List of (label, sequence) tuples
            
        Returns:
            Array of embeddings
        """
        # Sort sequences by length for better memory estimation
        sequences = sorted(sequences, key=lambda x: len(x[1]))
        max_seq_len = len(sequences[-1][1])
        
        # Very conservative memory estimation for safety
        # Each sequence needs roughly max_seq_len^2 * 4 bytes per attention layer
        safe_batch_size = 1
        if max_seq_len < 500:
            safe_batch_size = 8
        elif max_seq_len < 1000:
            safe_batch_size = 4
        elif max_seq_len < 2000:
            safe_batch_size = 2
        elif max_seq_len < 5000:
            safe_batch_size = 1
        
        # Process in smaller batches if needed
        if len(sequences) > safe_batch_size:
            self.logger.info(f"Processing {len(sequences)} sequences in batches of {safe_batch_size} (max_len={max_seq_len})")
            all_embeddings = []
            
            for i in range(0, len(sequences), safe_batch_size):
                batch = sequences[i:i+safe_batch_size]
                try:
                    # Recursively process smaller batch
                    batch_emb = self.embed_batch(batch)
                    all_embeddings.extend(batch_emb)
                except torch.cuda.OutOfMemoryError:
                    # If even smaller batch fails, process individually
                    self.logger.warning(f"Batch of {len(batch)} failed, processing individually")
                    for seq in batch:
                        emb = self.embed_batch([seq])
                        all_embeddings.extend(emb)
                
                # Clear cache between batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            return np.array(all_embeddings)
        
        # Try to process the batch
        try:
            batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
            batch_tokens = batch_tokens.to(self.device)
            
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_embeddings = results["representations"][33]
            
            # Extract embeddings and apply mean pooling
            embeddings = []
            for i, (_, seq) in enumerate(sequences):
                seq_len = len(seq)
                # Remove BOS/EOS tokens and mean pool
                seq_embeddings = token_embeddings[i, 1:seq_len+1]
                pooled = seq_embeddings.mean(dim=0)
                embeddings.append(pooled.cpu().numpy())
            
            return np.array(embeddings)
            
        except torch.cuda.OutOfMemoryError as e:
            # If batch processing fails, fall back to individual processing
            self.logger.warning(f"Batch processing failed with OOM, processing {len(sequences)} sequences individually")
            embeddings = []
            
            for label, seq in sequences:
                try:
                    # Clear cache before each sequence
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Process single sequence
                    single_batch = [(label, seq)]
                    batch_labels, batch_strs, batch_tokens = self.batch_converter(single_batch)
                    batch_tokens = batch_tokens.to(self.device)
                    
                    with torch.no_grad():
                        results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
                        token_embeddings = results["representations"][33]
                    
                    # Extract embedding
                    seq_len = len(seq)
                    seq_embeddings = token_embeddings[0, 1:seq_len+1]
                    pooled = seq_embeddings.mean(dim=0)
                    embeddings.append(pooled.cpu().numpy())
                    
                except Exception as e2:
                    self.logger.error(f"Failed to process sequence of length {len(seq)}: {e2}")
                    # Return zero embedding for failed sequences
                    embeddings.append(np.zeros(self.embed_dim))
                    
            return np.array(embeddings)
    
    def process_sequences(self, sequences: Dict[str, str], output_path: str, 
                         desc: str = "Processing", resume: bool = True):
        """
        Process all sequences and save embeddings
        
        Args:
            sequences: Dictionary mapping hash/id to sequence
            output_path: Path to save HDF5 file
            desc: Description for progress bar
            resume: Whether to resume from existing file
        """
        if self.model is None:
            self.load_model()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check for existing embeddings
        processed_hashes = set()
        if resume and output_path.exists():
            with h5py.File(output_path, 'r') as f:
                if 'hashes' in f:
                    processed_hashes = set(f['hashes'][:].astype(str))
                    self.logger.info(f"Found {len(processed_hashes)} existing embeddings")
        
        # Filter sequences to process
        to_process = {h: s for h, s in sequences.items() if h not in processed_hashes}
        
        if not to_process:
            self.logger.info("All sequences already processed")
            return
        
        self.logger.info(f"Processing {len(to_process)} sequences")
        
        # Sort sequences by length for efficient batching
        # This ensures similar-length sequences are processed together
        seq_with_hash = [(hash_val, seq, len(seq)) for hash_val, seq in to_process.items()]
        seq_with_hash.sort(key=lambda x: x[2])  # Sort by sequence length
        
        # Group sequences by length buckets for optimal memory usage
        length_buckets = []
        current_bucket = []
        current_max_len = 0
        
        for hash_val, seq, seq_len in seq_with_hash:
            # Start new bucket if length difference is too large
            if current_bucket and (seq_len > current_max_len * 1.5 or seq_len > current_max_len + 500):
                length_buckets.append(current_bucket)
                current_bucket = []
                current_max_len = 0
            
            current_bucket.append((hash_val, seq))
            current_max_len = max(current_max_len, seq_len)
        
        if current_bucket:
            length_buckets.append(current_bucket)
        
        self.logger.info(f"Organized sequences into {len(length_buckets)} length-based buckets")
        
        # Flatten buckets back for processing but maintain length ordering
        sorted_data = []
        for bucket in length_buckets:
            sorted_data.extend(bucket)
        
        hash_list = [item[0] for item in sorted_data]
        seq_list = [item[1] for item in sorted_data]
        
        # Open HDF5 file
        mode = 'a' if resume and output_path.exists() else 'w'
        with h5py.File(output_path, mode) as f:
            # Create or get datasets
            if 'embeddings' not in f:
                embeddings_ds = f.create_dataset(
                    'embeddings',
                    shape=(0, self.embed_dim),
                    maxshape=(None, self.embed_dim),
                    dtype='float32',
                    chunks=(100, self.embed_dim)
                )
                hashes_ds = f.create_dataset(
                    'hashes',
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype()
                )
            else:
                embeddings_ds = f['embeddings']
                hashes_ds = f['hashes']
            
            # Process in batches with dynamic batch sizing based on sequence lengths
            processed = 0
            pbar = tqdm(total=len(hash_list), desc=desc)
            
            i = 0
            while i < len(hash_list):
                # Determine adaptive batch size based on sequence lengths
                max_len_in_batch = len(seq_list[i])
                
                # Adjust batch size based on max length
                if max_len_in_batch < 500:
                    adaptive_batch_size = min(self.batch_size * 2, 16)  # Can handle more short sequences
                elif max_len_in_batch < 1000:
                    adaptive_batch_size = self.batch_size
                elif max_len_in_batch < 2000:
                    adaptive_batch_size = max(self.batch_size // 2, 2)
                else:
                    adaptive_batch_size = 1  # Process very long sequences individually
                
                # Get batch
                end_idx = min(i + adaptive_batch_size, len(hash_list))
                batch_hashes = hash_list[i:end_idx]
                batch_seqs = seq_list[i:end_idx]
                
                # Log batch info for very long sequences
                max_batch_len = max(len(seq) for seq in batch_seqs)
                if max_batch_len > 3000:
                    self.logger.info(f"Processing batch with {len(batch_seqs)} sequences, max length: {max_batch_len}")
                
                # Prepare sequences with labels
                labeled_seqs = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]
                
                try:
                    # Generate embeddings
                    batch_embeddings = self.embed_batch(labeled_seqs)
                    
                    # Resize datasets
                    current_size = embeddings_ds.shape[0]
                    new_size = current_size + len(batch_hashes)
                    
                    embeddings_ds.resize((new_size, self.embed_dim))
                    hashes_ds.resize((new_size,))
                    
                    # Add data
                    embeddings_ds[current_size:new_size] = batch_embeddings
                    hashes_ds[current_size:new_size] = batch_hashes
                    
                    # Flush periodically
                    if processed % 100 == 0:
                        f.flush()
                    
                    # Update progress
                    pbar.update(len(batch_hashes))
                    processed += len(batch_hashes)
                    
                    # Move to next batch
                    i = end_idx
                        
                except Exception as e:
                    self.logger.error(f"Error processing batch at index {i}: {e}")
                    raise
                finally:
                    # Clear GPU cache after each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            pbar.close()
        
        self.logger.info(f"Embeddings saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ESM-2 embeddings from sequence files")
    parser.add_argument("--host_sequences", type=str, required=True,
                       help="Path to host sequences JSON file")
    parser.add_argument("--phage_sequences", type=str, required=True,
                       help="Path to phage sequences JSON file")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to ESM-2 checkpoint file")
    parser.add_argument("--output_dir", type=str, default="data/embeddings",
                       help="Output directory for embeddings")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--no_resume", action="store_true",
                       help="Don't resume from existing embeddings")
    
    args = parser.parse_args()
    
    # Load sequences
    print(f"Loading host sequences from {args.host_sequences}")
    with open(args.host_sequences, 'r') as f:
        host_sequences = json.load(f)
    print(f"Loaded {len(host_sequences)} host sequences")
    
    print(f"Loading phage sequences from {args.phage_sequences}")
    with open(args.phage_sequences, 'r') as f:
        phage_sequences = json.load(f)
    print(f"Loaded {len(phage_sequences)} phage sequences")
    
    # Create embedder
    embedder = ESM2Embedder(
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Process sequences
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process host sequences
    host_output = output_dir / "host_embeddings.h5"
    embedder.process_sequences(
        host_sequences, 
        host_output,
        desc="Embedding host sequences",
        resume=not args.no_resume
    )
    
    # Process phage sequences
    phage_output = output_dir / "phage_embeddings.h5"
    embedder.process_sequences(
        phage_sequences,
        phage_output,
        desc="Embedding phage sequences",
        resume=not args.no_resume
    )
    
    print("\nEmbedding generation completed!")
    print(f"Host embeddings: {host_output}")
    print(f"Phage embeddings: {phage_output}")


if __name__ == "__main__":
    main()