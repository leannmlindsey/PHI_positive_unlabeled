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
        Generate embeddings for a batch of sequences
        
        Args:
            sequences: List of (label, sequence) tuples
            
        Returns:
            Array of embeddings
        """
        # Check if batch would be too large
        max_seq_len = max(len(seq) for _, seq in sequences)
        estimated_memory_gb = (len(sequences) * max_seq_len * max_seq_len * 4) / (1024**3)
        
        # If estimated memory > 40GB, process sequences individually
        if estimated_memory_gb > 40:
            self.logger.warning(f"Batch too large ({estimated_memory_gb:.1f}GB), processing individually")
            embeddings = []
            for label, seq in sequences:
                try:
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
                    
                    # Clear cache after each sequence
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    self.logger.error(f"Failed to process sequence of length {len(seq)}: {e}")
                    # Return zero embedding for failed sequences
                    embeddings.append(np.zeros(self.embed_dim))
                    
            return np.array(embeddings)
        
        # Normal batch processing for smaller batches
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
        
        # Prepare batches
        hash_list = list(to_process.keys())
        seq_list = list(to_process.values())
        
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
            
            # Process in batches
            for i in tqdm(range(0, len(hash_list), self.batch_size), 
                         desc=desc, 
                         total=(len(hash_list) + self.batch_size - 1) // self.batch_size):
                
                batch_hashes = hash_list[i:i + self.batch_size]
                batch_seqs = seq_list[i:i + self.batch_size]
                
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
                    if (i + self.batch_size) % (self.batch_size * 10) == 0:
                        f.flush()
                        
                except Exception as e:
                    self.logger.error(f"Error processing batch at index {i}: {e}")
                    raise
        
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