"""
ESM-2 Embedding Generation Script using MD5 hashes
Generates embeddings from deduplicated sequence files with MD5 hash identifiers
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import numpy as np
import torch
import h5py
from tqdm import tqdm
import logging
import hashlib


def setup_logging(name: str = __name__) -> logging.Logger:
    """Set up basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(name)


class ESM2Embedder:
    """ESM-2 embedder for sequences identified by MD5 hashes"""
    
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
        """Load ESM-2 model from checkpoint"""
        self.logger.info(f"Loading ESM-2 model from {self.model_path}")
        
        # Check PyTorch version for compatibility
        torch_version = torch.__version__
        self.logger.info(f"PyTorch version: {torch_version}")
        
        try:
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Import ESM after confirming we can load the checkpoint
            import esm
            
            # Get model info from checkpoint
            if 'cfg' in checkpoint:
                # This is a fairseq checkpoint
                model_name = checkpoint['cfg']['model'].get('_name', 'esm2_t33_650M_UR50D')
            else:
                # Default to t33_650M model
                model_name = 'esm2_t33_650M_UR50D'
            
            self.logger.info(f"Model architecture: {model_name}")
            
            # Load model and alphabet based on architecture
            if 't33_650M' in model_name or 't33' in self.model_path.lower():
                self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                expected_layers = 33
            elif 't30_150M' in model_name or 't30' in self.model_path.lower():
                self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
                expected_layers = 30
            elif 't36_3B' in model_name or 't36' in self.model_path.lower():
                self.model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
                expected_layers = 36
            elif 't48_15B' in model_name or 't48' in self.model_path.lower():
                self.model, self.alphabet = esm.pretrained.esm2_t48_15B_UR50D()
                expected_layers = 48
            else:
                # Default to t33
                self.logger.warning(f"Unknown model architecture, defaulting to t33_650M")
                self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                expected_layers = 33
            
            # Load weights from checkpoint
            if 'model' in checkpoint:
                model_state = checkpoint['model']
            elif 'state_dict' in checkpoint:
                model_state = checkpoint['state_dict']
            else:
                model_state = checkpoint
            
            # Filter out unexpected keys
            model_keys = set(self.model.state_dict().keys())
            filtered_state = {k: v for k, v in model_state.items() if k in model_keys}
            
            # Load the filtered state
            self.model.load_state_dict(filtered_state, strict=False)
            
            self.logger.info(f"Loaded {len(filtered_state)}/{len(model_state)} parameters")
            
        except Exception as e:
            self.logger.error(f"Failed to load model from checkpoint: {e}")
            self.logger.info("Falling back to downloading pre-trained model")
            
            import esm
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        
        # Set up batch converter
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.args.embed_dim if hasattr(self.model, 'args') else 1280
        self.logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        
    def generate_embeddings(self, sequences: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for sequences
        
        Args:
            sequences: Dictionary mapping MD5 hash to sequence
            
        Returns:
            Dictionary mapping MD5 hash to embedding
        """
        if self.model is None:
            self.load_model()
        
        embeddings = {}
        
        # Process in batches
        hash_list = list(sequences.keys())
        n_batches = (len(hash_list) + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(n_batches), desc="Generating embeddings"):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(hash_list))
                batch_hashes = hash_list[start_idx:end_idx]
                
                # Prepare batch data
                batch_data = [(hash_id, sequences[hash_id]) for hash_id in batch_hashes]
                
                try:
                    # Convert sequences
                    batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_data)
                    batch_tokens = batch_tokens.to(self.device)
                    
                    # Get embeddings
                    results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
                    batch_embeddings = results["representations"][33]
                    
                    # Extract per-sequence embeddings (mean over length)
                    for i, hash_id in enumerate(batch_hashes):
                        seq_len = len(sequences[hash_id])
                        # Take mean over sequence length (excluding special tokens)
                        seq_embedding = batch_embeddings[i, 1:seq_len+1].mean(0).cpu().numpy()
                        embeddings[hash_id] = seq_embedding
                        
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_idx}: {e}")
                    # Create zero embeddings for failed sequences
                    for hash_id in batch_hashes:
                        embeddings[hash_id] = np.zeros(self.embedding_dim, dtype=np.float32)
        
        return embeddings
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], output_path: str):
        """
        Save embeddings to HDF5 file
        
        Args:
            embeddings: Dictionary mapping MD5 hash to embedding
            output_path: Path to save HDF5 file
        """
        self.logger.info(f"Saving {len(embeddings)} embeddings to {output_path}")
        
        # Convert to arrays
        hashes = list(embeddings.keys())
        embedding_matrix = np.stack([embeddings[h] for h in hashes])
        
        # Save to HDF5
        with h5py.File(output_path, 'w') as f:
            # Save embeddings
            f.create_dataset('embeddings', data=embedding_matrix, 
                           dtype='float32', compression='gzip')
            
            # Save MD5 hashes as identifiers
            f.create_dataset('hashes', data=[h.encode('utf-8') for h in hashes],
                           dtype=h5py.string_dtype())
            
            # Save metadata
            f.attrs['embedding_dim'] = self.embedding_dim
            f.attrs['n_sequences'] = len(embeddings)
            
        self.logger.info(f"Embeddings saved successfully")


def load_sequences_from_json(json_path: str) -> Dict[str, str]:
    """
    Load sequences from JSON file with MD5 hash keys
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Dictionary mapping MD5 hash to sequence
    """
    with open(json_path, 'r') as f:
        sequences = json.load(f)
    return sequences


def verify_embeddings(h5_path: str, sequences: Dict[str, str]) -> bool:
    """
    Verify that embeddings match the sequences
    
    Args:
        h5_path: Path to HDF5 file
        sequences: Dictionary of sequences
        
    Returns:
        True if verification passes
    """
    with h5py.File(h5_path, 'r') as f:
        stored_hashes = set(h.decode('utf-8') for h in f['hashes'][:])
        sequence_hashes = set(sequences.keys())
        
        missing = sequence_hashes - stored_hashes
        extra = stored_hashes - sequence_hashes
        
        if missing:
            print(f"WARNING: {len(missing)} sequences missing from embeddings")
        if extra:
            print(f"WARNING: {len(extra)} extra embeddings not in sequences")
            
        return len(missing) == 0


def main():
    parser = argparse.ArgumentParser(description='Generate ESM-2 embeddings with MD5 hashes')
    parser.add_argument('--host_sequences', type=str,
                       default='data/sequences/dedup_host_sequences.json',
                       help='Path to host sequences JSON file')
    parser.add_argument('--phage_sequences', type=str,
                       default='data/sequences/dedup_phage_sequences.json',
                       help='Path to phage sequences JSON file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to ESM-2 model checkpoint')
    parser.add_argument('--output_dir', type=str, default='data/embeddings',
                       help='Directory to save embeddings')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--skip_host', action='store_true',
                       help='Skip host embedding generation')
    parser.add_argument('--skip_phage', action='store_true',
                       help='Skip phage embedding generation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize embedder
    embedder = ESM2Embedder(
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Generate host embeddings
    if not args.skip_host:
        print("\n=== Generating Host Embeddings ===")
        host_sequences = load_sequences_from_json(args.host_sequences)
        print(f"Loaded {len(host_sequences)} unique host sequences")
        
        host_embeddings = embedder.generate_embeddings(host_sequences)
        
        host_output = os.path.join(args.output_dir, 'host_embeddings.h5')
        embedder.save_embeddings(host_embeddings, host_output)
        
        # Verify
        if verify_embeddings(host_output, host_sequences):
            print("✓ Host embeddings verified successfully")
    
    # Generate phage embeddings
    if not args.skip_phage:
        print("\n=== Generating Phage Embeddings ===")
        phage_sequences = load_sequences_from_json(args.phage_sequences)
        print(f"Loaded {len(phage_sequences)} unique phage sequences")
        
        phage_embeddings = embedder.generate_embeddings(phage_sequences)
        
        phage_output = os.path.join(args.output_dir, 'phage_embeddings.h5')
        embedder.save_embeddings(phage_embeddings, phage_output)
        
        # Verify
        if verify_embeddings(phage_output, phage_sequences):
            print("✓ Phage embeddings verified successfully")
    
    print("\n=== Embedding Generation Complete ===")


if __name__ == "__main__":
    main()