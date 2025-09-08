"""
ESM-2 Embedding Generation Script using MD5 hashes
Generates embeddings from deduplicated sequence files with MD5 hash identifiers
Supports resuming from partial runs and handles OOM errors
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Set
import numpy as np
import torch
import h5py
from tqdm import tqdm
import logging
import hashlib
import pickle


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
        
        # Import ESM
        import esm
        
        # Determine model architecture from path
        model_path_lower = self.model_path.lower()
        if 't33' in model_path_lower or '650m' in model_path_lower:
            self.logger.info("Loading ESM-2 t33_650M model")
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            expected_layers = 33
        elif 't30' in model_path_lower or '150m' in model_path_lower:
            self.logger.info("Loading ESM-2 t30_150M model")
            self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            expected_layers = 30
        elif 't36' in model_path_lower or '3b' in model_path_lower:
            self.logger.info("Loading ESM-2 t36_3B model")
            self.model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            expected_layers = 36
        elif 't48' in model_path_lower or '15b' in model_path_lower:
            self.logger.info("Loading ESM-2 t48_15B model")
            self.model, self.alphabet = esm.pretrained.esm2_t48_15B_UR50D()
            expected_layers = 48
        else:
            # Default to t33
            self.logger.info("Could not determine model type from path, defaulting to t33_650M")
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            expected_layers = 33
        
        # Now load the checkpoint weights if the file exists
        if os.path.exists(self.model_path):
            try:
                self.logger.info(f"Loading weights from checkpoint: {self.model_path}")
                
                # Load checkpoint
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                
                # Extract state dict
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Clean up state dict keys if needed
                new_state_dict = {}
                for key, value in state_dict.items():
                    # Remove 'module.' prefix if present (from DataParallel)
                    if key.startswith('module.'):
                        new_key = key[7:]
                    else:
                        new_key = key
                    new_state_dict[new_key] = value
                
                # Load the weights
                self.model.load_state_dict(new_state_dict, strict=False)
                self.logger.info("Successfully loaded checkpoint weights")
                
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint weights: {e}")
                self.logger.info("Using downloaded pre-trained weights instead")
        else:
            self.logger.warning(f"Checkpoint file not found at {self.model_path}")
            self.logger.info("Using downloaded pre-trained weights")
        
        # Set up batch converter
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension and layer count
        self.embedding_dim = self.model.args.embed_dim if hasattr(self.model, 'args') else 1280
        self.repr_layers = [expected_layers]  # Use the last layer
        
        self.logger.info(f"Model ready. Embedding dimension: {self.embedding_dim}, "
                        f"Using layer: {expected_layers}")
    
    def process_single_sequence(self, hash_id: str, sequence: str) -> Optional[np.ndarray]:
        """
        Process a single sequence (for handling OOM on batches)
        
        Args:
            hash_id: MD5 hash identifier
            sequence: Protein sequence
            
        Returns:
            Embedding array or None if failed
        """
        try:
            # Process as single item batch
            batch_labels, batch_strs, batch_tokens = self.batch_converter([(hash_id, sequence)])
            batch_tokens = batch_tokens.to(self.device)
            
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=self.repr_layers, return_contacts=False)
                batch_embeddings = results["representations"][self.repr_layers[0]]
                
                seq_len = len(sequence)
                # Take mean over sequence length (excluding special tokens)
                seq_embedding = batch_embeddings[0, 1:seq_len+1].mean(0).cpu().numpy()
                
            return seq_embedding
            
        except torch.cuda.OutOfMemoryError:
            self.logger.error(f"OOM even for single sequence {hash_id} (length: {len(sequence)})")
            torch.cuda.empty_cache()
            return None
        except Exception as e:
            self.logger.error(f"Error processing single sequence {hash_id}: {e}")
            return None
    
    def generate_embeddings(self, sequences: Dict[str, str], resume: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for sequences with OOM handling and resume capability
        
        Args:
            sequences: Dictionary mapping MD5 hash to sequence
            resume: Whether to resume from existing partial results
            
        Returns:
            Dictionary mapping MD5 hash to embedding
        """
        if self.model is None:
            self.load_model()
        
        # Check for existing partial results
        embeddings = {}
        processed_hashes = set()
        checkpoint_file = "embedding_checkpoint.pkl"
        
        if resume and os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    embeddings = checkpoint_data['embeddings']
                    processed_hashes = set(checkpoint_data['processed_hashes'])
                self.logger.info(f"Resuming from checkpoint: {len(processed_hashes)} sequences already processed")
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint: {e}")
        
        # Filter out already processed sequences
        remaining_sequences = {k: v for k, v in sequences.items() if k not in processed_hashes}
        
        if not remaining_sequences:
            self.logger.info("All sequences already processed!")
            return embeddings
        
        # Sort by sequence length for better batching
        sorted_items = sorted(remaining_sequences.items(), key=lambda x: len(x[1]))
        hash_list = [item[0] for item in sorted_items]
        
        # Process in batches with adaptive batch size
        current_batch_size = self.batch_size
        i = 0
        pbar = tqdm(total=len(hash_list), initial=len(processed_hashes), desc="Generating embeddings")
        
        while i < len(hash_list):
            # Get batch
            batch_end = min(i + current_batch_size, len(hash_list))
            batch_hashes = hash_list[i:batch_end]
            batch_data = [(hash_id, remaining_sequences[hash_id]) for hash_id in batch_hashes]
            
            # Check batch sequence lengths
            max_len = max(len(remaining_sequences[h]) for h in batch_hashes)
            
            # Skip very long sequences in batch processing
            if max_len > 1000 and current_batch_size > 1:
                self.logger.warning(f"Long sequences detected (max {max_len}), reducing batch size")
                current_batch_size = max(1, current_batch_size // 2)
                continue
            
            try:
                # Try to process batch
                with torch.no_grad():
                    batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_data)
                    batch_tokens = batch_tokens.to(self.device)
                    
                    results = self.model(batch_tokens, repr_layers=self.repr_layers, return_contacts=False)
                    batch_embeddings = results["representations"][self.repr_layers[0]]
                    
                    # Extract embeddings
                    for j, hash_id in enumerate(batch_hashes):
                        seq_len = len(remaining_sequences[hash_id])
                        seq_embedding = batch_embeddings[j, 1:seq_len+1].mean(0).cpu().numpy()
                        embeddings[hash_id] = seq_embedding
                        processed_hashes.add(hash_id)
                
                # Success - try increasing batch size
                if current_batch_size < self.batch_size:
                    current_batch_size = min(current_batch_size + 1, self.batch_size)
                
                # Update progress
                pbar.update(len(batch_hashes))
                i = batch_end
                
                # Save checkpoint periodically
                if len(processed_hashes) % 100 == 0:
                    checkpoint_data = {
                        'embeddings': embeddings,
                        'processed_hashes': list(processed_hashes)
                    }
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(checkpoint_data, f)
                        
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.logger.warning(f"OOM with batch size {current_batch_size}, reducing...")
                
                if current_batch_size > 1:
                    # Reduce batch size and retry
                    current_batch_size = max(1, current_batch_size // 2)
                else:
                    # Process sequences individually
                    self.logger.info(f"Processing {len(batch_hashes)} sequences individually...")
                    for hash_id in batch_hashes:
                        seq = remaining_sequences[hash_id]
                        self.logger.info(f"Processing {hash_id} (length: {len(seq)})")
                        
                        embedding = self.process_single_sequence(hash_id, seq)
                        if embedding is not None:
                            embeddings[hash_id] = embedding
                            processed_hashes.add(hash_id)
                        else:
                            # Create zero embedding for failed sequences
                            embeddings[hash_id] = np.zeros(self.embedding_dim, dtype=np.float32)
                            processed_hashes.add(hash_id)
                        
                        pbar.update(1)
                    
                    i = batch_end
                    
            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")
                # Process failed batch individually
                for hash_id in batch_hashes:
                    if hash_id not in processed_hashes:
                        embedding = self.process_single_sequence(hash_id, remaining_sequences[hash_id])
                        if embedding is not None:
                            embeddings[hash_id] = embedding
                        else:
                            embeddings[hash_id] = np.zeros(self.embedding_dim, dtype=np.float32)
                        processed_hashes.add(hash_id)
                        pbar.update(1)
                i = batch_end
        
        pbar.close()
        
        # Save final results
        checkpoint_data = {
            'embeddings': embeddings,
            'processed_hashes': list(processed_hashes)
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        self.logger.info(f"Completed. Processed {len(embeddings)} sequences total")
        
        # Clean up checkpoint file if all done
        if len(embeddings) == len(sequences):
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                self.logger.info("Removed checkpoint file (all sequences processed)")
        
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


def load_existing_embeddings(h5_path: str) -> Set[str]:
    """
    Load hashes of already computed embeddings from HDF5 file
    
    Args:
        h5_path: Path to HDF5 file
        
    Returns:
        Set of MD5 hashes that already have embeddings
    """
    if not os.path.exists(h5_path):
        return set()
    
    try:
        with h5py.File(h5_path, 'r') as f:
            existing_hashes = set(h.decode('utf-8') for h in f['hashes'][:])
        return existing_hashes
    except Exception as e:
        print(f"Could not load existing embeddings: {e}")
        return set()


def merge_embeddings(existing_path: str, new_embeddings: Dict[str, np.ndarray], 
                     output_path: str = None) -> None:
    """
    Merge new embeddings with existing ones
    
    Args:
        existing_path: Path to existing HDF5 file
        new_embeddings: New embeddings to add
        output_path: Where to save merged embeddings (defaults to existing_path)
    """
    if output_path is None:
        output_path = existing_path
    
    # Load existing
    existing_embeddings = {}
    if os.path.exists(existing_path):
        with h5py.File(existing_path, 'r') as f:
            hashes = f['hashes'][:]
            embeddings = f['embeddings'][:]
            for i, h in enumerate(hashes):
                hash_str = h.decode('utf-8') if isinstance(h, bytes) else h
                existing_embeddings[hash_str] = embeddings[i]
    
    # Merge
    all_embeddings = {**existing_embeddings, **new_embeddings}
    
    # Save
    hashes = list(all_embeddings.keys())
    embedding_matrix = np.stack([all_embeddings[h] for h in hashes])
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('embeddings', data=embedding_matrix, 
                       dtype='float32', compression='gzip')
        f.create_dataset('hashes', data=[h.encode('utf-8') for h in hashes],
                       dtype=h5py.string_dtype())
        f.attrs['embedding_dim'] = embedding_matrix.shape[1]
        f.attrs['n_sequences'] = len(hashes)


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
            print(f"INFO: {len(extra)} extra embeddings in file (from previous runs)")
            
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
    parser.add_argument('--no_resume', action='store_true',
                       help='Do not resume from checkpoint')
    parser.add_argument('--merge', action='store_true',
                       help='Merge with existing embeddings file')
    
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
        
        host_output = os.path.join(args.output_dir, 'host_embeddings.h5')
        
        # Check what's already done
        if args.merge and os.path.exists(host_output):
            existing = load_existing_embeddings(host_output)
            remaining = {k: v for k, v in host_sequences.items() if k not in existing}
            print(f"Found {len(existing)} existing embeddings, {len(remaining)} to compute")
            
            if remaining:
                new_embeddings = embedder.generate_embeddings(remaining, resume=not args.no_resume)
                merge_embeddings(host_output, new_embeddings)
                print(f"Merged {len(new_embeddings)} new embeddings with existing")
        else:
            host_embeddings = embedder.generate_embeddings(host_sequences, resume=not args.no_resume)
            embedder.save_embeddings(host_embeddings, host_output)
        
        # Verify
        if verify_embeddings(host_output, host_sequences):
            print("✓ Host embeddings verified successfully")
    
    # Generate phage embeddings
    if not args.skip_phage:
        print("\n=== Generating Phage Embeddings ===")
        phage_sequences = load_sequences_from_json(args.phage_sequences)
        print(f"Loaded {len(phage_sequences)} unique phage sequences")
        
        phage_output = os.path.join(args.output_dir, 'phage_embeddings.h5')
        
        # Check what's already done
        if args.merge and os.path.exists(phage_output):
            existing = load_existing_embeddings(phage_output)
            remaining = {k: v for k, v in phage_sequences.items() if k not in existing}
            print(f"Found {len(existing)} existing embeddings, {len(remaining)} to compute")
            
            if remaining:
                new_embeddings = embedder.generate_embeddings(remaining, resume=not args.no_resume)
                merge_embeddings(phage_output, new_embeddings)
                print(f"Merged {len(new_embeddings)} new embeddings with existing")
        else:
            phage_embeddings = embedder.generate_embeddings(phage_sequences, resume=not args.no_resume)
            embedder.save_embeddings(phage_embeddings, phage_output)
        
        # Verify
        if verify_embeddings(phage_output, phage_sequences):
            print("✓ Phage embeddings verified successfully")
    
    print("\n=== Embedding Generation Complete ===")


if __name__ == "__main__":
    main()