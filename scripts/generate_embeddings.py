"""
ESM-2 Embedding Generation Script
Generates protein embeddings for all unique sequences in the dataset
Designed for execution on HPC cluster with GPU support
"""

import os
import sys
import logging
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import h5py
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

# Configure logging
def setup_logging(log_dir: Path, log_name: str = None) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        log_dir: Directory for log files
        log_name: Optional specific log file name
        
    Returns:
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if log_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"embedding_generation_{timestamp}.log"
    
    log_path = log_dir / log_name
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_path}")
    return logger


class ProteinSequenceExtractor:
    """
    Extracts unique protein sequences from the dataset
    """
    
    def __init__(self, data_path: str, logger: logging.Logger):
        """
        Initialize the extractor
        
        Args:
            data_path: Path to the TSV data file
            logger: Logger instance
        """
        self.data_path = data_path
        self.logger = logger
        self.df = None
        self.unique_sequences = {}
        
    def load_data(self) -> None:
        """Load the data file"""
        self.logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path, sep='\t')
        self.logger.info(f"Loaded {len(self.df)} interactions")
        
    def extract_unique_sequences(self) -> Dict[str, str]:
        """
        Extract all unique protein sequences with their MD5 hashes
        
        Returns:
            Dictionary mapping MD5 hash to protein sequence
        """
        if self.df is None:
            self.load_data()
            
        self.logger.info("Extracting unique protein sequences...")
        
        # Process marker sequences
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing markers"):
            marker_seqs = row['marker_gene_seq'].split(',')
            marker_hashes = row['marker_md5'].split(',') if ',' in str(row['marker_md5']) else [row['marker_md5']]
            
            # Ensure we have the same number of sequences and hashes
            if len(marker_seqs) != len(marker_hashes):
                self.logger.warning(f"Row {idx}: Mismatch in marker counts - {len(marker_seqs)} seqs, {len(marker_hashes)} hashes")
                continue
                
            for seq, hash_val in zip(marker_seqs, marker_hashes):
                seq = seq.strip()  # Remove any whitespace
                if hash_val not in self.unique_sequences:
                    self.unique_sequences[hash_val] = seq
                    
        # Process RBP sequences
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing RBPs"):
            rbp_seqs = row['rbp_seq'].split(',')
            rbp_hashes = row['rbp_md5'].split(',') if ',' in str(row['rbp_md5']) else [row['rbp_md5']]
            
            # Ensure we have the same number of sequences and hashes
            if len(rbp_seqs) != len(rbp_hashes):
                self.logger.warning(f"Row {idx}: Mismatch in RBP counts - {len(rbp_seqs)} seqs, {len(rbp_hashes)} hashes")
                continue
                
            for seq, hash_val in zip(rbp_seqs, rbp_hashes):
                seq = seq.strip()  # Remove any whitespace
                if hash_val not in self.unique_sequences:
                    self.unique_sequences[hash_val] = seq
                    
        self.logger.info(f"Found {len(self.unique_sequences)} unique protein sequences")
        
        # Verify MD5 hashes
        self.logger.info("Verifying MD5 hashes...")
        mismatches = 0
        for hash_val, seq in self.unique_sequences.items():
            computed_hash = hashlib.md5(seq.encode()).hexdigest()
            if computed_hash != hash_val:
                mismatches += 1
                self.logger.warning(f"Hash mismatch for {hash_val}: computed {computed_hash}")
                
        if mismatches > 0:
            self.logger.warning(f"Found {mismatches} hash mismatches")
        else:
            self.logger.info("All hashes verified successfully")
            
        return self.unique_sequences
    
    def save_sequences(self, output_path: str) -> None:
        """
        Save sequences to file for reference
        
        Args:
            output_path: Path to save the sequences
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.unique_sequences, f, indent=2)
        
        self.logger.info(f"Saved {len(self.unique_sequences)} sequences to {output_path}")


class ESM2EmbeddingGenerator:
    """
    Generates ESM-2 embeddings for protein sequences
    """
    
    def __init__(self, 
                 model_name: str = "facebook/esm2_t33_650M_UR50D",
                 model_path: str = None,
                 batch_size: int = 8,
                 max_length: int = 1024,
                 device: str = None,
                 logger: logging.Logger = None):
        """
        Initialize the embedding generator
        
        Args:
            model_name: ESM-2 model name from HuggingFace
            model_path: Local path to pre-downloaded model (optional)
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            device: Device to use (cuda/cpu)
            logger: Logger instance
        """
        self.model_name = model_name
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.logger = logger or logging.getLogger(__name__)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        
    def load_model(self) -> None:
        """Load the ESM-2 model and tokenizer"""
        # Check if model_path is a .pt checkpoint file (fair-esm format)
        if self.model_path and Path(self.model_path).exists() and self.model_path.endswith('.pt'):
            self.logger.info(f"Loading ESM-2 model from checkpoint: {self.model_path}")
            
            # Use fair-esm library for .pt checkpoints
            try:
                import esm
                
                # For PyTorch 2.6+, we need to handle weights_only issue
                # Temporarily monkey-patch the load function to use weights_only=False
                import torch
                original_load = torch.load
                
                def patched_load(f, *args, **kwargs):
                    # Force weights_only=False for fair-esm models
                    kwargs['weights_only'] = False
                    return original_load(f, *args, **kwargs)
                
                # Apply patch
                torch.load = patched_load
                
                try:
                    # Load the model using fair-esm
                    model, alphabet = esm.pretrained.load_model_and_alphabet_local(self.model_path)
                    self.model = model
                    self.alphabet = alphabet
                    self.tokenizer = alphabet  # Use alphabet as tokenizer for fair-esm
                    self.use_fair_esm = True
                    
                    # Get embedding dimension
                    if hasattr(self.model, 'embed_dim'):
                        self.embedding_dim = self.model.embed_dim
                    elif hasattr(self.model, 'args') and hasattr(self.model.args, 'embed_dim'):
                        self.embedding_dim = self.model.args.embed_dim
                    else:
                        # Default for ESM2_t33_650M
                        self.embedding_dim = 1280
                finally:
                    # Restore original load function
                    torch.load = original_load
                    
            except ImportError:
                self.logger.error("fair-esm library not found. Install with: pip install fair-esm")
                raise
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint with fair-esm: {e}")
                self.logger.info("Falling back to HuggingFace model")
                self._load_huggingface_model()
                
        elif self.model_path and Path(self.model_path).exists() and Path(self.model_path).is_dir():
            # Load from a directory with HuggingFace format
            self.logger.info(f"Loading ESM-2 model from local directory: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self.model = EsmModel.from_pretrained(self.model_path, local_files_only=True)
            self.use_fair_esm = False
            self.embedding_dim = self.model.config.hidden_size
        else:
            # Download from HuggingFace
            self._load_huggingface_model()
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self.logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def _load_huggingface_model(self) -> None:
        """Load model from HuggingFace"""
        self.logger.info(f"Loading ESM-2 model from HuggingFace: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = EsmModel.from_pretrained(self.model_name)
        self.use_fair_esm = False
        self.embedding_dim = self.model.config.hidden_size
        
    def generate_embedding(self, sequence: str) -> np.ndarray:
        """
        Generate embedding for a single sequence
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Embedding array of shape (embedding_dim,)
        """
        if hasattr(self, 'use_fair_esm') and self.use_fair_esm:
            # Use fair-esm processing
            import esm
            
            batch_converter = self.alphabet.get_batch_converter()
            batch_labels, batch_strs, batch_tokens = batch_converter([("protein", sequence)])
            batch_tokens = batch_tokens.to(self.device)
            
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
                # Extract embeddings from the last layer
                embeddings = results["representations"][33]
                
            # Remove BOS/EOS tokens and mean pool
            embeddings = embeddings[0, 1:-1]  # Remove first and last tokens
            embedding = embeddings.mean(dim=0)
            
            return embedding.cpu().numpy()
        else:
            # Use HuggingFace processing
            inputs = self.tokenizer(
                sequence,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use mean pooling over sequence length
            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Mask padding tokens
            embeddings = embeddings * attention_mask.unsqueeze(-1)
            
            # Mean pooling
            embedding = embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            
            return embedding.cpu().numpy()[0]
    
    def generate_embeddings_batch(self, sequences: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of sequences
        
        Args:
            sequences: List of protein sequences
            
        Returns:
            Array of embeddings with shape (n_sequences, embedding_dim)
        """
        if hasattr(self, 'use_fair_esm') and self.use_fair_esm:
            # Use fair-esm processing
            import esm
            
            batch_converter = self.alphabet.get_batch_converter()
            batch_labels = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_labels)
            batch_tokens = batch_tokens.to(self.device)
            
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
                # Extract embeddings from the last layer
                token_embeddings = results["representations"][33]
                
            # Process each sequence
            embeddings_list = []
            for i, seq in enumerate(sequences):
                seq_len = len(seq)
                # Remove BOS/EOS tokens and mean pool
                embeddings = token_embeddings[i, 1:seq_len+1]  # Skip BOS, take seq_len tokens
                embedding = embeddings.mean(dim=0)
                embeddings_list.append(embedding)
                
            embeddings = torch.stack(embeddings_list)
            return embeddings.cpu().numpy()
        else:
            # Use HuggingFace processing
            inputs = self.tokenizer(
                sequences,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use mean pooling over sequence length
            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Mask padding tokens
            embeddings = embeddings * attention_mask.unsqueeze(-1)
            
            # Mean pooling
            embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            
            return embeddings.cpu().numpy()
    
    def process_all_sequences(self, 
                            sequences_dict: Dict[str, str],
                            output_path: str,
                            checkpoint_dir: str = None) -> None:
        """
        Process all sequences and save embeddings
        
        Args:
            sequences_dict: Dictionary mapping hash to sequence
            output_path: Path to save HDF5 file
            checkpoint_dir: Directory for checkpoints (optional)
        """
        if self.model is None:
            self.load_model()
            
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up checkpointing
        checkpoint_file = None
        processed_hashes = set()
        
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_file = checkpoint_dir / "embedding_checkpoint.pkl"
            
            if checkpoint_file.exists():
                with open(checkpoint_file, 'rb') as f:
                    processed_hashes = pickle.load(f)
                self.logger.info(f"Resuming from checkpoint: {len(processed_hashes)} already processed")
        
        # Prepare sequences to process
        sequences_to_process = {
            h: s for h, s in sequences_dict.items() 
            if h not in processed_hashes
        }
        
        if len(sequences_to_process) == 0:
            self.logger.info("All sequences already processed")
            return
        
        # Create/open HDF5 file
        with h5py.File(output_path, 'a') as h5f:
            # Create dataset if it doesn't exist
            if 'embeddings' not in h5f:
                h5f.create_dataset(
                    'embeddings',
                    shape=(0, self.embedding_dim),
                    maxshape=(None, self.embedding_dim),
                    dtype='float32'
                )
                h5f.create_dataset(
                    'hashes',
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype()
                )
            
            # Process in batches
            hash_list = list(sequences_to_process.keys())
            seq_list = list(sequences_to_process.values())
            
            n_batches = (len(hash_list) + self.batch_size - 1) // self.batch_size
            
            for i in tqdm(range(0, len(hash_list), self.batch_size), 
                         total=n_batches, 
                         desc="Generating embeddings"):
                
                batch_hashes = hash_list[i:i + self.batch_size]
                batch_seqs = seq_list[i:i + self.batch_size]
                
                try:
                    # Generate embeddings
                    batch_embeddings = self.generate_embeddings_batch(batch_seqs)
                    
                    # Resize datasets
                    current_size = h5f['embeddings'].shape[0]
                    new_size = current_size + len(batch_hashes)
                    
                    h5f['embeddings'].resize((new_size, self.embedding_dim))
                    h5f['hashes'].resize((new_size,))
                    
                    # Add to HDF5
                    h5f['embeddings'][current_size:new_size] = batch_embeddings
                    h5f['hashes'][current_size:new_size] = batch_hashes
                    
                    # Update processed set
                    processed_hashes.update(batch_hashes)
                    
                    # Save checkpoint
                    if checkpoint_file and (i + self.batch_size) % (self.batch_size * 10) == 0:
                        with open(checkpoint_file, 'wb') as f:
                            pickle.dump(processed_hashes, f)
                        self.logger.info(f"Checkpoint saved: {len(processed_hashes)} processed")
                        
                except Exception as e:
                    self.logger.error(f"Error processing batch {i}: {e}")
                    # Save checkpoint on error
                    if checkpoint_file:
                        with open(checkpoint_file, 'wb') as f:
                            pickle.dump(processed_hashes, f)
                    raise
            
        self.logger.info(f"Embeddings saved to {output_path}")
        self.logger.info(f"Total embeddings: {len(processed_hashes)}")
        
        # Clean up checkpoint
        if checkpoint_file and checkpoint_file.exists():
            checkpoint_file.unlink()
            self.logger.info("Checkpoint file removed")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Generate ESM-2 embeddings for protein sequences")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to the TSV data file")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                       help="Output directory for embeddings")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D",
                       help="ESM-2 model name")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Local path to pre-downloaded ESM-2 model (optional)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Directory for checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Directory for log files")
    
    args = parser.parse_args()
    
    # Set up paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(Path(args.log_dir))
    logger.info(f"Arguments: {vars(args)}")
    
    # Extract sequences
    extractor = ProteinSequenceExtractor(args.data_path, logger)
    unique_sequences = extractor.extract_unique_sequences()
    
    # Save sequences for reference
    sequences_path = output_dir / "unique_sequences.json"
    extractor.save_sequences(sequences_path)
    
    # Generate embeddings
    generator = ESM2EmbeddingGenerator(
        model_name=args.model_name,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        logger=logger
    )
    
    embeddings_path = output_dir / "protein_embeddings.h5"
    generator.process_all_sequences(
        unique_sequences,
        embeddings_path,
        checkpoint_dir=args.checkpoint_dir
    )
    
    logger.info("Embedding generation completed successfully")


if __name__ == "__main__":
    main()