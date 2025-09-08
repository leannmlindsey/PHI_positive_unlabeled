"""
Test script to verify XGBoost baseline data loading works correctly
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import h5py
import logging
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def setup_logger():
    """Set up logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_data_loading():
    """Test that data loading works with new format"""
    logger = setup_logger()
    
    logger.info("=" * 50)
    logger.info("Testing XGBoost Data Loading")
    logger.info("=" * 50)
    
    # Check if data file exists
    data_path = Path("data/dedup.labeled_marker_rbp_phageID_with_hashes.tsv")
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run: python scripts/extract_sequences.py first")
        return False
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, sep='\t')
    
    # Check columns
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Check for required columns
    required_cols = ['host_md5_set', 'phage_md5_set']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    # Analyze hash columns
    logger.info("\nAnalyzing hash columns:")
    
    # Host hashes
    host_empty = df['host_md5_set'].isna() | (df['host_md5_set'] == '')
    logger.info(f"  Host hashes: {(~host_empty).sum()} non-empty, {host_empty.sum()} empty")
    
    # Count multi-host entries
    multi_host = df[~host_empty]['host_md5_set'].str.contains(',').sum()
    logger.info(f"  Multi-host entries: {multi_host}")
    
    # Phage hashes
    phage_empty = df['phage_md5_set'].isna() | (df['phage_md5_set'] == '')
    logger.info(f"  Phage hashes: {(~phage_empty).sum()} non-empty, {phage_empty.sum()} empty")
    
    # Count multi-phage entries
    multi_phage = df[~phage_empty]['phage_md5_set'].str.contains(',').sum()
    logger.info(f"  Multi-phage entries: {multi_phage}")
    
    # Sample some entries
    logger.info("\nSample entries:")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        host_hashes = str(row['host_md5_set']).split(',') if pd.notna(row['host_md5_set']) else []
        phage_hashes = str(row['phage_md5_set']).split(',') if pd.notna(row['phage_md5_set']) else []
        logger.info(f"  Row {i}: {len(host_hashes)} host hashes, {len(phage_hashes)} phage hashes")
    
    # Check splits file
    splits_path = Path("data/processed/splits.pkl")
    if not splits_path.exists():
        logger.error(f"Splits file not found: {splits_path}")
        logger.info("Please run: python scripts/simple_splitting.py first")
        return False
    
    logger.info(f"\nLoading splits from {splits_path}")
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    
    logger.info(f"Split keys: {list(splits.keys())}")
    
    # Check split format
    for split_name in ['train', 'val', 'test']:
        if split_name in splits:
            split_data = splits[split_name]
            if isinstance(split_data, pd.DataFrame):
                logger.info(f"  {split_name}: DataFrame with {len(split_data)} samples")
            else:
                logger.info(f"  {split_name}: {type(split_data)}")
        elif f'{split_name}_idx' in splits:
            indices = splits[f'{split_name}_idx']
            logger.info(f"  {split_name}_idx: {len(indices)} indices")
    
    # Check embeddings
    embeddings_dir = Path("data/embeddings")
    host_emb_path = embeddings_dir / "host_embeddings.h5"
    phage_emb_path = embeddings_dir / "phage_embeddings.h5"
    
    if not host_emb_path.exists() or not phage_emb_path.exists():
        logger.error(f"Embedding files not found in {embeddings_dir}")
        logger.info("Please run: python scripts/generate_embeddings.py first")
        return False
    
    logger.info(f"\nChecking embeddings:")
    
    # Load host embeddings
    with h5py.File(host_emb_path, 'r') as f:
        host_hashes = f['hashes'][:].astype(str)
        host_embeddings_shape = f['embeddings'].shape
        logger.info(f"  Host embeddings: {host_embeddings_shape[0]} proteins, {host_embeddings_shape[1]} dimensions")
    
    # Load phage embeddings
    with h5py.File(phage_emb_path, 'r') as f:
        phage_hashes = f['hashes'][:].astype(str)
        phage_embeddings_shape = f['embeddings'].shape
        logger.info(f"  Phage embeddings: {phage_embeddings_shape[0]} proteins, {phage_embeddings_shape[1]} dimensions")
    
    # Check if hashes in data match embeddings
    logger.info("\nChecking hash coverage:")
    
    # Get all unique hashes from data
    all_host_hashes = set()
    all_phage_hashes = set()
    
    for _, row in df.iterrows():
        if pd.notna(row['host_md5_set']) and row['host_md5_set']:
            for h in str(row['host_md5_set']).split(','):
                if h.strip():
                    all_host_hashes.add(h.strip())
        
        if pd.notna(row['phage_md5_set']) and row['phage_md5_set']:
            for p in str(row['phage_md5_set']).split(','):
                if p.strip():
                    all_phage_hashes.add(p.strip())
    
    # Check coverage
    host_hashes_set = set(host_hashes)
    phage_hashes_set = set(phage_hashes)
    
    host_coverage = len(all_host_hashes & host_hashes_set) / len(all_host_hashes) if all_host_hashes else 0
    phage_coverage = len(all_phage_hashes & phage_hashes_set) / len(all_phage_hashes) if all_phage_hashes else 0
    
    logger.info(f"  Host hash coverage: {host_coverage:.1%} ({len(all_host_hashes & host_hashes_set)}/{len(all_host_hashes)})")
    logger.info(f"  Phage hash coverage: {phage_coverage:.1%} ({len(all_phage_hashes & phage_hashes_set)}/{len(all_phage_hashes)})")
    
    if host_coverage < 0.9 or phage_coverage < 0.9:
        logger.warning("Low embedding coverage detected! Some proteins may be missing embeddings.")
    
    logger.info("\n" + "=" * 50)
    logger.info("Data Loading Test Complete!")
    logger.info("=" * 50)
    
    return True

def test_simple_training():
    """Test that we can load and prepare a small batch for XGBoost"""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 50)
    logger.info("Testing XGBoost Data Preparation")
    logger.info("=" * 50)
    
    # Import the baseline functions
    from scripts.simple_baseline import load_embeddings, prepare_simple_dataset
    
    try:
        # Load embeddings
        logger.info("Loading embeddings...")
        host_embeddings, phage_embeddings = load_embeddings("data/embeddings")
        logger.info(f"Loaded {len(host_embeddings)} host and {len(phage_embeddings)} phage embeddings")
        
        # Prepare dataset
        logger.info("\nPreparing dataset...")
        datasets = prepare_simple_dataset(
            "data/dedup.labeled_marker_rbp_phageID_with_hashes.tsv",
            "data/processed/splits.pkl",
            host_embeddings,
            phage_embeddings
        )
        
        # Check dataset shapes
        logger.info("\nDataset shapes:")
        for split_name in ['train', 'val', 'test']:
            if split_name in datasets:
                X = datasets[split_name]['X']
                y = datasets[split_name]['y']
                logger.info(f"  {split_name}: X shape={X.shape}, y shape={y.shape}")
                logger.info(f"    Class balance: {np.mean(y):.2%} positive")
        
        logger.info("\nData preparation successful!")
        return True
        
    except Exception as e:
        logger.error(f"Error during data preparation: {e}", exc_info=True)
        return False

def main():
    """Main test function"""
    logger = setup_logger()
    
    logger.info("Starting XGBoost baseline tests...")
    
    # Test data loading
    if not test_data_loading():
        logger.error("Data loading test failed!")
        return 1
    
    # Test data preparation
    if not test_simple_training():
        logger.error("Data preparation test failed!")
        return 1
    
    logger.info("\nAll tests passed! XGBoost baseline is ready to run.")
    logger.info("Run with: python scripts/simple_baseline.py")
    
    return 0

if __name__ == "__main__":
    exit(main())