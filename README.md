# Phage-Host Interaction Prediction using Positive-Unlabeled Learning

## Project Overview

This project implements a deep learning model to predict interactions between bacteriophages and their bacterial hosts using a novel approach that combines:
- **Positive-Unlabeled (PU) Learning**: Training with only positive interaction data
- **Multi-Instance Learning**: Handling variable numbers of proteins on both phage and host sides
- **Noisy-OR Aggregation**: Modeling uncertainty in which specific proteins mediate the interaction

### Biological Context

Phage-host interactions are mediated by:
- **Host side**: Surface proteins (wzx and wzm) that serve as receptors
  - wzx: Flippase protein (K-type marker)
  - wzm: O-type marker protein
- **Phage side**: Receptor Binding Proteins (RBPs) that recognize host receptors

The key challenge is that both hosts and phages can have multiple proteins, and we don't know which specific protein pairs interact.

## Repository Structure

```
phi_pos_unlabeled/
├── data/                          # Data files and processed outputs
│   ├── dedup.phage_marker_rbp_with_phage_entropy.tsv  # Original interaction data
│   └── processed/                 # Generated splits and embeddings
├── scripts/                       # Executable scripts
│   ├── simple_splitting.py        # Data splitting with RBP deduplication  ✅
│   ├── graph_based_splitting.py  # Alternative graph-based splitting  ✅
│   ├── extract_sequences.py      # Extract and deduplicate sequences  ✅
│   └── generate_embeddings.py    # ESM-2 embedding generation  ✅
├── slurm/                         # HPC job submission scripts
│   └── generate_embeddings.sh    # SLURM script for Biowulf  ✅
├── models/                        # Model architecture
│   ├── __init__.py               # Module initialization  ✅
│   ├── encoders.py               # Two-tower encoder architecture  ✅
│   ├── mil_model.py             # MIL model with noisy-OR  ✅
│   └── losses.py                 # nnPU loss implementation  ✅
├── training/                      # Training pipeline
│   ├── __init__.py               # Module initialization  ✅
│   └── dataset.py                # PyTorch dataset classes  ✅
├── utils/                         # Utility functions
│   ├── __init__.py               # Module initialization  ✅
│   ├── data_utils.py             # Data processing utilities  ✅
│   └── logging_utils.py          # Logging and metrics tracking  ✅
├── configs/                       # Configuration files
│   └── default_config.yaml       # Default training configuration  ✅
├── logs/                          # Execution logs
├── checkpoints/                   # Model and process checkpoints
├── requirements.txt               # Python dependencies  ✅
├── README.md                      # Project documentation  ✅
├── IMPLEMENTATION_PLAN_FINAL.md  # Detailed implementation plan  ✅
└── CLAUDE.md                      # Development guidelines  ✅
```

## Input Data Description

### Main Data File: `dedup.labeled_marker_rbp_phageID.tsv`

Tab-separated file with 25,470 known positive interactions containing:

| Column | Description | Format |
|--------|-------------|--------|
| `wzx_gene_seq` | Host wzx (flippase) protein sequences | Protein sequence string |
| `wzm_gene_seq` | Host wzm protein sequences | Protein sequence string |
| `rbp_seq` | Phage RBP sequences | Protein sequence string |
| `phage_ids` | Unique phage identifier | String (e.g., GCA_022150005.1_41) |

Note: This file does not have a header row. All entries represent positive interactions (prophages found in host genomes).

### Alternative Data File (if available): `all_info.labeled_marker.tsv`

More detailed file with additional metadata:

| Column | Description | Format |
|--------|-------------|--------|
| `genome_id` | Host genome identifier | String |
| `wzx_gene_id` | Wzx gene identifier | String |
| `wzx_gene_seq` | Host wzx protein sequence | Protein sequence string |
| `wzm_gene_id` | Wzm gene identifier | String |
| `wzm_gene_seq` | Host wzm protein sequence | Protein sequence string |
| `phage_id` | Phage identifier | String |
| `rbp_id` | RBP identifier | String |
| `rbp_seq` | Phage RBP sequence | Protein sequence string |

### Data Characteristics
- **All interactions are positive** (prophages found in host genomes)
- **Multi-instance nature**: 
  - Hosts have 1-2 marker proteins (wzx and/or wzm) - in this dataset all have both
  - Phages have 1-18 RBPs per interaction:
    - 18,888 rows (74.2%) have 1 RBP
    - 5,394 rows (21.2%) have 2 RBPs
    - 1,029 rows (4.0%) have 3 RBPs
    - 158 rows (0.6%) have 4+ RBPs (max: 18)
- **Total interactions**: 25,470 positive samples
- **Protein redundancy**: Same proteins appear in multiple interactions

## Setup Instructions

### 1. Environment Setup

Create and activate a conda environment:

```bash
# TODO: Add specific conda environment creation commands
# Example structure:
# conda create -n phi_pu_xor python=3.10
# conda activate phi_pu_xor
# pip install -r requirements.txt
```

### 2. Data Preparation

#### Step 1: Split the Data

Create train/validation/test splits with RBP deduplication to prevent data leakage:

```bash
python scripts/simple_splitting.py --data_path data/dedup.labeled_marker_rbp_phageID.tsv
```

This will:
- Create 60:20:20 train/val/test splits
- Remove val/test samples containing RBPs seen in training
- Save splits to `data/processed/`
- Generate split statistics

Expected output:
```
Train: ~14,876 samples (60%)
Val: ~4,375 samples (17.6%)
Test: ~4,363 samples (17.6%)
```

#### Step 2: Extract and Deduplicate Sequences

Extract all unique protein sequences from the dataset:

```bash
python scripts/extract_sequences.py \
    --data_path data/dedup.labeled_marker_rbp_phageID.tsv \
    --output_dir data/sequences
```

This will:
- Extract and deduplicate all host and phage protein sequences
- Handle problematic comma-separated sequences
- Save to separate JSON and FASTA files
- Create mapping file for reconstruction

Output files:
- `data/sequences/host_sequences.json` - 2,907 unique host proteins
- `data/sequences/phage_sequences.json` - 22,310 unique phage proteins
- `data/sequences/sequence_mappings.tsv` - Interaction mappings

#### Step 3: Generate Protein Embeddings

Generate ESM-2 embeddings for the extracted sequences:

```bash
# For local testing (if you have GPU)
python scripts/generate_embeddings.py \
    --host_sequences data/sequences/host_sequences.json \
    --phage_sequences data/sequences/phage_sequences.json \
    --model_path /path/to/esm2_t33_650M_UR50D.pt \
    --output_dir data/embeddings \
    --batch_size 8 \
    --device cuda

# For HPC cluster (Biowulf)
sbatch slurm/generate_embeddings.sh
```

Parameters:
- `--model_path`: Path to ESM-2 checkpoint file (.pt format)
- `--batch_size`: Number of sequences to process at once (adjust based on GPU memory)
- `--device`: cuda or cpu
- `--no_resume`: Don't resume from existing embeddings

The script will:
- Load ESM-2 model from local checkpoint
- Generate 1280-dimensional embeddings separately for host and phage proteins
- Save embeddings to HDF5 files with resumable processing
- Support PyTorch 2.6+ compatibility

Output files:
- `data/embeddings/host_embeddings.h5` - Host protein embeddings ([HDF5 format](#hdf5-file-format))
- `data/embeddings/phage_embeddings.h5` - Phage protein embeddings ([HDF5 format](#hdf5-file-format))

### 3. Monitoring Progress

Check embedding generation progress:
```bash
# View latest log
tail -f logs/embedding_generation_*.log

# Check HPC job status (Biowulf)
squeue -u $USER
```

## Training the Model

### Configuration Validation

Before training, validate your configuration file to catch errors early:

```bash
# Validate the default configuration
python scripts/validate_config.py configs/default_config.yaml

# Validate with verbose output (shows configuration summary)
python scripts/validate_config.py configs/default_config.yaml --verbose
```

The validator checks:
- **Required sections**: model, training, data, evaluation, loss, dataset
- **ESM-2 dimensions**: Ensures input_dim matches ESM-2 model (320, 480, 640, 1280, or 2560)
- **File paths**: Verifies data files and embeddings exist
- **Parameter ranges**: Validates learning rates, dropout, batch sizes, etc.
- **Scheduler settings**: Checks scheduler-specific parameters
- **Device compatibility**: Validates CUDA/CPU/MPS settings

Example output:
```
✓ Configuration file 'configs/default_config.yaml' is valid!

Configuration summary:
  Model type: balanced
  Input dimension: 1280
  Batch size: 32
  Learning rate: 0.0001
  Epochs: 100
  Optimizer: adamw
  Scheduler: warmup_cosine
  Class prior: 0.3
  Negative ratio: 1.0
```

### Running Training

Once configuration is validated, start training:

```bash
# Train with default configuration
python scripts/train.py --config configs/default_config.yaml

# Train with custom experiment name
python scripts/train.py --config configs/default_config.yaml \
    --experiment_name my_experiment

# Resume from checkpoint
python scripts/train.py --config configs/default_config.yaml \
    --checkpoint checkpoints/best_model.pt

# Test only (requires checkpoint)
python scripts/train.py --config configs/default_config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --test_only
```

The training script will:
1. Automatically validate configuration before starting
2. Create output directories if they don't exist
3. Set up logging and experiment tracking
4. Initialize the model and data loaders
5. Run training with early stopping and checkpointing
6. Perform model calibration if enabled
7. Evaluate on test set after training

## Model Architecture

The model uses a two-tower architecture with noisy-OR aggregation:

1. **Input**: Multi-instance bags of protein embeddings (1280-dim from ESM-2)
2. **Two-Tower Encoders**: Separate networks for host and phage proteins
   - Configurable depth: Conservative (1280→1024→512), Balanced (1280→768→512→256), or Aggressive (1280→512→256→128)
   - Layer normalization and dropout for regularization
3. **Pairwise Scoring**: Scaled dot product between all protein pairs with temperature scaling
4. **Noisy-OR Aggregation**: Combine pairwise scores into bag-level prediction
   - Models the probability that at least one protein pair interacts
5. **nnPU Loss**: Non-negative risk estimation for positive-unlabeled learning
   - Handles class prior estimation and risk correction

## Key Features

- **No data leakage**: Careful splitting ensures test RBPs are never seen in training
- **Handles variable protein counts**: Multi-instance learning approach
- **Uncertainty modeling**: Noisy-OR captures interaction uncertainty
- **Scalable**: Checkpoint support and batch processing for large datasets
- **Reproducible**: Fixed seeds and comprehensive logging

## Dependencies

See `requirements.txt` for full list. Key packages:
- PyTorch (deep learning)
- Transformers & fair-esm (ESM-2 models)
- H5py (efficient embedding storage)
- Pandas & NumPy (data processing)
- Scikit-learn (metrics)

## HDF5 File Format

The embedding files use HDF5 (Hierarchical Data Format version 5), a binary format designed for efficient storage of large scientific datasets. Think of it as a filesystem inside a single file.

### Structure of Embedding Files

Each embedding file contains:
```
host_embeddings.h5
├── embeddings (dataset)     # Shape: (n_proteins, 1280) float32
└── hashes (dataset)         # Shape: (n_proteins,) strings (MD5 hashes)
```

### Key Benefits

- **Efficient Storage**: Data is compressed, reducing ~30GB to manageable size
- **Fast Access**: Can read specific batches without loading entire file
- **Memory-Friendly**: Works with datasets larger than RAM via memory mapping
- **Resumable**: Can append new embeddings without rewriting file

### Accessing HDF5 Files

```python
import h5py
import numpy as np

# Read embeddings
with h5py.File('data/embeddings/host_embeddings.h5', 'r') as f:
    # See contents
    print(f.keys())  # ['embeddings', 'hashes']
    
    # Get info
    print(f['embeddings'].shape)  # (2907, 1280)
    
    # Load all data
    all_embeddings = f['embeddings'][:]
    all_hashes = f['hashes'][:].astype(str)
    
    # Load specific items
    first_10 = f['embeddings'][:10]
    
    # Find embedding for specific protein
    target_hash = "abc123..."
    idx = np.where(f['hashes'][:].astype(str) == target_hash)[0][0]
    embedding = f['embeddings'][idx]
```

### Chunking Strategy

The files use `chunks=(100, 1280)` meaning data is stored in 100-row blocks, optimized for batch loading during training where typical batch sizes are 8-32 samples.

## Citation

If you use this code, please cite:
```
[Citation information to be added]
```

## Contact

[Contact information to be added]

## License

[License information to be added]