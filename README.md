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
│   ├── simple_splitting.py        # Data splitting with RBP deduplication
│   └── generate_embeddings.py    # ESM-2 embedding generation
├── slurm/                         # HPC job submission scripts
├── models/                        # Model architecture (to be implemented)
├── training/                      # Training pipeline (to be implemented)
├── utils/                         # Utility functions
│   └── data_utils.py             # Data processing utilities
├── configs/                       # Configuration files
├── logs/                          # Execution logs
└── checkpoints/                   # Model and process checkpoints
```

## Input Data Description

### Main Data File: `dedup.phage_marker_rbp_with_phage_entropy.tsv`

Tab-separated file with 24,794 known positive interactions containing:

| Column | Description | Format |
|--------|-------------|--------|
| `marker_gene_seq` | Host marker protein sequences | Comma-separated if multiple |
| `rbp_seq` | Phage RBP sequences | Comma-separated if multiple |
| `phage_id` | Unique phage identifier | String |
| `rbp_length` | Length of RBP sequence | Integer |
| `marker_md5` | MD5 hash of marker sequences | Comma-separated if multiple |
| `rbp_md5` | MD5 hash of RBP sequences | Comma-separated if multiple |
| `shannon_entropy` | Sequence entropy measure | Float |
| `n_unique_markers` | Number of unique markers | Integer |

### Data Characteristics
- **All interactions are positive** (prophages found in host genomes)
- **Multi-instance nature**: 
  - Hosts have 1-2 marker proteins
  - Phages have 1-18 RBPs (typically 1-4)
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
python scripts/simple_splitting.py
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

#### Step 2: Generate Protein Embeddings

Generate ESM-2 embeddings for all unique protein sequences:

```bash
# For local testing (if you have GPU)
python scripts/generate_embeddings.py \
    --data_path data/dedup.phage_marker_rbp_with_phage_entropy.tsv \
    --output_dir data/processed \
    --batch_size 8 \
    --device cuda

# For HPC cluster (Biowulf)
sbatch slurm/generate_embeddings.sh
```

Parameters:
- `--batch_size`: Number of sequences to process at once (adjust based on GPU memory)
- `--max_length`: Maximum sequence length (default: 1024)
- `--checkpoint_dir`: Directory for resumable checkpoints
- `--device`: cuda or cpu

The script will:
- Extract all unique protein sequences (~25,000)
- Generate 1280-dimensional ESM-2 embeddings
- Save embeddings to `data/processed/protein_embeddings.h5`
- Support checkpoint/resume for long runs
- Log progress to `logs/`

### 3. Monitoring Progress

Check embedding generation progress:
```bash
# View latest log
tail -f logs/embedding_generation_*.log

# Check HPC job status (Biowulf)
squeue -u $USER
```

## Model Architecture

The model uses a two-tower architecture with noisy-OR aggregation:

1. **Input**: Multi-instance bags of protein embeddings
2. **Two-Tower Encoders**: Separate networks for host and phage proteins
3. **Pairwise Scoring**: Dot product between all protein pairs
4. **Noisy-OR Aggregation**: Combine pairwise scores into bag-level prediction
5. **nnPU Loss**: Non-negative risk estimation for positive-unlabeled learning

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

## Citation

If you use this code, please cite:
```
[Citation information to be added]
```

## Contact

[Contact information to be added]

## License

[License information to be added]