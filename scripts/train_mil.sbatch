#!/bin/bash
#SBATCH --job-name=phi_mil_train
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_mil_%j.out
#SBATCH --error=logs/train_mil_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Load modules if needed
module load python/3.9
module load cuda/11.8

# Activate conda environment
source /data/lindseylm/conda/etc/profile.d/conda.sh
conda activate phi_pos_unlabeled_env

# Create log directory if it doesn't exist
mkdir -p logs
mkdir -p checkpoints
mkdir -p outputs

# Set environment variables for better GPU performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Log system info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Change to project directory
cd /data/lindseylm/PHI_positive_unlabeled

# Run training script
echo "Starting training..."
python scripts/train.py --config configs/default_config.yaml

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully at: $(date)"
else
    echo "Training failed with exit code $? at: $(date)"
fi

# Log final GPU memory usage
echo "Final GPU memory usage:"
nvidia-smi