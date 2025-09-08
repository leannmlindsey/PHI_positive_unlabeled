#!/bin/bash
#SBATCH --job-name=phage_host_train
#SBATCH --time=48:00:00
#SBATCH --mem=64g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100x:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Load required modules (adjust based on Biowulf's available modules)
module load python/3.10
module load cuda/11.8

# Activate conda environment
source /data/$USER/conda/etc/profile.d/conda.sh
conda activate phi_pu_xor

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME=/data/$USER/.cache/torch
export WANDB_DIR=/data/$USER/.wandb
export WANDB_CACHE_DIR=/data/$USER/.cache/wandb

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p outputs

# Set experiment name with timestamp
EXPERIMENT_NAME="phage_host_$(date +%Y%m%d_%H%M%S)"

echo "Starting training experiment: $EXPERIMENT_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Run training
python scripts/train.py \
    --config configs/default_config.yaml \
    --experiment_name $EXPERIMENT_NAME

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
    
    # Run evaluation on the best model
    echo "Running evaluation on best model..."
    python scripts/evaluate.py \
        --config configs/default_config.yaml \
        --checkpoint checkpoints/best_model.pt \
        --output_dir outputs/${EXPERIMENT_NAME}_evaluation
else
    echo "Training failed with exit code $?"
    exit 1
fi

echo "All tasks completed"