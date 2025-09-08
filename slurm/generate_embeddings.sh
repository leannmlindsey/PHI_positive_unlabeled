#!/bin/bash
#SBATCH --job-name=esm2_embeddings
#SBATCH --time=24:00:00
#SBATCH --mem=64g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100x:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/embeddings_%j.out
#SBATCH --error=logs/embeddings_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Load required modules (adjust based on Biowulf's available modules)
module load python/3.10
module load cuda/11.8

# Activate conda environment (adjust path as needed)
source /data/$USER/conda/etc/profile.d/conda.sh
conda activate phi_pu_xor

# Set environment variables
export TRANSFORMERS_CACHE=/data/$USER/.cache/huggingface
export HF_HOME=/data/$USER/.cache/huggingface
export TORCH_HOME=/data/$USER/.cache/torch

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p data/processed

# Set the local model path (update this to your actual path)
# You can pass this as an environment variable or command line argument
ESM2_MODEL_PATH="${ESM2_MODEL_PATH:-/gpfs/gsfs12/users/lindseylm/PHAGE_HOST_INTERACTION/esm_models/checkpoints/esm2_t33_650M_UR50D.pt}"

echo "Using ESM-2 model from: $ESM2_MODEL_PATH"

# Run the embedding generation script
python scripts/generate_embeddings.py \
    --data_path data/dedup.phage_marker_rbp_with_phage_entropy.tsv \
    --output_dir data/processed \
    --model_name facebook/esm2_t33_650M_UR50D \
    --model_path "$ESM2_MODEL_PATH" \
    --batch_size 16 \
    --max_length 1024 \
    --device cuda \
    --checkpoint_dir checkpoints \
    --log_dir logs

# Check exit status
if [ $? -eq 0 ]; then
    echo "Embedding generation completed successfully"
else
    echo "Embedding generation failed with exit code $?"
    exit 1
fi