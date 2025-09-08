#!/bin/bash
#SBATCH --job-name=phage_host_predict
#SBATCH --time=1:00:00
#SBATCH --mem=16g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100x:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/prediction_%j.out
#SBATCH --error=logs/prediction_%j.err

# Load required modules
module load python/3.10
module load cuda/11.8

# Activate conda environment
source /data/$USER/conda/etc/profile.d/conda.sh
conda activate phi_pu_xor

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# Create necessary directories
mkdir -p logs
mkdir -p predictions

# Parse command line arguments
INPUT_FILE=${1:-"data/new_interactions.tsv"}
OUTPUT_FILE=${2:-"predictions/predictions_$(date +%Y%m%d_%H%M%S).tsv"}
CHECKPOINT_PATH=${3:-"checkpoints/best_model.pt"}
CONFIG_PATH=${4:-"configs/default_config.yaml"}
EMBEDDINGS_PATH=${5:-"data/processed/protein_embeddings.h5"}

echo "Making predictions"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Model: $CHECKPOINT_PATH"

# Run prediction
python scripts/predict.py \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --embeddings $EMBEDDINGS_PATH \
    batch \
    --input $INPUT_FILE \
    --output $OUTPUT_FILE

# Check exit status
if [ $? -eq 0 ]; then
    echo "Predictions completed successfully"
    echo "Results saved to: $OUTPUT_FILE"
else
    echo "Prediction failed with exit code $?"
    exit 1
fi