#!/bin/bash
#SBATCH --job-name=phage_host_eval
#SBATCH --time=2:00:00
#SBATCH --mem=32g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100x:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/evaluation_%j.out
#SBATCH --error=logs/evaluation_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

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
mkdir -p evaluation_results

# Parse command line arguments
CHECKPOINT_PATH=${1:-"checkpoints/best_model.pt"}
CONFIG_PATH=${2:-"configs/default_config.yaml"}
OUTPUT_DIR=${3:-"evaluation_results/$(date +%Y%m%d_%H%M%S)"}

echo "Evaluating model"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Config: $CONFIG_PATH"
echo "Output directory: $OUTPUT_DIR"

# Run evaluation
python scripts/evaluate.py \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --output_dir $OUTPUT_DIR

# Check exit status
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "Evaluation failed with exit code $?"
    exit 1
fi