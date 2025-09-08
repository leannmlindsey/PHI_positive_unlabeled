#!/bin/bash

# Pipeline for extracting sequences and generating embeddings

echo "Step 1: Extracting and deduplicating sequences..."
python scripts/extract_sequences.py \
    --data_path data/dedup.phage_marker_rbp_with_phage_entropy.tsv \
    --output_dir data/sequences

if [ $? -ne 0 ]; then
    echo "Sequence extraction failed!"
    exit 1
fi

echo -e "\nStep 2: Generating embeddings..."
python scripts/generate_embeddings_simple.py \
    --host_sequences data/sequences/host_sequences.json \
    --phage_sequences data/sequences/phage_sequences.json \
    --model_path /gpfs/gsfs12/users/lindseylm/PHAGE_HOST_INTERACTION/esm_models/checkpoints/esm2_t33_650M_UR50D.pt \
    --output_dir data/embeddings \
    --batch_size 8 \
    --device cuda

if [ $? -ne 0 ]; then
    echo "Embedding generation failed!"
    exit 1
fi

echo -e "\nPipeline completed successfully!"
echo "Sequence files in: data/sequences/"
echo "Embedding files in: data/embeddings/"