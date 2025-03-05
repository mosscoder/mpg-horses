#!/bin/bash

# Create directories for models and results
mkdir -p models
mkdir -p results/figures
mkdir -p data/cached_datasets

# Set the dataset path
DATASET_PATH="mpg-ranch/horse-detection"
CACHE_FILE="data/cached_datasets/${DATASET_PATH/\//_}.parquet"

# Check if cached dataset exists
if [ -f "$CACHE_FILE" ]; then
    echo "Found cached dataset at $CACHE_FILE"
    echo "Will use cached dataset instead of downloading from Hugging Face"
else
    echo "No cached dataset found. Will download from Hugging Face."
    
    # Check if HUGGINGFACE_TOKEN is set
    if [ -z "$HUGGINGFACE_TOKEN" ]; then
        echo "Warning: HUGGINGFACE_TOKEN environment variable is not set."
        echo "If you need to access private datasets, please set it with:"
        echo "export HUGGINGFACE_TOKEN=your_token_here"
    else
        echo "HUGGINGFACE_TOKEN is set. Will use for authentication."
    fi
fi

# Run the horse detection model training
echo "Running horse detection model training..."
python src/horse_detection.py \
    --dataset_path $DATASET_PATH \
    --use_auth \
    --model_type cnn \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --num_epochs 30 \
    --patience 5 \
    --cache_dir "data/cached_datasets" \
    --save_model \
    --plot_history

echo "Horse detection model training completed. Check results in results/figures/" 