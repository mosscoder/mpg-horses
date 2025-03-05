#!/bin/bash

# Create necessary directories
mkdir -p models
mkdir -p results/figures
mkdir -p data/cached_datasets
mkdir -p logs

# Set dataset path
DATASET_PATH="mpg-ranch/horse-detection"
CACHE_FILE="data/cached_datasets/${DATASET_PATH/\//_}.parquet"
LOG_FILE="logs/horse_detection_$(date +%Y%m%d_%H%M%S).log"

# Check if cached dataset exists
if [ -f "$CACHE_FILE" ]; then
    echo "Found cached dataset at $CACHE_FILE"
    echo "Will use cached dataset instead of downloading from Hugging Face"
else
    echo "No cached dataset found at $CACHE_FILE"
    echo "Will download dataset from Hugging Face"
    
    # Check if HUGGINGFACE_TOKEN is set
    if [ -z "$HUGGINGFACE_TOKEN" ]; then
        echo "Warning: HUGGINGFACE_TOKEN environment variable is not set."
        echo "You may need to set it to access private datasets:"
        echo "export HUGGINGFACE_TOKEN=your_token_here"
    else
        echo "HUGGINGFACE_TOKEN is set. Will use for authentication."
    fi
fi

echo "Running horse detection model training..."
echo "Logging output to $LOG_FILE"

# Kill any existing training processes
pkill -f "python src/horse_detection.py" || echo "No existing process found"

# Run the horse detection model training with proper logging
{
    python src/horse_detection.py \
        --dataset_path $DATASET_PATH \
        --use_auth \
        --model_type cnn \
        --batch_size 8 \
        --learning_rate 0.0001 \
        --num_epochs 10 \
        --patience 3 \
        --gradient_accumulation_steps 4 \
        --subset_size 1000 \
        --cache_dir data/cached_datasets \
        --save_model \
        --plot_history
} 2>&1 | tee $LOG_FILE

echo "Horse detection model training completed. Check results in results/figures/" 