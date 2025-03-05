#!/bin/bash

# Create directories for models, results, and cached datasets
mkdir -p models
mkdir -p results/figures
mkdir -p data/cached_datasets
mkdir -p logs

# Set dataset path
DATASET_PATH="mpg-ranch/horse-detection"

# Set cache file path
CACHE_FILE="data/cached_datasets/${DATASET_PATH/\//_}.parquet"

# Check if cached dataset exists
if [ -f "$CACHE_FILE" ]; then
    echo "Found cached dataset at $CACHE_FILE. Will use cached dataset."
else
    echo "No cached dataset found. Will download from Hugging Face."
    
    # Check if HUGGINGFACE_TOKEN is set
    if [ -z "$HUGGINGFACE_TOKEN" ]; then
        echo "Warning: HUGGINGFACE_TOKEN environment variable not set."
        echo "You may need to set it if the dataset requires authentication:"
        echo "export HUGGINGFACE_TOKEN=your_token_here"
    else
        echo "HUGGINGFACE_TOKEN is set. Will use for authentication."
    fi
fi

# Get timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/horse_detection_${TIMESTAMP}.log"

# Run the horse detection model
echo "Running horse detection model training. Log will be saved to $LOG_FILE"
python src/horse_detection.py \
    --model_type cnn \
    --cnn_model_type resnet50 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --num_epochs 10 \
    --patience 3 \
    --gradient_accumulation_steps 2 \
    --subset_size 0 \
    --save_model \
    --plot_history \
    --seed 42 \
    2>&1 | tee "$LOG_FILE"

echo "Horse detection model training completed. Check results in results/figures/" 