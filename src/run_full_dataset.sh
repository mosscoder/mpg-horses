#!/bin/bash

# Create necessary directories
mkdir -p models
mkdir -p results/figures
mkdir -p logs
mkdir -p data/cached_datasets

# Set timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/horse_detection_full_${TIMESTAMP}.log"

echo "Running horse detection model training on FULL DATASET..."
echo "Log will be saved to $LOG_FILE"
echo "This may take a while. You can monitor progress with: tail -f $LOG_FILE"

# Run the horse detection script with optimized settings for M3 chip
# Note: No subset_size parameter means using the full dataset
python src/horse_detection.py \
  --batch_size 16 \
  --num_epochs 10 \
  --patience 3 \
  --save_model \
  --plot_history \
  2>&1 | tee "$LOG_FILE"

echo "Training complete. Check results in results/figures/"
echo "Model saved in models/ directory" 