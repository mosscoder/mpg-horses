#!/bin/bash

# Create necessary directories
mkdir -p models
mkdir -p results/figures
mkdir -p logs
mkdir -p data/cached_datasets

# Set timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/horse_detection_${TIMESTAMP}.log"

echo "Running horse detection model training..."
echo "Log will be saved to $LOG_FILE"

# Run the horse detection script with optimized settings for M3 chip
python src/horse_detection.py \
  --batch_size 64 \
  --num_epochs 25 \
  --patience 8 \
  --learning_rate 0.0003 \
  --weight_decay 0.02 \
  --dropout_rate 0.5 \
  --grad_clip 1.0 \
  --use_class_weights \
  --subset_size 2000 \
  --save_model \
  --plot_history \
  2>&1 | tee "$LOG_FILE"

echo "Training complete. Check results in results/figures/" 