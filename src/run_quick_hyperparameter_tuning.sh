#!/bin/bash

# Create necessary directories
mkdir -p models
mkdir -p results/figures
mkdir -p results/hyperparameter_tuning
mkdir -p logs
mkdir -p data/cached_datasets

# Set timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/quick_hyperparameter_tuning_${TIMESTAMP}.log"

echo "Running quick hyperparameter tuning for horse detection model..."
echo "Log will be saved to $LOG_FILE"

# Create a parameter grid file with fewer combinations for quick testing
PARAM_GRID_FILE="results/hyperparameter_tuning/quick_param_grid_${TIMESTAMP}.json"

cat > $PARAM_GRID_FILE << EOL
{
    "learning_rate": [0.001, 0.0001],
    "weight_decay": [0.01],
    "dropout_rate": [0.5],
    "batch_size": [64],
    "num_epochs": [10],
    "patience": [3],
    "grad_clip": [1.0],
    "use_class_weights": [true]
}
EOL

echo "Parameter grid saved to $PARAM_GRID_FILE"

# Run the hyperparameter tuning with a smaller subset
python src/horse_detection.py \
  --tune_hyperparams \
  --param_grid_file $PARAM_GRID_FILE \
  --subset_size 1000 \
  --num_workers 4 \
  --seed 42 \
  2>&1 | tee "$LOG_FILE"

echo "Quick hyperparameter tuning complete. Check results in results/hyperparameter_tuning/"

# Note: This will run 2×1×1×1×1×1×1×1 = 2 combinations of hyperparameters.
# Each combination will train a model for up to 10 epochs with early stopping.
# This should complete in a reasonable amount of time for testing purposes. 