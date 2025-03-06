#!/bin/bash

# Create necessary directories
mkdir -p models
mkdir -p results/figures
mkdir -p results/hyperparameter_tuning
mkdir -p logs
mkdir -p data/cached_datasets

# Set timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/hyperparameter_tuning_${TIMESTAMP}.log"

echo "Running hyperparameter tuning for horse detection model..."
echo "Log will be saved to $LOG_FILE"

# Create a parameter grid file
PARAM_GRID_FILE="results/hyperparameter_tuning/param_grid_${TIMESTAMP}.json"

cat > $PARAM_GRID_FILE << EOL
{
    "learning_rate": [0.001, 0.0003, 0.0001],
    "weight_decay": [0.02, 0.01, 0.001],
    "dropout_rate": [0.3, 0.5, 0.7],
    "batch_size": [32, 64],
    "num_epochs": [15],
    "patience": [5],
    "grad_clip": [1.0],
    "use_class_weights": [true, false]
}
EOL

echo "Parameter grid saved to $PARAM_GRID_FILE"

# Run the hyperparameter tuning
python src/horse_detection.py \
  --tune_hyperparams \
  --param_grid_file $PARAM_GRID_FILE \
  --subset_size 2000 \
  --num_workers 4 \
  --seed 42 \
  2>&1 | tee "$LOG_FILE"

echo "Hyperparameter tuning complete. Check results in results/hyperparameter_tuning/"

# Note: This will run 3×3×3×2×1×1×1×2 = 108 combinations of hyperparameters.
# Each combination will train a model for up to 15 epochs with early stopping.
# The entire process may take several hours to complete.
# 
# To run a smaller grid, you can modify the param_grid_file to include fewer values.
# For example, to run a quick test with just 4 combinations:
# {
#     "learning_rate": [0.001, 0.0001],
#     "weight_decay": [0.01],
#     "dropout_rate": [0.5],
#     "batch_size": [64],
#     "num_epochs": [10],
#     "patience": [3],
#     "grad_clip": [1.0],
#     "use_class_weights": [true]
# } 