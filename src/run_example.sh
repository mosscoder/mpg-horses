#!/bin/bash

# Create necessary directories
mkdir -p ../models
mkdir -p ../results/figures

# Run the example script
python src/example.py

# Print a message
echo "Example completed! Check the results in ../results/figures/" 