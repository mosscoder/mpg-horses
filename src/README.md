# PyTorch Model Training Framework

This directory contains a framework for training PyTorch models on Hugging Face datasets with a focus on optimizing F1 score.

## Overview

The framework is designed to be modular and extensible, with separate modules for:

- Data preprocessing and loading
- Model definitions
- Training and evaluation utilities
- Visualization and metrics

## Files

- `pytorch_model.py`: Main script for training models with command-line arguments
- `model_utils.py`: Model definitions and training utilities
- `data_utils.py`: Data preprocessing and loading utilities
- `example.py`: Example script demonstrating how to use the framework

## Usage

### Running the Example Script

The example script demonstrates how to use the framework with the Iris dataset:

```bash
python example.py
```

This will:
1. Load the Iris dataset from Hugging Face
2. Train a simple linear model and a deep neural network
3. Evaluate both models and compare their performance
4. Save the trained models and visualizations

### Training a Model with Custom Dataset

To train a model on your own dataset, use the `pytorch_model.py` script with appropriate command-line arguments:

```bash
python pytorch_model.py --dataset your_dataset_name --input_cols feature1 feature2 --label_col target --model_type deep
```

#### Command-Line Arguments

- Dataset arguments:
  - `--dataset`: Name of the dataset on Hugging Face (required)
  - `--input_cols`: List of column names to use as input features (required)
  - `--label_col`: Column name to use as label (required)

- Model arguments:
  - `--model_type`: Type of model to train (`simple` or `deep`, default: `simple`)
  - `--hidden_size`: Size of the hidden layers for deep model (default: 128)
  - `--dropout_rate`: Dropout rate for regularization for deep model (default: 0.2)

- Training arguments:
  - `--batch_size`: Batch size for DataLoader (default: 32)
  - `--learning_rate`: Learning rate for optimizer (default: 0.001)
  - `--num_epochs`: Maximum number of epochs to train for (default: 50)
  - `--patience`: Number of epochs to wait for improvement before stopping (default: 5)
  - `--seed`: Random seed for reproducibility (default: 42)

## Extending the Framework

### Adding a New Model

To add a new model, define it in `model_utils.py` by creating a new class that inherits from `nn.Module`:

```python
class YourModel(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(YourModel, self).__init__()
        # Define your model architecture
        
    def forward(self, x):
        # Define the forward pass
        return output
```

Then, add a new function in `pytorch_model.py` to train your model:

```python
def train_your_model(dataset, input_cols, label_col, ...):
    # Implementation similar to train_simple_model or train_deep_model
    # ...
    model = YourModel(input_size=input_size, num_classes=num_classes)
    # ...
    return model, history
```

### Working with Text Data

For text data, use the `prepare_text_data_for_model` function in `data_utils.py`:

```python
train_loader, test_loader, input_size = prepare_text_data_for_model(
    dataset=dataset,
    tokenizer_name='bert-base-uncased',
    text_column='text',
    label_column='label',
    max_length=128,
    batch_size=16
)
```

## Best Practices

1. **Reproducibility**: Always set a random seed for reproducible results.
2. **Early Stopping**: Use early stopping to prevent overfitting.
3. **Learning Rate**: The learning rate is one of the most important hyperparameters. Start with 0.001 and adjust as needed.
4. **Evaluation**: Always evaluate your model on a separate test set.
5. **F1 Score**: For imbalanced datasets, F1 score is a better metric than accuracy.
6. **Visualization**: Visualize training history and confusion matrices to understand model performance.

## Dependencies

- PyTorch
- Hugging Face Datasets and Transformers
- scikit-learn
- matplotlib
- seaborn
- tqdm
- numpy
- pandas 