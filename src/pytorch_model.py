#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch Model Training Script

This script implements a structured approach to using PyTorch with a Hugging Face dataset
to achieve the best F1 score. It starts with fundamental PyTorch operations and progresses
toward advanced techniques, including hyperparameter tuning and model optimization.
"""

import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import argparse
from datetime import datetime

# Import utility modules
from data_utils import create_dataloaders, get_input_size, prepare_text_data_for_model
from model_utils import (
    SimpleModel,
    DeepModel,
    train_model_with_early_stopping,
    plot_training_history,
    plot_confusion_matrix,
)


# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")


# Check for available devices (CUDA, MPS, or CPU)
def get_device():
    """Determine the available device for PyTorch."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


# Phase 1: Loading and exploring the dataset
def load_and_explore_dataset(dataset_name, verbose=True):
    """
    Load a dataset from Hugging Face and explore its structure.

    Args:
        dataset_name (str): Name of the dataset on Hugging Face
        verbose (bool): Whether to print dataset information

    Returns:
        datasets.DatasetDict: The loaded dataset
    """
    print(f"Loading dataset: {dataset_name}")
    try:
        dataset = load_dataset(dataset_name)

        if verbose:
            print("\nDataset structure:")
            print(dataset)

            # Print information about each split
            for split in dataset.keys():
                print(f"\n{split} split:")
                print(f"  Number of examples: {len(dataset[split])}")
                print(f"  Features: {dataset[split].features}")

                # Show a sample from each split
                print(f"\nSample from {split} split:")
                sample = dataset[split][0]
                for key, value in sample.items():
                    if isinstance(value, (str, int, float, bool)):
                        print(f"  {key}: {value}")
                    else:
                        print(f"  {key}: {type(value)}")

        return dataset

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


# Phase 2: Data preprocessing
def preprocess_dataset(dataset, test_size=0.2, seed=42):
    """
    Preprocess the dataset by splitting it into train and test sets.

    Args:
        dataset: The dataset to preprocess
        test_size (float): The proportion of the dataset to include in the test split
        seed (int): Random seed for reproducibility

    Returns:
        datasets.DatasetDict: The preprocessed dataset
    """
    # If the dataset doesn't already have a train/test split, create one
    if "test" not in dataset.keys():
        print(f"Creating train/test split with test_size={test_size}")
        dataset = dataset.train_test_split(test_size=test_size, seed=seed)

    return dataset


def save_model(model, model_name, save_dir="../models"):
    """
    Save a trained model.

    Args:
        model: The PyTorch model to save
        model_name (str): Name of the model
        save_dir (str): Directory to save the model
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the model
    save_path = os.path.join(save_dir, f"{model_name}_{timestamp}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def train_simple_model(
    dataset,
    input_cols,
    label_col,
    batch_size=32,
    learning_rate=0.001,
    num_epochs=50,
    patience=5,
):
    """
    Train a simple model on the dataset.

    Args:
        dataset: Hugging Face dataset with 'train' and 'test' splits
        input_cols (list): List of column names to use as input features
        label_col (str): Column name to use as label
        batch_size (int): Batch size for DataLoader
        learning_rate (float): Learning rate for optimizer
        num_epochs (int): Maximum number of epochs to train for
        patience (int): Number of epochs to wait for improvement before stopping

    Returns:
        tuple: (trained model, training history)
    """
    # Get device
    device = get_device()

    # Create DataLoaders
    train_loader, test_loader = create_dataloaders(
        dataset, input_cols=input_cols, label_col=label_col, batch_size=batch_size
    )

    # Get input size
    input_size = get_input_size(dataset, input_cols)
    print(f"Input size: {input_size}")

    # Get number of classes
    num_classes = len(set(dataset["train"][label_col]))
    print(f"Number of classes: {num_classes}")

    # Create model
    model = SimpleModel(input_size=input_size, num_classes=num_classes)
    model = model.to(device)
    print(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model with early stopping
    model, history = train_model_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
    )

    # Plot training history
    plot_training_history(
        train_losses=history["train_losses"],
        val_losses=history["val_losses"],
        f1_scores=history["f1_scores"],
        save_path="../results/figures/simple_model_history.png",
    )

    # Save model
    save_model(model, model_name="simple_model")

    return model, history


def train_deep_model(
    dataset,
    input_cols,
    label_col,
    batch_size=32,
    learning_rate=0.001,
    hidden_size=128,
    dropout_rate=0.2,
    num_epochs=50,
    patience=5,
):
    """
    Train a deep model on the dataset.

    Args:
        dataset: Hugging Face dataset with 'train' and 'test' splits
        input_cols (list): List of column names to use as input features
        label_col (str): Column name to use as label
        batch_size (int): Batch size for DataLoader
        learning_rate (float): Learning rate for optimizer
        hidden_size (int): Size of the hidden layers
        dropout_rate (float): Dropout rate for regularization
        num_epochs (int): Maximum number of epochs to train for
        patience (int): Number of epochs to wait for improvement before stopping

    Returns:
        tuple: (trained model, training history)
    """
    # Get device
    device = get_device()

    # Create DataLoaders
    train_loader, test_loader = create_dataloaders(
        dataset, input_cols=input_cols, label_col=label_col, batch_size=batch_size
    )

    # Get input size
    input_size = get_input_size(dataset, input_cols)
    print(f"Input size: {input_size}")

    # Get number of classes
    num_classes = len(set(dataset["train"][label_col]))
    print(f"Number of classes: {num_classes}")

    # Create model
    model = DeepModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )
    model = model.to(device)
    print(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=3, verbose=True
    )

    # Train model with early stopping
    model, history = train_model_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        scheduler=scheduler,
    )

    # Plot training history
    plot_training_history(
        train_losses=history["train_losses"],
        val_losses=history["val_losses"],
        f1_scores=history["f1_scores"],
        save_path="../results/figures/deep_model_history.png",
    )

    # Save model
    save_model(model, model_name="deep_model")

    return model, history


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PyTorch Model Training Script")

    # Dataset arguments
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset on Hugging Face"
    )
    parser.add_argument(
        "--input_cols",
        type=str,
        nargs="+",
        required=True,
        help="List of column names to use as input features",
    )
    parser.add_argument(
        "--label_col", type=str, required=True, help="Column name to use as label"
    )

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["simple", "deep"],
        default="simple",
        help="Type of model to train",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Size of the hidden layers (for deep model)",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        help="Dropout rate for regularization (for deep model)",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Maximum number of epochs to train for",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs to wait for improvement before stopping",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


# Main function to run the script
def main():
    """Main function to run the script."""
    # Parse command line arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    print("Starting PyTorch model training script")

    # Load and explore the dataset
    dataset = load_and_explore_dataset(args.dataset)

    if dataset is not None:
        # Preprocess the dataset
        dataset = preprocess_dataset(dataset)
        print("\nPreprocessed dataset structure:")
        print(dataset)

        # Create results directory if it doesn't exist
        os.makedirs("../results/figures", exist_ok=True)

        # Train model based on model_type
        if args.model_type == "simple":
            model, history = train_simple_model(
                dataset=dataset,
                input_cols=args.input_cols,
                label_col=args.label_col,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                patience=args.patience,
            )
        else:  # deep model
            model, history = train_deep_model(
                dataset=dataset,
                input_cols=args.input_cols,
                label_col=args.label_col,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                hidden_size=args.hidden_size,
                dropout_rate=args.dropout_rate,
                num_epochs=args.num_epochs,
                patience=args.patience,
            )

        print(f"Best F1 score: {history['best_f1']:.4f}")
    else:
        print("Failed to load dataset. Please check the dataset name and try again.")


if __name__ == "__main__":
    main()
