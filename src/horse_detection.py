#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Horse Detection Model

This script applies the PyTorch model training framework to the Horse Detection dataset.
It trains models to detect the presence of horses in aerial imagery.

Note on authentication:
- The script can use Hugging Face authentication for private datasets
- Tokens are read from environment variables or the Hugging Face CLI cache
- NEVER hardcode tokens in this script or print them to the console
"""

import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from datetime import datetime
import base64
from PIL import Image
import io
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import traceback
import random

# Import utility modules
from data_utils import create_dataloaders, get_input_size
from model_utils import (
    SimpleModel,
    DeepModel,
    train_model_with_early_stopping,
    plot_training_history,
    plot_confusion_matrix,
)
from pytorch_model import set_seed, get_device, save_model


# CNN model for image classification
class CNNModel(nn.Module):
    """A CNN model for image classification."""

    def __init__(self, num_classes=2, pretrained=True):
        """
        Initialize the model.

        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
        """
        super(CNNModel, self).__init__()

        # Load a pretrained ResNet18 model
        self.model = models.resnet18(pretrained=pretrained)

        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)


# Custom dataset class for horse detection
class HorseDetectionDataset(torch.utils.data.Dataset):
    """
    Dataset for horse detection from aerial imagery.
    """

    def __init__(self, data, transform=None, debug=False):
        self.data = data
        self.transform = transform
        self.error_count = 0
        self.total_count = 0
        # Create a default image to use when loading fails
        self.default_image = np.ones((224, 224, 3), dtype=np.uint8) * 128  # Gray image
        # Print dataset column names for debugging
        if hasattr(data, "column_names"):
            print(f"Dataset columns: {data.column_names}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.total_count += 1

        try:
            # Get the image and label
            item = self.data[idx]

            # Use 'Presence' as the label field (instead of 'label')
            if "Presence" in item:
                label = item["Presence"]
            else:
                print(
                    f"Warning: No 'Presence' field at index {idx}. Available fields: {list(item.keys())}"
                )
                label = 0  # Default label

            # Handle image loading based on the format
            if "encoded_tile" in item and isinstance(item["encoded_tile"], str):
                # This is likely a base64 encoded image
                try:
                    # Try to decode base64 image
                    image_data = base64.b64decode(item["encoded_tile"])

                    # Check if the decoded data is valid
                    if len(image_data) < 100:  # Arbitrary small size check
                        print(
                            f"Warning: Very small image data at index {idx} (size: {len(image_data)})"
                        )
                        image = Image.fromarray(self.default_image)
                    else:
                        # Try to open the image from bytes
                        image = Image.open(io.BytesIO(image_data))
                        # Convert to RGB if needed
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                except Exception as e:
                    self.error_count += 1
                    error_rate = (self.error_count / self.total_count) * 100
                    print(
                        f"Error decoding base64 image at index {idx}: {e} (Error rate: {error_rate:.2f}%)"
                    )
                    image = Image.fromarray(self.default_image)

            elif "tile_path" in item:
                # This is a path to an image file
                try:
                    image_path = item["tile_path"]
                    if os.path.exists(image_path):
                        image = Image.open(image_path)
                        # Convert to RGB if needed
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                    else:
                        self.error_count += 1
                        error_rate = (self.error_count / self.total_count) * 100
                        print(
                            f"Image file not found at index {idx}: {image_path} (Error rate: {error_rate:.2f}%)"
                        )
                        image = Image.fromarray(self.default_image)
                except Exception as e:
                    self.error_count += 1
                    error_rate = (self.error_count / self.total_count) * 100
                    print(
                        f"Error loading image file at index {idx}: {e} (Error rate: {error_rate:.2f}%)"
                    )
                    image = Image.fromarray(self.default_image)

            elif "image" in item and isinstance(item["image"], str):
                # This is likely a base64 encoded image
                try:
                    # Try to decode base64 image
                    image_data = base64.b64decode(item["image"])

                    # Check if the decoded data is valid
                    if len(image_data) < 100:  # Arbitrary small size check
                        print(
                            f"Warning: Very small image data at index {idx} (size: {len(image_data)})"
                        )
                        image = Image.fromarray(self.default_image)
                    else:
                        # Try to open the image from bytes
                        image = Image.open(io.BytesIO(image_data))
                        # Convert to RGB if needed
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                except Exception as e:
                    self.error_count += 1
                    error_rate = (self.error_count / self.total_count) * 100
                    print(
                        f"Error decoding base64 image at index {idx}: {e} (Error rate: {error_rate:.2f}%)"
                    )
                    image = Image.fromarray(self.default_image)

            else:
                # Unknown format, use default image
                self.error_count += 1
                error_rate = (self.error_count / self.total_count) * 100
                print(
                    f"Unknown image format at index {idx}. Available fields: {list(item.keys())} (Error rate: {error_rate:.2f}%)"
                )
                image = Image.fromarray(self.default_image)

            # Apply transformations if any
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            self.error_count += 1
            error_rate = (self.error_count / self.total_count) * 100
            print(
                f"Unexpected error at index {idx}: {e} (Error rate: {error_rate:.2f}%)"
            )

            # Return a default image and label
            default_image = Image.fromarray(self.default_image)
            if self.transform:
                default_image = self.transform(default_image)

            # Use 0 as default label (assuming binary classification)
            return default_image, 0


def create_image_dataloaders(dataset, batch_size=16, test_size=0.2, random_state=42):
    """
    Create data loaders for training and testing.

    Args:
        dataset (pd.DataFrame): The dataset to use
        batch_size (int): The batch size
        test_size (float): The proportion of the dataset to use for testing
        random_state (int): The random state for reproducibility

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Print dataset info
    print(f"Dataset type: {type(dataset)}")
    print(f"Dataset columns: {dataset.columns.tolist()}")
    print(f"Dataset size: {len(dataset)}")

    # Print label distribution
    print(f"Label distribution: {dataset['Presence'].value_counts()}")

    # Split dataset into training and testing sets
    train_df, test_df = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset["Presence"],
    )

    print(f"Training set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")

    # Create datasets
    train_dataset = HorseDetectionDataset(
        train_df,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    test_dataset = HorseDetectionDataset(
        test_df,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    # Determine number of workers based on device
    # When using MPS (Apple Silicon), set num_workers=0 to avoid multiprocessing issues
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        num_workers = 0
        print("Using MPS device: setting num_workers=0 to avoid multiprocessing issues")
    else:
        num_workers = min(os.cpu_count(), 4)
        print(f"Using num_workers={num_workers} for data loading")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def train_model(
    model,
    train_loader,
    test_loader,
    learning_rate=0.0001,
    num_epochs=30,
    patience=5,
    device=None,
):
    """
    Train a model on the given data loaders.

    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): The training data loader
        test_loader (DataLoader): The testing data loader
        learning_rate (float): The learning rate
        num_epochs (int): The number of epochs to train for
        patience (int): The number of epochs to wait for improvement before early stopping
        device (str): The device to train on (cuda, mps, or cpu)

    Returns:
        tuple: (model, history) where history is a dictionary of training metrics
    """
    # Set device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Using device: {device}")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize variables for early stopping
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    # Initialize history dictionary
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Use tqdm for progress bar
        train_iterator = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
        )

        for inputs, labels in train_iterator:
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device).long()  # Ensure labels are long tensors

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar
            train_iterator.set_postfix(loss=loss.item())

        # Calculate average training loss and accuracy
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            # Use tqdm for progress bar
            val_iterator = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

            for inputs, labels in val_iterator:
                # Move data to device
                inputs = inputs.to(device)
                labels = labels.to(device).long()  # Ensure labels are long tensors

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Update statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Update progress bar
                val_iterator.set_postfix(loss=loss.item())

        # Calculate average validation loss and accuracy
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Print epoch statistics
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load the best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def plot_training_history(history, output_path):
    """
    Plot the training history.

    Args:
        history (dict): Dictionary containing training metrics
        output_path (str): Path to save the plot
    """
    try:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        ax1.plot(history["train_loss"], label="Train Loss")
        ax1.plot(history["val_loss"], label="Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True)

        # Plot accuracy
        ax2.plot(history["train_acc"], label="Train Accuracy")
        ax2.plot(history["val_acc"], label="Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True)

        # Adjust layout and save figure
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"Training history plot saved to {output_path}")

    except Exception as e:
        print(f"Error plotting training history: {str(e)}")
        traceback.print_exc()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Horse Detection Model")

    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="mpg-ranch/horse-detection",
        help="Path to the dataset on Hugging Face or local directory",
    )
    parser.add_argument(
        "--use_auth",
        action="store_true",
        help="Whether to use authentication for Hugging Face",
    )
    parser.add_argument(
        "--use_local_dataset",
        action="store_true",
        help="Whether to use a local dataset instead of Hugging Face",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/cached_datasets",
        help="Directory to cache the dataset",
    )

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="cnn",
        choices=["cnn"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=30, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for early stopping"
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )

    # Output arguments
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/figures",
        help="Directory to save results",
    )
    parser.add_argument(
        "--models_dir", type=str, default="models", help="Directory to save models"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Whether to save the trained model",
    )
    parser.add_argument(
        "--plot_history",
        action="store_true",
        help="Whether to plot the training history",
    )

    return parser.parse_args()


def download_and_cache_dataset(
    dataset_path, use_auth=False, cache_dir="data/cached_datasets"
):
    """
    Download a dataset from Hugging Face and cache it locally.

    Args:
        dataset_path (str): Path to the dataset on Hugging Face
        use_auth (bool): Whether to use authentication
        cache_dir (str): Directory to cache the dataset

    Returns:
        pd.DataFrame: The cached dataset
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Generate a filename for the cached dataset
    cache_file = os.path.join(cache_dir, f"{dataset_path.replace('/', '_')}.parquet")

    # Check if the dataset is already cached
    if os.path.exists(cache_file):
        print(f"Loading dataset from cache: {cache_file}")
        try:
            # Load the cached dataset
            df = pd.read_parquet(cache_file)
            print(f"Loaded cached dataset with {len(df)} samples")
            return df
        except Exception as e:
            print(f"Error loading cached dataset: {str(e)}")
            print("Will download dataset again")

    # Download the dataset from Hugging Face
    print(f"Downloading dataset from Hugging Face: {dataset_path}")
    try:
        # Check if we need to use authentication
        if use_auth:
            # Get the Hugging Face token from environment variable
            token = os.environ.get("HUGGINGFACE_TOKEN")
            if not token:
                print("Warning: HUGGINGFACE_TOKEN environment variable not set")
                print("Attempting to download dataset without authentication")
                dataset = load_dataset(dataset_path)
            else:
                print("Using authentication token for Hugging Face")
                dataset = load_dataset(dataset_path, token=token)
        else:
            dataset = load_dataset(dataset_path)

        # Convert to pandas DataFrame
        print("Converting dataset to pandas DataFrame")
        if isinstance(dataset, dict):
            # If the dataset has splits, use the 'train' split
            if "train" in dataset:
                df = dataset["train"].to_pandas()
            else:
                # Otherwise, use the first split
                first_key = list(dataset.keys())[0]
                df = dataset[first_key].to_pandas()
        else:
            df = dataset.to_pandas()

        # Print dataset information
        print(f"Downloaded dataset with {len(df)} samples")
        print(f"Dataset columns: {df.columns.tolist()}")

        # Ensure the dataset has the required columns
        if "Presence" not in df.columns:
            print("Warning: 'Presence' column not found in dataset")
            if "label" in df.columns:
                print("Renaming 'label' column to 'Presence'")
                df["Presence"] = df["label"]

        # Check for image data
        image_column = None
        for col in ["image_base64", "encoded_tile", "image"]:
            if col in df.columns:
                image_column = col
                break

        if image_column is None:
            print("Warning: No image column found in dataset")
        else:
            print(f"Using '{image_column}' as image column")
            # If the image column is not 'image_base64', rename it
            if image_column != "image_base64":
                df["image_base64"] = df[image_column]

        # Save the dataset to cache
        print(f"Saving dataset to cache: {cache_file}")
        df.to_parquet(cache_file)

        return df

    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        traceback.print_exc()
        raise


def main():
    """
    Main function to run the horse detection model.
    """
    args = parse_args()

    # Set random seed for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    print(f"Random seed set to {args.random_seed}")

    # Set device
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed_all(args.random_seed)
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Create directories for results and models if they don't exist
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    try:
        # Download and cache the dataset
        print(f"Loading dataset from {args.dataset_path}")
        if args.use_auth:
            print("Using authentication for Hugging Face")
            dataset = download_and_cache_dataset(
                args.dataset_path, use_auth=True, cache_dir=args.cache_dir
            )
        else:
            dataset = download_and_cache_dataset(
                args.dataset_path, use_auth=False, cache_dir=args.cache_dir
            )

        # Create data loaders
        print("Creating data loaders")
        train_loader, test_loader = create_image_dataloaders(
            dataset,
            batch_size=args.batch_size,
            test_size=0.2,
            random_state=args.random_seed,
        )

        # Create model
        print(f"Creating {args.model_type} model")
        if args.model_type == "cnn":
            model = CNNModel(num_classes=2, pretrained=True)
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")

        # Train the model
        model, history = train_model(
            model,
            train_loader,
            test_loader,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            patience=args.patience,
            device=device,
        )

        # Save model if requested
        if args.save_model:
            model_path = os.path.join(
                args.models_dir, f"horse_detection_{args.model_type}.pt"
            )
            print(f"Saving model to {model_path}")
            torch.save(model.state_dict(), model_path)

        # Plot training history if requested
        if args.plot_history:
            plot_path = os.path.join(
                args.results_dir, f"horse_detection_{args.model_type}_history.png"
            )
            print(f"Plotting training history to {plot_path}")
            plot_training_history(history, plot_path)

        print("Horse detection model training completed")

    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
