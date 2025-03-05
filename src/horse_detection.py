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
import warnings
import time

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

# Suppress specific warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*The given NumPy array is not writable.*"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*Palette images with Transparency.*"
)


# CNN model for image classification
class CNNModel(nn.Module):
    """
    CNN model for horse detection.
    """

    def __init__(self, num_classes=2, pretrained=True, model_type="resnet18"):
        """
        Initialize the model.

        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            model_type (str): Type of model to use (resnet18, resnet50)
        """
        super(CNNModel, self).__init__()

        # Select the base model
        if model_type == "resnet50":
            # Load a pretrained ResNet50 model
            self.model = models.resnet50(pretrained=pretrained)
        else:
            # Default to ResNet18
            self.model = models.resnet18(pretrained=pretrained)

        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)


# Vision Transformer model for horse detection
class ViTModel(nn.Module):
    """
    Vision Transformer model for horse detection.
    """

    def __init__(self, num_classes=2, pretrained=True):
        """
        Initialize the model.

        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
        """
        super(ViTModel, self).__init__()

        # Load a pretrained ViT model
        self.model = models.vit_b_16(pretrained=pretrained)

        # Replace the final classification head
        self.model.heads = nn.Linear(self.model.hidden_dim, num_classes)

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)


# Custom dataset class for horse detection
class HorseDetectionDataset(torch.utils.data.Dataset):
    """
    Dataset for horse detection from aerial imagery.
    """

    def __init__(self, data, transform=None, debug=False):
        """
        Initialize the dataset.

        Args:
            data (pd.DataFrame): The dataset
            transform (callable, optional): Optional transform to be applied on a sample
            debug (bool): Whether to print debug information
        """
        self.data = data
        self.transform = transform
        self.debug = debug
        self.total_count = 0
        self.error_count = 0

        # Create a default image (black square)
        self.default_image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Add a text label to the default image
        if debug:
            print(f"Initialized dataset with {len(data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to get

        Returns:
            tuple: (image, label)
        """
        self.total_count += 1

        try:
            # Get the row from the dataframe
            row = self.data.iloc[idx]

            # Get the label (Presence column)
            label = int(row["Presence"])

            # Try to get the image from base64 encoding first
            image = None
            error_message = ""

            # Try to load from encoded_tile first (most likely to contain the image)
            if "encoded_tile" in row and pd.notna(row["encoded_tile"]):
                try:
                    # Decode base64 image
                    base64_content = row["encoded_tile"]
                    if base64_content.startswith("data:image"):
                        base64_content = base64_content.split(",", 1)[1]

                    img_data = base64.b64decode(base64_content)
                    image = Image.open(io.BytesIO(img_data))

                    # Convert to RGB if needed
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                except Exception as e:
                    error_message += f"encoded_tile decode error: {str(e)}. "

            # Try image_base64 next
            if (
                image is None
                and "image_base64" in row
                and pd.notna(row["image_base64"])
            ):
                try:
                    # Decode base64 image
                    base64_content = row["image_base64"]
                    if base64_content.startswith("data:image"):
                        base64_content = base64_content.split(",", 1)[1]

                    img_data = base64.b64decode(base64_content)
                    image = Image.open(io.BytesIO(img_data))

                    # Convert to RGB if needed
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                except Exception as e:
                    error_message += f"image_base64 decode error: {str(e)}. "

            # If base64 failed, try to load from file path
            if image is None and "tile_path" in row and pd.notna(row["tile_path"]):
                try:
                    # Try to load from file path
                    image_path = row["tile_path"]
                    if os.path.exists(image_path):
                        image = Image.open(image_path)

                        # Convert to RGB if needed
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                    else:
                        error_message += f"File not found: {image_path}. "
                except Exception as e:
                    error_message += f"File load error: {str(e)}. "

            # If both methods failed, create a default image
            if image is None:
                if self.debug:
                    print(f"Error loading image at index {idx}: {error_message}")

                # Create a default image (black square with label text)
                image = Image.fromarray(self.default_image)

            # Apply transformations if specified
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            if self.debug:
                print(f"Unexpected error at index {idx}: {str(e)}")

            # Return a default image and label
            default_image = Image.fromarray(self.default_image)
            if self.transform:
                default_image = self.transform(default_image)

            # Use 0 as default label (assuming binary classification)
            return default_image, 0


def create_image_dataloaders(
    dataset, batch_size=32, test_size=0.2, seed=42, device=None
):
    """
    Create data loaders for training and testing.

    Args:
        dataset: The dataset to use
        batch_size (int): Batch size for training
        test_size (float): Proportion of the dataset to use for testing
        seed (int): Random seed for reproducibility
        device (str): Device to use for training

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Determine the number of workers based on the device
    if device == "mps":
        # MPS (Apple Silicon) doesn't work well with multiprocessing
        num_workers = 0
    else:
        # Use multiple workers for CPU and CUDA
        num_workers = min(os.cpu_count(), 4)

    print(f"Using {num_workers} workers for data loading")

    # Create dataset
    horse_dataset = HorseDetectionDataset(dataset)

    # Split dataset into train and test sets
    dataset_size = len(horse_dataset)
    test_size_int = int(dataset_size * test_size)
    train_size = dataset_size - test_size_int

    # Get class distribution for stratified split
    labels = [horse_dataset[i][1] for i in range(dataset_size)]

    # Use sklearn for stratified split
    from sklearn.model_selection import train_test_split

    indices = list(range(dataset_size))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=seed
    )

    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    # Create data loaders with prefetching for better performance
    train_loader = DataLoader(
        horse_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )

    test_loader = DataLoader(
        horse_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )

    # Print dataset split information
    print(f"Dataset split: {train_size} train, {test_size_int} test")

    return train_loader, test_loader


def train_model(
    model,
    train_loader,
    test_loader,
    learning_rate=0.0001,
    num_epochs=30,
    patience=5,
    device=None,
    gradient_accumulation_steps=4,
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
        gradient_accumulation_steps (int): Number of steps to accumulate gradients

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

    # Set up optimizer and loss function
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    criterion = nn.CrossEntropyLoss()

    # Initialize variables for early stopping
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    # Initialize history dictionary
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    # Enable mixed precision training if available
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Use tqdm for progress bar
        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
            leave=True,
        )

        optimizer.zero_grad()  # Zero gradients at the beginning of epoch

        for batch_idx, (inputs, targets) in enumerate(train_pbar):
            # Move inputs and targets to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass with mixed precision if available
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss = loss / gradient_accumulation_steps  # Scale loss

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (
                    batch_idx + 1
                ) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Standard forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / gradient_accumulation_steps  # Scale loss

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (
                    batch_idx + 1
                ) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            # Calculate metrics
            train_loss += loss.item() * gradient_accumulation_steps
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            # Update progress bar
            train_acc = 100.0 * train_correct / train_total
            train_pbar.set_postfix(
                {
                    "loss": train_loss / (batch_idx + 1),
                    "acc": train_acc,
                }
            )

            # Clean up memory if using MPS
            if device == "mps":
                # Explicitly delete tensors to free memory
                del inputs, targets, outputs, loss
                if batch_idx % 10 == 0:  # Every 10 batches
                    torch.mps.empty_cache()

        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(
                test_loader,
                desc=f"Epoch {epoch+1}/{num_epochs} [Val]",
                leave=True,
            )

            for inputs, targets in val_pbar:
                # Move inputs and targets to device
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                # Calculate metrics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                # Update progress bar
                val_acc = 100.0 * val_correct / val_total
                val_pbar.set_postfix(
                    {
                        "loss": val_loss / (val_pbar.n + 1),
                        "acc": val_acc,
                    }
                )

                # Clean up memory if using MPS
                if device == "mps":
                    del inputs, targets, outputs, loss

            # Clean up cache after validation
            if device == "mps":
                torch.mps.empty_cache()

        # Calculate epoch metrics
        val_loss = val_loss / len(test_loader)
        val_acc = 100.0 * val_correct / val_total

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Print epoch results
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"Validation loss improved to {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
            )

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
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
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Horse Detection Model")

    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="mpg-ranch/horse-detection",
        help="Path to the dataset on Hugging Face Hub",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/cached_datasets",
        help="Directory to cache the dataset",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=0,
        help="Number of samples to use (0 for full dataset)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to use for testing",
    )

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="cnn",
        choices=["cnn", "vit"],
        help="Type of model to use",
    )
    parser.add_argument(
        "--cnn_model_type",
        type=str,
        default="resnet50",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
        ],
        help="Type of CNN model to use",
    )
    parser.add_argument(
        "--vit_model_name",
        type=str,
        default="google/vit-base-patch16-224",
        help="Name of ViT model to use",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Number of epochs to wait for improvement before early stopping",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of steps to accumulate gradients",
    )

    # Output arguments
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/figures",
        help="Directory to save results",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory to save models",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Whether to save the model",
    )
    parser.add_argument(
        "--plot_history",
        action="store_true",
        help="Whether to plot the training history",
    )

    # Misc arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--use_auth",
        action="store_true",
        help="Whether to use authentication for Hugging Face Hub",
    )

    return parser.parse_args()


def set_seed(seed):
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def download_and_cache_dataset(
    dataset_path, cache_dir="data/cached_datasets", use_auth=False
):
    """
    Download and cache a dataset from Hugging Face Hub.

    Args:
        dataset_path (str): Path to the dataset on Hugging Face Hub
        cache_dir (str): Directory to cache the dataset
        use_auth (bool): Whether to use authentication for Hugging Face Hub

    Returns:
        Dataset: The downloaded dataset
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Create a cache file path
    cache_file = os.path.join(cache_dir, f"{dataset_path.replace('/', '_')}.parquet")

    # Check if the dataset is already cached
    if os.path.exists(cache_file):
        print(f"Loading dataset from cache: {cache_file}")
        dataset = pd.read_parquet(cache_file)
        print(f"Loaded cached dataset with {len(dataset)} samples")
        return dataset

    # Download the dataset from Hugging Face Hub
    print(f"Downloading dataset from Hugging Face Hub: {dataset_path}")
    try:
        dataset = load_dataset(dataset_path, use_auth_token=use_auth)

        # Convert to pandas DataFrame and save to cache
        if isinstance(dataset, dict) and "train" in dataset:
            df = dataset["train"].to_pandas()
        else:
            df = dataset.to_pandas()

        print(f"Saving dataset to cache: {cache_file}")
        df.to_parquet(cache_file)
        print(f"Dataset cached with {len(df)} samples")
        return df
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        raise


def inspect_dataset(dataset, num_samples=5):
    """
    Inspect the dataset to verify image encoding.

    Args:
        dataset (pd.DataFrame): The dataset to inspect
        num_samples (int): Number of samples to inspect
    """
    print(f"\n{'='*50}")
    print("DATASET INSPECTION")
    print(f"{'='*50}")
    print(f"Dataset type: {type(dataset)}")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Dataset columns: {dataset.columns.tolist()}")

    # Check for image columns
    image_columns = []
    for col in dataset.columns:
        if "image" in col.lower() or "tile" in col.lower() or "encoded" in col.lower():
            image_columns.append(col)

    print(f"\nPotential image columns: {image_columns}")

    # Check for label column
    if "Presence" in dataset.columns:
        print(f"\nLabel column 'Presence' found")
        print(f"Label distribution:\n{dataset['Presence'].value_counts()}")
    else:
        print("\nWARNING: Label column 'Presence' not found!")

    # Inspect a few samples
    print(f"\nInspecting {num_samples} random samples:")
    sample_indices = np.random.choice(
        len(dataset), min(num_samples, len(dataset)), replace=False
    )

    for i, idx in enumerate(sample_indices):
        print(f"\nSample {i+1}/{num_samples} (Index {idx}):")
        row = dataset.iloc[idx]

        # Print non-image columns
        for col in dataset.columns:
            if col not in image_columns and not pd.isna(row[col]):
                print(f"  {col}: {row[col]}")

        # Check image columns
        for col in image_columns:
            if pd.isna(row[col]):
                print(f"  {col}: None")
            elif isinstance(row[col], str) and row[col].startswith(
                ("data:image", "/9j/", "iVBOR")
            ):
                print(f"  {col}: Base64 encoded image (length: {len(row[col])})")
                # Try to decode and verify
                try:
                    # Extract base64 content if it's a data URL
                    base64_content = row[col]
                    if base64_content.startswith("data:image"):
                        base64_content = base64_content.split(",", 1)[1]

                    # Decode and check if it's a valid image
                    img_data = base64.b64decode(base64_content)
                    img = Image.open(io.BytesIO(img_data))
                    print(
                        f"    Successfully decoded: {img.format} image, size {img.size}"
                    )
                except Exception as e:
                    print(f"    Failed to decode: {str(e)}")
            elif isinstance(row[col], str) and os.path.exists(row[col]):
                print(f"  {col}: File path (exists: {os.path.exists(row[col])})")
                try:
                    img = Image.open(row[col])
                    print(
                        f"    Successfully loaded: {img.format} image, size {img.size}"
                    )
                except Exception as e:
                    print(f"    Failed to load: {str(e)}")
            else:
                print(
                    f"  {col}: {type(row[col]).__name__} (length: {len(str(row[col]))})"
                )

    print(f"\n{'='*50}")


def main(args):
    """
    Main function to run the horse detection model.

    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    set_seed(args.seed)

    # Set device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Download and cache dataset
    dataset = download_and_cache_dataset(
        dataset_path=args.dataset_path,
        cache_dir=args.cache_dir,
    )

    # Print dataset info
    print(f"Dataset size: {len(dataset)}")
    if hasattr(dataset, "column_names"):
        print(f"Dataset columns: {dataset.column_names}")
    elif hasattr(dataset, "columns"):
        print(f"Dataset columns: {dataset.columns.tolist()}")

    # Create subset if specified
    if args.subset_size > 0:
        print(f"Using subset of {args.subset_size} samples")
        # Ensure stratified sampling by presence
        if hasattr(dataset, "value_counts"):
            presence_counts = dataset["Presence"].value_counts()
        else:
            presence_counts = dataset["Presence"].value_counts().to_dict()
        print(f"Original class distribution: {presence_counts}")

        # Calculate stratified sample sizes
        total_samples = len(dataset)
        subset_size = min(args.subset_size, total_samples)

        # Create stratified subset
        if hasattr(dataset, "train_test_split"):
            dataset = dataset.train_test_split(
                test_size=subset_size / total_samples,
                stratify_by_column="Presence",
                seed=args.seed,
            )["test"]
        else:
            # For pandas DataFrame
            from sklearn.model_selection import train_test_split

            _, dataset = train_test_split(
                dataset,
                test_size=subset_size / total_samples,
                stratify=dataset["Presence"],
                random_state=args.seed,
            )

        # Verify stratification
        if hasattr(dataset, "value_counts"):
            subset_presence_counts = dataset["Presence"].value_counts()
        else:
            subset_presence_counts = dataset["Presence"].value_counts().to_dict()
        print(f"Subset class distribution: {subset_presence_counts}")
    else:
        print("Using full dataset")

    # Create dataloaders
    train_loader, test_loader = create_image_dataloaders(
        dataset=dataset,
        batch_size=args.batch_size,
        test_size=args.test_size,
        seed=args.seed,
        device=device,
    )

    # Create model
    if args.model_type == "cnn":
        model = create_cnn_model(
            num_classes=2,
            model_type=args.cnn_model_type,
            pretrained=True,
        )
    elif args.model_type == "vit":
        model = create_vit_model(
            num_classes=2,
            model_name=args.vit_model_name,
            pretrained=True,
        )
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    # Print model summary
    print(f"Model type: {args.model_type}")
    if args.model_type == "cnn":
        print(f"CNN model type: {args.cnn_model_type}")
    elif args.model_type == "vit":
        print(f"ViT model name: {args.vit_model_name}")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    print("Training model...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,
        device=device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # Save model if specified
    if args.save_model:
        model_path = os.path.join(
            args.models_dir, f"horse_detection_{args.model_type}.pt"
        )
        print(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)

    # Plot training history if specified
    if args.plot_history:
        history_path = os.path.join(
            args.results_dir, f"horse_detection_{args.model_type}_history.png"
        )
        print(f"Plotting training history to {history_path}")
        plot_training_history(history, history_path)

    # Evaluate model on test set
    print("Evaluating model on test set...")
    model.eval()
    test_correct = 0
    test_total = 0

    # Calculate class distribution in test set
    test_class_counts = {0: 0, 1: 0}
    test_class_correct = {0: 0, 1: 0}

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            # Update overall metrics
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

            # Update class-specific metrics
            for i in range(targets.size(0)):
                label = targets[i].item()
                test_class_counts[label] += 1
                if predicted[i].item() == label:
                    test_class_correct[label] += 1

            # Clean up memory if using MPS
            if device == "mps":
                del inputs, targets, outputs

    # Calculate and print metrics
    test_acc = 100.0 * test_correct / test_total
    print(f"Test accuracy: {test_acc:.2f}%")

    # Print class-specific metrics
    for class_idx in test_class_counts:
        if test_class_counts[class_idx] > 0:
            class_acc = (
                100.0 * test_class_correct[class_idx] / test_class_counts[class_idx]
            )
            print(
                f"Class {class_idx} accuracy: {class_acc:.2f}% ({test_class_correct[class_idx]}/{test_class_counts[class_idx]})"
            )

    # Calculate confusion matrix
    print("Confusion matrix:")
    print(
        f"TN: {test_class_correct[0]}, FP: {test_class_counts[0] - test_class_correct[0]}"
    )
    print(
        f"FN: {test_class_counts[1] - test_class_correct[1]}, TP: {test_class_correct[1]}"
    )

    # Calculate precision, recall, and F1 score
    if test_class_correct[1] + (test_class_counts[0] - test_class_correct[0]) > 0:
        precision = test_class_correct[1] / (
            test_class_correct[1] + (test_class_counts[0] - test_class_correct[0])
        )
    else:
        precision = 0

    if test_class_correct[1] + (test_class_counts[1] - test_class_correct[1]) > 0:
        recall = test_class_correct[1] / (
            test_class_correct[1] + (test_class_counts[1] - test_class_correct[1])
        )
    else:
        recall = 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")

    return model, history


if __name__ == "__main__":
    try:
        # Parse arguments
        args = parse_args()

        # Create directories for results and models if they don't exist
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(args.models_dir, exist_ok=True)
        os.makedirs(args.cache_dir, exist_ok=True)

        # Run main function
        main(args)

        print("Horse detection model training completed successfully")
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()
