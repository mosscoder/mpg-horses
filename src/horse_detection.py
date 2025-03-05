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
from collections import Counter

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
    Dataset for horse detection.
    """

    def __init__(self, dataset, transform=None, debug=False):
        """
        Initialize the dataset.

        Args:
            dataset: The dataset to use (pandas DataFrame or Hugging Face dataset)
            transform: Optional transform to be applied to the images
            debug (bool): Whether to print debug information
        """
        self.dataset = dataset
        self.debug = debug

        # Identify image columns
        self.image_columns = []
        if isinstance(dataset, pd.DataFrame):
            for col in dataset.columns:
                if (
                    "image" in col.lower()
                    or "tile" in col.lower()
                    or "encoded" in col.lower()
                ):
                    self.image_columns.append(col)

        if self.debug:
            print(f"Identified image columns: {self.image_columns}")

        # Set up transforms
        if transform is None:
            # Default transforms for training
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),  # Resize to larger size for cropping
                    transforms.RandomCrop(224),  # Random crop for more variation
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.3),  # Add vertical flips
                    transforms.RandomRotation(15),  # Increase rotation range
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),  # Add color jitter
                    transforms.RandomAffine(
                        degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                    ),  # Add affine transformations
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    transforms.RandomErasing(
                        p=0.2
                    ),  # Add random erasing for robustness
                ]
            )
        else:
            self.transform = transform

        # Set up test transforms
        self.test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Print dataset info if debug is enabled
        if self.debug:
            if hasattr(dataset, "column_names"):
                print(f"Dataset columns: {dataset.column_names}")
            elif hasattr(dataset, "columns"):
                print(f"Dataset columns: {dataset.columns.tolist()}")
            print(f"Dataset size: {len(dataset)}")

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): The index of the item to get

        Returns:
            tuple: (image, label)
        """
        # Get the item from the dataset
        if isinstance(self.dataset, pd.DataFrame):
            # For pandas DataFrame, use iloc to access by position
            item = self.dataset.iloc[idx]
        else:
            # For Hugging Face datasets or other types
            item = self.dataset[idx]

        # Get the label
        if hasattr(item, "get"):
            # For Hugging Face datasets
            label = item.get("Presence", item.get("label", 0))
        else:
            # For pandas DataFrames
            label = item["Presence"] if "Presence" in item.index else 0

        # Convert label to integer
        if isinstance(label, bool):
            label = 1 if label else 0
        elif isinstance(label, str):
            label = 1 if label.lower() in ["true", "yes", "1"] else 0
        else:
            label = int(label)

        # Get the image
        try:
            # Try to get the image from various possible fields
            image = None

            # For pandas DataFrame
            if isinstance(self.dataset, pd.DataFrame):
                # Try each image column in order of preference
                for col in self.image_columns:
                    if col in item.index and pd.notna(item[col]):
                        image_data = item[col]

                        # Handle base64 encoded images
                        if isinstance(image_data, str) and (
                            image_data.startswith("data:image") or len(image_data) > 100
                        ):
                            try:
                                # Extract base64 content if it's a data URL
                                if image_data.startswith("data:image"):
                                    image_data = image_data.split(",")[1]

                                # Decode base64
                                img_data = base64.b64decode(image_data)
                                image = Image.open(io.BytesIO(img_data))

                                # Convert to RGB if needed
                                if image.mode != "RGB":
                                    image = image.convert("RGB")

                                if self.debug:
                                    print(
                                        f"Successfully loaded image from {col} for item {idx}"
                                    )
                                break
                            except Exception as e:
                                if self.debug:
                                    print(
                                        f"Error decoding base64 from {col} for item {idx}: {str(e)}"
                                    )
                                continue

                        # Handle file paths
                        elif isinstance(image_data, str) and os.path.exists(image_data):
                            try:
                                image = Image.open(image_data)

                                # Convert to RGB if needed
                                if image.mode != "RGB":
                                    image = image.convert("RGB")

                                if self.debug:
                                    print(
                                        f"Successfully loaded image from file {image_data} for item {idx}"
                                    )
                                break
                            except Exception as e:
                                if self.debug:
                                    print(
                                        f"Error loading image from file {image_data} for item {idx}: {str(e)}"
                                    )
                                continue

            # For Hugging Face datasets
            else:
                # Check for image fields
                for field in ["image_base64", "encoded_tile", "image"]:
                    if field in item:
                        image_data = item[field]

                        # Handle base64 encoded images
                        if isinstance(image_data, str) and (
                            image_data.startswith("data:image") or len(image_data) > 100
                        ):
                            try:
                                # Extract base64 content if it's a data URL
                                if image_data.startswith("data:image"):
                                    image_data = image_data.split(",")[1]

                                # Decode base64
                                img_data = base64.b64decode(image_data)
                                image = Image.open(io.BytesIO(img_data))

                                # Convert to RGB if needed
                                if image.mode != "RGB":
                                    image = image.convert("RGB")

                                if self.debug:
                                    print(
                                        f"Successfully loaded image from {field} for item {idx}"
                                    )
                                break
                            except Exception as e:
                                if self.debug:
                                    print(
                                        f"Error decoding base64 from {field} for item {idx}: {str(e)}"
                                    )
                                continue

                        # Handle Hugging Face image type
                        elif hasattr(image_data, "convert_to_pil"):
                            try:
                                image = image_data.convert_to_pil()

                                # Convert to RGB if needed
                                if image.mode != "RGB":
                                    image = image.convert("RGB")

                                if self.debug:
                                    print(
                                        f"Successfully loaded image from {field} for item {idx}"
                                    )
                                break
                            except Exception as e:
                                if self.debug:
                                    print(
                                        f"Error converting image from {field} for item {idx}: {str(e)}"
                                    )
                                continue

            # If no image found, create a blank image
            if image is None:
                if self.debug:
                    print(f"No image found for item {idx}, creating blank image")
                image = Image.new("RGB", (224, 224), color="gray")

            # Apply transforms
            if self.transform is not None:
                image = self.transform(image)

            return image, label

        except Exception as e:
            if self.debug:
                print(f"Error processing item {idx}: {str(e)}")

            # Return a blank image and the label
            blank_image = torch.zeros(3, 224, 224)
            return blank_image, label


def create_image_dataloaders(
    dataset, batch_size=32, test_size=0.2, seed=42, device=None, debug=False
):
    """
    Create data loaders for training and testing.

    Args:
        dataset: The dataset to use
        batch_size (int): Batch size for training
        test_size (float): Proportion of the dataset to use for testing
        seed (int): Random seed for reproducibility
        device (str): Device to use for training
        debug (bool): Whether to enable debug mode

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

    # Create dataset with debug mode to see what's happening
    horse_dataset = HorseDetectionDataset(dataset, debug=debug)

    # Split dataset into train and test sets
    dataset_size = len(horse_dataset)
    test_size_int = int(dataset_size * test_size)
    train_size = dataset_size - test_size_int

    # Get class distribution for stratified split
    # Use a try-except block to handle potential errors
    try:
        # Get labels for stratification
        if isinstance(dataset, pd.DataFrame):
            # For pandas DataFrame, use the Presence column directly
            labels = dataset["Presence"].values
        else:
            # For other dataset types, extract labels from the dataset
            labels = []
            for i in range(min(1000, dataset_size)):  # Sample a subset for efficiency
                try:
                    _, label = horse_dataset[i]
                    labels.append(label)
                except Exception as e:
                    print(f"Error getting label for item {i}: {str(e)}")
                    labels.append(0)  # Default to 0 if there's an error

            # If we sampled, repeat the pattern to match dataset size
            if len(labels) < dataset_size:
                labels = labels * (dataset_size // len(labels) + 1)
                labels = labels[:dataset_size]

        print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

        # Use sklearn for stratified split
        from sklearn.model_selection import train_test_split

        indices = list(range(dataset_size))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, stratify=labels, random_state=seed
        )
    except Exception as e:
        print(f"Error during stratified split: {str(e)}")
        print("Falling back to random split")

        # Fall back to random split
        indices = list(range(dataset_size))
        random.seed(seed)
        random.shuffle(indices)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

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
    Inspect the dataset to understand its structure.

    Args:
        dataset: The dataset to inspect
        num_samples (int): Number of samples to inspect
    """
    print("\n" + "=" * 50)
    print("Dataset Inspection")
    print("=" * 50)

    # Print dataset type and size
    print(f"Dataset type: {type(dataset)}")
    if hasattr(dataset, "column_names"):
        print(f"Dataset columns: {dataset.column_names}")
    elif hasattr(dataset, "columns"):
        print(f"Dataset columns: {dataset.columns.tolist()}")
    print(f"Dataset size: {len(dataset)}")

    # Print label distribution
    if isinstance(dataset, pd.DataFrame) and "Presence" in dataset.columns:
        print(f"Label distribution: {dataset['Presence'].value_counts().to_dict()}")

    # Check for image columns
    image_columns = []
    if isinstance(dataset, pd.DataFrame):
        for col in dataset.columns:
            if "image" in col.lower() or "tile" in col.lower():
                image_columns.append(col)

    print(f"Potential image columns: {image_columns}")

    # Inspect a few samples
    print("\nSample Inspection:")
    for i in range(min(num_samples, len(dataset))):
        print(f"\nSample {i+1}:")
        if isinstance(dataset, pd.DataFrame):
            row = dataset.iloc[i]
            print(f"  Presence: {row.get('Presence', 'N/A')}")

            # Check image columns
            for col in image_columns:
                if pd.notna(row[col]):
                    print(f"  {col}: {type(row[col]).__name__}")
                    if isinstance(row[col], str):
                        if row[col].startswith("data:image"):
                            print(f"    Data URL format, length: {len(row[col])}")
                        elif len(row[col]) > 100:
                            print(f"    Likely base64, length: {len(row[col])}")
                            try:
                                img_data = base64.b64decode(
                                    row[col].split(",")[-1]
                                    if "," in row[col]
                                    else row[col]
                                )
                                img = Image.open(io.BytesIO(img_data))
                                print(
                                    f"    Successfully decoded: {img.format} image, size {img.size}"
                                )
                            except Exception as e:
                                print(f"    Failed to decode: {str(e)}")
                        elif os.path.exists(row[col]):
                            print(
                                f"  {col}: File path (exists: {os.path.exists(row[col])})"
                            )
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
        else:
            # For Hugging Face datasets
            item = dataset[i]
            print(f"  Presence: {item.get('Presence', 'N/A')}")

            # Check for image fields
            for field in item:
                if "image" in field.lower() or "tile" in field.lower():
                    print(f"  {field}: {type(item[field]).__name__}")
                    if isinstance(item[field], str):
                        if item[field].startswith("data:image"):
                            print(f"    Data URL format, length: {len(item[field])}")
                        elif len(item[field]) > 100:
                            print(f"    Likely base64, length: {len(item[field])}")
                            try:
                                img_data = base64.b64decode(
                                    item[field].split(",")[-1]
                                    if "," in item[field]
                                    else item[field]
                                )
                                img = Image.open(io.BytesIO(img_data))
                                print(
                                    f"    Successfully decoded: {img.format} image, size {img.size}"
                                )
                            except Exception as e:
                                print(f"    Failed to decode: {str(e)}")

    print("\n" + "=" * 50)


def create_cnn_model(num_classes=2, model_type="resnet50", pretrained=True):
    """
    Create a CNN model for image classification.

    Args:
        num_classes (int): Number of output classes
        model_type (str): Type of CNN model to use
        pretrained (bool): Whether to use pretrained weights

    Returns:
        nn.Module: The CNN model
    """
    print(f"Creating CNN model: {model_type}")

    if model_type.startswith("resnet"):
        if model_type == "resnet18":
            base_model = models.resnet18(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif model_type == "resnet34":
            base_model = models.resnet34(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif model_type == "resnet50":
            base_model = models.resnet50(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif model_type == "resnet101":
            base_model = models.resnet101(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        else:
            raise ValueError(f"Unsupported ResNet model: {model_type}")

        # Modify the final layer for our classification task
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, num_classes)

    elif model_type.startswith("efficientnet"):
        if model_type == "efficientnet_b0":
            base_model = models.efficientnet_b0(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif model_type == "efficientnet_b1":
            base_model = models.efficientnet_b1(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif model_type == "efficientnet_b2":
            base_model = models.efficientnet_b2(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        else:
            raise ValueError(f"Unsupported EfficientNet model: {model_type}")

        # Modify the final layer for our classification task
        in_features = base_model.classifier[1].in_features
        base_model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return base_model


def create_vit_model(
    num_classes=2, model_name="google/vit-base-patch16-224", pretrained=True
):
    """
    Create a Vision Transformer (ViT) model for image classification.

    Args:
        num_classes (int): Number of output classes
        model_name (str): Name of the ViT model to use
        pretrained (bool): Whether to use pretrained weights

    Returns:
        nn.Module: The ViT model
    """
    print(f"Creating ViT model: {model_name}")

    try:
        from transformers import ViTForImageClassification, ViTConfig

        if pretrained:
            # Load pretrained model
            model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            # Create model with random weights
            config = ViTConfig.from_pretrained(
                model_name,
                num_labels=num_classes,
            )
            model = ViTForImageClassification(config)

        return model

    except ImportError:
        print("Error: transformers library not installed. Using CNN model instead.")
        return create_cnn_model(
            num_classes=num_classes, model_type="resnet50", pretrained=pretrained
        )


def main(args):
    """
    Main function.

    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    set_seed(args.seed)

    # Set up device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Set up logging
    if args.log_file:
        setup_logging(args.log_file)

    # Download and cache the dataset
    dataset = download_and_cache_dataset(
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        force_download=args.force_download,
    )

    print(f"Dataset size: {len(dataset)}")
    if hasattr(dataset, "column_names"):
        print(f"Dataset columns: {dataset.column_names}")
    elif hasattr(dataset, "columns"):
        print(f"Dataset columns: {dataset.columns.tolist()}")

    # Inspect dataset structure
    inspect_dataset(dataset, num_samples=3)

    # Create a subset of the dataset if specified
    if args.subset_size > 0:
        print(f"Creating subset of {args.subset_size} samples")

        # Get label distribution before subsetting
        if isinstance(dataset, pd.DataFrame):
            label_counts = dataset["Presence"].value_counts().to_dict()
        else:
            label_counts = Counter(dataset["Presence"])
        print(f"Label distribution before subsetting: {label_counts}")

        # Create stratified subset
        if isinstance(dataset, pd.DataFrame):
            # For pandas DataFrame
            from sklearn.model_selection import train_test_split

            # Get stratification column
            strat_col = "Presence"

            # Split the dataset
            _, dataset = train_test_split(
                dataset,
                test_size=args.subset_size / len(dataset),
                stratify=dataset[strat_col],
                random_state=args.seed,
            )
        else:
            # For Hugging Face datasets
            dataset = dataset.shuffle(seed=args.seed)
            dataset = dataset.select(range(args.subset_size))

        # Get label distribution after subsetting
        if isinstance(dataset, pd.DataFrame):
            label_counts = dataset["Presence"].value_counts().to_dict()
        else:
            label_counts = Counter(dataset["Presence"])
        print(f"Label distribution after subsetting: {label_counts}")

    # Create dataloaders
    train_dataloader, test_dataloader = create_image_dataloaders(
        dataset=dataset,
        batch_size=args.batch_size,
        test_size=args.test_size,
        num_workers=args.num_workers,
        seed=args.seed,
        debug=True,  # Enable debug mode
    )

    print(f"Train dataloader size: {len(train_dataloader.dataset)}")
    print(f"Test dataloader size: {len(test_dataloader.dataset)}")

    # Create model
    if args.model_type == "cnn":
        model = create_cnn_model(
            num_classes=2,
            model_type=args.cnn_model_type,
            pretrained=args.pretrained,
        )
    elif args.model_type == "vit":
        model = create_vit_model(
            num_classes=2,
            model_type=args.vit_model_type,
            pretrained=args.pretrained,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Print model summary
    print(f"Model type: {args.model_type}")
    if args.model_type == "cnn":
        print(f"CNN model type: {args.cnn_model_type}")
    elif args.model_type == "vit":
        print(f"ViT model type: {args.vit_model_type}")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Move model to device
    model = model.to(device)

    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        verbose=True,
    )

    # Set up loss function
    criterion = nn.CrossEntropyLoss()

    # Train model
    history = train_model(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        patience=args.patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # Save model
    if args.save_model:
        save_model(model, args.model_dir, args.model_name)

    # Plot training history
    if args.plot_history:
        plot_training_history(history, args.figure_dir, args.figure_name)


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
