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
import copy
import logging

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
    train_dataloader,
    test_dataloader,
    criterion,
    optimizer,
    scheduler=None,
    device="cpu",
    num_epochs=10,
    patience=3,
    gradient_accumulation_steps=1,
):
    """
    Train a model.

    Args:
        model: The model to train
        train_dataloader: The training dataloader
        test_dataloader: The test dataloader
        criterion: The loss function
        optimizer: The optimizer
        scheduler: The learning rate scheduler
        device: The device to use for training
        num_epochs: The number of epochs to train for
        patience: The number of epochs to wait for improvement before early stopping
        gradient_accumulation_steps: The number of steps to accumulate gradients over

    Returns:
        dict: Training history
    """
    # Initialize variables
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    no_improve_epochs = 0

    # Initialize history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    # Train model
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Use tqdm for progress bar
        train_pbar = tqdm(
            train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"
        )

        # Reset gradients
        optimizer.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(train_pbar):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Scale loss by gradient accumulation steps
            loss = loss / gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Update weights if we've accumulated enough gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (
                batch_idx + 1
            ) == len(train_dataloader):
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update weights
                optimizer.step()

                # Reset gradients
                optimizer.zero_grad()

            # Update metrics
            train_loss += loss.item() * gradient_accumulation_steps
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            # Update progress bar
            train_acc = 100.0 * train_correct / train_total
            train_pbar.set_postfix(
                {"loss": train_loss / (batch_idx + 1), "acc": f"{train_acc:.2f}%"}
            )

            # Clean up memory if using MPS
            if device == "mps":
                del inputs, targets, outputs, loss
                torch.mps.empty_cache()

        # Calculate epoch metrics
        train_loss = train_loss / len(train_dataloader)
        train_acc = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Calculate class-specific metrics
        val_class_counts = {0: 0, 1: 0}
        val_class_correct = {0: 0, 1: 0}

        # Use tqdm for progress bar
        val_pbar = tqdm(
            test_dataloader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"
        )

        with torch.no_grad():
            for inputs, targets in val_pbar:
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Update metrics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                # Update class-specific metrics
                for i in range(targets.size(0)):
                    label = targets[i].item()
                    val_class_counts[label] += 1
                    if predicted[i].item() == label:
                        val_class_correct[label] += 1

                # Update progress bar
                val_acc = 100.0 * val_correct / val_total
                val_pbar.set_postfix(
                    {"loss": val_loss / (val_pbar.n + 1), "acc": f"{val_acc:.2f}%"}
                )

                # Clean up memory if using MPS
                if device == "mps":
                    del inputs, targets, outputs, loss
                    torch.mps.empty_cache()

        # Calculate epoch metrics
        val_loss = val_loss / len(test_dataloader)
        val_acc = 100.0 * val_correct / val_total

        # Print epoch metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Print class-specific metrics
        for class_idx in val_class_counts:
            if val_class_counts[class_idx] > 0:
                class_acc = (
                    100.0 * val_class_correct[class_idx] / val_class_counts[class_idx]
                )
                print(
                    f"Class {class_idx} Val Acc: {class_acc:.2f}% ({val_class_correct[class_idx]}/{val_class_counts[class_idx]})"
                )

        # Calculate precision, recall, and F1 score
        if val_class_correct[1] + (val_class_counts[0] - val_class_correct[0]) > 0:
            precision = val_class_correct[1] / (
                val_class_correct[1] + (val_class_counts[0] - val_class_correct[0])
            )
        else:
            precision = 0

        if val_class_correct[1] + (val_class_counts[1] - val_class_correct[1]) > 0:
            recall = val_class_correct[1] / (
                val_class_correct[1] + (val_class_counts[1] - val_class_correct[1])
            )
        else:
            recall = 0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning rate: {current_lr:.6f}")

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Check if this is the best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Early stopping
        if no_improve_epochs >= patience:
            print(
                f"Early stopping at epoch {epoch + 1} as validation accuracy hasn't improved for {patience} epochs"
            )
            break

        # Print separator
        print("-" * 60)

    # Print training summary
    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best validation accuracy: {best_acc:.2f}% at epoch {best_epoch + 1}")

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return history


def plot_training_history(history, figure_dir, figure_name=None):
    """
    Plot training history.

    Args:
        history (dict): Training history
        figure_dir (str): The directory to save the figure to
        figure_name (str): The name of the figure file
    """
    # Create figure directory if it doesn't exist
    os.makedirs(figure_dir, exist_ok=True)

    # Set default figure name if not provided
    if figure_name is None:
        figure_name = f"training_history_{time.strftime('%Y%m%d_%H%M%S')}.png"

    # Add .png extension if not present
    if not figure_name.endswith(".png"):
        figure_name += ".png"

    # Create figure
    plt.figure(figsize=(12, 10))

    # Plot training and validation loss
    plt.subplot(2, 1, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot training and validation accuracy
    plt.subplot(2, 1, 2)
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    figure_path = os.path.join(figure_dir, figure_name)
    print(f"Saving figure to {figure_path}")
    plt.savefig(figure_path)
    plt.close()


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Horse Detection")

    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mpg-ranch/horse-detection",
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/cached_datasets",
        help="Directory to cache the dataset",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force download of the dataset",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=0,
        help="Size of the subset to use (0 for full dataset)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of the dataset to use for testing",
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
        "--vit_model_type",
        type=str,
        default="vit_base_patch16_224",
        help="Type of ViT model to use",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained model",
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
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for training",
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
        default=1,
        help="Number of steps to accumulate gradients over",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Output arguments
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the model after training",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory to save the model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the model file",
    )
    parser.add_argument(
        "--plot_history",
        action="store_true",
        help="Plot the training history",
    )
    parser.add_argument(
        "--figure_dir",
        type=str,
        default="results/figures",
        help="Directory to save the figure",
    )
    parser.add_argument(
        "--figure_name",
        type=str,
        default=None,
        help="Name of the figure file",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to the log file",
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
    dataset_name, cache_dir="data/cached_datasets", force_download=False
):
    """
    Download and cache a dataset from the Hugging Face Hub.

    Args:
        dataset_name (str): Name of the dataset on the Hugging Face Hub
        cache_dir (str): Directory to cache the dataset
        force_download (bool): Whether to force download the dataset

    Returns:
        pandas.DataFrame: The dataset
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Create cache file path
    dataset_slug = dataset_name.replace("/", "_")
    cache_file = os.path.join(cache_dir, f"{dataset_slug}.parquet")

    # Check if cache file exists
    if os.path.exists(cache_file) and not force_download:
        print(f"Loading dataset from cache: {cache_file}")
        return pd.read_parquet(cache_file)

    # Download dataset
    print(f"Downloading dataset {dataset_name}...")
    try:
        # Try to load from Hugging Face Hub
        dataset = load_dataset(dataset_name)

        # Convert to pandas DataFrame
        if "train" in dataset:
            df = dataset["train"].to_pandas()
        else:
            df = next(iter(dataset.values())).to_pandas()

        # Save to cache
        print(f"Saving dataset to cache: {cache_file}")
        df.to_parquet(cache_file)

        return df
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")

        # Check if cache file exists as fallback
        if os.path.exists(cache_file):
            print(f"Loading dataset from cache as fallback: {cache_file}")
            return pd.read_parquet(cache_file)

        raise e


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

    # Set up loss function
    criterion = nn.CrossEntropyLoss()

    # Train model
    history = train_model(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
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


def get_device(device=None):
    """
    Get the device to use for training.

    Args:
        device (str): The device to use (cuda, mps, or cpu)

    Returns:
        str: The device to use
    """
    if device is not None:
        return device

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def save_model(model, model_dir, model_name=None):
    """
    Save a model to disk.

    Args:
        model: The model to save
        model_dir (str): The directory to save the model to
        model_name (str): The name of the model file
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Set default model name if not provided
    if model_name is None:
        model_name = f"horse_detection_{time.strftime('%Y%m%d_%H%M%S')}.pt"

    # Add .pt extension if not present
    if not model_name.endswith(".pt"):
        model_name += ".pt"

    # Save model
    model_path = os.path.join(model_dir, model_name)
    print(f"Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)


def setup_logging(log_file):
    """
    Set up logging to a file.

    Args:
        log_file (str): The path to the log file
    """
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
