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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import base64
import io
import traceback
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from tqdm import tqdm
from torch.optim import Adam
import copy
from sklearn.model_selection import train_test_split
import zipfile
from sklearn.metrics import confusion_matrix, classification_report


# Set up device
def get_device():
    """
    Determine the available device for PyTorch.

    Returns:
        torch.device: The device to use for training
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


class HorseDetectionDataset(Dataset):
    """Dataset for horse detection."""

    def __init__(self, dataframe, transform=None):
        """
        Initialize the dataset with a dataframe containing image data and labels.

        Args:
            dataframe: Pandas DataFrame with image data and labels
            transform: Optional transform to be applied to images
        """
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.

        Args:
            idx: Index of the item to retrieve

        Returns:
            tuple: (image, label) where label is 1 if horses are present, 0 otherwise
        """
        try:
            # Get the row from the dataframe
            row = self.df.iloc[idx]

            # Try to load the image from the encoded_tile field
            img_data = None
            if "encoded_tile" in row and row["encoded_tile"] is not None:
                img_data = row["encoded_tile"]

                # Handle ZIP file containing NumPy arrays
                try:
                    with zipfile.ZipFile(io.BytesIO(img_data), "r") as zip_ref:
                        if "data.npy" in zip_ref.namelist():
                            with zip_ref.open("data.npy") as f:
                                # Load the NumPy array
                                arr = np.load(f, allow_pickle=True)

                                # Convert the array to an image
                                # The array shape is (4, 390, 390) - RGBA format with channels first
                                # We need to transpose it to (390, 390, 4) for PIL
                                if arr.shape[0] == 4 and len(arr.shape) == 3:
                                    # Transpose from (C, H, W) to (H, W, C)
                                    arr = np.transpose(arr, (1, 2, 0))

                                    # Convert to RGB if needed
                                    if arr.shape[2] == 4:  # RGBA
                                        img = Image.fromarray(arr, "RGBA").convert(
                                            "RGB"
                                        )
                                    else:
                                        img = Image.fromarray(arr)
                                else:
                                    # If shape is unexpected, try to create image directly
                                    img = Image.fromarray(arr)
                except Exception as e:
                    print(f"Error processing ZIP file for item {idx}: {str(e)}")
                    # Try other methods if ZIP processing fails
                    img = None

                # If ZIP processing failed, try other methods
                if img is None:
                    # If img_data is binary data, try to open directly
                    if isinstance(img_data, bytes):
                        try:
                            img = Image.open(io.BytesIO(img_data))
                        except Exception:
                            # If that fails, try to decode as base64
                            try:
                                base64_str = img_data.decode("utf-8", errors="ignore")
                                if base64_str.startswith("data:"):
                                    base64_str = base64_str.split(",", 1)[1]
                                decoded_data = base64.b64decode(base64_str)
                                img = Image.open(io.BytesIO(decoded_data))
                            except Exception:
                                img = None
                    # If it's a string, it's likely base64-encoded
                    elif isinstance(img_data, str):
                        try:
                            if img_data.startswith("data:"):
                                img_data = img_data.split(",", 1)[1]
                            decoded_data = base64.b64decode(img_data)
                            img = Image.open(io.BytesIO(decoded_data))
                        except Exception:
                            img = None
                    else:
                        img = None

            # If encoded_tile failed, try image_base64
            if (
                img is None
                and "image_base64" in row
                and row["image_base64"] is not None
            ):
                img_data = row["image_base64"]

                # Handle ZIP file containing NumPy arrays
                try:
                    with zipfile.ZipFile(io.BytesIO(img_data), "r") as zip_ref:
                        if "data.npy" in zip_ref.namelist():
                            with zip_ref.open("data.npy") as f:
                                # Load the NumPy array
                                arr = np.load(f, allow_pickle=True)

                                # Convert the array to an image
                                # The array shape is (4, 390, 390) - RGBA format with channels first
                                # We need to transpose it to (390, 390, 4) for PIL
                                if arr.shape[0] == 4 and len(arr.shape) == 3:
                                    # Transpose from (C, H, W) to (H, W, C)
                                    arr = np.transpose(arr, (1, 2, 0))

                                    # Convert to RGB if needed
                                    if arr.shape[2] == 4:  # RGBA
                                        img = Image.fromarray(arr, "RGBA").convert(
                                            "RGB"
                                        )
                                    else:
                                        img = Image.fromarray(arr)
                                else:
                                    # If shape is unexpected, try to create image directly
                                    img = Image.fromarray(arr)
                except Exception as e:
                    print(f"Error processing ZIP file for item {idx}: {str(e)}")
                    # Try other methods if ZIP processing fails
                    img = None

                # If ZIP processing failed, try other methods
                if img is None:
                    # If img_data is binary data, try to open directly
                    if isinstance(img_data, bytes):
                        try:
                            img = Image.open(io.BytesIO(img_data))
                        except Exception:
                            # If that fails, try to decode as base64
                            try:
                                base64_str = img_data.decode("utf-8", errors="ignore")
                                if base64_str.startswith("data:"):
                                    base64_str = base64_str.split(",", 1)[1]
                                decoded_data = base64.b64decode(base64_str)
                                img = Image.open(io.BytesIO(decoded_data))
                            except Exception:
                                img = None
                    # If it's a string, it's likely base64-encoded
                    elif isinstance(img_data, str):
                        try:
                            if img_data.startswith("data:"):
                                img_data = img_data.split(",", 1)[1]
                            decoded_data = base64.b64decode(img_data)
                            img = Image.open(io.BytesIO(decoded_data))
                        except Exception:
                            img = None
                    else:
                        img = None

            # If no image was successfully loaded, create a blank image
            if img is None:
                img = Image.new("RGB", (224, 224), color="gray")

            # Apply transformations if specified
            if self.transform:
                img = self.transform(img)

            # Get the label (presence of horses)
            label = int(row["Presence"])

            return img, label

        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            # Return a blank image and 0 label in case of error
            blank_img = Image.new("RGB", (224, 224), color="gray")
            if self.transform:
                blank_img = self.transform(blank_img)
            return blank_img, 0


def inspect_dataset(dataset, num_samples=3):
    """
    Inspect the dataset structure to understand the format of the data.

    Args:
        dataset: The dataset to inspect (DataFrame or HorseDetectionDataset)
        num_samples: Number of samples to inspect
    """
    print("\n===== DATASET INSPECTION =====")

    # If it's a HorseDetectionDataset, get the underlying dataframe
    if isinstance(dataset, HorseDetectionDataset):
        df = dataset.df
        print(f"Dataset type: HorseDetectionDataset with {len(dataset)} samples")
    else:
        df = dataset
        print(f"Dataset type: {type(dataset).__name__} with {len(df)} samples")

    # Print column names
    print(f"\nColumns: {df.columns.tolist()}")

    # Check for image columns
    image_columns = [
        col for col in df.columns if "image" in col.lower() or "tile" in col.lower()
    ]
    if image_columns:
        print(f"\nPotential image columns: {image_columns}")

    # Print label distribution if it's a DataFrame with 'Presence' column
    if isinstance(df, pd.DataFrame) and "Presence" in df.columns:
        print(f"\nLabel distribution: {df['Presence'].value_counts().to_dict()}")

    # Sample a few rows to inspect
    print(f"\nInspecting {num_samples} random samples:")
    sample_indices = np.random.choice(len(df), min(num_samples, len(df)), replace=False)

    for i, idx in enumerate(sample_indices):
        print(f"\nSample {i+1} (index {idx}):")
        row = df.iloc[idx]

        # Print non-image columns
        for col in df.columns:
            if col not in image_columns and not isinstance(
                row[col], (bytes, bytearray)
            ):
                print(f"  {col}: {row[col]}")

        # Check image data format
        for col in image_columns:
            if pd.notna(row[col]):
                img_data = row[col]
                data_type = type(img_data).__name__

                # Check if it's a data URL
                is_data_url = False
                if isinstance(img_data, str) and img_data.startswith("data:image"):
                    is_data_url = True
                    print(f"  {col}: Data URL (length: {len(img_data)})")
                # Check if it's base64 encoded
                elif isinstance(img_data, str):
                    print(f"  {col}: String (length: {len(img_data)})")
                    # Try to show the beginning of the string
                    if len(img_data) > 0:
                        print(
                            f"    Starts with: {img_data[:min(30, len(img_data))]}..."
                        )
                # Check if it's bytes
                elif isinstance(img_data, (bytes, bytearray)):
                    print(f"  {col}: Binary data (length: {len(img_data)})")
                else:
                    print(f"  {col}: {data_type}")
            else:
                print(f"  {col}: None/NaN")

    print("\n===== END OF INSPECTION =====\n")


def create_model(num_classes=2, dropout_rate=0.5):
    """Create a ResNet50 model for classification with regularization to prevent overfitting."""
    model = models.resnet50(weights="DEFAULT")

    # Freeze early layers
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False

    # Replace the final fully connected layer with dropout for regularization
    num_features = model.fc.in_features

    # For binary classification with BCEWithLogitsLoss, we need a single output
    if num_classes == 2:
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),  # Add dropout with specified probability
            nn.Linear(num_features, 1),  # Single output for binary classification
        )
    else:
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),  # Add dropout with specified probability
            nn.Linear(num_features, num_classes),
        )

    return model


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=10,
    patience=3,
    learning_rate=0.0005,
    weight_decay=0.01,
    grad_clip=1.0,
    class_weights=None,
    save_best=True,
    model_path="models/best_model.pth",
):
    """
    Train the model with early stopping based on validation accuracy.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on ('cuda', 'mps', or 'cpu')
        num_epochs: Maximum number of epochs to train
        patience: Number of epochs to wait for improvement before stopping
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay (L2 regularization) for optimizer
        grad_clip: Maximum norm of gradients for clipping
        class_weights: Optional tensor of class weights for loss function
        save_best: Whether to save the best model during training
        model_path: Path to save the best model

    Returns:
        dict: Training history with loss and accuracy metrics
    """
    # Move model to device
    model = model.to(device)

    # Define loss function and optimizer
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=class_weights
        )  # Use class weights if provided
    else:
        criterion = (
            nn.BCEWithLogitsLoss()
        )  # Binary Cross Entropy with Logits for binary classification

    optimizer = Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )  # Add L2 regularization with weight decay

    # Learning rate scheduler to reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,  # More aggressive reduction (from 0.5 to 0.3)
        patience=3,  # Reduce patience to detect plateaus earlier
        verbose=True,
        min_lr=1e-6,  # Set a minimum learning rate
    )

    # Initialize variables for early stopping
    best_val_acc = 0.0
    epochs_no_improve = 0

    # Initialize history dictionary
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("Training model...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Create progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}\nTraining")

        for inputs, labels in train_pbar:
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Reshape outputs and labels for BCEWithLogitsLoss
            outputs = outputs.view(-1)  # Flatten to [batch_size]
            labels = labels.float()  # Convert to float for BCEWithLogitsLoss

            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()

            # Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # Calculate statistics
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0).int()  # Threshold at 0 for binary prediction
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar
            train_pbar.set_postfix(
                {
                    "loss": train_loss / train_total,
                    "acc": 100 * train_correct / train_total,
                }
            )

        # Calculate epoch statistics
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Create progress bar for validation
        val_pbar = tqdm(val_loader, desc="Validation")

        with torch.no_grad():
            for inputs, labels in val_pbar:
                # Move data to device
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)

                # Reshape outputs and labels for BCEWithLogitsLoss
                outputs = outputs.view(-1)  # Flatten to [batch_size]
                labels = labels.float()  # Convert to float for BCEWithLogitsLoss

                loss = criterion(outputs, labels)

                # Calculate statistics
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0).int()  # Threshold at 0 for binary prediction
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Update progress bar
                val_pbar.set_postfix(
                    {"loss": val_loss / val_total, "acc": 100 * val_correct / val_total}
                )

        # Calculate epoch statistics
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = 100 * val_correct / val_total

        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

        # Update history
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        # Check for early stopping
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            epochs_no_improve = 0
            # Save the best model weights
            best_model_weights = copy.deepcopy(model.state_dict())

            # Save the best model if requested
            if save_best:
                print(
                    f"Saving best model with validation accuracy: {best_val_acc:.2f}%"
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": best_model_weights,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": best_val_acc,
                        "val_loss": epoch_val_loss,
                    },
                    model_path,
                )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model weights
    model.load_state_dict(best_model_weights)

    return history


def plot_history(history, save_path=None):
    """Plot training history."""
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")

    plt.show()


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test set.

    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: Device to evaluate on ('cuda', 'mps', or 'cpu')

    Returns:
        tuple: (test_loss, test_accuracy)
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize variables
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    # Define loss function
    criterion = nn.BCEWithLogitsLoss()

    # Create confusion matrix
    all_labels = []
    all_predictions = []

    # Create progress bar for testing
    test_pbar = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for inputs, labels in test_pbar:
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Reshape outputs and labels for BCEWithLogitsLoss
            outputs = outputs.view(-1)  # Flatten to [batch_size]
            labels = labels.float()  # Convert to float for BCEWithLogitsLoss

            loss = criterion(outputs, labels)

            # Calculate statistics
            test_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0).int()  # Threshold at 0 for binary prediction
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            # Update progress bar
            test_pbar.set_postfix(
                {"loss": test_loss / test_total, "acc": 100 * test_correct / test_total}
            )

            # Store labels and predictions for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate test statistics
    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = 100 * test_correct / test_total

    # Print test statistics
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

    # Print confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("\nConfusion Matrix:")
    print(cm)

    # Print classification report
    print("\nClassification Report:")
    print(
        classification_report(
            all_labels, all_predictions, target_names=["No Horse", "Horse"]
        )
    )

    return test_loss, test_accuracy


def main():
    """Main function to run the horse detection model."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a horse detection model")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="Patience for early stopping"
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="Size of subset to use (for testing)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0005,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (L2 regularization) for optimizer",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.5,
        help="Dropout rate for regularization",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Maximum norm for gradient clipping",
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Use class weights to handle imbalanced data",
    )
    parser.add_argument(
        "--save_model", action="store_true", help="Save the trained model"
    )
    parser.add_argument(
        "--plot_history", action="store_true", help="Plot training history"
    )
    args = parser.parse_args()

    try:
        # Check if cached dataset exists
        dataset_path = "data/cached_datasets/mpg-ranch_horse-detection.parquet"
        if not os.path.exists(dataset_path):
            print(f"Dataset not found at {dataset_path}")
            print("Please run the data preparation script first.")
            return

        # Load dataset
        df = pd.read_parquet(dataset_path)
        print(f"Dataset size: {len(df)}")
        print(f"Label distribution: {dict(df['Presence'].value_counts())}")

        # Inspect dataset
        inspect_dataset(df, num_samples=3)

        # Create subset if specified
        if args.subset_size:
            print(f"Creating subset of {args.subset_size} samples")
            # Create a stratified sample based on the presence of horses
            df_horses = df[df["Presence"] == 1]
            df_no_horses = df[df["Presence"] == 0]

            # Calculate how many samples of each class to include
            n_horses = min(len(df_horses), args.subset_size // 2)
            n_no_horses = min(len(df_no_horses), args.subset_size // 2)

            # Sample from each class
            df_horses_sample = df_horses.sample(n_horses, random_state=42)
            df_no_horses_sample = df_no_horses.sample(n_no_horses, random_state=42)

            # Combine the samples
            df = pd.concat([df_horses_sample, df_no_horses_sample])

            print(f"Subset size: {len(df)}")
            print(f"Subset label distribution: {dict(df['Presence'].value_counts())}")

        # Define transforms for training and validation
        train_transform = transforms.Compose(
            [
                transforms.Resize(
                    (256, 256)
                ),  # Resize larger than final size for random cropping
                transforms.RandomResizedCrop(
                    224, scale=(0.8, 1.0)
                ),  # Random crop with zoom
                transforms.RandomHorizontalFlip(
                    p=0.5
                ),  # Horizontal flip with 50% probability
                transforms.RandomVerticalFlip(
                    p=0.5
                ),  # Vertical flip with 50% probability
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),  # Color jittering
                transforms.RandomRotation(15),  # Random rotation up to 15 degrees
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Validation transform without augmentation
        val_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Split into train and test sets
        train_df, test_df = train_test_split(
            df, test_size=args.test_size, stratify=df["Presence"], random_state=42
        )

        # Further split train into train and validation
        train_df, val_df = train_test_split(
            train_df, test_size=0.15, stratify=train_df["Presence"], random_state=42
        )

        # Create datasets
        train_dataset = HorseDetectionDataset(train_df, transform=train_transform)
        val_dataset = HorseDetectionDataset(val_df, transform=val_transform)
        test_dataset = HorseDetectionDataset(test_df, transform=val_transform)

        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

        # Calculate class weights for imbalanced dataset
        class_counts = train_df["Presence"].value_counts().to_dict()
        print(f"Class distribution in training set: {class_counts}")

        # Calculate weight for positive class (presence of horses)
        # Formula: weight = n_samples / (n_classes * n_samples_for_class)
        if args.use_class_weights and 1 in class_counts and 0 in class_counts:
            n_samples = len(train_df)
            n_classes = len(class_counts)
            pos_weight = n_samples / (n_classes * class_counts[1])
            class_weights = torch.tensor([pos_weight])
            print(f"Using class weight for positive class: {pos_weight:.4f}")
        else:
            class_weights = None
            print(
                "Not using class weights"
                + (
                    " (disabled by command line)"
                    if not args.use_class_weights
                    else " (missing class in training data)"
                )
            )

        # Create data loaders optimized for M3 chip
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,  # Parallel loading
            pin_memory=True,  # Pin memory for faster transfer to GPU
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,  # Use multiple workers for parallel loading
            pin_memory=True,  # Faster data transfer to GPU
        )

        # Create model
        model = create_model(num_classes=2, dropout_rate=args.dropout_rate)
        print(
            f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
        )

        # Train the model
        history = train_model(
            model,
            train_loader,
            val_loader,
            device=get_device(),
            num_epochs=args.num_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            class_weights=class_weights,  # Pass class weights to training function
            save_best=args.save_model,
            model_path="models/best_model.pth",
        )

        # Evaluate model on test set
        print("\nEvaluating model on test set...")
        device = get_device()
        test_loss, test_accuracy = evaluate_model(model, test_loader, device)

        # Save model if specified
        if args.save_model:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_path = f"models/horse_detection_{timestamp}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

        # Plot history if specified
        if args.plot_history:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plot_path = f"results/figures/horse_detection_history_{timestamp}.png"
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plot_history(history, save_path=plot_path)
            print(f"Training history plot saved to {plot_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"Error: {str(e)}")
        traceback.print_exc()
