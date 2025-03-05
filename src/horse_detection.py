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
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from tqdm import tqdm

# Set up device
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"Using device: {device}")


class HorseDetectionDataset(Dataset):
    """Dataset for horse detection."""

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get row
        row = self.dataframe.iloc[idx]

        # Get label (1 if horse present, 0 if not)
        label = int(row["Presence"])

        # Try to get image from base64 encoding
        try:
            if "image_base64" in row and pd.notna(row["image_base64"]):
                img_data = row["image_base64"]
                # Handle data URL format
                if img_data.startswith("data:image"):
                    img_data = img_data.split(",")[1]
                # Decode base64
                image = Image.open(io.BytesIO(base64.b64decode(img_data)))
            elif "encoded_tile" in row and pd.notna(row["encoded_tile"]):
                img_data = row["encoded_tile"]
                if img_data.startswith("data:image"):
                    img_data = img_data.split(",")[1]
                image = Image.open(io.BytesIO(base64.b64decode(img_data)))
            else:
                # Create a blank image if no image data found
                print(f"No image found for item {idx}, creating blank image")
                image = Image.new("RGB", (224, 224), color="gray")

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            # Return a blank image and the label
            blank_image = torch.zeros(3, 224, 224)
            return blank_image, label


def create_model(num_classes=2):
    """Create a ResNet50 model for classification."""
    model = models.resnet50(weights="DEFAULT")

    # Freeze early layers
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False

    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def train_model(model, train_loader, val_loader, num_epochs=10, patience=3):
    """Train the model."""
    # Move model to device
    model = model.to(device)

    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Initialize variables for training
    best_val_acc = 0.0
    best_model_weights = model.state_dict().copy()
    patience_counter = 0

    # Training history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)

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

            # Clean up memory if using MPS
            if device.type == "mps":
                torch.mps.empty_cache()

        # Calculate epoch statistics
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                # Move data to device
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Update statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Clean up memory if using MPS
                if device.type == "mps":
                    torch.mps.empty_cache()

        # Calculate epoch statistics
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total

        # Print epoch statistics
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

        # Update history
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        # Check if this is the best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        print("-" * 50)

    # Load best model weights
    model.load_state_dict(best_model_weights)

    return model, history


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


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Horse Detection")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--patience", type=int, default=3, help="Patience for early stopping"
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=0,
        help="Size of subset to use (0 for full dataset)",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Proportion of data for testing"
    )
    parser.add_argument("--save_model", action="store_true", help="Save the model")
    parser.add_argument(
        "--plot_history", action="store_true", help="Plot training history"
    )
    args = parser.parse_args()

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("data/cached_datasets", exist_ok=True)

    # Load dataset
    cache_file = "data/cached_datasets/mpg-ranch_horse-detection.parquet"
    if os.path.exists(cache_file):
        print(f"Loading dataset from cache: {cache_file}")
        df = pd.read_parquet(cache_file)
    else:
        print("Cached dataset not found. Please run the data preparation script first.")
        return

    print(f"Dataset size: {len(df)}")
    print(f"Label distribution: {df['Presence'].value_counts().to_dict()}")

    # Create subset if specified
    if args.subset_size > 0:
        print(f"Creating subset of {args.subset_size} samples")
        # Stratified sampling
        presence_1 = df[df["Presence"] == 1].sample(
            n=min(args.subset_size // 2, len(df[df["Presence"] == 1]))
        )
        presence_0 = df[df["Presence"] == 0].sample(
            n=min(args.subset_size // 2, len(df[df["Presence"] == 0]))
        )
        df = pd.concat([presence_1, presence_0])
        print(f"Subset size: {len(df)}")
        print(f"Subset label distribution: {df['Presence'].value_counts().to_dict()}")

    # Create dataset
    dataset = HorseDetectionDataset(df)

    # Split dataset
    train_size = int((1 - args.test_size) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for MPS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 for MPS
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create model
    model = create_model(num_classes=2)
    print(
        f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
    )

    # Train model
    print("Training model...")
    model, history = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=args.num_epochs,
        patience=args.patience,
    )

    # Save model
    if args.save_model:
        model_path = (
            f"models/horse_detection_resnet50_{time.strftime('%Y%m%d_%H%M%S')}.pt"
        )
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Plot history
    if args.plot_history:
        plot_path = f"results/figures/horse_detection_history_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plot_history(history, save_path=plot_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"Error: {str(e)}")
        traceback.print_exc()
