#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch Model Utilities

This module contains utility functions and model definitions for PyTorch model training.
It includes simple and deep models, training and evaluation functions, and metrics calculation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# Simple PyTorch model
class SimpleModel(nn.Module):
    """A simple linear model for binary classification."""

    def __init__(self, input_size, num_classes=2):
        """
        Initialize the model.

        Args:
            input_size (int): Size of the input features
            num_classes (int): Number of output classes
        """
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        """Forward pass through the model."""
        return self.fc(x)


# Deep PyTorch model
class DeepModel(nn.Module):
    """A deep neural network with multiple layers for classification."""

    def __init__(self, input_size, hidden_size=128, num_classes=2, dropout_rate=0.2):
        """
        Initialize the model.

        Args:
            input_size (int): Size of the input features
            hidden_size (int): Size of the hidden layers
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate for regularization
        """
        super(DeepModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        """Forward pass through the model."""
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model to train
        dataloader (DataLoader): DataLoader for the training data
        criterion: Loss function
        optimizer: Optimizer for updating model weights
        device: Device to run the model on (cuda, mps, or cpu)

    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0

    # Use tqdm for a progress bar
    for batch in tqdm(dataloader, desc="Training"):
        # Get inputs and labels from batch
        # Note: Adjust these keys based on your actual dataset structure
        inputs = batch["input_features"].to(device)
        labels = batch["label"].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

    # Return average loss
    return total_loss / len(dataloader)


# Evaluation function
def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on a dataset.

    Args:
        model (nn.Module): The PyTorch model to evaluate
        dataloader (DataLoader): DataLoader for the evaluation data
        criterion: Loss function
        device: Device to run the model on (cuda, mps, or cpu)

    Returns:
        tuple: (average loss, F1 score, predictions, true labels)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get inputs and labels from batch
            # Note: Adjust these keys based on your actual dataset structure
            inputs = batch["input_features"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get predictions
            _, preds = torch.max(outputs, 1)

            # Store predictions and labels for metrics calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate F1 score
    f1 = f1_score(all_labels, all_preds, average="macro")

    # Return average loss and F1 score
    return total_loss / len(dataloader), f1, all_preds, all_labels


# Function to plot training history
def plot_training_history(train_losses, val_losses, f1_scores, save_path=None):
    """
    Plot the training history.

    Args:
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
        f1_scores (list): List of F1 scores
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Plot F1 scores
    plt.subplot(1, 2, 2)
    plt.plot(f1_scores, label="F1 Score", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 Score")
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes=None, save_path=None):
    """
    Plot the confusion matrix.

    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        classes (list, optional): List of class names
        save_path (str, optional): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path)

    plt.show()


# Function to train model with early stopping
def train_model_with_early_stopping(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=50,
    patience=5,
    scheduler=None,
):
    """
    Train a model with early stopping based on validation F1 score.

    Args:
        model (nn.Module): The PyTorch model to train
        train_loader (DataLoader): DataLoader for the training data
        val_loader (DataLoader): DataLoader for the validation data
        criterion: Loss function
        optimizer: Optimizer for updating model weights
        device: Device to run the model on (cuda, mps, or cpu)
        num_epochs (int): Maximum number of epochs to train for
        patience (int): Number of epochs to wait for improvement before stopping
        scheduler: Learning rate scheduler (optional)

    Returns:
        tuple: (trained model, training history)
    """
    # Initialize variables for early stopping
    best_f1 = 0
    counter = 0
    best_model_state = None

    # Initialize lists to store training history
    train_losses = []
    val_losses = []
    f1_scores = []

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Evaluate on validation set
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        f1_scores.append(val_f1)

        # Print epoch results
        print(
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1 Score: {val_f1:.4f}"
        )

        # Check if this is the best model so far
        if val_f1 > best_f1:
            best_f1 = val_f1
            counter = 0
            best_model_state = model.state_dict().copy()
            print(f"New best F1 score: {best_f1:.4f}")
        else:
            counter += 1
            print(f"F1 score did not improve. Counter: {counter}/{patience}")

        # Step the scheduler if provided
        if scheduler is not None:
            # For ReduceLROnPlateau, pass the validation metric
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_f1)  # Pass the F1 score as the metric
            else:
                scheduler.step()  # For other schedulers that don't need a metric

        # Check for early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Return the trained model and training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "f1_scores": f1_scores,
        "best_f1": best_f1,
    }

    return model, history
