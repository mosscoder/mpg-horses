#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example Script for PyTorch Model Training

This script demonstrates how to use the PyTorch model training framework
with a specific dataset from Hugging Face.
"""

import os
import torch
import numpy as np
from datasets import load_dataset, Dataset
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Import utility modules
from data_utils import create_dataloaders, get_input_size
from model_utils import (
    SimpleModel,
    DeepModel,
    train_model_with_early_stopping,
    plot_training_history,
    plot_confusion_matrix,
    evaluate,
)
from pytorch_model import set_seed, get_device


def main():
    """Main function to run the example script."""
    # Set random seed for reproducibility
    set_seed(42)

    # Get device
    device = get_device()

    print("Starting example script for PyTorch model training")

    # Load the iris dataset from scikit-learn instead of Hugging Face
    print("Loading iris dataset from scikit-learn")
    iris = load_iris()
    data = {
        "sepal_length": iris.data[:, 0],
        "sepal_width": iris.data[:, 1],
        "petal_length": iris.data[:, 2],
        "petal_width": iris.data[:, 3],
        "label": iris.target,
    }

    # Convert to Hugging Face dataset
    dataset = Dataset.from_dict(data)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # Print dataset information
    print("\nDataset structure:")
    print(dataset)

    # Print a sample from the dataset
    print("\nSample from the dataset:")
    print(dataset["train"][0])

    # Define input columns and label column
    input_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    label_col = "label"  # Changed from 'target' to 'label'

    # Create DataLoaders
    batch_size = 16
    train_loader, test_loader = create_dataloaders(
        dataset, input_cols=input_cols, label_col=label_col, batch_size=batch_size
    )

    # Get input size
    input_size = get_input_size(dataset, input_cols)
    print(f"Input size: {input_size}")

    # Get number of classes
    num_classes = len(set(dataset["train"][label_col]))
    print(f"Number of classes: {num_classes}")

    # Create results directory if it doesn't exist
    os.makedirs("../results/figures", exist_ok=True)

    # Train a simple model
    print("\nTraining a simple model")
    simple_model = SimpleModel(input_size=input_size, num_classes=num_classes)
    simple_model = simple_model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)

    # Train model with early stopping
    simple_model, simple_history = train_model_with_early_stopping(
        model=simple_model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=50,
        patience=5,
    )

    # Plot training history
    plot_training_history(
        train_losses=simple_history["train_losses"],
        val_losses=simple_history["val_losses"],
        f1_scores=simple_history["f1_scores"],
        save_path="../results/figures/simple_model_iris_history.png",
    )

    # Evaluate the model
    _, f1, preds, labels = evaluate(
        model=simple_model, dataloader=test_loader, criterion=criterion, device=device
    )

    # Print classification report
    print("\nClassification Report (Simple Model):")
    target_names = ["setosa", "versicolor", "virginica"]
    print(classification_report(labels, preds, target_names=target_names))

    # Plot confusion matrix
    plot_confusion_matrix(
        y_true=labels,
        y_pred=preds,
        classes=target_names,
        save_path="../results/figures/simple_model_iris_confusion_matrix.png",
    )

    # Train a deep model
    print("\nTraining a deep model")
    deep_model = DeepModel(
        input_size=input_size, hidden_size=64, num_classes=num_classes, dropout_rate=0.2
    )
    deep_model = deep_model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(deep_model.parameters(), lr=0.01)

    # Define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=3
    )

    # Train model with early stopping
    deep_model, deep_history = train_model_with_early_stopping(
        model=deep_model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=50,
        patience=5,
        scheduler=scheduler,
    )

    # Plot training history
    plot_training_history(
        train_losses=deep_history["train_losses"],
        val_losses=deep_history["val_losses"],
        f1_scores=deep_history["f1_scores"],
        save_path="../results/figures/deep_model_iris_history.png",
    )

    # Evaluate the model
    _, f1, preds, labels = evaluate(
        model=deep_model, dataloader=test_loader, criterion=criterion, device=device
    )

    # Print classification report
    print("\nClassification Report (Deep Model):")
    print(classification_report(labels, preds, target_names=target_names))

    # Plot confusion matrix
    plot_confusion_matrix(
        y_true=labels,
        y_pred=preds,
        classes=target_names,
        save_path="../results/figures/deep_model_iris_confusion_matrix.png",
    )

    # Compare models
    print("\nModel Comparison:")
    print(f"Simple Model Best F1 Score: {simple_history['best_f1']:.4f}")
    print(f"Deep Model Best F1 Score: {deep_history['best_f1']:.4f}")

    # Save models
    os.makedirs("../models", exist_ok=True)
    torch.save(simple_model.state_dict(), "../models/simple_model_iris.pt")
    torch.save(deep_model.state_dict(), "../models/deep_model_iris.pt")
    print("\nModels saved to ../models/")


if __name__ == "__main__":
    main()
