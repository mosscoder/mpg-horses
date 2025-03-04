#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Preprocessing Utilities

This module contains utility functions for preprocessing datasets and creating DataLoaders
for PyTorch model training.
"""

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset


class PyTorchDataset(Dataset):
    """
    A PyTorch Dataset wrapper for Hugging Face datasets.

    This class converts a Hugging Face dataset into a PyTorch Dataset,
    which can be used with PyTorch DataLoader.
    """

    def __init__(self, hf_dataset, input_cols, label_col, transform=None):
        """
        Initialize the dataset.

        Args:
            hf_dataset: Hugging Face dataset
            input_cols (list): List of column names to use as input features
            label_col (str): Column name to use as label
            transform (callable, optional): Optional transform to apply to the input features
        """
        self.dataset = hf_dataset
        self.input_cols = input_cols
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to get

        Returns:
            dict: Dictionary containing input features and label
        """
        # Get the example from the dataset
        example = self.dataset[idx]

        # Extract input features
        if len(self.input_cols) == 1:
            # Single input column
            input_features = example[self.input_cols[0]]
        else:
            # Multiple input columns
            input_features = [example[col] for col in self.input_cols]

        # Extract label
        label = example[self.label_col]

        # Apply transform if provided
        if self.transform:
            input_features = self.transform(input_features)

        # Convert to tensors
        if isinstance(input_features, list):
            input_features = torch.tensor(input_features, dtype=torch.float32)
        else:
            input_features = torch.tensor(input_features, dtype=torch.float32)

        label = torch.tensor(label, dtype=torch.long)

        return {"input_features": input_features, "label": label}


def tokenize_text_dataset(
    dataset, tokenizer_name, text_column, max_length=128, batch_size=32
):
    """
    Tokenize a text dataset using a Hugging Face tokenizer.

    Args:
        dataset: Hugging Face dataset
        tokenizer_name (str): Name of the tokenizer to use
        text_column (str): Name of the column containing text
        max_length (int): Maximum sequence length
        batch_size (int): Batch size for tokenization

    Returns:
        datasets.Dataset: Tokenized dataset
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, batch_size=batch_size
    )

    return tokenized_dataset


def create_dataloaders(
    dataset, input_cols, label_col, batch_size=32, transform=None, shuffle_train=True
):
    """
    Create PyTorch DataLoaders from a Hugging Face dataset.

    Args:
        dataset: Hugging Face dataset with 'train' and 'test' splits
        input_cols (list): List of column names to use as input features
        label_col (str): Column name to use as label
        batch_size (int): Batch size for DataLoader
        transform (callable, optional): Optional transform to apply to the input features
        shuffle_train (bool): Whether to shuffle the training data

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create PyTorch datasets
    train_dataset = PyTorchDataset(
        dataset["train"],
        input_cols=input_cols,
        label_col=label_col,
        transform=transform,
    )

    test_dataset = PyTorchDataset(
        dataset["test"], input_cols=input_cols, label_col=label_col, transform=transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_input_size(dataset, input_cols):
    """
    Get the input size for a model based on the dataset.

    Args:
        dataset: Hugging Face dataset
        input_cols (list): List of column names to use as input features

    Returns:
        int: Input size for the model
    """
    # Get a sample from the dataset
    sample = dataset["train"][0]

    # Calculate input size
    if len(input_cols) == 1:
        # Single input column
        input_features = sample[input_cols[0]]
        if isinstance(input_features, list):
            input_size = len(input_features)
        else:
            input_size = 1
    else:
        # Multiple input columns
        input_size = sum(
            len(sample[col]) if isinstance(sample[col], list) else 1
            for col in input_cols
        )

    return input_size


def prepare_text_data_for_model(
    dataset, tokenizer_name, text_column, label_column, max_length=128, batch_size=32
):
    """
    Prepare a text dataset for model training.

    Args:
        dataset: Hugging Face dataset with 'train' and 'test' splits
        tokenizer_name (str): Name of the tokenizer to use
        text_column (str): Name of the column containing text
        label_column (str): Name of the column containing labels
        max_length (int): Maximum sequence length
        batch_size (int): Batch size for DataLoader

    Returns:
        tuple: (train_loader, test_loader, input_size)
    """
    # Tokenize dataset
    tokenized_dataset = tokenize_text_dataset(
        dataset,
        tokenizer_name=tokenizer_name,
        text_column=text_column,
        max_length=max_length,
        batch_size=batch_size,
    )

    # Create DataLoaders
    train_loader, test_loader = create_dataloaders(
        tokenized_dataset,
        input_cols=["input_ids"],
        label_col=label_column,
        batch_size=batch_size,
    )

    # Get input size
    input_size = max_length

    return train_loader, test_loader, input_size
