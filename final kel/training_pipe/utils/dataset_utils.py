#!/usr/bin/env python3
"""
Dataset Utilities for Kelly MIDI Companion ML Training
========================================================
Provides dataset splitting, validation, and data loading utilities.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional
import numpy as np
from pathlib import Path


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split a dataset into train, validation, and test sets.

    Args:
        dataset: Dataset to split
        train_ratio: Ratio of training data (default: 0.8)
        val_ratio: Ratio of validation data (default: 0.1)
        test_ratio: Ratio of test data (default: 0.1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    return train_dataset, val_dataset, test_dataset


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create data loaders for train, validation, and test sets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory (useful for GPU)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    return train_loader, val_loader, test_loader
