#!/usr/bin/env python3
"""
Training Utilities for Kelly MIDI Companion ML Training
========================================================
Provides training metrics, validation, early stopping, and evaluation utilities.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import time


class TrainingMetrics:
    """Tracks training metrics over epochs."""

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.epoch_times = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
        val_acc: Optional[float] = None,
        epoch_time: Optional[float] = None
    ):
        """Update metrics for an epoch."""
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
        if train_acc is not None:
            self.train_accuracies.append(train_acc)
        if val_acc is not None:
            self.val_accuracies.append(val_acc)
        if epoch_time is not None:
            self.epoch_times.append(epoch_time)

    def plot_metrics(self, save_path: Optional[Path] = None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        if self.train_accuracies:
            axes[0, 1].plot(self.train_accuracies, label='Train Acc')
        if self.val_accuracies:
            axes[0, 1].plot(self.val_accuracies, label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Epoch times
        if self.epoch_times:
            axes[1, 0].plot(self.epoch_times)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Time (s)')
            axes[1, 0].set_title('Epoch Duration')
            axes[1, 0].grid(True)

        # Learning curve (loss vs time)
        if self.epoch_times:
            cumulative_time = np.cumsum(self.epoch_times)
            axes[1, 1].plot(cumulative_time, self.train_losses, label='Train')
            if self.val_losses:
                axes[1, 1].plot(cumulative_time, self.val_losses, label='Val')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Learning Curve')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Metrics plot saved to {save_path}")
        else:
            plt.show()

    def save(self, filepath: Path):
        """Save metrics to JSON file."""
        data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'epoch_times': self.epoch_times,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Path):
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.train_losses = data['train_losses']
        self.val_losses = data.get('val_losses', [])
        self.train_accuracies = data.get('train_accuracies', [])
        self.val_accuracies = data.get('val_accuracies', [])
        self.epoch_times = data.get('epoch_times', [])
        self.best_val_loss = data.get('best_val_loss', float('inf'))
        self.best_epoch = data.get('best_epoch', 0)


class EarlyStopping:
    """Early stopping callback to prevent overfitting."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_weights = None
        self.stopped_early = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        Returns True if training should stop.
        """
        if self.mode == 'min':
            is_better = score < (self.best_score - self.min_delta)
        else:
            is_better = score > (self.best_score + self.min_delta)

        if is_better:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_early = True
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True

        return False


def validate_model(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = 'cpu'
) -> Tuple[float, Optional[float]]:
    """
    Validate a model on validation set.
    Returns (val_loss, val_accuracy).
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            # Get inputs and targets (adapt based on dataset structure)
            if 'mel_features' in batch:
                inputs = batch['mel_features'].to(device)
                targets = batch['emotion'].to(device)
            elif 'emotion' in batch and 'notes' in batch:
                inputs = batch['emotion'].to(device)
                targets = batch['notes'].to(device)
            elif 'context' in batch and 'chords' in batch:
                inputs = batch['context'].to(device)
                targets = batch['chords'].to(device)
            elif 'context' in batch and 'dynamics' in batch:
                inputs = batch['context'].to(device)
                targets = batch['dynamics'].to(device)
            elif 'emotion' in batch and 'groove' in batch:
                inputs = batch['emotion'].to(device)
                targets = batch['groove'].to(device)
            else:
                # Fallback
                inputs = list(batch.values())[0].to(device)
                targets = list(batch.values())[1].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Calculate accuracy (for classification tasks)
            if len(targets.shape) == 1 or targets.shape[-1] == 1:
                # Binary or single output
                preds = (outputs > 0.5).float()
                total_correct += (preds == targets).sum().item()
                total_samples += targets.numel()
            elif len(targets.shape) == 2:
                # Multi-output (e.g., note probabilities)
                # Use cosine similarity or other metric
                pass

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples if total_samples > 0 else None

    return avg_loss, accuracy


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cpu',
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    Returns dictionary of metrics.
    """
    if metrics is None:
        metrics = ['loss', 'accuracy', 'mse', 'mae']

    model.eval()
    results = defaultdict(list)

    with torch.no_grad():
        for batch in test_loader:
            # Get inputs and targets
            if 'mel_features' in batch:
                inputs = batch['mel_features'].to(device)
                targets = batch['emotion'].to(device)
            elif 'emotion' in batch and 'notes' in batch:
                inputs = batch['emotion'].to(device)
                targets = batch['notes'].to(device)
            elif 'context' in batch and 'chords' in batch:
                inputs = batch['context'].to(device)
                targets = batch['chords'].to(device)
            elif 'context' in batch and 'dynamics' in batch:
                inputs = batch['context'].to(device)
                targets = batch['dynamics'].to(device)
            elif 'emotion' in batch and 'groove' in batch:
                inputs = batch['emotion'].to(device)
                targets = batch['groove'].to(device)
            else:
                inputs = list(batch.values())[0].to(device)
                targets = list(batch.values())[1].to(device)

            outputs = model(inputs)

            # Calculate metrics
            if 'loss' in metrics or 'mse' in metrics:
                mse = nn.MSELoss()(outputs, targets).item()
                results['mse'].append(mse)
                results['loss'].append(mse)

            if 'mae' in metrics:
                mae = nn.L1Loss()(outputs, targets).item()
                results['mae'].append(mae)

            if 'accuracy' in metrics:
                if len(targets.shape) == 1:
                    preds = (outputs > 0.5).float()
                    acc = (preds == targets).float().mean().item()
                    results['accuracy'].append(acc)

    # Average metrics
    final_results = {k: np.mean(v) for k, v in results.items()}
    return final_results


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: Path,
    metrics: Optional[TrainingMetrics] = None
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    if metrics:
        checkpoint['metrics'] = {
            'train_losses': metrics.train_losses,
            'val_losses': metrics.val_losses,
            'best_val_loss': metrics.best_val_loss,
            'best_epoch': metrics.best_epoch
        }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def create_train_val_split(
    dataset: torch.utils.data.Dataset,
    val_split: float = 0.2,
    shuffle: bool = True
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Split dataset into train and validation sets."""
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    if shuffle:
        np.random.shuffle(indices)

    split = int(np.floor(val_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    return train_sampler, val_sampler
