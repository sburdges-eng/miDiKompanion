#!/usr/bin/env python3
"""
Training Utilities for Kelly MIDI Companion ML Training
========================================================
Provides utilities for training: metrics tracking, early stopping, checkpoints, etc.
"""

import json
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class TrainingMetrics:
    """Track training metrics across epochs."""

    def __init__(self):
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []
        self.epoch_times: List[float] = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
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
        
        if val_acc is not None:
            self.val_accuracies.append(val_acc)
        
        if epoch_time is not None:
            self.epoch_times.append(epoch_time)

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'epoch_times': self.epoch_times,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.train_losses)
        }

    def save(self, path: Path):
        """Save metrics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def plot_metrics(self, path: Path):
        """Plot training curves."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        ax = axes[0, 0]
        ax.plot(self.train_losses, label='Train Loss', alpha=0.7)
        if self.val_losses:
            ax.plot(self.val_losses, label='Val Loss', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy (if available)
        ax = axes[0, 1]
        if self.val_accuracies:
            ax.plot(self.val_accuracies, label='Val Accuracy', alpha=0.7, color='green')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Validation Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No accuracy data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Validation Accuracy')
        
        # Epoch times
        ax = axes[1, 0]
        if self.epoch_times:
            ax.plot(self.epoch_times, label='Epoch Time', alpha=0.7, color='orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Epoch Duration')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No timing data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Epoch Duration')
        
        # Summary stats
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = f"""
Training Summary:
-----------------
Total Epochs: {len(self.train_losses)}
Best Epoch: {self.best_epoch + 1}
Best Val Loss: {self.best_val_loss:.6f}
Final Train Loss: {self.train_losses[-1]:.6f}
"""
        if self.val_losses:
            summary_text += f"Final Val Loss: {self.val_losses[-1]:.6f}\n"
        if self.epoch_times:
            total_time = sum(self.epoch_times)
            summary_text += f"Total Time: {total_time:.1f}s\n"
            summary_text += f"Avg Time/Epoch: {total_time/len(self.epoch_times):.2f}s\n"
        
        ax.text(0.1, 0.5, summary_text, fontsize=10, 
               family='monospace', verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()


class EarlyStopping:
    """Early stopping callback."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better)
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0

    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score (loss or accuracy)
            model: Model to save weights from

        Returns:
            True if training should stop, False otherwise
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
            self.stopped_epoch = self.counter
            if self.restore_best_weights and self.best_weights is not None:
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

    Returns:
        Tuple of (validation_loss, validation_accuracy)
        Accuracy is None if not applicable to the loss function
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Get inputs and targets (handle different batch formats)
            inputs = None
            targets = None
            
            # Try common keys
            if 'mel_features' in batch:
                inputs = batch['mel_features'].to(device)
                targets = batch['emotion'].to(device)
            elif 'emotion' in batch and 'notes' in batch:
                inputs = batch['emotion'].to(device)
                targets = batch['notes'].to(device)
            elif 'context' in batch:
                inputs = batch['context'].to(device)
                if 'chords' in batch:
                    targets = batch['chords'].to(device)
                elif 'expression' in batch:
                    targets = batch['expression'].to(device)
            elif 'groove' in batch:
                inputs = batch['emotion'].to(device)
                targets = batch['groove'].to(device)
            else:
                # Fallback: use first two items
                items = list(batch.values())
                if len(items) >= 2:
                    inputs = items[0].to(device)
                    targets = items[1].to(device)
            
            if inputs is None or targets is None:
                continue
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            # Calculate accuracy for classification tasks
            if isinstance(criterion, (nn.CrossEntropyLoss, nn.BCEWithLogitsLoss)):
                if outputs.dim() > 1:
                    predicted = torch.argmax(outputs, dim=1)
                    if targets.dim() > 1:
                        targets = torch.argmax(targets, dim=1)
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)
    
    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total if total > 0 else None
    
    return avg_loss, accuracy


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate a model on test set.

    Returns:
        Dictionary with evaluation metrics
    """
    loss, accuracy = validate_model(model, test_loader, criterion, device)
    
    return {
        'loss': loss,
        'accuracy': accuracy if accuracy is not None else 0.0
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
    metrics: Optional[TrainingMetrics] = None
):
    """Save a training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if metrics:
        checkpoint['metrics'] = metrics.to_dict()
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict:
    """
    Load a training checkpoint.

    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def create_train_val_split(
    dataset: torch.utils.data.Dataset,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """
    Split dataset into train and validation sets.

    Returns:
        Tuple of (train_subset, val_subset)
    """
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Set random seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    
    train_subset, val_subset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )
    
    return train_subset, val_subset
