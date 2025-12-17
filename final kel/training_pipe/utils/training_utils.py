#!/usr/bin/env python3
"""
Training Utilities for Kelly MIDI Companion ML Training
=========================================================
Provides early stopping, metrics tracking, checkpoint management, and more.
"""

import torch
import torch.nn as nn
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    train_loss: List[float] = None
    val_loss: List[float] = None
    train_accuracy: List[float] = None
    val_accuracy: List[float] = None
    learning_rate: List[float] = None
    epoch_times: List[float] = None

    def __post_init__(self):
        if self.train_loss is None:
            self.train_loss = []
        if self.val_loss is None:
            self.val_loss = []
        if self.train_accuracy is None:
            self.train_accuracy = []
        if self.val_accuracy is None:
            self.val_accuracy = []
        if self.learning_rate is None:
            self.learning_rate = []
        if self.epoch_times is None:
            self.epoch_times = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def save(self, filepath: Path):
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> 'TrainingMetrics':
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better) or 'max' for accuracy (higher is better)
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score (loss or accuracy)
            model: Model to track weights for

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self._save_weights(model)
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self._save_weights(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    self._restore_weights(model)

        return self.early_stop

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best."""
        if self.mode == 'min':
            return current < (best - self.min_delta)
        else:  # mode == 'max'
            return current > (best + self.min_delta)

    def _save_weights(self, model: nn.Module):
        """Save current model weights."""
        self.best_weights = {
            name: param.clone().cpu()
            for name, param in model.named_parameters()
        }

    def _restore_weights(self, model: nn.Module):
        """Restore best model weights."""
        if self.best_weights is not None:
            for name, param in model.named_parameters():
                if name in self.best_weights:
                    param.data = self.best_weights[name].to(param.device)


class CheckpointManager:
    """Manages model checkpoints during training."""

    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 5):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_history = []

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: TrainingMetrics,
        is_best: bool = False,
        additional_info: Optional[Dict] = None
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch number
            metrics: Training metrics
            is_best: Whether this is the best model so far
            additional_info: Any additional information to save

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics.to_dict(),
            'additional_info': additional_info or {}
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

        # Track checkpoint history
        self.checkpoint_history.append({
            'epoch': epoch,
            'path': checkpoint_path,
            'is_best': is_best
        })

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load(
        self,
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
    ) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int, TrainingMetrics]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            device: Device to load model to

        Returns:
            Tuple of (model, optimizer, epoch, metrics)
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        metrics = TrainingMetrics(**checkpoint.get('metrics', {}))

        return model, optimizer, epoch, metrics

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
    ) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int, TrainingMetrics]:
        """Load the latest checkpoint."""
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        if latest_path.exists():
            return self.load(latest_path, model, optimizer, device)
        else:
            raise FileNotFoundError(f"No checkpoint found at {latest_path}")

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
    ) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int, TrainingMetrics]:
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            return self.load(best_path, model, optimizer, device)
        else:
            raise FileNotFoundError(f"No best model found at {best_path}")

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return

        # Sort by epoch (newest first)
        sorted_checkpoints = sorted(
            self.checkpoint_history,
            key=lambda x: x['epoch'],
            reverse=True
        )

        # Keep best model and latest, plus max_checkpoints-2 others
        to_keep = {sorted_checkpoints[0]['path']}  # Latest
        for cp in sorted_checkpoints:
            if cp['is_best']:
                to_keep.add(cp['path'])

        # Keep top N checkpoints
        for cp in sorted_checkpoints[:self.max_checkpoints]:
            to_keep.add(cp['path'])

        # Remove old checkpoints
        for cp in self.checkpoint_history:
            if cp['path'] not in to_keep and cp['path'].exists():
                cp['path'].unlink()

        # Update history
        self.checkpoint_history = [
            cp for cp in sorted_checkpoints
            if cp['path'] in to_keep
        ]


class LearningRateScheduler:
    """Learning rate scheduler with multiple strategies."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = 'reduce_on_plateau',
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-6,
        **kwargs
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            mode: 'reduce_on_plateau', 'step', 'cosine', or 'exponential'
            factor: Factor to reduce LR by (for reduce_on_plateau, step, exponential)
            patience: Patience for reduce_on_plateau
            min_lr: Minimum learning rate
            **kwargs: Additional arguments for specific schedulers
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr

        if mode == 'reduce_on_plateau':
            # verbose parameter was added in PyTorch 1.4, use try/except for compatibility
            try:
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=factor, patience=patience,
                    min_lr=min_lr, verbose=True
                )
            except TypeError:
                # Fallback for older PyTorch versions
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=factor, patience=patience,
                    min_lr=min_lr
                )
        elif mode == 'step':
            step_size = kwargs.get('step_size', 10)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=factor
            )
        elif mode == 'cosine':
            T_max = kwargs.get('T_max', 50)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=min_lr
            )
        elif mode == 'exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=factor
            )
        else:
            raise ValueError(f"Unknown scheduler mode: {mode}")

    def step(self, metrics: Optional[float] = None):
        """
        Update learning rate.

        Args:
            metrics: Validation loss (for reduce_on_plateau) or None
        """
        if self.mode == 'reduce_on_plateau':
            if metrics is None:
                raise ValueError("metrics required for reduce_on_plateau mode")
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


def calculate_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    task_type: str = 'regression'
) -> float:
    """
    Calculate accuracy for different task types.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        task_type: 'regression', 'classification', or 'multilabel'

    Returns:
        Accuracy score
    """
    if task_type == 'regression':
        # For regression, use R² score or MSE-based accuracy
        mse = torch.mean((predictions - targets) ** 2)
        var = torch.var(targets)
        if var == 0:
            return 1.0 if mse < 1e-6 else 0.0
        r2 = 1 - (mse / var)
        return max(0.0, min(1.0, r2.item()))

    elif task_type == 'classification':
        # For classification, use top-1 accuracy
        pred_classes = torch.argmax(predictions, dim=1)
        target_classes = torch.argmax(targets, dim=1) if targets.dim() > 1 else targets
        correct = (pred_classes == target_classes).float()
        return correct.mean().item()

    elif task_type == 'multilabel':
        # For multilabel, use threshold-based accuracy
        threshold = 0.5
        pred_binary = (predictions > threshold).float()
        target_binary = (targets > threshold).float()
        correct = (pred_binary == target_binary).float()
        return correct.mean().item()

    else:
        raise ValueError(f"Unknown task_type: {task_type}")
