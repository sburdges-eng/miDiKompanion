#!/usr/bin/env python3
"""
Enhanced Training Functions for Kelly MIDI Companion ML Training
==================================================================
Wraps model training with validation, early stopping, metrics tracking, etc.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.training_utils import (
    TrainingMetrics,
    EarlyStopping,
    CheckpointManager,
    LearningRateScheduler,
    calculate_accuracy
)
from utils.dataset_utils import split_dataset, create_data_loaders
from utils.metrics_visualization import (
    plot_training_curves,
    create_training_summary
)


def train_model_with_validation(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str = 'cpu',
    model_name: str = "Model",
    output_dir: Optional[Path] = None,
    early_stopping: Optional[EarlyStopping] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    lr_scheduler: Optional[LearningRateScheduler] = None,
    task_type: str = 'regression',
    log_interval: int = 10,
    resume_from_checkpoint: bool = False
) -> TrainingMetrics:
    """
    Train a model with validation, early stopping, and metrics tracking.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of epochs to train
        device: Device to train on
        model_name: Name of the model (for logging)
        output_dir: Directory to save checkpoints and metrics
        early_stopping: EarlyStopping instance (optional)
        checkpoint_manager: CheckpointManager instance (optional)
        lr_scheduler: LearningRateScheduler instance (optional)
        task_type: Type of task ('regression', 'classification', 'multilabel')
        log_interval: Log every N epochs
        resume_from_checkpoint: Whether to resume from latest checkpoint

    Returns:
        TrainingMetrics object with training history
    """
    model = model.to(device)
    metrics = TrainingMetrics()

    start_epoch = 0

    # Resume from checkpoint if requested
    if resume_from_checkpoint and checkpoint_manager is not None:
        try:
            model, optimizer, start_epoch, metrics = checkpoint_manager.load_latest(
                model, optimizer, device
            )
            print(f"Resumed training from epoch {start_epoch}")
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch")

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs} (starting from {start_epoch})")
    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"{'='*60}\n")

    best_val_loss = float('inf')

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        num_train_batches = 0

        for batch in train_loader:
            # Get batch data (handle different batch formats)
            if isinstance(batch, dict):
                # Determine input key based on task type
                inputs = None
                if task_type == 'regression':
                    # For regression (emotion recognition), input is mel_features
                    input_keys = ['mel_features', 'context', 'input']
                elif task_type == 'multilabel':
                    # For multilabel (melody), input is emotion
                    input_keys = ['emotion', 'context', 'input']
                else:
                    input_keys = ['mel_features', 'emotion', 'context', 'input']
                
                for key in input_keys:
                    if key in batch:
                        inputs = batch[key]
                        break
                
                # Determine target key - prioritize task-specific keys
                targets = None
                if task_type == 'regression':
                    # For emotion recognition, target is emotion
                    target_keys = ['emotion', 'target']
                elif task_type == 'multilabel':
                    # For melody, target is notes (NOT emotion, even though emotion is in batch)
                    target_keys = ['notes', 'target']
                else:
                    # Default: try all target keys except the one used for input
                    target_keys = ['notes', 'chords', 'dynamics', 'groove', 'emotion', 'target']
                    # Remove input key from target keys
                    input_key_used = None
                    for key in input_keys:
                        if key in batch and batch[key] is inputs:
                            input_key_used = key
                            break
                    if input_key_used:
                        target_keys = [k for k in target_keys if k != input_key_used]
                
                for key in target_keys:
                    if key in batch:
                        targets = batch[key]
                        break
                
                if inputs is None or targets is None:
                    available_keys = [k for k in batch.keys()] if hasattr(batch, 'keys') else 'unknown'
                    raise ValueError(f"Could not find input/target in batch. Available keys: {available_keys}, inputs={inputs is not None}, targets={targets is not None}, task_type={task_type}")
            else:
                # Assume tuple/list format: (inputs, targets)
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                else:
                    raise ValueError(f"Unknown batch format: {type(batch)}")

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_accuracy += calculate_accuracy(outputs, targets, task_type)
            num_train_batches += 1

        avg_train_loss = train_loss / num_train_batches
        avg_train_accuracy = train_accuracy / num_train_batches

        metrics.train_loss.append(avg_train_loss)
        metrics.train_accuracy.append(avg_train_accuracy)

        # Validation phase
        val_loss = 0.0
        val_accuracy = 0.0
        num_val_batches = 0

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, dict):
                        # Use same logic as training phase
                        inputs = None
                        if task_type == 'regression':
                            input_keys = ['mel_features', 'context', 'input']
                        elif task_type == 'multilabel':
                            input_keys = ['emotion', 'context', 'input']
                        else:
                            input_keys = ['mel_features', 'emotion', 'context', 'input']
                        
                        for key in input_keys:
                            if key in batch:
                                inputs = batch[key]
                                break
                        
                        targets = None
                        if task_type == 'regression':
                            target_keys = ['emotion', 'target']
                        elif task_type == 'multilabel':
                            # For melody, target is notes (NOT emotion)
                            target_keys = ['notes', 'target']
                        else:
                            target_keys = ['notes', 'chords', 'dynamics', 'groove', 'emotion', 'target']
                            input_key_used = None
                            for key in input_keys:
                                if key in batch and batch[key] is inputs:
                                    input_key_used = key
                                    break
                            if input_key_used:
                                target_keys = [k for k in target_keys if k != input_key_used]
                        
                        for key in target_keys:
                            if key in batch:
                                targets = batch[key]
                                break
                        
                        if inputs is None or targets is None:
                            available_keys = [k for k in batch.keys()] if hasattr(batch, 'keys') else 'unknown'
                            raise ValueError(f"Could not find input/target in batch. Available keys: {available_keys}, task_type={task_type}")
                    else:
                        # Assume tuple/list format: (inputs, targets)
                        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                            inputs, targets = batch[0], batch[1]
                        else:
                            raise ValueError(f"Unknown batch format: {type(batch)}")

                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    val_accuracy += calculate_accuracy(outputs, targets, task_type)
                    num_val_batches += 1

            avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0.0
            avg_val_accuracy = val_accuracy / num_val_batches if num_val_batches > 0 else 0.0

            metrics.val_loss.append(avg_val_loss)
            metrics.val_accuracy.append(avg_val_accuracy)

            best_val_loss = min(best_val_loss, avg_val_loss)
        else:
            metrics.val_loss.append(0.0)
            metrics.val_accuracy.append(0.0)

        # Learning rate tracking
        current_lr = optimizer.param_groups[0]['lr']
        metrics.learning_rate.append(current_lr)

        # Epoch time tracking
        epoch_time = time.time() - epoch_start_time
        metrics.epoch_times.append(epoch_time)

        # Logging
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            log_str = f"Epoch {epoch+1}/{epochs} | "
            log_str += f"Train Loss: {avg_train_loss:.6f} | "
            if val_loader:
                log_str += f"Val Loss: {avg_val_loss:.6f} | "
            log_str += f"LR: {current_lr:.6f} | "
            log_str += f"Time: {epoch_time:.2f}s"
            print(log_str)

        # Learning rate scheduling
        if lr_scheduler is not None:
            if val_loader:
                lr_scheduler.step(avg_val_loss)
            else:
                lr_scheduler.step()

        # Early stopping check
        if early_stopping is not None and val_loader is not None:
            if early_stopping(avg_val_loss, model):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best validation loss: {early_stopping.best_score:.6f}")
                break

        # Checkpointing
        if checkpoint_manager is not None:
            is_best = avg_val_loss < best_val_loss if val_loader else False
            checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics=metrics,
                is_best=is_best
            )

    # Save final metrics
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics JSON
        metrics_path = output_dir / f"{model_name.lower()}_metrics.json"
        metrics.save(metrics_path)

        # Save training summary
        summary_path = output_dir / f"{model_name.lower()}_summary.txt"
        create_training_summary(metrics, summary_path, model_name)

        # Plot training curves
        plot_path = output_dir / f"{model_name.lower()}_curves.png"
        plot_training_curves(metrics, plot_path, model_name)

    print(f"\n{'='*60}")
    print(f"Training Complete: {model_name}")
    print(f"{'='*60}")
    print(f"Final Train Loss: {metrics.train_loss[-1]:.6f}")
    if val_loader:
        print(f"Final Val Loss: {metrics.val_loss[-1]:.6f}")
        print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"{'='*60}\n")

    return metrics
