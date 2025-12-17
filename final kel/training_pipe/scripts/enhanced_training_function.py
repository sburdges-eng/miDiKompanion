#!/usr/bin/env python3
"""
Enhanced Training Function Template
==================================
Shows how to use all advanced training features together.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict
import time

from training_utils import (
    TrainingMetrics,
    EarlyStopping,
    validate_model,
    save_checkpoint
)
from advanced_training_utils import (
    LearningRateScheduler,
    GradientClipper,
    MixedPrecisionTrainer,
    DataAugmentation,
    TrainingStability
)


def train_model_advanced(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    criterion: nn.Module,
    epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    checkpoint_dir: Optional[Path] = None,
    model_name: str = "model",
    # Advanced features
    use_scheduler: bool = True,
    scheduler_type: str = 'plateau',
    gradient_clip: Optional[float] = 1.0,
    use_mixed_precision: bool = False,
    use_augmentation: bool = False,
    early_stop_patience: int = 10,
    check_stability: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Enhanced training function with all advanced features.

    Returns:
        Dictionary with training results and metrics
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize advanced utilities
    metrics = TrainingMetrics()
    early_stopping = EarlyStopping(patience=early_stop_patience, min_delta=0.001)

    # Learning rate scheduler
    scheduler = None
    if use_scheduler:
        scheduler = LearningRateScheduler(
            optimizer,
            scheduler_type=scheduler_type,
            patience=5 if scheduler_type == 'plateau' else None,
            T_max=epochs if scheduler_type == 'cosine' else None
        )

    # Gradient clipping
    grad_clipper = None
    if gradient_clip is not None:
        grad_clipper = GradientClipper(max_norm=gradient_clip)

    # Mixed precision training
    mp_trainer = MixedPrecisionTrainer(
        enabled=use_mixed_precision,
        device=device
    )

    # Data augmentation
    augmenter = DataAugmentation() if use_augmentation else None

    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Device: {device}")
        print(f"Train samples: {len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 'N/A'}")
        if val_loader:
            print(f"Val samples: {len(val_loader.dataset) if hasattr(val_loader, 'dataset') else 'N/A'}")
        print(f"\nAdvanced Features:")
        print(f"  Scheduler: {scheduler_type if use_scheduler else 'None'}")
        print(f"  Gradient clipping: {gradient_clip if gradient_clip else 'None'}")
        print(f"  Mixed precision: {use_mixed_precision}")
        print(f"  Data augmentation: {use_augmentation}")
        print(f"  Early stopping: patience={early_stop_patience}")
        print(f"{'='*60}\n")

    best_val_loss = float('inf')

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Extract inputs and targets
            if 'mel_features' in batch:
                inputs = batch['mel_features'].to(device)
                targets = batch['emotion'].to(device)
                # Apply augmentation if enabled
                if augmenter:
                    inputs = augmenter.augment_audio_features(inputs)
                    targets = augmenter.augment_emotion_embedding(targets)
            elif 'emotion' in batch and 'notes' in batch:
                inputs = batch['emotion'].to(device)
                targets = batch['notes'].to(device)
                if augmenter:
                    inputs = augmenter.augment_emotion_embedding(inputs)
                    targets = augmenter.augment_note_probabilities(targets)
            else:
                inputs = list(batch.values())[0].to(device)
                targets = list(batch.values())[1].to(device)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with mp_trainer.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Backward pass with scaling
            scaled_loss = mp_trainer.scale_loss(loss)
            scaled_loss.backward()

            # Gradient clipping
            if grad_clipper:
                grad_norm = grad_clipper.clip(model)
                if verbose and epoch == 0 and batch_idx == 0:
                    print(f"  Gradient norm: {grad_norm:.4f}")

            # Optimizer step with scaling
            mp_trainer.step_optimizer(optimizer)

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches

        # Validation phase
        val_loss = None
        val_acc = None
        if val_loader is not None:
            val_loss, val_acc = validate_model(model, val_loader, criterion, device)

        # Learning rate scheduling
        if scheduler:
            if scheduler_type == 'plateau' and val_loss is not None:
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Update metrics
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_lr() if scheduler else learning_rate
        metrics.update(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            epoch_time=epoch_time
        )

        # Check training stability
        if check_stability and (epoch + 1) % 10 == 0:
            grad_stats = TrainingStability.check_gradients(model, verbose=False)
            weight_stats = TrainingStability.check_weights(model, verbose=False)
            if verbose:
                print(f"  Gradient stats: max={grad_stats['max_grad']:.4f}, "
                      f"mean={grad_stats['mean_grad']:.4f}")
                if grad_stats['nan_grads'] > 0 or grad_stats['inf_grads'] > 0:
                    print(f"  WARNING: {grad_stats['nan_grads']} NaN, "
                          f"{grad_stats['inf_grads']} Inf gradients!")

        # Early stopping
        if early_stopping and val_loss is not None:
            if early_stopping(val_loss, model):
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Best validation loss: {early_stopping.best_score:.6f}")
                break

        # Save checkpoint (best model)
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            if checkpoint_dir:
                best_checkpoint_path = checkpoint_dir / f"{model_name}_best.pt"
                checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': val_loss,
                    'metrics': {
                        'train_losses': metrics.train_losses,
                        'val_losses': metrics.val_losses,
                        'best_val_loss': metrics.best_val_loss,
                        'best_epoch': metrics.best_epoch
                    }
                }
                if scheduler:
                    checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                if mp_trainer.scaler:
                    checkpoint_data['scaler_state_dict'] = mp_trainer.state_dict()

                torch.save(checkpoint_data, best_checkpoint_path)
                if verbose:
                    print(f"  Saved best checkpoint: {best_checkpoint_path}")

        # Print progress
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            lr_str = f", LR: {current_lr:.6f}" if scheduler else ""
            acc_str = f", Val Acc: {val_acc:.4f}" if val_acc is not None else ""
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f if val_loss else 'N/A'}{acc_str}{lr_str}")

    # Save final metrics
    if checkpoint_dir:
        metrics_path = checkpoint_dir / f"{model_name}_metrics.json"
        metrics.save(metrics_path)

        plot_path = checkpoint_dir / f"{model_name}_metrics.png"
        metrics.plot_metrics(plot_path)

        if verbose:
            print(f"\nMetrics saved to {metrics_path}")
            print(f"Metrics plot saved to {plot_path}")
            print(f"Best validation loss: {metrics.best_val_loss:.6f} at epoch {metrics.best_epoch}")

    # Return results
    results = {
        'metrics': metrics,
        'best_val_loss': metrics.best_val_loss,
        'best_epoch': metrics.best_epoch,
        'gradient_stats': grad_clipper.get_stats() if grad_clipper else {},
        'final_lr': scheduler.get_lr() if scheduler else learning_rate
    }

    return results
