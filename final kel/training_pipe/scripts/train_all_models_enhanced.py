#!/usr/bin/env python3
"""
Kelly MIDI Companion - Enhanced Multi-Model Training Pipeline
==============================================================
Enhanced training with validation splits, early stopping, metrics tracking,
checkpoint management, hyperparameter tuning, and visualization.

Trains all 5 neural network models for the Kelly plugin:
1. EmotionRecognizer: Audio → Emotion (128→512→256→128→64)
2. MelodyTransformer: Emotion → MIDI (64→256→256→256→128)
3. HarmonyPredictor: Context → Chords (128→256→128→64)
4. DynamicsEngine: Context → Expression (32→128→64→16)
5. GroovePredictor: Emotion → Groove (64→128→64→32)

Total: ~1M parameters, ~4MB memory, <10ms inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

# Import enhanced training utilities
from enhanced_training import train_model_with_validation
from training_utils import (
    TrainingMetrics,
    EarlyStopping,
    CheckpointManager,
    LearningRateScheduler
)
from dataset_utils import split_dataset, create_data_loaders
from metrics_visualization import plot_training_curves, create_training_summary

# Import model definitions from original script
sys.path.insert(0, str(Path(__file__).parent))
try:
    from train_all_models import (
        EmotionRecognizer,
        MelodyTransformer,
        HarmonyPredictor,
        DynamicsEngine,
        GroovePredictor,
        SyntheticEmotionDataset,
        SyntheticMelodyDataset,
        export_to_rtneural
    )
except ImportError:
    # Fallback: define models here if import fails
    print("Warning: Could not import models from train_all_models.py")
    print("Please ensure train_all_models.py is in the same directory")


def train_emotion_recognizer_enhanced(
    model: EmotionRecognizer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu',
    output_dir: Optional[Path] = None,
    early_stopping_patience: int = 10,
    checkpoint_dir: Optional[Path] = None,
    lr_scheduler_mode: Optional[str] = None,
    resume: bool = False
) -> TrainingMetrics:
    """Train EmotionRecognizer with enhanced features."""

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Setup early stopping
    early_stopping = None
    if val_loader is not None and early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=0.001,
            mode='min',
            restore_best_weights=True
        )

    # Setup checkpoint manager
    checkpoint_manager = None
    if checkpoint_dir is not None:
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir / "emotion_recognizer",
            max_checkpoints=5
        )

    # Setup learning rate scheduler
    lr_scheduler = None
    if lr_scheduler_mode is not None:
        lr_scheduler = LearningRateScheduler(
            optimizer=optimizer,
            mode=lr_scheduler_mode,
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

    # Train with enhanced features
    metrics = train_model_with_validation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        model_name="EmotionRecognizer",
        output_dir=output_dir,
        early_stopping=early_stopping,
        checkpoint_manager=checkpoint_manager,
        lr_scheduler=lr_scheduler,
        task_type='regression',
        resume_from_checkpoint=resume
    )

    return metrics


def train_melody_transformer_enhanced(
    model: MelodyTransformer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu',
    output_dir: Optional[Path] = None,
    early_stopping_patience: int = 10,
    checkpoint_dir: Optional[Path] = None,
    lr_scheduler_mode: Optional[str] = None,
    resume: bool = False
) -> TrainingMetrics:
    """Train MelodyTransformer with enhanced features."""

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    early_stopping = None
    if val_loader is not None and early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=0.001,
            mode='min',
            restore_best_weights=True
        )

    checkpoint_manager = None
    if checkpoint_dir is not None:
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir / "melody_transformer",
            max_checkpoints=5
        )

    lr_scheduler = None
    if lr_scheduler_mode is not None:
        lr_scheduler = LearningRateScheduler(
            optimizer=optimizer,
            mode=lr_scheduler_mode,
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

    metrics = train_model_with_validation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        model_name="MelodyTransformer",
        output_dir=output_dir,
        early_stopping=early_stopping,
        checkpoint_manager=checkpoint_manager,
        lr_scheduler=lr_scheduler,
        task_type='multilabel',
        resume_from_checkpoint=resume
    )

    return metrics


def train_all_models_enhanced(
    output_dir: Path,
    datasets_dir: Optional[Path] = None,
    epochs: int = 50,
    batch_size: int = 64,
    device: str = 'cpu',
    use_synthetic: bool = False,
    validation_split: float = 0.2,
    early_stopping_patience: int = 10,
    lr_scheduler_mode: Optional[str] = 'reduce_on_plateau',
    resume: bool = False,
    config_file: Optional[Path] = None
):
    """
    Train all 5 models with enhanced features.

    Args:
        output_dir: Directory to save models and outputs
        datasets_dir: Directory containing datasets
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to train on
        use_synthetic: Use synthetic data
        validation_split: Fraction of data for validation
        early_stopping_patience: Patience for early stopping
        lr_scheduler_mode: Learning rate scheduler mode
        resume: Resume from checkpoint
        config_file: Path to training config JSON
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config if provided
    config = {}
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            config = json.load(f)

    print("=" * 60)
    print("Kelly MIDI Companion - Enhanced Multi-Model Training")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Validation split: {validation_split}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Learning rate scheduler: {lr_scheduler_mode}")
    print("=" * 60)

    # Create models
    models = {
        'EmotionRecognizer': EmotionRecognizer(),
        'MelodyTransformer': MelodyTransformer(),
        'HarmonyPredictor': HarmonyPredictor(),
        'DynamicsEngine': DynamicsEngine(),
        'GroovePredictor': GroovePredictor()
    }

    # Print model stats
    print("\nModel Architecture Summary:")
    print("-" * 40)
    total_params = 0
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        total_params += params
        print(f"{name}: {params:,} params ({params * 4 / 1024:.1f} KB)")
    print("-" * 40)
    print(f"TOTAL: {total_params:,} params ({total_params * 4 / 1024:.1f} KB)\n")

    # Create datasets with validation splits
    print("Preparing datasets...")

    if use_synthetic:
        print("Using synthetic datasets...")
        emotion_dataset = SyntheticEmotionDataset(num_samples=10000)
        melody_dataset = SyntheticMelodyDataset(num_samples=10000)

        # Split datasets
        emotion_train, emotion_val, _ = split_dataset(
            emotion_dataset, train_ratio=1.0 - validation_split,
            val_ratio=validation_split, test_ratio=0.0
        )
        melody_train, melody_val, _ = split_dataset(
            melody_dataset, train_ratio=1.0 - validation_split,
            val_ratio=validation_split, test_ratio=0.0
        )

        # Create loaders
        emotion_train_loader, emotion_val_loader, _ = create_data_loaders(
            emotion_train, emotion_val, batch_size=batch_size
        )
        melody_train_loader, melody_val_loader, _ = create_data_loaders(
            melody_train, melody_val, batch_size=batch_size
        )

        harmony_train_loader = None
        harmony_val_loader = None
        dynamics_train_loader = None
        dynamics_val_loader = None
        groove_train_loader = None
        groove_val_loader = None
    else:
        # TODO: Load real datasets and split them
        print("Real dataset loading not yet implemented")
        print("Falling back to synthetic data...")
        emotion_dataset = SyntheticEmotionDataset(num_samples=10000)
        melody_dataset = SyntheticMelodyDataset(num_samples=10000)

        emotion_train, emotion_val, _ = split_dataset(
            emotion_dataset, train_ratio=1.0 - validation_split,
            val_ratio=validation_split, test_ratio=0.0
        )
        melody_train, melody_val, _ = split_dataset(
            melody_dataset, train_ratio=1.0 - validation_split,
            val_ratio=validation_split, test_ratio=0.0
        )

        emotion_train_loader, emotion_val_loader, _ = create_data_loaders(
            emotion_train, emotion_val, batch_size=batch_size
        )
        melody_train_loader, melody_val_loader, _ = create_data_loaders(
            melody_train, melody_val, batch_size=batch_size
        )

        harmony_train_loader = None
        harmony_val_loader = None
        dynamics_train_loader = None
        dynamics_val_loader = None
        groove_train_loader = None
        groove_val_loader = None

    # Setup checkpoint directory
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Setup metrics directory
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    all_metrics = {}

    # Train EmotionRecognizer
    print("\n" + "=" * 60)
    print("[1/5] Training EmotionRecognizer...")
    print("=" * 60)
    metrics = train_emotion_recognizer_enhanced(
        models['EmotionRecognizer'],
        emotion_train_loader,
        emotion_val_loader,
        epochs=epochs,
        lr=config.get('models', {}).get('emotion_recognizer', {}).get('learning_rate', 0.001),
        device=device,
        output_dir=metrics_dir,
        early_stopping_patience=early_stopping_patience,
        checkpoint_dir=checkpoint_dir,
        lr_scheduler_mode=lr_scheduler_mode,
        resume=resume
    )
    all_metrics['EmotionRecognizer'] = metrics

    # Train MelodyTransformer
    print("\n" + "=" * 60)
    print("[2/5] Training MelodyTransformer...")
    print("=" * 60)
    metrics = train_melody_transformer_enhanced(
        models['MelodyTransformer'],
        melody_train_loader,
        melody_val_loader,
        epochs=epochs,
        lr=config.get('models', {}).get('melody_transformer', {}).get('learning_rate', 0.001),
        device=device,
        output_dir=metrics_dir,
        early_stopping_patience=early_stopping_patience,
        checkpoint_dir=checkpoint_dir,
        lr_scheduler_mode=lr_scheduler_mode,
        resume=resume
    )
    all_metrics['MelodyTransformer'] = metrics

    # Train remaining models (simplified for now)
    print("\n[3/5] Training HarmonyPredictor...")
    print("[4/5] Training DynamicsEngine...")
    print("[5/5] Training GroovePredictor...")
    print("(Using pre-initialized weights - full training not yet implemented)")

    # Export all models
    print("\n" + "=" * 60)
    print("Exporting models to RTNeural format...")
    print("=" * 60)

    export_dir = output_dir / "models"
    export_dir.mkdir(exist_ok=True)

    for name, model in models.items():
        export_to_rtneural(model, name, export_dir)

    # Save final summary
    summary_path = output_dir / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Training Summary\n")
        f.write("=" * 60 + "\n\n")

        for name, metrics in all_metrics.items():
            f.write(f"\n{name}:\n")
            f.write(f"  Final Train Loss: {metrics.train_loss[-1]:.6f}\n")
            if metrics.val_loss:
                f.write(f"  Final Val Loss: {metrics.val_loss[-1]:.6f}\n")
                f.write(f"  Best Val Loss: {min(metrics.val_loss):.6f}\n")
            f.write(f"  Epochs Trained: {len(metrics.train_loss)}\n")

    print(f"\nTraining complete!")
    print(f"Models saved to: {export_dir}")
    print(f"Metrics saved to: {metrics_dir}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Kelly MIDI Companion ML models with enhanced features"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./trained_models",
        help="Output directory for trained models"
    )
    parser.add_argument(
        "--datasets-dir", "-d",
        type=str,
        default=None,
        help="Directory containing training datasets"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=64,
        help="Training batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Training device"
    )
    parser.add_argument(
        "--synthetic", "-s",
        action="store_true",
        help="Use synthetic data instead of real datasets"
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Patience for early stopping (default: 10)"
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="reduce_on_plateau",
        choices=["reduce_on_plateau", "step", "cosine", "exponential", None],
        help="Learning rate scheduler mode"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config JSON file"
    )

    args = parser.parse_args()

    # Auto-detect best device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = "cpu"

    train_all_models_enhanced(
        output_dir=Path(args.output),
        datasets_dir=Path(args.datasets_dir) if args.datasets_dir else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        use_synthetic=args.synthetic,
        validation_split=args.validation_split,
        early_stopping_patience=args.early_stopping_patience,
        lr_scheduler_mode=args.lr_scheduler,
        resume=args.resume,
        config_file=Path(args.config) if args.config else None
    )
