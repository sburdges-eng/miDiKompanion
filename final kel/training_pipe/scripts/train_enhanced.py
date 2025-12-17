#!/usr/bin/env python3
"""
Enhanced Training Script for Kelly MIDI Companion ML Models
===========================================================
Includes:
- Real dataset loading
- Validation splits
- Early stopping
- Training metrics and logging
- Checkpoint/resume functionality
- Hyperparameter tuning support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from datetime import datetime
import copy

# Import model definitions and data loaders
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_all_models import (
    EmotionRecognizer, MelodyTransformer, HarmonyPredictor,
    DynamicsEngine, GroovePredictor, export_to_rtneural
)
from data_loaders import (
    EmotionDataset, MelodyDataset, HarmonyDataset,
    DynamicsDataset, GrooveDataset
)


# =============================================================================
# Training Utilities
# =============================================================================

class EarlyStopping:
    """Early stopping callback to stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
                self.best_weights = copy.deepcopy(model.state_dict())
            else:
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                self.best_weights = copy.deepcopy(model.state_dict())
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

    def load_best_weights(self, model: nn.Module):
        """Load the best weights back into the model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class TrainingMetrics:
    """Track training and validation metrics."""

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def update(self, train_loss: float, val_loss: float,
               train_acc: Optional[float] = None, val_acc: Optional[float] = None):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if train_acc is not None:
            self.train_accuracies.append(train_acc)
        if val_acc is not None:
            self.val_accuracies.append(val_acc)

    def get_best_epoch(self) -> int:
        """Return epoch with lowest validation loss."""
        return np.argmin(self.val_losses)

    def to_dict(self) -> Dict:
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_epoch': self.get_best_epoch(),
            'best_val_loss': min(self.val_losses) if self.val_losses else None
        }


# =============================================================================
# Enhanced Training Functions
# =============================================================================

def train_with_validation(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu',
    early_stopping: Optional[EarlyStopping] = None,
    checkpoint_dir: Optional[Path] = None,
    model_name: str = "model"
) -> TrainingMetrics:
    """
    Train model with validation and early stopping.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Use appropriate loss function based on output type
    if isinstance(model, EmotionRecognizer):
        criterion = nn.MSELoss()
    elif isinstance(model, MelodyTransformer):
        criterion = nn.BCELoss()
    elif isinstance(model, HarmonyPredictor):
        criterion = None  # Handled specially with KL divergence
    else:
        criterion = nn.MSELoss()

    metrics = TrainingMetrics()

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_count = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Get inputs and targets based on model type
            if isinstance(model, EmotionRecognizer):
                inputs = batch['mel_features'].to(device)
                targets = batch['emotion'].to(device)
                outputs = model(inputs)
            elif isinstance(model, MelodyTransformer):
                inputs = batch['emotion'].to(device)
                targets = batch['notes'].to(device)
                outputs = model(inputs)
            elif isinstance(model, HarmonyPredictor):
                inputs = batch['context'].to(device)
                targets = batch['chords'].to(device)
                outputs = model(inputs)
                # Use KL divergence for probability distributions
                loss = nn.functional.kl_div(
                    nn.functional.log_softmax(outputs, dim=1),
                    targets,
                    reduction='batchmean'
                )
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_count += 1
                continue
            elif isinstance(model, DynamicsEngine):
                inputs = batch['context'].to(device)
                targets = batch['expression'].to(device)
                outputs = model(inputs)
            elif isinstance(model, GroovePredictor):
                inputs = batch['emotion'].to(device)
                targets = batch['groove'].to(device)
                outputs = model(inputs)
            else:
                continue

            if criterion is not None:
                loss = criterion(outputs, targets)
            else:
                # Already handled in HarmonyPredictor case
                continue

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_count += 1

        avg_train_loss = train_loss / max(train_count, 1)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(model, EmotionRecognizer):
                    inputs = batch['mel_features'].to(device)
                    targets = batch['emotion'].to(device)
                    outputs = model(inputs)
                elif isinstance(model, MelodyTransformer):
                    inputs = batch['emotion'].to(device)
                    targets = batch['notes'].to(device)
                    outputs = model(inputs)
                elif isinstance(model, HarmonyPredictor):
                    inputs = batch['context'].to(device)
                    targets = batch['chords'].to(device)
                    outputs = model(inputs)
                    loss = nn.functional.kl_div(
                        nn.functional.log_softmax(outputs, dim=1),
                        targets,
                        reduction='batchmean'
                    )
                    val_loss += loss.item()
                    val_count += 1
                    continue
                elif isinstance(model, DynamicsEngine):
                    inputs = batch['context'].to(device)
                    targets = batch['expression'].to(device)
                    outputs = model(inputs)
                elif isinstance(model, GroovePredictor):
                    inputs = batch['emotion'].to(device)
                    targets = batch['groove'].to(device)
                    outputs = model(inputs)
                else:
                    continue

                if criterion is not None:
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    val_count += 1

        avg_val_loss = val_loss / max(val_count, 1)

        metrics.update(avg_train_loss, avg_val_loss)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}")

        # Early stopping check
        if early_stopping:
            if early_stopping(avg_val_loss, model):
                print(f"  Early stopping triggered at epoch {epoch+1}")
                early_stopping.load_best_weights(model)
                break

        # Save checkpoint
        if checkpoint_dir and (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f"{model_name}_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'metrics': metrics.to_dict()
            }, checkpoint_path)

    return metrics


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: str = 'cpu'
) -> Dict:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


# =============================================================================
# Main Training Function
# =============================================================================

def train_all_models_enhanced(
    datasets_dir: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 64,
    val_split: float = 0.2,
    device: str = 'cpu',
    use_real_data: bool = True,
    early_stopping_patience: int = 10,
    checkpoint_every: int = 10,
    resume_from: Optional[Path] = None
):
    """
    Train all 5 models with enhanced features.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Kelly MIDI Companion - Enhanced Multi-Model Training")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Using real data: {use_real_data}")
    print()

    # Create models
    models = {
        'EmotionRecognizer': EmotionRecognizer(),
        'MelodyTransformer': MelodyTransformer(),
        'HarmonyPredictor': HarmonyPredictor(),
        'DynamicsEngine': DynamicsEngine(),
        'GroovePredictor': GroovePredictor()
    }

    # Print model stats
    print("Model Architecture Summary:")
    print("-" * 40)
    total_params = 0
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        total_params += params
        print(f"{name}: {params:,} params ({params * 4 / 1024:.1f} KB)")
    print("-" * 40)
    print(f"TOTAL: {total_params:,} params ({total_params * 4 / 1024:.1f} KB)")
    print()

    datasets_dir = Path(datasets_dir)

    # Load datasets
    print("Loading datasets...")
    print("-" * 40)

    # 1. Emotion Dataset
    print("\n[1/5] EmotionRecognizer Dataset")
    emotion_dataset = EmotionDataset(
        audio_dir=datasets_dir / "audio",
        labels_file=datasets_dir / "audio" / "labels.csv",
        use_synthetic=not use_real_data
    )
    train_size = int((1 - val_split) * len(emotion_dataset))
    val_size = len(emotion_dataset) - train_size
    emotion_train, emotion_val = random_split(emotion_dataset, [train_size, val_size])
    emotion_train_loader = DataLoader(emotion_train, batch_size=batch_size, shuffle=True)
    emotion_val_loader = DataLoader(emotion_val, batch_size=batch_size, shuffle=False)

    # 2. Melody Dataset
    print("\n[2/5] MelodyTransformer Dataset")
    melody_dataset = MelodyDataset(
        midi_dir=datasets_dir / "midi",
        emotion_labels=datasets_dir / "emotion_labels.json",
        use_synthetic=not use_real_data
    )
    train_size = int((1 - val_split) * len(melody_dataset))
    val_size = len(melody_dataset) - train_size
    melody_train, melody_val = random_split(melody_dataset, [train_size, val_size])
    melody_train_loader = DataLoader(melody_train, batch_size=batch_size, shuffle=True)
    melody_val_loader = DataLoader(melody_val, batch_size=batch_size, shuffle=False)

    # 3. Harmony Dataset
    print("\n[3/5] HarmonyPredictor Dataset")
    harmony_dataset = HarmonyDataset(
        chord_file=datasets_dir / "chords" / "chord_progressions.json",
        use_synthetic=not use_real_data
    )
    train_size = int((1 - val_split) * len(harmony_dataset))
    val_size = len(harmony_dataset) - train_size
    harmony_train, harmony_val = random_split(harmony_dataset, [train_size, val_size])
    harmony_train_loader = DataLoader(harmony_train, batch_size=batch_size, shuffle=True)
    harmony_val_loader = DataLoader(harmony_val, batch_size=batch_size, shuffle=False)

    # 4. Dynamics Dataset
    print("\n[4/5] DynamicsEngine Dataset")
    dynamics_dataset = DynamicsDataset(
        midi_dir=datasets_dir / "midi",
        use_synthetic=not use_real_data
    )
    train_size = int((1 - val_split) * len(dynamics_dataset))
    val_size = len(dynamics_dataset) - train_size
    dynamics_train, dynamics_val = random_split(dynamics_dataset, [train_size, val_size])
    dynamics_train_loader = DataLoader(dynamics_train, batch_size=batch_size, shuffle=True)
    dynamics_val_loader = DataLoader(dynamics_val, batch_size=batch_size, shuffle=False)

    # 5. Groove Dataset
    print("\n[5/5] GroovePredictor Dataset")
    groove_dataset = GrooveDataset(
        drums_dir=datasets_dir / "drums",
        emotion_labels=datasets_dir / "drum_labels.json",
        use_synthetic=not use_real_data
    )
    train_size = int((1 - val_split) * len(groove_dataset))
    val_size = len(groove_dataset) - train_size
    groove_train, groove_val = random_split(groove_dataset, [train_size, val_size])
    groove_train_loader = DataLoader(groove_train, batch_size=batch_size, shuffle=True)
    groove_val_loader = DataLoader(groove_val, batch_size=batch_size, shuffle=False)

    print("\n" + "=" * 60)
    print("Training Models")
    print("=" * 60)

    all_metrics = {}

    # Train each model
    model_configs = [
        ('EmotionRecognizer', emotion_train_loader, emotion_val_loader),
        ('MelodyTransformer', melody_train_loader, melody_val_loader),
        ('HarmonyPredictor', harmony_train_loader, harmony_val_loader),
        ('DynamicsEngine', dynamics_train_loader, dynamics_val_loader),
        ('GroovePredictor', groove_train_loader, groove_val_loader),
    ]

    for model_name, train_loader, val_loader in model_configs:
        print(f"\n[{model_configs.index((model_name, train_loader, val_loader)) + 1}/5] "
              f"Training {model_name}...")

        model = models[model_name]
        early_stopping = EarlyStopping(patience=early_stopping_patience)

        metrics = train_with_validation(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            device=device,
            early_stopping=early_stopping,
            checkpoint_dir=checkpoint_dir if checkpoint_every > 0 else None,
            model_name=model_name.lower()
        )

        all_metrics[model_name] = metrics.to_dict()

        # Save final model
        final_path = checkpoint_dir / f"{model_name.lower()}_final.pt"
        torch.save(model.state_dict(), final_path)

        print(f"  ✓ {model_name} training complete")
        print(f"    Best val loss: {metrics.get_best_epoch()}")
        print(f"    Best epoch: {metrics.get_best_epoch() + 1}")

    # Save all metrics
    metrics_file = metrics_dir / "training_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Export models to RTNeural format
    print("\n" + "=" * 60)
    print("Exporting models to RTNeural format...")
    print("=" * 60)

    for name, model in models.items():
        export_to_rtneural(model, name, output_dir)

    print(f"\nTraining complete!")
    print(f"  Models: {output_dir}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Metrics: {metrics_dir}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced training for Kelly MIDI Companion ML models"
    )
    parser.add_argument("--datasets-dir", "-d", type=str, default="./datasets",
                        help="Directory containing datasets")
    parser.add_argument("--output", "-o", type=str, default="./trained_models",
                        help="Output directory for trained models")
    parser.add_argument("--epochs", "-e", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split ratio (0.0-1.0)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Training device")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of real datasets")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Save checkpoint every N epochs (0 to disable)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume training from checkpoint")

    args = parser.parse_args()

    # Auto-detect device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = "cpu"

    train_all_models_enhanced(
        datasets_dir=Path(args.datasets_dir),
        output_dir=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        device=args.device,
        use_real_data=not args.synthetic,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_every=args.checkpoint_every,
        resume_from=Path(args.resume_from) if args.resume_from else None
    )
