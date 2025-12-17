#!/usr/bin/env python3
"""
Example Enhanced Training Script
================================
Demonstrates how to use the enhanced training utilities including:
- TrainingMetrics for tracking
- EarlyStopping for preventing overfitting
- Validation splits
- Checkpointing
- Model evaluation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

# Import models and datasets
from train_all_models import EmotionRecognizer, MelodyTransformer
from data_loaders import EmotionDataset, MelodyDataset
from training_utils import (
    TrainingMetrics,
    EarlyStopping,
    validate_model,
    evaluate_model,
    save_checkpoint,
    load_checkpoint,
    create_train_val_split
)


def train_with_enhanced_utilities(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu',
    checkpoint_dir: Path = None,
    model_name: str = "model"
):
    """
    Example training loop using enhanced training utilities.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Use appropriate loss function
    if isinstance(model, EmotionRecognizer):
        criterion = nn.MSELoss()
    elif isinstance(model, MelodyTransformer):
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    # Initialize utilities
    metrics = TrainingMetrics()
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining {model_name}...")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Device: {device}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    best_val_loss = float('inf')

    for epoch in range(epochs):
        import time
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            # Extract inputs and targets
            if 'mel_features' in batch:
                inputs = batch['mel_features'].to(device)
                targets = batch['emotion'].to(device)
            elif 'emotion' in batch and 'notes' in batch:
                inputs = batch['emotion'].to(device)
                targets = batch['notes'].to(device)
            else:
                inputs = list(batch.values())[0].to(device)
                targets = list(batch.values())[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)

        # Update metrics
        epoch_time = time.time() - epoch_start
        metrics.update(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            epoch_time=epoch_time
        )

        # Early stopping
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"Best validation loss: {early_stopping.best_score:.6f}")
            break

        # Save checkpoint (save best model)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if checkpoint_dir:
                best_checkpoint_path = checkpoint_dir / f"{model_name}_best.pt"
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    best_checkpoint_path, metrics
                )

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc_str = f", Val Acc: {val_acc:.4f}" if val_acc else ""
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}{acc_str}")

    # Save final metrics
    if checkpoint_dir:
        metrics_path = checkpoint_dir / f"{model_name}_metrics.json"
        metrics.save(metrics_path)

        plot_path = checkpoint_dir / f"{model_name}_metrics.png"
        metrics.plot_metrics(plot_path)

        print(f"\nMetrics saved to {metrics_path}")
        print(f"Metrics plot saved to {plot_path}")
        print(f"Best validation loss: {metrics.best_val_loss:.6f} at epoch {metrics.best_epoch}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Example enhanced training with utilities"
    )
    parser.add_argument(
        "--datasets-dir", "-d",
        type=str,
        default="./datasets/training",
        help="Directory containing datasets"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./trained_models",
        help="Output directory for models and checkpoints"
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
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Training device"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="emotion",
        choices=["emotion", "melody"],
        help="Which model to train"
    )

    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect device
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        print("CUDA not available, using CPU")
    elif args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"
        print("MPS not available, using CPU")

    # Load dataset
    if args.model == "emotion":
        audio_dir = datasets_dir / "audio"
        labels_csv = audio_dir / "labels.csv"

        if not audio_dir.exists():
            print(f"Error: Audio directory not found: {audio_dir}")
            print("Please download datasets first using download_datasets.py")
            return

        dataset = EmotionDataset(audio_dir, labels_csv)
        model = EmotionRecognizer()
        model_name = "emotion_recognizer"

    elif args.model == "melody":
        midi_dir = datasets_dir / "midi"
        emotion_labels = datasets_dir / "emotion_labels.json"

        if not midi_dir.exists():
            print(f"Error: MIDI directory not found: {midi_dir}")
            print("Please download datasets first using download_datasets.py")
            return

        dataset = MelodyDataset(midi_dir, emotion_labels)
        model = MelodyTransformer()
        model_name = "melody_transformer"

    # Create train/val split (returns samplers)
    train_sampler, val_sampler = create_train_val_split(
        dataset, val_split=0.2, shuffle=True
    )

    # Create data loaders (use samplers, don't shuffle in DataLoader when using sampler)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=val_sampler
    )

    # Train model
    checkpoint_dir = output_dir / "checkpoints" / model_name
    metrics = train_with_enhanced_utilities(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=0.001,
        device=args.device,
        checkpoint_dir=checkpoint_dir,
        model_name=model_name
    )

    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)

    final_metrics = evaluate_model(model, val_loader, device=args.device)
    print("\nTest Metrics:")
    for metric_name, value in final_metrics.items():
        print(f"  {metric_name}: {value:.6f}")

    print(f"\nTraining complete! Checkpoints saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
