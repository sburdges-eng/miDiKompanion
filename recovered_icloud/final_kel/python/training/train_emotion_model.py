#!/usr/bin/env python3
"""
Train emotion model for RTNeural integration.

This script trains a neural network that maps 128-dimensional audio
features to 64-dimensional emotion vectors. The model architecture
matches RTNeural's compile-time optimized structure:
128→256 (Dense+Tanh) → 128 (LSTM) → 64 (Dense).

Usage:
    python train_emotion_model.py --data dataset.json --epochs 50 \\
        --output emotion_model.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore
from tqdm import tqdm  # type: ignore


class EmotionDataset(Dataset):
    """Dataset for emotion model training."""

    def __init__(self, data_file: str):
        """
        Load dataset from JSON file.

        Expected format:
        {
            "samples": [
                {
                    "features": [128 float values],
                    "emotion": [64 float values]  # Target emotion vector
                },
                ...
            ]
        }
        """
        with open(data_file, 'r') as f:
            data = json.load(f)

        self.samples = data.get('samples', data)  # Support both formats

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.tensor(sample['features'], dtype=torch.float32)
        emotion = torch.tensor(sample['emotion'], dtype=torch.float32)
        return features, emotion


class EmotionModel(nn.Module):
    """
    Emotion model architecture matching RTNeural structure.

    Architecture: 128 → 256 (Dense + Tanh) → 128 (LSTM) → 64 (Dense)
    """

    def __init__(self, input_size: int = 128, hidden_size: int = 256,
                 lstm_size: int = 128, output_size: int = 64):
        super().__init__()

        # First dense layer: 128 → 256
        self.dense1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()

        # LSTM layer: 256 → 128
        # Note: RTNeural uses LSTM, but for batch processing we use it
        # differently
        self.lstm = nn.LSTM(hidden_size, lstm_size, batch_first=True)

        # Final dense layer: 128 → 64
        self.dense2 = nn.Linear(lstm_size, output_size)

    def forward(self, x):
        # x shape: (batch, 128)
        x = self.tanh(self.dense1(x))  # (batch, 256)

        # LSTM expects (batch, seq_len, features)
        # We treat the 256 features as a sequence of length 1
        x = x.unsqueeze(1)  # (batch, 1, 256)
        x, _ = self.lstm(x)  # (batch, 1, 128)
        x = x.squeeze(1)  # (batch, 128)

        # Final dense layer
        x = self.dense2(x)  # (batch, 64)

        return x


def train_epoch(model: nn.Module, dataloader: DataLoader,
                optimizer: optim.Optimizer, criterion: nn.Module,
                device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for features, emotion in tqdm(dataloader, desc="Training"):
        features = features.to(device)
        emotion = emotion.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, emotion)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model: nn.Module, dataloader: DataLoader,
             criterion: nn.Module, device: torch.device) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for features, emotion in dataloader:
            features = features.to(device)
            emotion = emotion.to(device)

            output = model(features)
            loss = criterion(output, emotion)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def create_dummy_dataset(output_file: str, num_samples: int = 1000):
    """
    Create a dummy dataset for testing.

    In production, you would use real audio features and emotion labels.
    """
    print(f"Creating dummy dataset with {num_samples} samples...")

    samples = []
    for i in range(num_samples):
        # Generate random features (128-dim)
        features = np.random.randn(128).astype(np.float32).tolist()

        # Generate dummy emotion vector (64-dim)
        # In real training, this would come from emotion labels
        emotion = np.random.randn(64).astype(np.float32).tolist()

        samples.append({
            'features': features,
            'emotion': emotion
        })

    data = {'samples': samples}

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Created dataset: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Train emotion model for RTNeural')
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to training data JSON file')
    parser.add_argument(
        '--output', type=str, default='emotion_model.pt',
        help='Output path for trained model')
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs')
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size')
    parser.add_argument(
        '--learning-rate', type=float, default=0.001,
        help='Learning rate')
    parser.add_argument(
        '--create-dummy', action='store_true',
        help='Create dummy dataset for testing')
    parser.add_argument(
        '--dummy-samples', type=int, default=1000,
        help='Number of dummy samples to create')

    args = parser.parse_args()

    # Create dummy dataset if requested
    if args.create_dummy:
        create_dummy_dataset(args.data, args.dummy_samples)
        print("Dummy dataset created. Run again without --create-dummy to train.")
        return

    # Check if data file exists
    if not Path(args.data).exists():
        print(f"Error: Data file not found: {args.data}")
        print("Use --create-dummy to create a test dataset.")
        return

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {args.data}...")
    dataset = EmotionDataset(args.data)
    print(f"Dataset size: {len(dataset)} samples")

    # Split into train/validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    model = EmotionModel().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output)
            print(f"Saved best model to {args.output}")

    print(f"\nTraining complete! Best validation loss: "
          f"{best_val_loss:.6f}")
    print(f"Model saved to: {args.output}")
    print("\nNext step: Export to RTNeural JSON format using "
          "export_to_rtneural.py")


if __name__ == '__main__':
    main()
