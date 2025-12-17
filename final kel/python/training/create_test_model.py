#!/usr/bin/env python3
"""
Create a minimal test model for plugin testing.

This script creates a simple, pre-trained model that can be used immediately
for testing the plugin's ML inference integration without full training.

Usage:
    python create_test_model.py --output test_emotion_model.pt
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from train_emotion_model import EmotionModel


def create_test_model(output_path: str):
    """Create a minimal test model with initialized weights."""
    print("Creating test emotion model...")

    model = EmotionModel()

    # Initialize weights with small random values
    # This creates a model that will run but won't be accurate
    # It's useful for testing the plugin integration
    for param in model.parameters():
        nn.init.normal_(param, mean=0.0, std=0.01)

    # Save model
    torch.save(model.state_dict(), output_path)

    print(f"Test model created: {output_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nNote: This is a test model with random weights.")
    print("For production, train the model with real data.")


def main():
    parser = argparse.ArgumentParser(
        description='Create a test emotion model for plugin testing')
    parser.add_argument(
        '--output', type=str, default='test_emotion_model.pt',
        help='Output path for test model')

    args = parser.parse_args()

    create_test_model(args.output)

    print(f"\nNext steps:")
    print(f"1. Export to RTNeural: python export_to_rtneural.py --model {args.output}")
    print(f"2. Copy to plugin data: cp emotion_model.json ../../data/emotion_model.json")


if __name__ == '__main__':
    main()

