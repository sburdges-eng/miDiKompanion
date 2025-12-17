#!/usr/bin/env python3
"""
Test emotion model inference.

This script tests the trained emotion model to verify it works correctly
before exporting to RTNeural format.

Usage:
    python test_emotion_model.py --model emotion_model.pt
"""

import argparse
import sys
from pathlib import Path

import torch  # type: ignore

sys.path.insert(0, str(Path(__file__).parent))
from train_emotion_model import EmotionModel  # type: ignore


def test_model(model_path: str):
    """Test the trained model."""
    print(f"Loading model from {model_path}...")

    # Load model
    model = EmotionModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    print("Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create test input (128-dim features)
    print("\nTesting with random input...")
    test_input = torch.randn(1, 128)  # Batch size 1

    with torch.no_grad():
        output = model(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, "
          f"{output.max().item():.4f}]")
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std: {output.std().item():.4f}")

    # Test with multiple samples
    print("\nTesting with batch of 10 samples...")
    batch_input = torch.randn(10, 128)

    with torch.no_grad():
        batch_output = model(batch_input)

    print(f"Batch input shape: {batch_input.shape}")
    print(f"Batch output shape: {batch_output.shape}")
    print("Batch output stats:")
    print(f"  Mean: {batch_output.mean().item():.4f}")
    print(f"  Std: {batch_output.std().item():.4f}")
    print(f"  Min: {batch_output.min().item():.4f}")
    print(f"  Max: {batch_output.max().item():.4f}")

    # Test inference speed
    print("\nTesting inference speed...")
    import time

    num_iterations = 100
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(test_input)

    elapsed = time.time() - start_time
    avg_time = elapsed / num_iterations * 1000  # Convert to ms

    print(f"Average inference time: {avg_time:.4f} ms")
    print(f"Throughput: {1000 / avg_time:.1f} inferences/second")

    if avg_time < 1.0:
        print("✓ Inference is fast enough for real-time audio "
              "processing!")
    else:
        print("⚠ Inference may be too slow for real-time processing.")
        print("  Consider optimizing the model or using RTNeural's "
              "compile-time optimizations.")

    print("\n✓ Model test complete!")


def main():
    parser = argparse.ArgumentParser(description='Test emotion model')
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to trained PyTorch model (.pt file)')

    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return

    test_model(args.model)


if __name__ == '__main__':
    main()
