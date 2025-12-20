#!/usr/bin/env python3
"""
Test plugin integration with model.

This script verifies that:
1. Model can be loaded
2. Model produces valid outputs
3. Output format matches plugin expectations
4. Model can be exported to RTNeural format

Usage:
    python test_plugin_integration.py --model emotion_model.pt
"""

import argparse
import torch
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from train_emotion_model import EmotionModel
from export_to_rtneural import export_model_to_rtneural


def test_model_compatibility(model_path: str):
    """Test that model is compatible with plugin requirements."""
    print("=" * 60)
    print("Plugin Integration Test")
    print("=" * 60)

    # Load model
    print(f"\n1. Loading model from {model_path}...")
    model = EmotionModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("   ✓ Model loaded successfully")

    # Test input/output dimensions
    print("\n2. Testing input/output dimensions...")
    test_input = torch.randn(1, 128)
    with torch.no_grad():
        output = model(test_input)

    assert test_input.shape == (1, 128), "Input must be 128-dimensional"
    assert output.shape == (1, 64), "Output must be 64-dimensional"
    print("   ✓ Input: 128 dimensions")
    print("   ✓ Output: 64 dimensions")

    # Test batch processing
    print("\n3. Testing batch processing...")
    batch_input = torch.randn(10, 128)
    with torch.no_grad():
        batch_output = model(batch_input)

    assert batch_output.shape == (10, 64), "Batch output must be (batch, 64)"
    print("   ✓ Batch processing works")

    # Test output range
    print("\n4. Testing output range...")
    with torch.no_grad():
        test_output = model(test_input)

    output_min = test_output.min().item()
    output_max = test_output.max().item()
    output_mean = test_output.mean().item()
    output_std = test_output.std().item()

    print(f"   Output range: [{output_min:.4f}, {output_max:.4f}]")
    print(f"   Output mean: {output_mean:.4f}")
    print(f"   Output std: {output_std:.4f}")

    # Check for NaN or Inf
    if torch.isnan(test_output).any() or torch.isinf(test_output).any():
        print("   ⚠ Warning: Output contains NaN or Inf values")
        return False
    else:
        print("   ✓ Output values are valid")

    # Test export to RTNeural
    print("\n5. Testing RTNeural export...")
    export_path = "test_export.json"
    try:
        export_model_to_rtneural(model, export_path)

        # Verify JSON file exists and is valid
        if not Path(export_path).exists():
            print("   ✗ Export file not created")
            return False

        with open(export_path, 'r') as f:
            exported_data = json.load(f)

        # Verify structure
        assert 'input_size' in exported_data, "Missing input_size"
        assert 'output_size' in exported_data, "Missing output_size"
        assert 'layers' in exported_data, "Missing layers"
        assert exported_data['input_size'] == 128, "Input size must be 128"
        assert exported_data['output_size'] == 64, "Output size must be 64"

        print("   ✓ RTNeural export successful")
        print(f"   ✓ Export file: {export_path}")

        # Cleanup
        Path(export_path).unlink()

    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        return False

    # Test inference speed
    print("\n6. Testing inference speed...")
    import time

    num_iterations = 1000
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(test_input)

    elapsed = time.time() - start_time
    avg_time = elapsed / num_iterations * 1000  # ms

    print(f"   Average inference: {avg_time:.4f} ms")

    if avg_time < 1.0:
        print("   ✓ Fast enough for real-time processing")
    elif avg_time < 10.0:
        print("   ⚠ May be slow for real-time (RTNeural will optimize)")
    else:
        print("   ✗ Too slow for real-time processing")
        return False

    print("\n" + "=" * 60)
    print("✓ All tests passed! Model is ready for plugin integration.")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Export model: python export_to_rtneural.py --model", model_path)
    print("2. Copy to plugin: cp emotion_model.json ../../data/emotion_model.json")
    print("3. Build plugin with RTNeural enabled")
    print("4. Enable ML inference in plugin settings")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Test model compatibility with plugin')
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to trained PyTorch model (.pt file)')

    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return 1

    success = test_model_compatibility(args.model)
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

