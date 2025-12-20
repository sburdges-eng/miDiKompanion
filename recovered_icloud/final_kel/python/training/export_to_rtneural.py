#!/usr/bin/env python3
"""
Export PyTorch emotion model to RTNeural JSON format.

RTNeural uses a JSON format to describe model architecture and weights.
This script converts a trained PyTorch model to RTNeural-compatible JSON.

Usage:
    python export_to_rtneural.py --model emotion_model.pt \\
        --output emotion_model.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch  # type: ignore
import torch.nn as nn  # type: ignore

# Import the model architecture
sys.path.insert(0, str(Path(__file__).parent))
from train_emotion_model import EmotionModel  # type: ignore


def export_dense_layer(layer: nn.Linear, layer_name: str) -> dict:
    """Export a dense/linear layer to RTNeural format."""
    weights = layer.weight.data.cpu().numpy().T  # RTNeural expects (out, in)
    bias = layer.bias.data.cpu().numpy() if layer.bias is not None else None

    return {
        'type': 'dense',
        'name': layer_name,
        'weights': weights.tolist(),
        'bias': bias.tolist() if bias is not None else None
    }


def export_lstm_layer(lstm: nn.LSTM, layer_name: str) -> dict:
    """Export an LSTM layer to RTNeural format."""
    # LSTM has 4 weight matrices: W_ii, W_if, W_ig, W_io (input)
    # and 4 bias vectors: b_ii, b_if, b_ig, b_io

    # Get weights and biases
    weight_ih = lstm.weight_ih_l0.data.cpu().numpy()  # (4*hidden, input)
    weight_hh = lstm.weight_hh_l0.data.cpu().numpy()  # (4*hidden, hidden)
    bias_ih = (lstm.bias_ih_l0.data.cpu().numpy()
               if lstm.bias_ih_l0 is not None else None)
    bias_hh = (lstm.bias_hh_l0.data.cpu().numpy()
               if lstm.bias_hh_l0 is not None else None)

    # RTNeural LSTM format
    # Note: RTNeural's LSTM format may differ - this is a simplified export
    # For full compatibility, you may need to adjust based on RTNeural's
    # exact format

    return {
        'type': 'lstm',
        'name': layer_name,
        'input_size': lstm.input_size,
        'hidden_size': lstm.hidden_size,
        'weight_ih': weight_ih.tolist(),
        'weight_hh': weight_hh.tolist(),
        'bias_ih': bias_ih.tolist() if bias_ih is not None else None,
        'bias_hh': bias_hh.tolist() if bias_hh is not None else None
    }


def export_activation(activation_type: str, layer_name: str) -> dict:
    """Export activation function."""
    return {
        'type': activation_type.lower(),  # 'tanh', 'relu', etc.
        'name': layer_name
    }


def export_model_to_rtneural(model: nn.Module, output_file: str):
    """
    Export PyTorch model to RTNeural JSON format.

    RTNeural JSON structure:
    {
        "layers": [
            {"type": "dense", "weights": [...], "bias": [...]},
            {"type": "tanh"},
            {"type": "lstm", ...},
            {"type": "dense", "weights": [...], "bias": [...]}
        ]
    }
    """
    model.eval()

    layers = []
    layer_idx = 0

    # Export layers in order
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            layers.append(export_dense_layer(module, f"dense_{layer_idx}"))
            layer_idx += 1
        elif isinstance(module, nn.LSTM):
            layers.append(export_lstm_layer(module, f"lstm_{layer_idx}"))
            layer_idx += 1
        elif isinstance(module, nn.Tanh):
            layers.append(export_activation('tanh', f"tanh_{layer_idx}"))
            layer_idx += 1
        elif isinstance(module, nn.ReLU):
            layers.append(export_activation('relu', f"relu_{layer_idx}"))
            layer_idx += 1
        elif isinstance(module, nn.Sigmoid):
            layers.append(export_activation('sigmoid', f"sigmoid_{layer_idx}"))
            layer_idx += 1

    # Create RTNeural JSON structure
    rtneural_model = {
        'version': '1.0',
        'input_size': 128,
        'output_size': 64,
        'layers': layers
    }

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(rtneural_model, f, indent=2)

    print(f"Exported model to: {output_file}")
    print(f"  Input size: {rtneural_model['input_size']}")
    print(f"  Output size: {rtneural_model['output_size']}")
    print(f"  Layers: {len(layers)}")

    # Note: RTNeural's exact JSON format may require adjustments
    # Check RTNeural documentation for the exact format expected
    print("\nNote: Verify JSON format matches RTNeural's expected "
          "structure.")
    print("You may need to adjust the export format based on "
          "RTNeural version.")


def main():
    parser = argparse.ArgumentParser(
        description='Export PyTorch model to RTNeural JSON')
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to trained PyTorch model (.pt file)')
    parser.add_argument(
        '--output', type=str, default='emotion_model.json',
        help='Output path for RTNeural JSON file')

    args = parser.parse_args()

    # Check if model file exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return

    # Load model
    print(f"Loading model from {args.model}...")
    model = EmotionModel()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()

    print("Model architecture:")
    print(model)

    # Export to RTNeural format
    print("\nExporting to RTNeural format...")
    export_model_to_rtneural(model, args.output)

    print(f"\nExport complete! Place {args.output} in the plugin's "
          f"data/ directory.")


if __name__ == '__main__':
    main()
