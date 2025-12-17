#!/usr/bin/env python3
"""
RTNeural Export Format Verification Script
==========================================
Verifies that exported RTNeural JSON files are correctly formatted
and compatible with the C++ MultiModelProcessor.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def verify_rtneural_json(json_path: Path) -> Tuple[bool, List[str]]:
    """
    Verify RTNeural JSON export format.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    except Exception as e:
        return False, [f"Error reading file: {e}"]

    # Check required top-level fields
    required_fields = ['model_name', 'model_type', 'input_size', 'output_size', 'layers']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # Verify model_type
    if 'model_type' in data and data['model_type'] != 'sequential':
        errors.append(f"Expected model_type='sequential', got '{data['model_type']}'")

    # Verify layers structure
    if 'layers' in data:
        if not isinstance(data['layers'], list):
            errors.append("'layers' must be a list")
        else:
            for i, layer in enumerate(data['layers']):
                if not isinstance(layer, dict):
                    errors.append(f"Layer {i} must be a dictionary")
                    continue

                # Check layer type
                if 'type' not in layer:
                    errors.append(f"Layer {i} missing 'type' field")
                else:
                    layer_type = layer['type']

                    # Verify layer fields based on type
                    if layer_type == 'dense':
                        required_dense = ['in_size', 'out_size', 'weights']
                        for field in required_dense:
                            if field not in layer:
                                errors.append(f"Layer {i} (dense) missing '{field}'")

                        # Verify dimensions
                        if 'weights' in layer and 'in_size' in layer and 'out_size' in layer:
                            weights = layer['weights']
                            if isinstance(weights, list):
                                if len(weights) != layer['out_size']:
                                    errors.append(
                                        f"Layer {i}: weights length ({len(weights)}) "
                                        f"!= out_size ({layer['out_size']})"
                                    )
                                else:
                                    # Check inner dimensions
                                    for j, row in enumerate(weights):
                                        if isinstance(row, list) and len(row) != layer['in_size']:
                                            errors.append(
                                                f"Layer {i}, row {j}: weight vector length "
                                                f"({len(row)}) != in_size ({layer['in_size']})"
                                            )

                    elif layer_type == 'lstm':
                        required_lstm = ['in_size', 'out_size']
                        for field in required_lstm:
                            if field not in layer:
                                errors.append(f"Layer {i} (lstm) missing '{field}'")
                    else:
                        errors.append(f"Layer {i}: Unknown layer type '{layer_type}'")

                # Check activation (for dense layers)
                if layer.get('type') == 'dense':
                    if 'activation' in layer:
                        valid_activations = ['tanh', 'relu', 'sigmoid', 'linear']
                        if layer['activation'] not in valid_activations:
                            errors.append(
                                f"Layer {i}: Invalid activation '{layer['activation']}'. "
                                f"Must be one of {valid_activations}"
                            )

    # Verify input/output sizes match layers
    if 'input_size' in data and 'output_size' in data and 'layers' in data:
        layers = data['layers']
        if len(layers) > 0:
            first_layer = layers[0]
            if 'in_size' in first_layer and first_layer['in_size'] != data['input_size']:
                errors.append(
                    f"First layer in_size ({first_layer['in_size']}) "
                    f"!= model input_size ({data['input_size']})"
                )

            last_layer = layers[-1]
            if 'out_size' in last_layer and last_layer['out_size'] != data['output_size']:
                errors.append(
                    f"Last layer out_size ({last_layer['out_size']}) "
                    f"!= model output_size ({data['output_size']})"
                )

    # Check metadata
    if 'metadata' in data:
        metadata = data['metadata']
        if 'parameter_count' in metadata:
            # Could verify this matches actual layer dimensions
            pass

    is_valid = len(errors) == 0
    return is_valid, errors


def verify_model_compatibility(json_path: Path, expected_input: int, expected_output: int) -> Tuple[bool, List[str]]:
    """
    Verify model is compatible with expected input/output sizes.
    """
    errors = []

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"Error reading file: {e}"]

    if 'input_size' in data:
        if data['input_size'] != expected_input:
            errors.append(
                f"Input size mismatch: expected {expected_input}, "
                f"got {data['input_size']}"
            )

    if 'output_size' in data:
        if data['output_size'] != expected_output:
            errors.append(
                f"Output size mismatch: expected {expected_output}, "
                f"got {data['output_size']}"
            )

    return len(errors) == 0, errors


def verify_all_models(models_dir: Path) -> Dict[str, Tuple[bool, List[str]]]:
    """
    Verify all exported models in a directory.
    """
    models_dir = Path(models_dir)

    # Expected model configurations (from MultiModelProcessor.h)
    model_specs = {
        'emotionrecognizer.json': (128, 64),
        'melodytransformer.json': (64, 128),
        'harmonypredictor.json': (128, 64),
        'dynamicsengine.json': (32, 16),
        'groovepredictor.json': (64, 32),
    }

    results = {}

    for model_file, (expected_input, expected_output) in model_specs.items():
        json_path = models_dir / model_file

        if not json_path.exists():
            results[model_file] = (False, [f"File not found: {json_path}"])
            continue

        # Verify format
        format_valid, format_errors = verify_rtneural_json(json_path)

        # Verify compatibility
        compat_valid, compat_errors = verify_model_compatibility(
            json_path, expected_input, expected_output
        )

        all_errors = format_errors + compat_errors
        is_valid = format_valid and compat_valid

        results[model_file] = (is_valid, all_errors)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Verify RTNeural JSON export format"
    )
    parser.add_argument("--models-dir", "-m", type=str, default="./trained_models",
                        help="Directory containing exported model JSON files")
    parser.add_argument("--file", "-f", type=str, default=None,
                        help="Verify a single JSON file")

    args = parser.parse_args()

    print("=" * 60)
    print("RTNeural Export Format Verification")
    print("=" * 60)
    print()

    if args.file:
        # Verify single file
        json_path = Path(args.file)
        print(f"Verifying: {json_path}")
        print("-" * 60)

        is_valid, errors = verify_rtneural_json(json_path)

        if is_valid:
            print("✓ Format is valid")
        else:
            print("✗ Format errors found:")
            for error in errors:
                print(f"  - {error}")
    else:
        # Verify all models
        models_dir = Path(args.models_dir)
        print(f"Verifying models in: {models_dir}")
        print()

        results = verify_all_models(models_dir)

        all_valid = True
        for model_file, (is_valid, errors) in results.items():
            status = "✓" if is_valid else "✗"
            print(f"{status} {model_file}")

            if not is_valid:
                all_valid = False
                for error in errors:
                    print(f"    - {error}")

        print()
        if all_valid:
            print("✓ All models verified successfully!")
            print("\nModels are ready for use in C++ MultiModelProcessor.")
        else:
            print("✗ Some models have validation errors.")
            print("Please check the errors above and fix export issues.")
            sys.exit(1)


if __name__ == "__main__":
    main()
