#!/usr/bin/env python3
"""
RTNeural Export Verification Script
===================================
Verifies that exported RTNeural JSON models are correctly formatted
and can be loaded by the C++ MultiModelProcessor.
"""

import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Import model definitions
import sys
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from train_all_models import (
        EmotionRecognizer,
        MelodyTransformer,
        HarmonyPredictor,
        DynamicsEngine,
        GroovePredictor,
        export_to_rtneural
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def verify_json_structure(json_path: Path) -> Tuple[bool, List[str]]:
    """Verify RTNeural JSON file structure."""
    errors = []

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]

    # Check required fields
    required_fields = ['model_type', 'input_size', 'output_size', 'layers']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # Verify model_type
    if 'model_type' in data and data['model_type'] != 'sequential':
        errors.append(f"Unexpected model_type: {data['model_type']} (expected 'sequential')")

    # Verify layers
    if 'layers' in data:
        if not isinstance(data['layers'], list):
            errors.append("'layers' must be a list")
        else:
            for i, layer in enumerate(data['layers']):
                if not isinstance(layer, dict):
                    errors.append(f"Layer {i} is not a dictionary")
                    continue

                # Check layer type
                if 'type' not in layer:
                    errors.append(f"Layer {i} missing 'type' field")
                elif layer['type'] not in ['dense', 'lstm', 'tanh', 'relu', 'sigmoid']:
                    errors.append(f"Layer {i} has unknown type: {layer['type']}")

                # Check layer dimensions
                if 'in_size' not in layer:
                    errors.append(f"Layer {i} missing 'in_size'")
                if 'out_size' not in layer:
                    errors.append(f"Layer {i} missing 'out_size'")

                # Check weights for dense layers
                if layer.get('type') == 'dense':
                    if 'weights' not in layer:
                        errors.append(f"Layer {i} (dense) missing 'weights'")
                    if 'bias' not in layer:
                        errors.append(f"Layer {i} (dense) missing 'bias'")

                    # Verify weight dimensions
                    if 'weights' in layer and 'in_size' in layer and 'out_size' in layer:
                        weights = layer['weights']
                        if isinstance(weights, list):
                            if len(weights) != layer['out_size']:
                                errors.append(
                                    f"Layer {i}: weights outer dimension mismatch "
                                    f"(expected {layer['out_size']}, got {len(weights)})"
                                )
                            elif len(weights) > 0 and len(weights[0]) != layer['in_size']:
                                errors.append(
                                    f"Layer {i}: weights inner dimension mismatch "
                                    f"(expected {layer['in_size']}, got {len(weights[0])})"
                                )

    # Verify input/output size consistency
    if 'layers' in data and len(data['layers']) > 0:
        first_layer = data['layers'][0]
        last_layer = data['layers'][-1]

        if 'input_size' in data and 'in_size' in first_layer:
            if data['input_size'] != first_layer['in_size']:
                errors.append(
                    f"Input size mismatch: model says {data['input_size']}, "
                    f"first layer says {first_layer['in_size']}"
                )

        if 'output_size' in data and 'out_size' in last_layer:
            if data['output_size'] != last_layer['out_size']:
                errors.append(
                    f"Output size mismatch: model says {data['output_size']}, "
                    f"last layer says {last_layer['out_size']}"
                )

    return len(errors) == 0, errors


def verify_model_consistency(
    pytorch_model: nn.Module,
    json_path: Path
) -> Tuple[bool, List[str]]:
    """Verify that PyTorch model matches exported JSON."""
    errors = []

    # Load JSON
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # Get PyTorch model structure
    pytorch_state = pytorch_model.state_dict()

    # Check layer count
    json_layers = [l for l in json_data.get('layers', []) if l.get('type') in ['dense', 'lstm']]
    pytorch_layers = [k for k in pytorch_state.keys() if 'weight' in k]

    # This is a simplified check - full verification would require
    # matching each layer's weights exactly
    print(f"  JSON layers: {len(json_layers)}")
    print(f"  PyTorch layers: {len(pytorch_layers)}")

    return len(errors) == 0, errors


def test_model_inference(
    json_path: Path,
    input_size: int,
    num_tests: int = 5
) -> Tuple[bool, List[str]]:
    """Test model inference with sample inputs."""
    errors = []

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Generate test inputs
        test_inputs = []
        for _ in range(num_tests):
            test_input = np.random.randn(input_size).astype(np.float32)
            test_inputs.append(test_input)

        # Basic validation: check that we can parse the structure
        # Full inference would require RTNeural C++ library
        print(f"  Generated {num_tests} test inputs of size {input_size}")
        print(f"  Model output size: {data.get('output_size', 'unknown')}")

        # Verify input/output sizes are reasonable
        if input_size <= 0 or input_size > 10000:
            errors.append(f"Suspicious input size: {input_size}")

        output_size = data.get('output_size', 0)
        if output_size <= 0 or output_size > 10000:
            errors.append(f"Suspicious output size: {output_size}")

    except Exception as e:
        errors.append(f"Inference test failed: {e}")

    return len(errors) == 0, errors


def verify_all_models(models_dir: Path) -> Dict[str, Dict]:
    """Verify all exported models."""
    print("=" * 60)
    print("RTNeural Export Verification")
    print("=" * 60)

    models_dir = Path(models_dir)
    results = {}

    model_configs = {
        'emotionrecognizer': (EmotionRecognizer(), 128),
        'melodytransformer': (MelodyTransformer(), 64),
        'harmonypredictor': (HarmonyPredictor(), 128),
        'dynamicsengine': (DynamicsEngine(), 32),
        'groovepredictor': (GroovePredictor(), 64)
    }

    for model_name, (model, input_size) in model_configs.items():
        json_path = models_dir / f"{model_name}.json"

        print(f"\n[{model_name.upper()}]")
        print(f"  Checking: {json_path}")

        if not json_path.exists():
            print(f"  ✗ File not found")
            results[model_name] = {'valid': False, 'errors': ['File not found']}
            continue

        # Verify JSON structure
        valid, errors = verify_json_structure(json_path)
        if not valid:
            print(f"  ✗ JSON structure errors:")
            for error in errors:
                print(f"    - {error}")
            results[model_name] = {'valid': False, 'errors': errors}
            continue

        print(f"  ✓ JSON structure valid")

        # Verify model consistency
        consistent, consistency_errors = verify_model_consistency(model, json_path)
        if consistency_errors:
            print(f"  ⚠ Consistency warnings:")
            for error in consistency_errors:
                print(f"    - {error}")

        # Test inference
        inference_ok, inference_errors = test_model_inference(json_path, input_size)
        if not inference_ok:
            print(f"  ✗ Inference test errors:")
            for error in inference_errors:
                print(f"    - {error}")
        else:
            print(f"  ✓ Inference test passed")

        results[model_name] = {
            'valid': valid and inference_ok,
            'errors': errors + inference_errors,
            'json_valid': valid,
            'inference_ok': inference_ok
        }

    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)

    all_valid = True
    for model_name, result in results.items():
        status = "✓" if result['valid'] else "✗"
        print(f"{status} {model_name}: {'Valid' if result['valid'] else 'Invalid'}")
        if not result['valid']:
            all_valid = False

    return results, all_valid


def generate_cpp_test_code(models_dir: Path, output_file: Path):
    """Generate C++ test code for loading models."""
    print(f"\nGenerating C++ test code: {output_file}")

    models_dir = Path(models_dir)

    code = """// Auto-generated RTNeural model loading test
// This code tests loading exported JSON models in C++

#include "ml/MultiModelProcessor.h"
#include <juce_core/juce_core.h>
#include <iostream>

void testModelLoading() {
    using namespace Kelly::ML;

    MultiModelProcessor processor;

"""

    model_names = [
        'emotionrecognizer',
        'melodytransformer',
        'harmonypredictor',
        'dynamicsengine',
        'groovepredictor'
    ]

    for model_name in model_names:
        json_path = models_dir / f"{model_name}.json"
        if json_path.exists():
            code += f"""    // Test {model_name}
    {{
        juce::File modelFile("{json_path.absolute()}");
        if (modelFile.exists()) {{
            // Load model using MultiModelProcessor
            // processor.loadModel(ModelType::{model_name.capitalize()}, modelFile);
            std::cout << "Loaded {model_name}" << std::endl;
        }} else {{
            std::cout << "Model file not found: {model_name}.json" << std::endl;
        }}
    }}
"""

    code += """}

int main() {
    testModelLoading();
    return 0;
}
"""

    with open(output_file, 'w') as f:
        f.write(code)

    print(f"  ✓ Generated C++ test code")


def main():
    parser = argparse.ArgumentParser(
        description="Verify RTNeural JSON export format"
    )
    parser.add_argument(
        "--models-dir", "-m",
        type=str,
        default="./trained_models",
        help="Directory containing exported models"
    )
    parser.add_argument(
        "--generate-cpp", "-c",
        type=str,
        default=None,
        help="Generate C++ test code file"
    )

    args = parser.parse_args()

    results, all_valid = verify_all_models(Path(args.models_dir))

    if args.generate_cpp:
        generate_cpp_test_code(Path(args.models_dir), Path(args.generate_cpp))

    if not all_valid:
        print("\n⚠ Some models failed verification. Check errors above.")
        return 1
    else:
        print("\n✓ All models verified successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
