#!/usr/bin/env python3
"""
Model Evaluation Script for Kelly MIDI Companion ML Models
===========================================================
Evaluates trained models with various metrics:
- Loss (MSE, BCE, KL divergence)
- Accuracy metrics
- Inference latency
- Model size
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import time
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_all_models import (
    EmotionRecognizer, MelodyTransformer, HarmonyPredictor,
    DynamicsEngine, GroovePredictor
)
from data_loaders import (
    EmotionDataset, MelodyDataset, HarmonyDataset,
    DynamicsDataset, GrooveDataset
)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu',
    model_name: str = "model"
) -> Dict:
    """
    Evaluate a model on test data.
    """
    model.eval()
    model = model.to(device)

    # Determine appropriate loss function
    if isinstance(model, EmotionRecognizer):
        criterion = nn.MSELoss()
        metric_name = "MSE"
    elif isinstance(model, MelodyTransformer):
        criterion = nn.BCELoss()
        metric_name = "BCE"
    elif isinstance(model, HarmonyPredictor):
        criterion = None  # KL divergence
        metric_name = "KL_Divergence"
    else:
        criterion = nn.MSELoss()
        metric_name = "MSE"

    total_loss = 0.0
    total_samples = 0

    # Measure inference latency
    latencies = []

    with torch.no_grad():
        for batch in test_loader:
            # Get inputs and targets
            if isinstance(model, EmotionRecognizer):
                inputs = batch['mel_features'].to(device)
                targets = batch['emotion'].to(device)
            elif isinstance(model, MelodyTransformer):
                inputs = batch['emotion'].to(device)
                targets = batch['notes'].to(device)
            elif isinstance(model, HarmonyPredictor):
                inputs = batch['context'].to(device)
                targets = batch['chords'].to(device)
            elif isinstance(model, DynamicsEngine):
                inputs = batch['context'].to(device)
                targets = batch['expression'].to(device)
            elif isinstance(model, GroovePredictor):
                inputs = batch['emotion'].to(device)
                targets = batch['groove'].to(device)
            else:
                continue

            # Measure inference time
            start_time = time.perf_counter()
            outputs = model(inputs)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

            # Calculate loss
            if isinstance(model, HarmonyPredictor):
                loss = nn.functional.kl_div(
                    nn.functional.log_softmax(outputs, dim=1),
                    targets,
                    reduction='batchmean'
                )
            else:
                loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    # Calculate additional metrics
    metrics = {
        'model_name': model_name,
        'test_loss': avg_loss,
        'metric_name': metric_name,
        'avg_inference_latency_ms': avg_latency,
        'p95_latency_ms': p95_latency,
        'p99_latency_ms': p99_latency,
        'total_samples': total_samples,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'model_size_mb': sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
    }

    return metrics


def benchmark_inference_speed(
    model: nn.Module,
    input_size: Tuple[int, ...],
    device: str = 'cpu',
    num_runs: int = 1000
) -> Dict:
    """
    Benchmark inference speed with dummy inputs.
    """
    model.eval()
    model = model.to(device)

    # Create dummy input
    dummy_input = torch.randn(1, *input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

    return {
        'avg_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99)
    }


def evaluate_all_models(
    models_dir: Path,
    datasets_dir: Path,
    output_file: Optional[Path] = None,
    device: str = 'cpu',
    use_real_data: bool = True
):
    """
    Evaluate all trained models.
    """
    models_dir = Path(models_dir)
    datasets_dir = Path(datasets_dir)

    print("=" * 60)
    print("Kelly MIDI Companion - Model Evaluation")
    print("=" * 60)
    print(f"Models directory: {models_dir}")
    print(f"Device: {device}")
    print()

    # Model configurations
    model_classes = {
        'EmotionRecognizer': (EmotionRecognizer, (128,)),
        'MelodyTransformer': (MelodyTransformer, (64,)),
        'HarmonyPredictor': (HarmonyPredictor, (128,)),
        'DynamicsEngine': (DynamicsEngine, (32,)),
        'GroovePredictor': (GroovePredictor, (64,))
    }

    all_results = {}

    # Load and evaluate each model
    for model_name, (model_class, input_shape) in model_classes.items():
        print(f"\nEvaluating {model_name}...")
        print("-" * 40)

        # Load model
        checkpoint_path = models_dir / "checkpoints" / f"{model_name.lower()}_final.pt"
        if not checkpoint_path.exists():
            print(f"  ⚠️  Checkpoint not found: {checkpoint_path}")
            print(f"  Using untrained model for benchmarking only")
            model = model_class()
        else:
            model = model_class()
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"  ✓ Loaded checkpoint: {checkpoint_path}")

        # Benchmark inference speed
        print("  Benchmarking inference speed...")
        speed_results = benchmark_inference_speed(model, input_shape, device)
        print(f"    Avg latency: {speed_results['avg_latency_ms']:.3f} ms")
        print(f"    P95 latency: {speed_results['p95_latency_ms']:.3f} ms")
        print(f"    P99 latency: {speed_results['p99_latency_ms']:.3f} ms")

        # Evaluate on test data if available
        test_results = {}
        try:
            if model_name == 'EmotionRecognizer':
                test_dataset = EmotionDataset(
                    audio_dir=datasets_dir / "audio",
                    labels_file=datasets_dir / "audio" / "labels.csv",
                    use_synthetic=not use_real_data
                )
            elif model_name == 'MelodyTransformer':
                test_dataset = MelodyDataset(
                    midi_dir=datasets_dir / "midi",
                    emotion_labels=datasets_dir / "emotion_labels.json",
                    use_synthetic=not use_real_data
                )
            elif model_name == 'HarmonyPredictor':
                test_dataset = HarmonyDataset(
                    chord_file=datasets_dir / "chords" / "chord_progressions.json",
                    use_synthetic=not use_real_data
                )
            elif model_name == 'DynamicsEngine':
                test_dataset = DynamicsDataset(
                    midi_dir=datasets_dir / "midi",
                    use_synthetic=not use_real_data
                )
            elif model_name == 'GroovePredictor':
                test_dataset = GrooveDataset(
                    drums_dir=datasets_dir / "drums",
                    emotion_labels=datasets_dir / "drum_labels.json",
                    use_synthetic=not use_real_data
                )

            # Use last 20% for testing
            test_size = int(len(test_dataset) * 0.2)
            if test_size > 0:
                from torch.utils.data import random_split
                _, test_data = random_split(test_dataset, [len(test_dataset) - test_size, test_size])
                test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

                print("  Evaluating on test data...")
                test_results = evaluate_model(model, test_loader, device, model_name)
                print(f"    Test loss ({test_results['metric_name']}): {test_results['test_loss']:.6f}")
        except Exception as e:
            print(f"  ⚠️  Could not evaluate on test data: {e}")

        # Combine results
        results = {
            'inference_benchmark': speed_results,
            'test_evaluation': test_results if test_results else None,
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
        }

        all_results[model_name] = results

    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    total_params = 0
    total_size_mb = 0

    for model_name, results in all_results.items():
        params = results['parameter_count']
        size_mb = results['model_size_mb']
        latency = results['inference_benchmark']['avg_latency_ms']

        total_params += params
        total_size_mb += size_mb

        print(f"\n{model_name}:")
        print(f"  Parameters: {params:,}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Avg latency: {latency:.3f} ms")

        if results['test_evaluation']:
            test_loss = results['test_evaluation'].get('test_loss', 'N/A')
            print(f"  Test loss: {test_loss:.6f}")

    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Total Size: {total_size_mb:.2f} MB")
    print(f"Target: <10ms inference (all models combined)")

    # Save results
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained Kelly MIDI Companion ML models"
    )
    parser.add_argument("--models-dir", "-m", type=str, default="./trained_models",
                        help="Directory containing trained models")
    parser.add_argument("--datasets-dir", "-d", type=str, default="./datasets",
                        help="Directory containing test datasets")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file for evaluation results (JSON)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Evaluation device")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for testing")

    args = parser.parse_args()

    # Auto-detect device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = "cpu"

    evaluate_all_models(
        models_dir=Path(args.models_dir),
        datasets_dir=Path(args.datasets_dir),
        output_file=Path(args.output) if args.output else None,
        device=args.device,
        use_real_data=not args.synthetic
    )
