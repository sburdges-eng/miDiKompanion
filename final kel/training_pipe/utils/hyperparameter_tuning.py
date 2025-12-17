#!/usr/bin/env python3
"""
Hyperparameter Tuning Framework for Kelly MIDI Companion ML Training
======================================================================
Provides grid search, random search, and Bayesian optimization for hyperparameters.
"""

import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import itertools
import random
from .training_utils import TrainingMetrics, CheckpointManager


@dataclass
class HyperparameterConfig:
    """Configuration for a single hyperparameter."""
    name: str
    type: str  # 'float', 'int', 'choice', 'log_float', 'log_int'
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None

    def sample(self) -> Any:
        """Sample a random value for this hyperparameter."""
        if self.type == 'float':
            return random.uniform(self.min_value, self.max_value)
        elif self.type == 'int':
            return random.randint(int(self.min_value), int(self.max_value))
        elif self.type == 'choice':
            return random.choice(self.choices)
        elif self.type == 'log_float':
            log_min = np.log(self.min_value)
            log_max = np.log(self.max_value)
            return np.exp(random.uniform(log_min, log_max))
        elif self.type == 'log_int':
            log_min = np.log(self.min_value)
            log_max = np.log(self.max_value)
            return int(np.exp(random.uniform(log_min, log_max)))
        else:
            raise ValueError(f"Unknown hyperparameter type: {self.type}")


@dataclass
class TrialResult:
    """Result of a single hyperparameter trial."""
    trial_id: int
    hyperparameters: Dict[str, Any]
    best_val_loss: float
    best_val_accuracy: Optional[float]
    final_train_loss: float
    final_val_loss: float
    epochs_trained: int
    training_time: float
    metrics: Optional[TrainingMetrics] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        if self.metrics is not None:
            result['metrics'] = self.metrics.to_dict()
        return result


class HyperparameterTuner:
    """Base class for hyperparameter tuning."""

    def __init__(
        self,
        hyperparameter_configs: List[HyperparameterConfig],
        output_dir: Path,
        max_trials: int = 50
    ):
        """
        Args:
            hyperparameter_configs: List of hyperparameter configurations
            output_dir: Directory to save trial results
            max_trials: Maximum number of trials to run
        """
        self.hyperparameter_configs = hyperparameter_configs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_trials = max_trials
        self.trial_results: List[TrialResult] = []

    def sample_hyperparameters(self) -> Dict[str, Any]:
        """Sample a random set of hyperparameters."""
        return {
            config.name: config.sample()
            for config in self.hyperparameter_configs
        }

    def save_results(self):
        """Save all trial results to JSON."""
        results_path = self.output_dir / "trial_results.json"
        results_data = {
            'total_trials': len(self.trial_results),
            'trials': [result.to_dict() for result in self.trial_results]
        }
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        # Also save best trial
        if self.trial_results:
            best_trial = min(
                self.trial_results,
                key=lambda x: x.best_val_loss
            )
            best_path = self.output_dir / "best_trial.json"
            with open(best_path, 'w') as f:
                json.dump(best_trial.to_dict(), f, indent=2)

    def load_results(self) -> List[TrialResult]:
        """Load trial results from JSON."""
        results_path = self.output_dir / "trial_results.json"
        if not results_path.exists():
            return []

        with open(results_path, 'r') as f:
            data = json.load(f)

        results = []
        for trial_data in data['trials']:
            metrics = None
            if 'metrics' in trial_data and trial_data['metrics']:
                metrics = TrainingMetrics(**trial_data['metrics'])
            result = TrialResult(
                trial_id=trial_data['trial_id'],
                hyperparameters=trial_data['hyperparameters'],
                best_val_loss=trial_data['best_val_loss'],
                best_val_accuracy=trial_data.get('best_val_accuracy'),
                final_train_loss=trial_data['final_train_loss'],
                final_val_loss=trial_data['final_val_loss'],
                epochs_trained=trial_data['epochs_trained'],
                training_time=trial_data['training_time'],
                metrics=metrics
            )
            results.append(result)

        self.trial_results = results
        return results


class GridSearchTuner(HyperparameterTuner):
    """Grid search hyperparameter tuning."""

    def __init__(
        self,
        hyperparameter_configs: List[HyperparameterConfig],
        output_dir: Path
    ):
        """
        Args:
            hyperparameter_configs: List of hyperparameter configurations
            output_dir: Directory to save trial results
        """
        # For grid search, only 'choice' type is supported
        for config in hyperparameter_configs:
            if config.type != 'choice':
                raise ValueError(
                    f"Grid search only supports 'choice' type. "
                    f"Found '{config.type}' for '{config.name}'"
                )

        # Calculate total combinations
        choices_list = [config.choices for config in hyperparameter_configs]
        total_combinations = len(list(itertools.product(*choices_list)))

        super().__init__(hyperparameter_configs, output_dir, max_trials=total_combinations)
        self.total_combinations = total_combinations

    def get_all_combinations(self) -> List[Dict[str, Any]]:
        """Get all hyperparameter combinations for grid search."""
        config_names = [config.name for config in self.hyperparameter_configs]
        choices_list = [config.choices for config in self.hyperparameter_configs]

        combinations = []
        for combo in itertools.product(*choices_list):
            combinations.append(dict(zip(config_names, combo)))

        return combinations


class RandomSearchTuner(HyperparameterTuner):
    """Random search hyperparameter tuning."""

    def __init__(
        self,
        hyperparameter_configs: List[HyperparameterConfig],
        output_dir: Path,
        max_trials: int = 50
    ):
        """
        Args:
            hyperparameter_configs: List of hyperparameter configurations
            output_dir: Directory to save trial results
            max_trials: Maximum number of random trials
        """
        super().__init__(hyperparameter_configs, output_dir, max_trials)


def run_hyperparameter_tuning(
    tuner: HyperparameterTuner,
    train_fn: Callable,
    model_class: type,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: str = 'cpu',
    epochs_per_trial: int = 20
) -> TrialResult:
    """
    Run hyperparameter tuning.

    Args:
        tuner: HyperparameterTuner instance
        train_fn: Training function that takes (model, train_loader, val_loader, hyperparams, epochs, device)
        model_class: Model class to instantiate
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs_per_trial: Number of epochs per trial

    Returns:
        Best trial result
    """
    if isinstance(tuner, GridSearchTuner):
        hyperparameter_sets = tuner.get_all_combinations()
    else:  # RandomSearchTuner
        hyperparameter_sets = [
            tuner.sample_hyperparameters()
            for _ in range(tuner.max_trials)
        ]

    print(f"\n{'='*60}")
    print(f"Starting Hyperparameter Tuning")
    print(f"{'='*60}")
    print(f"Total trials: {len(hyperparameter_sets)}")
    print(f"Epochs per trial: {epochs_per_trial}")
    print(f"{'='*60}\n")

    for trial_id, hyperparams in enumerate(hyperparameter_sets, 1):
        print(f"\nTrial {trial_id}/{len(hyperparameter_sets)}")
        print(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}")

        # Create model with hyperparameters
        model = model_class(**hyperparams)

        # Train model
        import time
        start_time = time.time()

        metrics = train_fn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs_per_trial,
            device=device,
            **hyperparams
        )

        training_time = time.time() - start_time

        # Extract results
        best_val_loss = min(metrics.val_loss) if metrics.val_loss else float('inf')
        best_val_accuracy = max(metrics.val_accuracy) if metrics.val_accuracy else None

        result = TrialResult(
            trial_id=trial_id,
            hyperparameters=hyperparams,
            best_val_loss=best_val_loss,
            best_val_accuracy=best_val_accuracy,
            final_train_loss=metrics.train_loss[-1] if metrics.train_loss else float('inf'),
            final_val_loss=metrics.val_loss[-1] if metrics.val_loss else float('inf'),
            epochs_trained=epochs_per_trial,
            training_time=training_time,
            metrics=metrics
        )

        tuner.trial_results.append(result)

        print(f"  Best Val Loss: {best_val_loss:.6f}")
        if best_val_accuracy is not None:
            print(f"  Best Val Accuracy: {best_val_accuracy:.4f}")
        print(f"  Training Time: {training_time:.2f}s")

    # Save results
    tuner.save_results()

    # Find and return best trial
    best_trial = min(tuner.trial_results, key=lambda x: x.best_val_loss)

    print(f"\n{'='*60}")
    print(f"Hyperparameter Tuning Complete")
    print(f"{'='*60}")
    print(f"Best Trial: {best_trial.trial_id}")
    print(f"Best Val Loss: {best_trial.best_val_loss:.6f}")
    print(f"Hyperparameters: {json.dumps(best_trial.hyperparameters, indent=2)}")
    print(f"{'='*60}\n")

    return best_trial
