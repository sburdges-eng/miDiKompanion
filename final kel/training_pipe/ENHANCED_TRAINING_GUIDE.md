# Enhanced Training Pipeline Guide

This guide explains how to use the enhanced training pipeline with validation splits, early stopping, metrics tracking, checkpoint management, hyperparameter tuning, and visualization.

## Features

### ✅ Validation Splits

- Automatic train/validation/test dataset splitting
- Configurable split ratios
- Stratified splitting support

### ✅ Early Stopping

- Prevents overfitting by monitoring validation loss
- Configurable patience and minimum delta
- Automatic best model restoration

### ✅ Metrics Tracking

- Training and validation loss
- Accuracy metrics (for classification tasks)
- Learning rate tracking
- Epoch timing
- JSON export for analysis

### ✅ Checkpoint Management

- Automatic checkpoint saving
- Best model tracking
- Resume training from checkpoints
- Configurable checkpoint retention

### ✅ Learning Rate Scheduling

- Reduce on plateau
- Step decay
- Cosine annealing
- Exponential decay

### ✅ Metrics Visualization

- Training curves (loss, accuracy, LR, timing)
- Comparison plots for multiple runs
- Text summaries

### ✅ Hyperparameter Tuning

- Grid search
- Random search
- Bayesian optimization (planned)

## Quick Start

### Basic Training with Enhanced Features

```bash
# Train with default settings (validation split, early stopping, etc.)
python scripts/train_all_models_enhanced.py \
    --output ./trained_models \
    --epochs 100 \
    --batch-size 64 \
    --device mps \
    --validation-split 0.2 \
    --early-stopping-patience 10
```

### Resume Training from Checkpoint

```bash
python scripts/train_all_models_enhanced.py \
    --output ./trained_models \
    --resume \
    --epochs 100
```

### Training with Learning Rate Scheduling

```bash
python scripts/train_all_models_enhanced.py \
    --output ./trained_models \
    --lr-scheduler reduce_on_plateau \
    --epochs 100
```

### Using Configuration File

```bash
python scripts/train_all_models_enhanced.py \
    --config configs/training_config.json \
    --output ./trained_models
```

## Configuration File Format

Create a `training_config.json` file:

```json
{
  "training": {
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 0.001,
    "device": "mps",
    "validation_split": 0.2,
    "early_stopping_patience": 10,
    "lr_scheduler": "reduce_on_plateau"
  },
  "models": {
    "emotion_recognizer": {
      "learning_rate": 0.001,
      "weight_decay": 0.0001
    },
    "melody_transformer": {
      "learning_rate": 0.0005,
      "weight_decay": 0.0001
    }
  }
}
```

## Output Structure

After training, you'll find:

```
trained_models/
├── models/                    # RTNeural JSON models
│   ├── emotionrecognizer.json
│   ├── melodytransformer.json
│   └── ...
├── checkpoints/               # Model checkpoints
│   ├── emotion_recognizer/
│   │   ├── best_model.pt
│   │   ├── latest_checkpoint.pt
│   │   └── checkpoint_epoch_*.pt
│   └── ...
├── metrics/                    # Training metrics
│   ├── emotionrecognizer_metrics.json
│   ├── emotionrecognizer_curves.png
│   ├── emotionrecognizer_summary.txt
│   └── ...
└── training_summary.txt        # Overall summary
```

## Advanced Usage

### Custom Training Function

You can use the enhanced training utilities in your own code:

```python
from utils.enhanced_training import train_model_with_validation
from utils.training_utils import (
    EarlyStopping,
    CheckpointManager,
    LearningRateScheduler
)

# Setup utilities
early_stopping = EarlyStopping(patience=10, min_delta=0.001)
checkpoint_manager = CheckpointManager(checkpoint_dir="./checkpoints")
lr_scheduler = LearningRateScheduler(optimizer, mode='reduce_on_plateau')

# Train
metrics = train_model_with_validation(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=100,
    device='mps',
    early_stopping=early_stopping,
    checkpoint_manager=checkpoint_manager,
    lr_scheduler=lr_scheduler
)
```

### Hyperparameter Tuning

```python
from utils.hyperparameter_tuning import (
    HyperparameterConfig,
    RandomSearchTuner,
    run_hyperparameter_tuning
)

# Define hyperparameter search space
configs = [
    HyperparameterConfig('learning_rate', 'log_float', 1e-5, 1e-2),
    HyperparameterConfig('batch_size', 'choice', choices=[32, 64, 128]),
    HyperparameterConfig('weight_decay', 'log_float', 1e-6, 1e-3)
]

# Create tuner
tuner = RandomSearchTuner(
    hyperparameter_configs=configs,
    output_dir="./hyperparameter_tuning",
    max_trials=50
)

# Run tuning
best_trial = run_hyperparameter_tuning(
    tuner=tuner,
    train_fn=your_training_function,
    model_class=YourModel,
    train_loader=train_loader,
    val_loader=val_loader,
    device='mps',
    epochs_per_trial=20
)
```

### Dataset Splitting

```python
from utils.dataset_utils import split_dataset, create_data_loaders

# Split dataset
train_dataset, val_dataset, test_dataset = split_dataset(
    dataset=full_dataset,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
)

# Create loaders
train_loader, val_loader, test_loader = create_data_loaders(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=64,
    num_workers=2
)
```

### Metrics Visualization

```python
from utils.metrics_visualization import (
    plot_training_curves,
    create_training_summary
)

# Plot training curves
plot_training_curves(
    metrics=metrics,
    output_path="./curves.png",
    model_name="EmotionRecognizer"
)

# Create text summary
create_training_summary(
    metrics=metrics,
    output_path="./summary.txt",
    model_name="EmotionRecognizer"
)
```

## Command Line Options

### Main Options

- `--output, -o`: Output directory (default: `./trained_models`)
- `--datasets-dir, -d`: Directory containing datasets
- `--epochs, -e`: Number of epochs (default: 50)
- `--batch-size, -b`: Batch size (default: 64)
- `--device`: Device: `cpu`, `cuda`, or `mps` (default: `cpu`)
- `--synthetic, -s`: Use synthetic data

### Enhanced Features

- `--validation-split`: Validation split ratio (default: 0.2)
- `--early-stopping-patience`: Early stopping patience (default: 10)
- `--lr-scheduler`: LR scheduler mode: `reduce_on_plateau`, `step`, `cosine`, `exponential` (default: `reduce_on_plateau`)
- `--resume`: Resume from latest checkpoint
- `--config`: Path to training config JSON

## Best Practices

1. **Always use validation splits**: Set `--validation-split 0.2` to monitor overfitting
2. **Enable early stopping**: Use `--early-stopping-patience 10` to prevent overfitting
3. **Use learning rate scheduling**: `--lr-scheduler reduce_on_plateau` helps convergence
4. **Save checkpoints**: Checkpoints are automatically saved, use `--resume` to continue
5. **Monitor metrics**: Check the `metrics/` directory for training curves and summaries
6. **Start with fewer epochs**: Use `--epochs 20` for quick tests, then scale up

## Troubleshooting

### Import Errors

If you get import errors, make sure the `utils/` directory is in your Python path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
```

### Checkpoint Not Found

If `--resume` fails, check that checkpoints exist in `output_dir/checkpoints/`.

### Out of Memory

Reduce batch size: `--batch-size 32` or `--batch-size 16`

### Training Too Slow

- Use GPU: `--device cuda` or `--device mps`
- Reduce epochs for testing: `--epochs 10`
- Increase batch size if memory allows

## Examples

### Example 1: Quick Test Run

```bash
python scripts/train_all_models_enhanced.py \
    --output ./test_run \
    --epochs 10 \
    --batch-size 32 \
    --synthetic \
    --validation-split 0.2
```

### Example 2: Full Training Run

```bash
python scripts/train_all_models_enhanced.py \
    --output ./full_training \
    --epochs 200 \
    --batch-size 64 \
    --device mps \
    --validation-split 0.2 \
    --early-stopping-patience 15 \
    --lr-scheduler reduce_on_plateau \
    --config configs/training_config.json
```

### Example 3: Resume Training

```bash
python scripts/train_all_models_enhanced.py \
    --output ./full_training \
    --resume \
    --epochs 200
```

## Next Steps

1. **Dataset Integration**: Integrate real datasets (DEAM, Lakh MIDI, etc.)
2. **Hyperparameter Tuning**: Run hyperparameter search for optimal settings
3. **Model Evaluation**: Evaluate models on test set
4. **Production Export**: Export best models to RTNeural format

## See Also

- `training_pipe/README.md` - Original training pipeline documentation
- `training_pipe/configs/training_config.json` - Example configuration
- `training_pipe/utils/` - Utility module documentation
