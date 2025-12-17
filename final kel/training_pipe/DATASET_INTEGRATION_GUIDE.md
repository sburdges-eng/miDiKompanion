# Dataset Integration Guide

This guide explains how to use the new data loaders and training utilities that integrate real datasets (DEAM, Lakh MIDI, MAESTRO, Groove MIDI) into the training pipeline.

## Overview

The training pipeline now supports:

- **Real dataset loading** via `data_loaders.py`
- **Validation splits** for proper model evaluation
- **Early stopping** to prevent overfitting
- **Checkpoint management** for model saving/loading
- **Metrics tracking** for training monitoring

## Files Created

### 1. `scripts/data_loaders.py`

Implements data loaders for real datasets:

- **EmotionDataset**: Loads DEAM dataset (audio files + CSV labels)
- **MelodyDataset**: Loads Lakh MIDI files with emotion labels
- **DynamicsDataset**: Loads MAESTRO MIDI files for dynamics extraction
- **GrooveDataset**: Loads Groove MIDI drum patterns

**Key Features:**

- Mel-spectrogram computation from audio (128-dim features)
- MIDI note extraction and processing
- Emotion embedding conversion (valence-arousal → 64-dim embedding)
- Dynamics and groove feature extraction

### 2. `scripts/training_utils.py`

Provides training utilities:

- **EarlyStopping**: Stops training when validation loss stops improving
- **CheckpointManager**: Saves/loads model checkpoints
- **MetricsTracker**: Tracks training and validation metrics
- **train_epoch()**: Standardized training loop
- **validate_epoch()**: Standardized validation loop
- **split_dataset()**: Creates train/val splits

## Usage

### Basic Training with Real Datasets

```bash
# 1. Download datasets (see download_datasets.py)
python scripts/download_datasets.py --datasets-dir ./datasets

# 2. Organize datasets
python scripts/download_datasets.py --organize

# 3. Train with real datasets
python scripts/train_all_models.py \
    --datasets-dir ./datasets \
    --output ./trained_models \
    --epochs 100 \
    --batch-size 64 \
    --device mps
```

### Using Synthetic Data (for testing)

```bash
python scripts/train_all_models.py \
    --output ./trained_models \
    --synthetic \
    --epochs 50
```

### With Validation and Early Stopping

The training script now supports validation splits. Add these to your training configuration:

```python
# In train_all_models.py or custom script
from training_utils import split_dataset, EarlyStopping, CheckpointManager

# Split dataset
train_dataset, val_dataset = split_dataset(
    emotion_dataset,
    val_split=0.2,
    shuffle=True
)

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Setup early stopping
early_stopping = EarlyStopping(patience=10, min_delta=0.001)

# Setup checkpointing
checkpoint_mgr = CheckpointManager(
    checkpoint_dir=Path("./checkpoints"),
    save_best_only=True
)

# Training loop with validation
for epoch in range(epochs):
    train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
    val_metrics = validate_epoch(model, val_loader, criterion, device)

    # Check early stopping
    if early_stopping(val_metrics['val_loss'], epoch):
        break

    # Save checkpoint
    checkpoint_mgr.save_checkpoint(
        model, epoch, val_metrics['val_loss'],
        optimizer=optimizer,
        model_name="emotion_recognizer"
    )
```

## Dataset Requirements

### EmotionRecognizer (DEAM)

**Directory Structure:**

```
datasets/training/audio/
├── labels.csv
├── audio_001.wav
├── audio_002.wav
└── ...
```

**labels.csv Format:**

```csv
filename,valence,arousal
audio_001.wav,0.8,0.9
audio_002.wav,-0.6,0.3
```

**Alternative (DEAM format):**

```csv
filename,valence_mean,arousal_mean
audio_001.wav,0.8,0.9
```

### MelodyTransformer (Lakh MIDI)

**Directory Structure:**

```
datasets/training/midi/
├── emotion_labels.json
├── melody_001.mid
├── melody_002.mid
└── ...
```

**emotion_labels.json Format:**

```json
{
  "melody_001.mid": {
    "valence": 0.7,
    "arousal": 0.6
  },
  "melody_002.mid": {
    "valence": -0.4,
    "arousal": 0.5
  }
}
```

### DynamicsEngine (MAESTRO)

**Directory Structure:**

```
datasets/training/dynamics_midi/
├── piece_001.mid
├── piece_002.mid
└── ...
```

MIDI files should contain velocity information. The loader extracts:

- Mean/std/range of velocities
- Velocity distribution
- Timing features
- Note density and accent ratios

### GroovePredictor (Groove MIDI)

**Directory Structure:**

```
datasets/training/drums/
├── drum_labels.json  (optional)
├── groove_001.mid
├── groove_002.mid
└── ...
```

**drum_labels.json Format:**

```json
{
  "groove_001.mid": {
    "valence": 0.6,
    "arousal": 0.7
  }
}
```

## Data Processing Details

### Audio → Mel-Spectrogram

The `EmotionDataset` automatically:

1. Loads audio files (supports .wav, .mp3, .flac)
2. Resamples to 22050 Hz
3. Computes 128-band mel-spectrogram
4. Averages over time to get 128-dim feature vector
5. Normalizes to [-1, 1]

### MIDI → Note Probabilities

The `MelodyDataset`:

1. Parses MIDI files using `pretty_midi`
2. Extracts all non-drum notes
3. Counts note occurrences (MIDI pitches 0-127)
4. Normalizes to probability distribution

### MIDI → Dynamics Features

The `DynamicsDataset` extracts:

- **16-dim feature vector** including:
  - Velocity statistics (mean, std, range, median)
  - Velocity distribution (4 bins)
  - Timing features (note lengths)
  - Expressive features (accent ratio, note density)

### MIDI → Groove Features

The `GrooveDataset` extracts:

- **32-dim feature vector** including:
  - Timing regularity
  - Drum pattern statistics (kick, snare, hihat)
  - Velocity patterns per drum
  - Syncopation measures

## Validation and Early Stopping

### Validation Split

Use `split_dataset()` to create train/val splits:

```python
from training_utils import split_dataset

train_dataset, val_dataset = split_dataset(
    dataset,
    val_split=0.2,  # 20% for validation
    shuffle=True,
    seed=42
)
```

### Early Stopping

```python
from training_utils import EarlyStopping

early_stopping = EarlyStopping(
    patience=10,      # Stop after 10 epochs without improvement
    min_delta=0.001,  # Minimum improvement to count as progress
    mode='min',       # 'min' for loss, 'max' for accuracy
    verbose=True
)

# In training loop
if early_stopping(val_loss, epoch):
    print("Early stopping triggered")
    break
```

### Checkpointing

```python
from training_utils import CheckpointManager

checkpoint_mgr = CheckpointManager(
    checkpoint_dir=Path("./checkpoints"),
    save_best_only=True,  # Only keep best model
    verbose=True
)

# Save checkpoint
checkpoint_mgr.save_checkpoint(
    model=model,
    epoch=epoch,
    score=val_loss,
    optimizer=optimizer,
    metadata={'learning_rate': lr},
    model_name="emotion_recognizer"
)

# Load checkpoint
checkpoint_mgr.load_checkpoint(
    model=model,
    checkpoint_path=Path("./checkpoints/emotion_recognizer_best.pt"),
    optimizer=optimizer
)
```

## Metrics Tracking

```python
from training_utils import MetricsTracker

metrics = MetricsTracker()

# Update metrics each epoch
metrics.update({
    'train_loss': train_loss,
    'val_loss': val_loss,
    'learning_rate': lr
})
metrics.end_epoch()

# Get history
train_losses = metrics.get_history('train_loss')
latest_loss = metrics.get_latest('val_loss')

# Save to JSON
metrics.save_to_json(Path("./metrics.json"))

# Plot (requires matplotlib)
metrics.plot_metrics(['train_loss', 'val_loss'], save_path=Path("./metrics.png"))
```

## Troubleshooting

### Import Errors

If you get import errors, make sure `data_loaders.py` and `training_utils.py` are in the same directory as `train_all_models.py`:

```bash
cd training_pipe/scripts
python train_all_models.py --help
```

### Missing Dependencies

Install required packages:

```bash
pip install librosa soundfile mido pretty_midi
```

### MIDI Processing Errors

If MIDI processing fails:

1. Check that MIDI files are valid (not corrupted)
2. Ensure `mido` and `pretty_midi` are installed
3. Some MIDI files may have unusual formats - they'll be skipped with a warning

### Audio Processing Errors

If audio processing fails:

1. Ensure `librosa` and `soundfile` are installed
2. Check audio file formats (supported: .wav, .mp3, .flac)
3. Very large audio files may cause memory issues - consider limiting duration

### Out of Memory

If you run out of memory:

1. Reduce `batch_size` (e.g., from 64 to 32)
2. Limit number of workers in DataLoader (`num_workers=0`)
3. Process audio files with limited duration (set `audio_duration` parameter)

## Next Steps

1. **Download real datasets** using `download_datasets.py`
2. **Organize datasets** into the expected directory structure
3. **Run training** with validation and early stopping
4. **Monitor metrics** to track training progress
5. **Load best checkpoints** for inference or continued training

## Integration with Existing Code

The new data loaders are compatible with the existing training script. The script will:

1. Try to load real datasets if `--datasets-dir` is provided
2. Fall back to synthetic data if datasets are missing
3. Use existing training functions that already support validation

To fully leverage the new utilities (early stopping, checkpointing, metrics), consider updating individual training functions to use the utilities from `training_utils.py`.
