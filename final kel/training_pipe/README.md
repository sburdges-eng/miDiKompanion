# Kelly MIDI Companion - ML Training Pipeline

Complete training pipeline for the 5-model neural network architecture.

## Quick Start

```bash
# 1. Extract this archive
unzip training_pipe.zip
cd training_pipe

# 2. Run setup
chmod +x setup.sh
./setup.sh

# 3. Activate environment
source venv/bin/activate

# 4. Prepare datasets (optional)
python scripts/prepare_datasets.py --datasets-dir ./datasets

# 5. Train all models
python scripts/train_all_models.py --output ../models --epochs 100 --device mps
```

## What's Included

```
training_pipe/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── setup.sh                       # Automated setup script
├── scripts/
│   ├── train_all_models.py       # Main training script
│   └── prepare_datasets.py       # Dataset preparation
├── configs/
│   └── training_config.json      # Training configuration
├── datasets/                      # Place your data here
│   ├── audio/                    # Audio files for EmotionRecognizer
│   ├── midi/                     # MIDI files for MelodyTransformer
│   ├── chords/                   # Chord progressions for HarmonyPredictor
│   └── drums/                    # Drum patterns for GroovePredictor
├── examples/
│   └── example_datasets.md       # Example dataset info
└── utils/
    └── export_helpers.py         # RTNeural export utilities
```

## The 5 Models

1. **EmotionRecognizer** (128→512→256→128→64) - ~500K params
   - Input: 128-dim mel-spectrogram features
   - Output: 64-dim emotion embedding

2. **MelodyTransformer** (64→256→256→256→128) - ~400K params
   - Input: 64-dim emotion embedding
   - Output: 128-dim MIDI note probabilities

3. **HarmonyPredictor** (128→256→128→64) - ~100K params
   - Input: 128-dim context (emotion + audio)
   - Output: 64-dim chord probabilities

4. **DynamicsEngine** (32→128→64→16) - ~20K params
   - Input: 32-dim compact context
   - Output: 16-dim velocity/timing/expression

5. **GroovePredictor** (64→128→64→32) - ~25K params
   - Input: 64-dim emotion embedding
   - Output: 32-dim groove parameters

**Total**: ~1M parameters, ~4MB memory, <10ms inference

## Training Options

### Basic Training

```bash
python scripts/train_all_models.py
```

### Advanced Options

```bash
python scripts/train_all_models.py \
    --output ../models \
    --epochs 200 \
    --batch-size 128 \
    --device cuda \
    --learning-rate 0.0001
```

### Available Options

- `--output`, `-o` - Output directory (default: `./trained_models`)
- `--epochs`, `-e` - Training epochs (default: `50`)
- `--batch-size`, `-b` - Batch size (default: `64`)
- `--device`, `-d` - Device: `cpu`, `cuda`, `mps` (default: `cpu`)

## Dataset Preparation

### 1. Emotion Recognition

Place audio files in `datasets/audio/` and create `labels.csv`:

```csv
filename,valence,arousal
happy_001.wav,0.8,0.9
sad_001.wav,-0.6,0.3
calm_001.wav,0.2,-0.5
```

### 2. Melody Generation

Place MIDI files in `datasets/midi/` and create `emotion_labels.json`:

```json
{
  "melody_001.mid": {"valence": 0.7, "arousal": 0.6},
  "melody_002.mid": {"valence": -0.4, "arousal": 0.5}
}
```

### 3. Chord Progressions

Create `datasets/chords/progressions.json`:

```json
{
  "progressions": [
    {
      "name": "I-V-vi-IV",
      "chords": ["C", "G", "Am", "F"],
      "emotion": {"valence": 0.7, "arousal": 0.6}
    }
  ]
}
```

### 4. Dynamics (Uses MIDI files)

The DynamicsEngine uses the same MIDI files from `datasets/midi/` and extracts velocity information automatically.

### 5. Drum Patterns

Place drum MIDI files in `datasets/drums/` with emotion labels.

## Recommended Datasets

### EmotionRecognizer
- **DEAM**: 14,000 audio clips with valence/arousal labels
  - https://cvml.unige.ch/databases/DEAM/
- **PMEmo**: 794 music tracks with emotion labels
  - http://pmemo.allmusic.top/

### MelodyTransformer
- **Lakh MIDI Dataset**: 176,581 MIDI files
  - https://colinraffel.com/projects/lmd/
- Combine with DEAM emotion labels

### HarmonyPredictor
- **iRealPro**: Jazz chord progressions
  - https://www.irealpro.com/
- **Hooktheory**: Analyzed chord progressions
  - https://www.hooktheory.com/theorytab

### DynamicsEngine
- **MAESTRO**: 200+ hours of piano with dynamics
  - https://magenta.tensorflow.org/datasets/maestro

### GroovePredictor
- **Groove MIDI Dataset**: 1,150 drum patterns
  - https://magenta.tensorflow.org/datasets/groove
- **E-GMD**: Electronic music drums
  - https://magenta.tensorflow.org/datasets/e-gmd

## Output Format

Training produces two sets of files:

### 1. RTNeural JSON (for C++ plugin)

```
../models/
├── emotionrecognizer.json
├── melodytransformer.json
├── harmonypredictor.json
├── dynamicsengine.json
└── groovepredictor.json
```

Copy these to your Kelly MIDI Companion plugin:

```bash
cp ../models/*.json "/path/to/Kelly MIDI Companion.app/Contents/Resources/models/"
```

### 2. PyTorch Checkpoints (for continued training)

```
../models/checkpoints/
├── emotionrecognizer.pt
├── melodytransformer.pt
└── ...
```

Use these to continue training or fine-tune models.

## Troubleshooting

### Setup Issues

**Problem**: `setup.sh` fails

**Solution**: Make it executable first:
```bash
chmod +x setup.sh
./setup.sh
```

### CUDA Out of Memory

**Solution**: Reduce batch size:
```bash
python scripts/train_all_models.py --batch-size 32
```

### MPS Not Available (Apple Silicon)

**Check**:
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

**Fix**: Install MPS-enabled PyTorch:
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Training Too Slow

**Use GPU**: Add `--device cuda` or `--device mps`

**Reduce epochs for testing**: `--epochs 10`

## Integration with Plugin

Once training is complete:

1. **Copy models to plugin**:
   ```bash
   cp ../models/*.json "/path/to/Kelly MIDI Companion.app/Contents/Resources/models/"
   ```

2. **Rebuild plugin** (optional, if you modified the architecture):
   ```bash
   cd ../
   cmake --build build --config Release
   ```

3. **Launch plugin** - Models will auto-load on startup

4. **Verify in plugin logs**:
   ```
   MultiModelProcessor initialized:
     Total params: 1,016,880
     Total memory: 3971 KB
   ```

## Performance Tips

### Training Speed
- Use GPU: 20-100x faster than CPU
- Increase batch size if memory allows
- Use mixed precision (CUDA only): `torch.cuda.amp`

### Model Quality
- More epochs (200+) for better convergence
- Data augmentation: pitch shift, time stretch
- Regularization: dropout, weight decay
- Early stopping: monitor validation loss

### Memory Usage
- Reduce batch size if OOM
- Gradient accumulation for large effective batch sizes
- Model checkpointing for very large models

## Advanced Usage

### Custom Model Architecture

To modify a model architecture:

1. Edit `scripts/train_all_models.py`
2. Update the model class (e.g., `EmotionRecognizer`)
3. Retrain with new architecture
4. Update C++ plugin if input/output sizes changed

### Fine-Tuning

Continue training from checkpoint:

```bash
# Train initial model
python scripts/train_all_models.py --output ../models --epochs 50

# Fine-tune with more data
python scripts/train_all_models.py \
    --output ../models \
    --checkpoint ../models/checkpoints/emotionrecognizer.pt \
    --epochs 50 \
    --learning-rate 0.00001
```

### Multi-GPU Training

```bash
# Use all available GPUs
python scripts/train_all_models.py --device cuda --multi-gpu
```

## Documentation

- **Main Guide**: `../MULTI_MODEL_ML_GUIDE.md`
- **Build Verification**: `../MARKDOWN/BUILD_VERIFICATION.md`
- **Project README**: `../README.md`

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the main documentation
3. Check plugin logs for error messages

## License

Part of the Kelly MIDI Companion project.
See main project LICENSE for details.

---

**Version**: 1.0.0
**Date**: December 16, 2024
**Status**: Ready for training with real data
