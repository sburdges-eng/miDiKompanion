# ML Training Guide

Complete guide for training the 5-model neural network architecture for Kelly MIDI Companion.

## Quick Start

```bash
# 1. Navigate to training directory
cd ml_training

# 2. Install dependencies (if not already installed)
pip install torch torchaudio librosa numpy matplotlib mido

# 3. Train all models with default settings
python train_all_models.py --output ../models --epochs 50

# 4. Validate exported models
python validate_models.py ../models/*.json --check-specs
```

## Prerequisites

### Python Environment

- Python 3.9 or higher
- PyTorch 1.12+ (with CUDA/MPS support if available)
- Required packages: `torch`, `torchaudio`, `librosa`, `numpy`, `matplotlib`, `mido`

### Datasets (Optional)

For best results, use real datasets:

1. **DEAM** (Emotion Recognition)
   - Download from: <https://cvml.unige.ch/databases/DEAM/>
   - Structure: `datasets/deam/audio/` and `datasets/deam/annotations/annotations.csv`

2. **Lakh MIDI** (Melody Generation)
   - Download from: <https://colinraffel.com/projects/lmd/>
   - Structure: `datasets/lakh_midi/*.mid`

3. **MAESTRO** (Dynamics)
   - Download from: <https://magenta.tensorflow.org/datasets/maestro>
   - Structure: `datasets/maestro/*.midi`

4. **Groove MIDI** (Groove Patterns)
   - Download from: <https://magenta.tensorflow.org/datasets/groove>
   - Structure: `datasets/groove_midi/*.midi`

5. **Chord Progressions** (Harmony)
   - Create JSON file: `datasets/chords/chord_progressions.json`
   - Format: `[{"chords": ["C", "Am", "F", "G"], "emotion": {"valence": 0.5, "arousal": 0.7}}, ...]`

## Training Configuration

Edit `ml_training/config.json` to customize training parameters. The configuration file supports per-model settings:

```json
{
  "training": {
    "default_epochs": 50,
    "default_batch_size": 64,
    "default_learning_rate": 0.001,
    "default_validation_split": 0.2,
    "early_stopping": {
      "patience": 10,
      "min_delta": 0.001,
      "mode": "min"
    },
    "checkpointing": {
      "save_best": true,
      "save_latest": true,
      "save_interval": 10
    }
  },
  "models": {
    "EmotionRecognizer": {
      "input_size": 128,
      "output_size": 64,
      "estimated_params": 403264,
      "loss": "mse",
      "learning_rate": 0.001,
      "dataset": "deam"
    },
    "MelodyTransformer": {
      "input_size": 64,
      "output_size": 128,
      "estimated_params": 641664,
      "loss": "bce",
      "learning_rate": 0.001,
      "dataset": "lakh"
    }
  },
  "datasets": {
    "deam": {
      "path": "datasets/deam",
      "annotations_file": "annotations/annotations.csv"
    }
  }
}
```

**Note**: The training script (`train_all_models.py`) loads configuration from `config.json` by default, with command-line arguments taking precedence. This allows easy customization without modifying code.

## Training Commands

### Basic Training

Train all 5 models with default settings:

```bash
python train_all_models.py --output ../models
```

### Advanced Training

```bash
python train_all_models.py \
    --output ../models \
    --epochs 100 \
    --batch-size 128 \
    --device mps \
    --learning-rate 0.0001 \
    --validation-split 0.2 \
    --early-stopping-patience 15 \
    --datasets-dir ../datasets
```

### Training Individual Models

For training individual models, you can modify `train_all_models.py` to train only specific models, or use the dedicated script:

```bash
# Train only EmotionRecognizer (if train_emotion_model.py exists)
python train_emotion_model.py \
    --dataset ../datasets/deam \
    --epochs 100 \
    --output ../models/emotionrecognizer.json
```

**Note**: The recommended approach is to use `train_all_models.py` which trains all 5 models in sequence. Individual model training scripts may have different argument formats.

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output`, `-o` | Output directory for models | `./trained_models` |
| `--epochs`, `-e` | Number of training epochs | `50` |
| `--batch-size`, `-b` | Training batch size | `64` |
| `--device`, `-d` | Device: `cpu`, `cuda`, `mps` | `cpu` |
| `--learning-rate`, `-lr` | Learning rate | `0.001` |
| `--validation-split`, `-v` | Validation set ratio | `0.2` |
| `--early-stopping-patience` | Early stopping patience | `10` |
| `--datasets-dir` | Directory containing datasets | None |
| `--use-synthetic` | Force synthetic data | False |
| `--no-history` | Don't save training history | False |
| `--no-plots` | Don't generate plots | False |

## Training Process

### 1. Dataset Preparation

The training script automatically detects and loads datasets:

```bash
# Test dataset loader
python -c "from dataset_loaders import create_dataset; ds = create_dataset('deam', 'datasets/deam'); print(f'Loaded {len(ds)} samples')"
```

### 2. Model Training

**All 5 models are now fully trainable** with both real datasets and synthetic fallback support. Training proceeds in order:

1. **EmotionRecognizer** - Trains on DEAM dataset (or synthetic fallback)
2. **MelodyTransformer** - Trains on Lakh MIDI dataset (or synthetic fallback)
3. **HarmonyPredictor** - Trains on chord progressions (or synthetic fallback)
4. **DynamicsEngine** - Trains on MAESTRO dataset (or synthetic fallback)
5. **GroovePredictor** - Trains on Groove MIDI dataset (or synthetic fallback)

Each model:

- Automatically detects and loads real datasets if available
- Falls back to synthetic datasets if real data is unavailable
- Splits data into train/validation sets
- Trains with early stopping and validation monitoring
- Saves checkpoints (best and latest)
- Tracks training metrics (loss curves, validation accuracy)
- Exports to RTNeural JSON format (v2.0)

### 3. Model Export

After training, models are automatically exported to RTNeural JSON format:

```
models/
├── emotionrecognizer.json
├── melodytransformer.json
├── harmonypredictor.json
├── dynamicsengine.json
├── groovepredictor.json
├── checkpoints/
│   ├── emotionrecognizer_best.pt
│   ├── emotionrecognizer_latest.pt
│   └── ...
├── history/
│   ├── emotionrecognizer_history.json
│   └── ...
└── plots/
    ├── emotionrecognizer_loss.png
    └── ...
```

## Model Validation

Validate exported models before use:

```bash
# Validate all models
python validate_models.py models/*.json --check-specs

# Validate specific model
python validate_models.py models/emotionrecognizer.json --verbose
```

The validation script checks:

- JSON structure correctness
- Layer dimensions match specifications
- LSTM weight splitting (4 gates)
- Parameter counts match expected values
- Input/output sizes match C++ ModelSpec definitions

## Integration with C++ Plugin

### 1. Copy Models to Plugin Resources

```bash
# macOS
cp models/*.json "Kelly MIDI Companion.app/Contents/Resources/models/"

# Or set in plugin code:
juce::File modelsDir = juce::File::getSpecialLocation(
    juce::File::currentExecutableFile
).getParentDirectory().getChildFile("Resources").getChildFile("models");
```

### 2. Verify Model Loading

Check plugin logs for:

```
MultiModelProcessor initialized:
  EmotionRecognizer: 403,264 params
  MelodyTransformer: 641,664 params
  HarmonyPredictor: 74,176 params
  DynamicsEngine: 13,520 params
  GroovePredictor: 18,656 params
  Total: 1,152,280 params (~4.6 MB)
```

### 3. Test Inference

```cpp
// In PluginProcessor::processBlock():
std::array<float, 128> features = extractMelFeatures(buffer);
auto result = multiModelProcessor_.runFullPipeline(features);
// Use result.emotionEmbedding, result.melodyProbabilities, etc.
```

## Troubleshooting

### "No audio files found"

- Check dataset directory structure
- Verify audio file extensions (.wav, .mp3, .flac, .m4a, .ogg)
- Ensure labels file exists and is properly formatted

### "CUDA out of memory"

- Reduce batch size: `--batch-size 32`
- Don't cache features: Remove `--cache-features` if using
- Reduce audio duration: Modify dataset loader `duration` parameter

### "Training too slow"

- Use GPU: `--device cuda` or `--device mps`
- Increase batch size (if memory allows)
- Enable feature caching in dataset loader

### "Model export failed"

- Check RTNeural export format matches C++ parser expectations (v2.0 format)
- Verify LSTM weight splitting (should be 4 gates: input, forget, cell, output)
- Run validation script: `python validate_models.py model.json`
- Check export version in metadata matches "2.0"

### "C++ model loading failed"

- Verify JSON file structure with validation script
- Check model file paths in plugin code
- Ensure RTNeural is enabled: `ENABLE_RTNEURAL=1` in CMakeLists.txt
- Check plugin logs for specific error messages

## Best Practices

1. **Use Real Datasets**: Synthetic data is fine for testing, but real datasets produce better models
2. **Monitor Training**: Watch training curves for overfitting
3. **Early Stopping**: Use validation-based early stopping to prevent overfitting
4. **Checkpoint Management**: Save both best and latest checkpoints
5. **Validate Exports**: Always validate exported models before integration
6. **Test Inference**: Test model inference in C++ before deploying

## Performance Optimization

### Training Speed

- Use GPU when available (`--device cuda` or `--device mps`)
- Increase batch size (if memory allows)
- Cache extracted features
- Use multiple data loader workers

### Model Size

- Current models: ~1M parameters total (~4MB)
- To reduce size: Decrease hidden layer dimensions
- Trade-off: Smaller models may have lower accuracy

### Inference Latency

- Target: <10ms per full pipeline run
- Use async inference pipeline for audio thread safety
- Profile with `benchmark_inference.py`

## Next Steps

After training:

1. Validate models: `python validate_models.py models/*.json --check-specs`
2. Copy to plugin: `cp models/*.json plugin/Resources/models/`
3. Rebuild plugin: `cmake --build build --config Release`
4. Test in DAW: Load plugin and verify model loading in logs
5. Test inference: Generate MIDI and verify output quality

## Recent Enhancements

### Complete Training Pipeline (2024)

All 5 models are now fully trainable with comprehensive support:

- **Training Functions**: Complete training functions for all 5 models:
  - `train_emotion_recognizer()` - EmotionRecognizer
  - `train_melody_transformer()` - MelodyTransformer
  - `train_harmony_predictor()` - HarmonyPredictor (NEW)
  - `train_dynamics_engine()` - DynamicsEngine (NEW)
  - `train_groove_predictor()` - GroovePredictor (NEW)

- **Synthetic Datasets**: Synthetic dataset support for all 5 models:
  - `SyntheticEmotionDataset` - For emotion recognition
  - `SyntheticMelodyDataset` - For melody generation
  - `SyntheticHarmonyDataset` - For harmony prediction (NEW)
  - `SyntheticDynamicsDataset` - For dynamics/expression (NEW)
  - `SyntheticGrooveDataset` - For groove prediction (NEW)

- **Enhanced Evaluation**: Updated `evaluate_model()` function supports all model types with proper batch key handling

- **RTNeural Export v2.0**: Improved export format with:
  - Proper LSTM weight splitting into 4 gates (input, forget, cell, output)
  - Automatic activation detection (tanh, relu, sigmoid, softmax)
  - Enhanced metadata with input/output sizes

- **Model Validation**: Validation scripts available:
  - `validate_models.py` - Validates exported RTNeural JSON models
  - `verify_model_architectures.py` - Verifies Python models match C++ specs

### Benefits

- **No Dataset Required**: Can train all models with synthetic data for testing
- **Automatic Fallback**: Gracefully falls back to synthetic data if real datasets unavailable
- **Complete Pipeline**: Single command trains all 5 models end-to-end
- **Validation Ready**: Models can be validated before C++ integration

## References

- Training Script: `ml_training/train_all_models.py`
- Dataset Loaders: `ml_training/dataset_loaders.py`
- Model Validation: `ml_training/validate_models.py`
- Architecture Verification: `ml_training/verify_model_architectures.py`
- Training Configuration: `ml_training/config.json`
- C++ Integration: `src/ml/MultiModelProcessor.h`
- Architecture Docs: `docs/ML_ARCHITECTURE.md`
