# ML Model Training Scripts

This directory contains scripts for training and exporting ML models for the Kelly MIDI Companion plugin.

## Quick Start

### 1. Install Dependencies

```bash
cd python
pip install -r requirements.txt
```

### 2. Create Test Model (Fast - No Training Required)

For immediate plugin testing, create a test model:

```bash
python training/create_test_model.py --output test_emotion_model.pt
python training/export_to_rtneural.py --model test_emotion_model.pt --output emotion_model.json
cp emotion_model.json ../../data/emotion_model.json
```

### 3. Train Real Model (Recommended for Production)

```bash
# Create dataset
python training/train_emotion_model.py \
    --data dummy_dataset.json \
    --create-dummy \
    --dummy-samples 1000

# Train model
python training/train_emotion_model.py \
    --data dummy_dataset.json \
    --output emotion_model.pt \
    --epochs 50 \
    --batch-size 32

# Test model
python training/test_emotion_model.py --model emotion_model.pt

# Test plugin integration
python training/test_plugin_integration.py --model emotion_model.pt

# Export to RTNeural
python training/export_to_rtneural.py \
    --model emotion_model.pt \
    --output emotion_model.json

# Deploy
cp emotion_model.json ../../data/emotion_model.json
```

### 4. Automated Quick Start

```bash
cd training
./quick_start.sh
```

## Scripts Overview

### `create_test_model.py`
Creates a minimal test model with random weights for immediate plugin testing.
- **Use when**: You want to test plugin integration without training
- **Output**: `.pt` PyTorch model file

### `train_emotion_model.py`
Trains the emotion model from a dataset.
- **Input**: JSON dataset with features and emotion vectors
- **Output**: Trained `.pt` PyTorch model
- **Options**: Epochs, batch size, learning rate, etc.

### `test_emotion_model.py`
Tests a trained model for correctness and performance.
- **Checks**: Input/output dimensions, inference speed, batch processing
- **Use when**: After training to verify model works

### `test_plugin_integration.py`
Comprehensive test for plugin compatibility.
- **Checks**: Model format, export compatibility, performance
- **Use when**: Before deploying to plugin

### `export_to_rtneural.py`
Exports PyTorch model to RTNeural JSON format.
- **Input**: `.pt` PyTorch model
- **Output**: `.json` RTNeural model file
- **Use when**: Ready to deploy to plugin

### `quick_start.sh`
Automated script that runs the full pipeline.
- Creates dataset, trains, tests, and exports
- **Use when**: You want everything automated

## Dataset Format

The training dataset should be a JSON file with this structure:

```json
{
  "samples": [
    {
      "features": [128 float values],  // Audio features from MLFeatureExtractor
      "emotion": [64 float values]      // Target emotion vector
    },
    ...
  ]
}
```

### Creating Real Datasets

For production, you need:

1. **Audio Files**: Collect audio samples with known emotions
2. **Feature Extraction**: Use `MLFeatureExtractor` to extract 128-dim features
3. **Emotion Labels**: Map to 64-dim emotion vectors (valence, arousal, etc.)

Example workflow:

```python
from ml_feature_extractor import extract_features
import json

samples = []
for audio_file in audio_files:
    # Extract 128-dim features
    features = extract_features(audio_file)

    # Get emotion label (64-dim vector)
    emotion = get_emotion_vector(audio_file)

    samples.append({
        'features': features.tolist(),
        'emotion': emotion.tolist()
    })

with open('dataset.json', 'w') as f:
    json.dump({'samples': samples}, f)
```

## Model Architecture

The emotion model uses this architecture (matching RTNeural):

- **Input**: 128-dimensional audio features
- **Layer 1**: Dense (128 → 256) + Tanh activation
- **Layer 2**: LSTM (256 → 128)
- **Layer 3**: Dense (128 → 64)
- **Output**: 64-dimensional emotion vector

## Training Tips

1. **Dataset Size**:
   - Minimum: 1000 samples
   - Recommended: 10,000+ samples
   - Per emotion: 1000+ samples

2. **Training Parameters**:
   - Epochs: Start with 50, increase if loss still decreasing
   - Batch Size: 32-64 (adjust for GPU memory)
   - Learning Rate: 0.001 (adjust if loss doesn't decrease)

3. **Validation**:
   - Script uses 80/20 train/validation split
   - Monitor validation loss
   - Stop if validation loss stops improving

4. **Performance**:
   - Target inference time: < 1ms
   - RTNeural will optimize further
   - Test with `test_emotion_model.py`

## Deployment

After training and exporting:

1. **Copy model to plugin data directory**:
   ```bash
   cp emotion_model.json ../../data/emotion_model.json
   ```

2. **Build plugin with RTNeural**:
   ```bash
   cmake -B build -DENABLE_RTNEURAL=ON
   cmake --build build
   ```

3. **Test in plugin**:
   - Load plugin in DAW
   - Enable ML inference in settings
   - Process audio input
   - Verify emotion detection

## Troubleshooting

### Import Errors
- Ensure you're in `python/training/` directory
- Install dependencies: `pip install -r ../requirements.txt`

### Model Not Loading in Plugin
- Verify JSON format matches RTNeural structure
- Check model file is in `data/` directory
- Enable plugin logging to see errors
- Run `test_plugin_integration.py` to verify export

### Poor Model Performance
- Increase dataset size
- Train for more epochs
- Adjust learning rate
- Try different architectures

### Export Errors
- Verify PyTorch model is saved correctly
- Check model architecture matches RTNeural
- Ensure all layers are exported

### Slow Inference
- Model should be < 1ms per inference
- RTNeural will optimize further
- Consider model quantization
- Reduce model size if needed

## Next Steps

- Train on real audio data
- Fine-tune architecture
- Optimize for real-time
- Train additional models (transformer, DDSP)

See `MARKDOWN/ML_MODEL_TRAINING.md` for comprehensive documentation.
