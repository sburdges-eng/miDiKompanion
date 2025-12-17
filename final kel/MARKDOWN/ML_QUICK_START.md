# ML Model Training - Quick Start Guide

This guide walks you through training and deploying an emotion model for the Kelly MIDI Companion plugin.

## Prerequisites

- Python 3.8+
- PyTorch (will be installed via requirements.txt)
- CUDA (optional, for GPU acceleration)

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
cd python
pip install -r requirements.txt
```

### 2. Run Quick Start Script

```bash
cd training
./quick_start.sh
```

This will:
- Create a dummy training dataset
- Train a simple emotion model
- Test the model
- Export to RTNeural JSON format

### 3. Deploy to Plugin

```bash
# Copy model to plugin data directory
cp emotion_model.json ../../data/emotion_model.json
```

### 4. Build Plugin with RTNeural

```bash
cd ../..
cmake -B build -DENABLE_RTNEURAL=ON
cmake --build build
```

### 5. Test in Plugin

1. Load plugin in your DAW
2. Enable ML inference in plugin settings
3. Process audio input
4. Verify emotion detection works

## Manual Steps (if quick start doesn't work)

### Step 1: Create Dataset

```bash
python training/train_emotion_model.py \
    --data dummy_dataset.json \
    --create-dummy \
    --dummy-samples 1000
```

### Step 2: Train Model

```bash
python training/train_emotion_model.py \
    --data dummy_dataset.json \
    --output emotion_model.pt \
    --epochs 50 \
    --batch-size 32
```

### Step 3: Test Model

```bash
python training/test_emotion_model.py --model emotion_model.pt
```

### Step 4: Export to RTNeural

```bash
python training/export_to_rtneural.py \
    --model emotion_model.pt \
    --output emotion_model.json
```

### Step 5: Deploy

```bash
cp emotion_model.json data/emotion_model.json
```

## Production Training

For production use, you'll need:

1. **Real Audio Data**: Collect audio samples with emotion labels
2. **Feature Extraction**: Use `MLFeatureExtractor` to extract 128-dim features
3. **Emotion Labels**: Map audio to 64-dim emotion vectors
4. **Larger Dataset**: Aim for 10,000+ samples
5. **More Epochs**: Train for 100+ epochs with validation

See `python/training/README.md` for detailed instructions.

## Troubleshooting

### Import Errors

If you get import errors, ensure you're in the correct directory:

```bash
cd python/training
python3 train_emotion_model.py --help
```

### PyTorch Not Found

Install PyTorch:

```bash
pip install torch torchvision torchaudio
```

### Model Export Fails

- Verify model file exists and is valid
- Check that model architecture matches RTNeural structure
- See `export_to_rtneural.py` for export format details

### Plugin Can't Load Model

- Verify JSON file is in `data/` directory
- Check plugin logs for error messages
- Ensure RTNeural is enabled in CMake build
- Verify JSON format matches RTNeural's expected structure

## Next Steps

- Train on real audio data
- Fine-tune model architecture
- Optimize for real-time performance
- Train additional models (transformer, DDSP)

See `MARKDOWN/ML_MODEL_TRAINING.md` for comprehensive documentation.

