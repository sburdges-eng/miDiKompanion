# Quick Test Model Creation

This guide shows you how to quickly create a test model for plugin integration testing.

## Prerequisites Check

First, verify your environment:

```bash
python3 verify_setup.py
```

If PyTorch is not installed, install it:

```bash
pip install torch
```

## Create Test Model (2 Steps)

### Step 1: Create Model

```bash
python3 create_test_model.py --output test_emotion_model.pt
```

This creates a minimal model with random weights (not trained, but functional for testing).

### Step 2: Export to RTNeural Format

```bash
python3 export_to_rtneural.py --model test_emotion_model.pt --output emotion_model.json
```

### Step 3: Deploy to Plugin

```bash
cp emotion_model.json ../../data/emotion_model.json
```

## Test Model in Plugin

1. Build plugin with RTNeural enabled:
   ```bash
   cd ../..
   cmake -B build -DENABLE_RTNEURAL=ON
   cmake --build build
   ```

2. Load plugin in your DAW

3. Enable ML inference in plugin settings

4. Process audio input - the model will run (though results won't be accurate since it's not trained)

## What This Tests

- ✅ Model loading in plugin
- ✅ RTNeural integration
- ✅ Feature extraction pipeline
- ✅ Inference thread communication
- ✅ Plugin ML infrastructure

## Next Steps

For production use, train a real model:

```bash
python3 train_emotion_model.py --data dataset.json --create-dummy
python3 train_emotion_model.py --data dataset.json --output emotion_model.pt --epochs 50
python3 export_to_rtneural.py --model emotion_model.pt --output emotion_model.json
```

See `README.md` for full training instructions.

