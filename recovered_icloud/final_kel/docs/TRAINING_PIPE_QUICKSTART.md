# Training Pipeline - Quick Start Guide

## 📦 What You Have

**File**: `training pipe.zip` (25 KB)

A complete, self-contained training pipeline for the Kelly MIDI Companion's 5-model ML architecture.

---

## 🚀 Quick Start (3 Steps)

### 1. Extract

```bash
cd "/Users/seanburdges/Desktop/final kel"
unzip "training pipe.zip"
cd training_pipe
```

### 2. Setup

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create Python virtual environment
- Install all dependencies (PyTorch, NumPy, librosa, etc.)
- Set up the training environment

### 3. Train

```bash
source venv/bin/activate
python scripts/train_all_models.py --output ../models --epochs 100 --device mps
```

**Done!** Models will be exported to `../models/` ready for the plugin.

---

## 📋 What's Inside

```
training_pipe/
├── README.md                  # Full documentation
├── setup.sh                   # Automated setup
├── requirements.txt           # Python dependencies
├── scripts/
│   ├── train_all_models.py   # Main training script (5 models)
│   └── prepare_datasets.py   # Dataset preparation helper
├── configs/
│   └── training_config.json  # Training configuration
├── datasets/                  # Place your data here
│   ├── audio/                # For EmotionRecognizer
│   ├── midi/                 # For MelodyTransformer
│   ├── chords/               # For HarmonyPredictor
│   └── drums/                # For GroovePredictor
└── examples/
    └── example_datasets.md   # Dataset format examples
```

---

## 🎯 Training Options

### Basic (CPU, synthetic data)
```bash
python scripts/train_all_models.py
```

### Production (GPU, more epochs)
```bash
python scripts/train_all_models.py \
    --output ../models \
    --epochs 200 \
    --batch-size 128 \
    --device cuda
```

### Apple Silicon (MPS)
```bash
python scripts/train_all_models.py --device mps --epochs 100
```

### Quick Test (10 epochs)
```bash
python scripts/train_all_models.py --epochs 10 --output ./test_models
```

---

## 📊 Add Your Data (Optional)

The training script works with synthetic data out of the box, but for production use:

### 1. Prepare Datasets

```bash
python scripts/prepare_datasets.py --datasets-dir ./datasets
```

This creates template files for your data.

### 2. Add Your Files

**EmotionRecognizer** (`datasets/audio/`):
```csv
filename,valence,arousal
happy_001.wav,0.8,0.9
sad_001.wav,-0.6,0.3
```

**MelodyTransformer** (`datasets/midi/` + `emotion_labels.json`):
```json
{
  "melody_001.mid": {"valence": 0.7, "arousal": 0.6}
}
```

**See** `examples/example_datasets.md` for complete format specs.

### 3. Train with Real Data

Once you've added your files, just run training as normal:
```bash
python scripts/train_all_models.py --output ../models --epochs 100
```

---

## 🔧 Recommended Datasets

### Free & Public

1. **DEAM** - 14,000 audio clips with emotion labels
   - https://cvml.unige.ch/databases/DEAM/

2. **Lakh MIDI** - 176,581 MIDI files
   - https://colinraffel.com/projects/lmd/

3. **Groove MIDI** - 1,150 drum patterns
   - https://magenta.tensorflow.org/datasets/groove

---

## 📤 Export to Plugin

After training, copy models to your plugin:

```bash
# Models are in RTNeural JSON format
cp ../models/*.json "/path/to/Kelly MIDI Companion.app/Contents/Resources/models/"
```

**Or** place next to the app:
```bash
cp -r ../models "/path/to/Kelly MIDI Companion.app/../"
```

Plugin will auto-load on startup!

---

## ✅ Verify Training Output

You should see:

```
============================================================
Kelly MIDI Companion - Multi-Model Training
============================================================

Model Architecture Summary:
----------------------------------------
EmotionRecognizer: 497,664 params (1944.0 KB)
MelodyTransformer: 412,672 params (1612.0 KB)
HarmonyPredictor: 74,048 params (289.0 KB)
DynamicsEngine: 13,456 params (52.0 KB)
GroovePredictor: 19,040 params (74.0 KB)
----------------------------------------
TOTAL: 1,016,880 params (3971.0 KB)

[1/5] Training EmotionRecognizer...
EmotionRecognizer Epoch 10/50, Loss: 0.042315
...

============================================================
Exporting models to RTNeural format...
============================================================
Exported EmotionRecognizer to ../models/emotionrecognizer.json
  Parameters: 497,664
  Memory: 1944.0 KB
...

Training complete! Models saved to ../models
```

---

## 🐛 Troubleshooting

### "setup.sh: Permission denied"
```bash
chmod +x setup.sh
./setup.sh
```

### "CUDA out of memory"
```bash
python scripts/train_all_models.py --batch-size 32
```

### "MPS not available" (Apple Silicon)
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Training too slow on CPU
Add `--device mps` (Apple Silicon) or `--device cuda` (NVIDIA GPU)

---

## 📚 Full Documentation

For complete details, see:
- `training_pipe/README.md` - Full training guide
- `examples/example_datasets.md` - Dataset format specs
- `../MULTI_MODEL_ML_GUIDE.md` - Complete ML architecture docs

---

## 🎉 That's It!

You now have a complete training pipeline for Kelly MIDI Companion's ML system.

**Next Steps**:
1. Extract and setup (done in 2 minutes)
2. Train with synthetic data to verify (10 minutes)
3. Gather real datasets (ongoing)
4. Retrain with real data for production quality

**Questions?** Check the full README in the extracted folder.

---

**Version**: 1.0.0
**Date**: December 16, 2024
**Size**: 25 KB (compressed), ~84 KB (extracted)
**Models**: 5 (EmotionRecognizer, MelodyTransformer, HarmonyPredictor, DynamicsEngine, GroovePredictor)
