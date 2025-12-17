# Dataset Download Guide

This guide will help you download and set up the datasets needed for training the Kelly ML models.

## Quick Start

Run the setup script:
```bash
cd training_pipe/scripts
./setup_datasets.sh
```

## Required Datasets

### 1. DEAM Dataset (Emotion Recognition)

**Purpose:** Train EmotionRecognizer model  
**Size:** ~2GB  
**Format:** Audio files (WAV/MP3) + CSV labels

**Download Steps:**
1. Visit: https://cvml.unige.ch/databases/DEAM/
2. Register for access (free, requires email)
3. Download the dataset
4. Extract to: `datasets/deam/`
5. Ensure structure:
   ```
   datasets/deam/
   ├── audio/
   │   ├── audio_001.wav
   │   ├── audio_002.wav
   │   └── ...
   └── annotations/
       └── annotations.csv  (or similar)
   ```

**Alternative:** Use synthetic data for testing (no download needed)

---

### 2. Lakh MIDI Dataset (Melody Generation)

**Purpose:** Train MelodyTransformer model  
**Size:** ~1.7GB (clean subset) or ~50GB (full)  
**Format:** MIDI files

**Download Steps:**
1. Visit: https://colinraffel.com/projects/lmd/
2. Download `lmd_clean.tar.gz` (recommended, ~1.7GB)
   - Or download full dataset if you have space
3. Extract to: `datasets/lakh_midi/`
4. Structure should be:
   ```
   datasets/lakh_midi/
   ├── 0/
   │   ├── *.mid
   │   └── ...
   ├── 1/
   └── ...
   ```

**Note:** The clean subset contains ~45,000 MIDI files, which is sufficient for training.

---

### 3. MAESTRO Dataset (Dynamics Engine)

**Purpose:** Train DynamicsEngine model  
**Size:** ~18GB  
**Format:** MIDI files with velocity data

**Option A: Using tensorflow-datasets (Recommended)**
```bash
pip install tensorflow-datasets
python -c "import tensorflow_datasets as tfds; tfds.load('maestro')"
```
This will automatically download and prepare the dataset.

**Option B: Manual Download**
1. Visit: https://magenta.tensorflow.org/datasets/maestro
2. Download the dataset
3. Extract to: `datasets/maestro/`

**Option C: Use Lakh MIDI as substitute**
The Lakh MIDI dataset also contains velocity information and can be used for dynamics training if MAESTRO is not available.

---

### 4. Groove MIDI Dataset (Groove Predictor)

**Purpose:** Train GroovePredictor model  
**Size:** ~50MB  
**Format:** Drum MIDI patterns

**Option A: Direct Download (Easiest)**
```bash
cd datasets
wget https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip
unzip groove-v1.0.0-midionly.zip -d groove
rm groove-v1.0.0-midionly.zip
```

**Option B: Using tensorflow-datasets**
```bash
pip install tensorflow-datasets
python -c "import tensorflow_datasets as tfds; tfds.load('groove')"
```

**Option C: Manual Download**
1. Visit: https://magenta.tensorflow.org/datasets/groove
2. Download `groove-v1.0.0-midionly.zip`
3. Extract to: `datasets/groove/`

---

### 5. Harmony Dataset (Harmony Predictor)

**Purpose:** Train HarmonyPredictor model  
**Format:** JSON chord progressions with emotion labels

**Option A: Use Existing Data**
The project already has chord progression data in `/data/chord_progressions_db.json`. This can be converted to the training format.

**Option B: Create from Existing Data**
```bash
python training_pipe/scripts/prepare_datasets.py --datasets-dir ./datasets
```
This will create a template that you can fill with chord progressions.

**Option C: Download from iRealPro or Hooktheory**
- iRealPro: Export chord progressions
- Hooktheory: API access for chord progressions

---

## Automated Setup

### Step 1: Run Setup Script
```bash
cd training_pipe/scripts
./setup_datasets.sh
```

### Step 2: Download Datasets
Follow the links provided by the setup script to download missing datasets.

### Step 3: Organize Datasets
```bash
python training_pipe/scripts/download_datasets.py \
  --datasets-dir ./datasets \
  --organize
```

### Step 4: Prepare for Training
```bash
python training_pipe/scripts/prepare_datasets.py \
  --datasets-dir ./datasets \
  --output-dir ./prepared_data
```

### Step 5: Start Training
```bash
python training_pipe/scripts/train_all_models.py \
  --datasets-dir ./datasets \
  --output ./trained_models \
  --epochs 50
```

---

## Using Synthetic Data (For Testing)

If you want to test the training pipeline without downloading datasets:

```bash
python training_pipe/scripts/train_all_models.py \
  --synthetic \
  --output ./trained_models \
  --epochs 10
```

This will generate synthetic data for all models and train them. Useful for:
- Testing the training pipeline
- Verifying model architectures
- Debugging training code
- Quick experiments

---

## Dataset Directory Structure

After setup, your directory structure should look like:

```
datasets/
├── deam/                    # DEAM emotion dataset
│   ├── audio/
│   └── annotations/
├── lakh_midi/               # Lakh MIDI dataset
│   ├── 0/
│   ├── 1/
│   └── ...
├── maestro/                 # MAESTRO dataset
│   └── *.mid
├── groove/                  # Groove MIDI dataset
│   └── *.mid
├── chords/                  # Chord progressions
│   └── chord_progressions.json
└── training/                # Organized for training
    ├── audio/
    ├── midi/
    ├── dynamics_midi/
    └── drums/
```

---

## Troubleshooting

### "Dataset not found" errors
- Check that datasets are extracted to the correct directories
- Verify file permissions
- Run `prepare_datasets.py` to create necessary label files

### "Out of memory" during training
- Reduce batch size: `--batch-size 32`
- Use fewer samples: modify data loaders to limit files
- Use synthetic data for testing: `--synthetic`

### "Missing labels" errors
- Run `prepare_datasets.py` to generate label templates
- Fill in the label files with your data
- Or use synthetic data which generates labels automatically

### Slow dataset loading
- Set `num_workers=0` in data loaders (already set by default)
- Use smaller subset of datasets for testing
- Pre-process datasets into faster formats (e.g., pre-extracted features)

---

## Next Steps

1. ✅ Download at least one dataset to test the pipeline
2. ✅ Run training with real data
3. ✅ Monitor training metrics and adjust hyperparameters
4. ✅ Export trained models to RTNeural format
5. ✅ Test models in C++ plugin

For questions or issues, check the training pipeline documentation or use synthetic data for testing.
