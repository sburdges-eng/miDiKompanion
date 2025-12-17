# Dataset Inventory and Location Report

## Summary

**Date:** Generated automatically
**Location:** `/Users/seanburdges/Desktop/final kel/`

## ZIP Files Found

### Root Directory ZIP Files

1. **`kell ml ai.zip`** (1.1 MB)
   - Contains: `KELLY_ML_TEACHING_GUIDE.md`, `KELLY_ML_COMPLETE.zip`
   - **No datasets found** - Contains documentation and nested zip

2. **`kell ml final.zip`** (21 KB)
   - Contains: C++ headers and implementation files
   - **No datasets found** - Code files only

3. **`ml model training.zip`** (16 KB)
   - Contains: Training scripts, model headers, documentation
   - **No datasets found** - Code files only

4. **`training pipe.zip`** (25 KB)
   - Contains: Training pipeline scripts and configuration
   - **No datasets found** - Code files only

### ZIPS Directory

1. **`kelly_vocal_lyric.zip`**
   - Contains: Vocal/lyric system code and documentation
   - **No datasets found**

2. **`phase2.zip`**
   - Contains: Phase 2 implementation files
   - **No datasets found**

3. **`VOCAL IMP.zip`**, **`VOCAL QUANTNUM.zip`**, **`VOCAL SYS IMP.zip`**
   - Contains: Vocal system implementation files
   - **No datasets found**

## Existing Data Files

### `/data/` Directory

The project contains JSON data files in `/data/` directory:

- Emotion data (angry.json, happy.json, sad.json, etc.)
- Chord progressions
- Phonemes
- Groove templates
- Lyric templates
- Voice types

**These are configuration/reference data, not training datasets.**

## Required Datasets (Not Found in ZIPs)

According to the training plan, the following datasets are needed but **NOT found in any ZIP files**:

### 1. **DEAM Dataset** (Emotion Recognition)

- **Status:** ❌ Not found
- **Required:** Audio files + CSV with valence/arousal labels
- **Size:** ~2GB
- **Download:** <https://cvml.unige.ch/databases/DEAM/>
- **Note:** Requires registration

### 2. **Lakh MIDI Dataset** (Melody Generation)

- **Status:** ❌ Not found
- **Required:** MIDI files (176,581 files)
- **Size:** ~1.7GB (clean subset)
- **Download:** <https://colinraffel.com/projects/lmd/>

### 3. **MAESTRO Dataset** (Dynamics Engine)

- **Status:** ❌ Not found
- **Required:** Piano MIDI files with velocity data
- **Size:** ~18GB
- **Download:** <https://magenta.tensorflow.org/datasets/maestro>

### 4. **Groove MIDI Dataset** (Groove Predictor)

- **Status:** ❌ Not found
- **Required:** Drum MIDI patterns
- **Size:** ~50MB
- **Download:** <https://magenta.tensorflow.org/datasets/groove>

### 5. **Harmony Dataset** (Harmony Predictor)

- **Status:** ❌ Not found
- **Required:** Chord progressions with emotion labels
- **Sources:** iRealPro or Hooktheory
- **Format:** JSON chord progressions

## Recommendations

### Option 1: Download Datasets

Use the existing download script:

```bash
python training_pipe/scripts/download_datasets.py --datasets-dir ./datasets
```

### Option 2: Use Synthetic Data (For Testing)

The training pipeline supports synthetic data:

```bash
python training_pipe/scripts/train_all_models.py --synthetic
```

### Option 3: Use Existing JSON Data

The `/data/` directory contains chord progressions and emotion mappings that could be used for:

- Harmony prediction (chord_progressions_db.json)
- Emotion reference (emotions/*.json)

However, these would need to be converted to the training format expected by the data loaders.

## Next Steps

1. **Download datasets** using the download script (requires manual steps for DEAM and Lakh MIDI)
2. **Organize datasets** using `prepare_datasets.py`
3. **Train models** using `train_all_models.py` with real data

## Dataset Directory Structure Expected

After downloading and organizing, datasets should be in:

```
datasets/
├── audio/              # DEAM audio files
│   └── labels.csv      # Valence/arousal labels
├── midi/               # Lakh MIDI files
├── emotion_labels.json # MIDI emotion mappings
├── chords/             # Chord progressions
│   └── chord_progressions.json
├── dynamics_midi/      # MAESTRO MIDI files
└── drums/              # Groove MIDI files
```

## Notes

- All ZIP files found contain **code and documentation only**, no training datasets
- The training pipeline is ready to use but requires actual datasets to be downloaded
- Synthetic data can be used for testing the pipeline without downloading datasets
