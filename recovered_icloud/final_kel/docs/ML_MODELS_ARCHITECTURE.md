# ML Models Architecture Documentation

## Overview

The iDAW system uses 5 specialized neural network models to convert emotional input into musical MIDI output. All models are optimized for real-time audio processing with strict performance requirements.

**Total Parameters**: ~1.15M  
**Total Memory**: ~4.39 MB  
**Max Latency**: <10ms (all models pass)  
**Target Accuracy**: >80% (needs real dataset validation)

## Model Specifications

### 1. EmotionRecognizer

**Purpose**: Converts audio features to emotion embedding

**Architecture**:
```
Input: 128-dim audio features (mel spectrogram)
  ↓
Dense: 128 → 512 (tanh)
  ↓
Dense: 512 → 256 (tanh)
  ↓
LSTM: 256 → 128 (batch_first=True)
  ↓
Dense: 128 → 64 (tanh)
  ↓
Output: 64-dim emotion embedding
```

**Parameters**: 403,264 (~500K)  
**Memory**: 1,575.2 KB  
**Latency**: 3.71 ms  
**Status**: ✅ Validated

**Input Format**: Mel spectrogram features (128 bins)  
**Output Format**: 64-dimensional emotion embedding vector

### 2. MelodyTransformer

**Purpose**: Converts emotion embedding to MIDI note probabilities

**Architecture**:
```
Input: 64-dim emotion embedding
  ↓
Dense: 64 → 256 (ReLU)
  ↓
LSTM: 256 → 256 (batch_first=True)
  ↓
Dense: 256 → 256 (ReLU)
  ↓
Dense: 256 → 128 (sigmoid)
  ↓
Output: 128-dim MIDI note probabilities
```

**Parameters**: 641,664 (~400K)  
**Memory**: 2,506.5 KB  
**Latency**: 1.98 ms  
**Status**: ✅ Validated

**Input Format**: 64-dim emotion embedding  
**Output Format**: 128-dimensional vector representing note probabilities (MIDI notes 0-127)

### 3. HarmonyPredictor

**Purpose**: Predicts chord progressions from context

**Architecture**:
```
Input: 128-dim context (emotion + state)
  ↓
Dense: 128 → 256 (tanh)
  ↓
Dense: 256 → 128 (tanh)
  ↓
Dense: 128 → 64 (softmax)
  ↓
Output: 64-dim chord probabilities
```

**Parameters**: 74,176 (~100K)  
**Memory**: 289.8 KB  
**Latency**: 1.26 ms  
**Status**: ✅ Validated

**Input Format**: 128-dim context vector (emotion embedding + current state)  
**Output Format**: 64-dimensional chord probability distribution

### 4. DynamicsEngine

**Purpose**: Generates expression parameters (velocity, timing, articulation)

**Architecture**:
```
Input: 32-dim compact context
  ↓
Dense: 32 → 128 (ReLU)
  ↓
Dense: 128 → 64 (ReLU)
  ↓
Dense: 64 → 16 (sigmoid)
  ↓
Output: 16-dim expression parameters
```

**Parameters**: 13,520 (~20K)  
**Memory**: 52.8 KB  
**Latency**: 0.27 ms  
**Status**: ✅ Validated

**Input Format**: 32-dim compact context vector  
**Output Format**: 16-dimensional expression parameters (velocity, timing, articulation, etc.)

### 5. GroovePredictor

**Purpose**: Predicts rhythm/groove patterns from emotion

**Architecture**:
```
Input: 64-dim emotion embedding
  ↓
Dense: 64 → 128 (tanh)
  ↓
Dense: 128 → 64 (tanh)
  ↓
Dense: 64 → 32 (tanh)
  ↓
Output: 32-dim groove parameters
```

**Parameters**: 18,656 (~25K)  
**Memory**: 72.9 KB  
**Latency**: 0.35 ms  
**Status**: ✅ Validated

**Input Format**: 64-dim emotion embedding  
**Output Format**: 32-dimensional groove/rhythm parameters

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
│                                                                   │
│  Option 1: Audio Input                                            │
│    Audio Signal → Mel Spectrogram (128-dim)                       │
│         ↓                                                          │
│    EmotionRecognizer → 64-dim emotion embedding                   │
│                                                                   │
│  Option 2: Direct Emotion Input                                    │
│    Text/Biometrics → 64-dim emotion embedding (direct)            │
│                                                                   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼ 64-dim emotion embedding
┌──────────────────────────────────────────────────────────────────┐
│                    ML MODEL PIPELINE                              │
│                                                                   │
│  ┌──────────────────────┐      ┌──────────────────────┐         │
│  │  MelodyTransformer   │      │   GroovePredictor    │         │
│  │  64 → 128 MIDI notes │      │  64 → 32 groove     │         │
│  │  Latency: 1.98ms     │      │  Latency: 0.35ms    │         │
│  └──────────┬───────────┘      └──────────┬───────────┘         │
│             │                              │                      │
│             │ 128-dim context              │                      │
│             ▼                              │                      │
│  ┌──────────────────────┐                 │                      │
│  │  HarmonyPredictor    │                 │                      │
│  │  128 → 64 chords     │                 │                      │
│  │  Latency: 1.26ms     │                 │                      │
│  └──────────┬───────────┘                 │                      │
│             │                              │                      │
│             │ 32-dim compact context       │                      │
│             ▼                              │                      │
│  ┌──────────────────────┐                 │                      │
│  │  DynamicsEngine      │                 │                      │
│  │  32 → 16 expression  │                 │                      │
│  │  Latency: 0.27ms     │                 │                      │
│  └──────────┬───────────┘                 │                      │
│             │                              │                      │
│             └──────────┬───────────────────┘                      │
│                        │                                          │
└────────────────────────┼──────────────────────────────────────────┘
                         │
                         ▼ Combined Output
┌──────────────────────────────────────────────────────────────────┐
│                    MUSIC THEORY LAYER                            │
│              (DAiW-Music-Brain Validation)                        │
│  - MIDI notes (128-dim probabilities)                            │
│  - Chords (64-dim probabilities)                                 │
│  - Groove (32-dim parameters)                                    │
│  - Expression (16-dim parameters)                                │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                   │
│                    Final MIDI with                               │
│              Emotional Intent Preserved                           │
└──────────────────────────────────────────────────────────────────┘
```

## Complete End-to-End Flow

### Phase 0: Emotion Input
```
User Input (text/audio/biometrics)
    ↓
Emotion Processing (CIF/LAS frameworks)
    ↓
64-dim emotion embedding
```

### Phase 1: ML Model Inference
```
64-dim emotion embedding
    ↓
┌─────────────────────────────────┐
│ Parallel Processing:            │
│  • MelodyTransformer (notes)    │ → 128-dim MIDI notes
│  • GroovePredictor (rhythm)     │ → 32-dim groove
└─────────────────────────────────┘
    ↓
128-dim context (emotion + notes)
    ↓
HarmonyPredictor → 64-dim chords
    ↓
32-dim compact context
    ↓
DynamicsEngine → 16-dim expression
```

### Phase 2: Music Theory Validation
```
ML Outputs:
  • MIDI notes (128-dim)
  • Chords (64-dim)
  • Groove (32-dim)
  • Expression (16-dim)
    ↓
Music Brain:
  • Intent-driven composition
  • Rule-breaking system
  • Groove extraction/application
  • Chord progression analysis
    ↓
Validated MIDI Output
```

## Performance Characteristics

### Latency Breakdown

| Model | Latency (ms) | Cumulative (ms) |
|-------|--------------|-----------------|
| EmotionRecognizer (if audio) | 3.71 | 3.71 |
| MelodyTransformer | 1.98 | 5.69 |
| GroovePredictor (parallel) | 0.35 | 5.69 |
| HarmonyPredictor | 1.26 | 6.95 |
| DynamicsEngine | 0.27 | 7.22 |
| **Total Pipeline** | **~7.22ms** | **✅ <10ms** |

### Memory Usage

| Model | Memory (KB) | Percentage |
|-------|-------------|------------|
| EmotionRecognizer | 1,575.2 | 35.9% |
| MelodyTransformer | 2,506.5 | 57.1% |
| HarmonyPredictor | 289.8 | 6.6% |
| DynamicsEngine | 52.8 | 1.2% |
| GroovePredictor | 72.9 | 1.7% |
| **Total** | **4,497.2 KB** | **100%** |

### Model Sizes (RTNeural JSON)

| Model | File Size | Compressed |
|-------|-----------|------------|
| EmotionRecognizer | 13 MB | - |
| MelodyTransformer | 21 MB | - |
| HarmonyPredictor | 2.3 MB | - |
| DynamicsEngine | 421 KB | - |
| GroovePredictor | 586 KB | - |
| **Total** | **~37 MB** | - |

*Note: JSON format includes full precision weights. Runtime memory is much smaller.*

## Model Formats

### Export Formats

1. **RTNeural JSON** (✅ Complete)
   - Format: JSON with layer definitions and weights
   - Location: `ml_training/trained_models/*.json`
   - Used for: C++ plugin integration

2. **PyTorch Checkpoints** (✅ Complete)
   - Format: `.pt` files with model state dict
   - Location: `ml_training/trained_models/checkpoints/`
   - Used for: Python inference, retraining

3. **ONNX** (⚠️ Pending)
   - Format: `.onnx` files
   - Status: Export script available, requires Python 3.11/3.12 or Docker
   - Used for: Cross-platform deployment

## Training Details

### Training Configuration

- **Epochs**: 10 (quick validation run)
- **Training Data**: Synthetic (for quick testing)
- **Validation Split**: 20%
- **Device**: CPU
- **Batch Size**: 32 (default)
- **Learning Rate**: 0.001 (default)

### Production Training Recommendations

For production use with real datasets:
```bash
cd ml_training
python train_all_models.py \
    --epochs 100 \
    --device mps \
    --batch-size 64 \
    --datasets-dir ../datasets \
    --output ./trained_models
```

### Training Metrics

All models trained with early stopping:
- **Patience**: 10 epochs
- **Min Delta**: 0.001

Best validation losses (from TRAINING_COMPLETE.md):
- EmotionRecognizer: 0.395090
- MelodyTransformer: 0.045704
- HarmonyPredictor: -3.981914
- DynamicsEngine: 0.083675
- GroovePredictor: 0.395527

## Integration Points

### With Emotion Frameworks

1. **CIF/LAS Input**:
   - Emotion embedding can come from LAS ESV (Emotional State Vector)
   - Or from EmotionRecognizer if audio input

2. **Output to Music Brain**:
   - MIDI notes → Intent-driven composition
   - Chords → Chord progression analysis
   - Groove → Groove extraction/application
   - Expression → Rule-breaking system

### With Plugins (C++)

Models are loaded as RTNeural JSON in JUCE plugins:
- Real-time safe inference
- Lock-free communication with Python (Side B)
- Dual-heap architecture (Side A: RT, Side B: AI)

## Model Files

### Location Structure

```
ml_training/
├── trained_models/
│   ├── emotionrecognizer.json      (13 MB)
│   ├── melodytransformer.json      (21 MB)
│   ├── harmonypredictor.json       (2.3 MB)
│   ├── dynamicsengine.json         (421 KB)
│   ├── groovepredictor.json        (586 KB)
│   ├── checkpoints/
│   │   ├── EmotionRecognizer_best.pt
│   │   ├── MelodyTransformer_best.pt
│   │   ├── HarmonyPredictor_best.pt
│   │   ├── DynamicsEngine_best.pt
│   │   └── GroovePredictor_best.pt
│   └── history/
│       ├── emotionrecognizer_history.json
│       ├── melodytransformer_history.json
│       └── ...
└── deployment/
    └── models/
        └── (copy of RTNeural JSON files)
```

## Validation Status

### Performance Validation ✅

- ✅ All models meet latency requirement (<10ms)
- ✅ All models meet memory requirement (<4MB per model)
- ✅ All models validated against C++ specifications

### Accuracy Validation ⚠️

- ⚠️ Needs real dataset validation for >80% accuracy target
- ⚠️ MIDI output quality needs subjective evaluation
- ⚠️ Emotional authenticity needs user testing

## Future Improvements

1. **Retrain with Real Datasets**
   - Use DEAM, EMOPIA, and custom emotion-labeled datasets
   - Target 100 epochs with real data

2. **Model Optimization**
   - Quantization for smaller memory footprint
   - Pruning for faster inference
   - Batch processing optimization

3. **Multi-Modal Input**
   - Support for text → emotion embedding
   - Biometric data integration
   - Multi-modal fusion

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-18  
**Validation Status**: Performance ✅ | Accuracy ⚠️
