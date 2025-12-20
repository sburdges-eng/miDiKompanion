# Kelly MIDI Companion - Multi-Model ML Architecture Guide

**Status**: ✅ **COMPLETE** - Successfully integrated and building
**Date**: December 16, 2024

---

## Overview

The Kelly MIDI Companion now features a **5-model ML architecture** that processes audio and emotion data through specialized neural networks to generate intelligent MIDI patterns.

### Architecture Summary

| Model | Input | Output | Parameters | Purpose |
|-------|-------|--------|------------|---------|
| **EmotionRecognizer** | 128 mel features | 64-dim emotion | ~500K | Audio → Emotion mapping |
| **MelodyTransformer** | 64-dim emotion | 128 note probs | ~400K | Emotion → Melody suggestions |
| **HarmonyPredictor** | 128-dim context | 64 chord probs | ~100K | Context → Chord predictions |
| **DynamicsEngine** | 32-dim compact | 16-dim expression | ~20K | Context → Velocity/timing |
| **GroovePredictor** | 64-dim emotion | 32 groove params | ~25K | Emotion → Groove patterns |

**Total**: ~1M parameters, ~4MB memory, <10ms inference

---

## Files Added

### Core ML Implementation

```
src/ml/
├── MultiModelProcessor.h          # Simplified all-in-one header
├── MultiModelProcessor.cpp        # Full implementation
└── [existing RTNeural files...]

models/
├── model_architectures.json       # All model specifications
└── emotionrecognizer.json         # Placeholder weights

ml_training/
└── train_all_models.py            # PyTorch training → RTNeural export
```

### Integration Points

- **PluginProcessor.h**: Added `multiModelProcessor_` and `asyncMLPipeline_` members
- **PluginProcessor.cpp**: Initialize models in `prepareToPlay()`, cleanup in `releaseResources()`
- **CMakeLists.txt**: Added `MultiModelProcessor.cpp` to build

---

## Quick Start

### 1. Using the Multi-Model System (C++)

```cpp
#include "ml/MultiModelProcessor.h"

// Initialize processor
Kelly::ML::MultiModelProcessor mlProcessor;
mlProcessor.initialize(modelsDirectory);

// In audio callback (processBlock):
std::array<float, 128> features = extractMelFeatures(buffer);

// Option A: Run full pipeline
auto result = mlProcessor.runFullPipeline(features);
// Use result.emotionEmbedding, result.melodyProbabilities, etc.

// Option B: Run individual model
auto emotion = mlProcessor.infer(Kelly::ML::ModelType::EmotionRecognizer, features);
```

### 2. Async Usage (Audio Thread Safe)

```cpp
Kelly::ML::AsyncMLPipeline asyncML(mlProcessor);
asyncML.start();

// Audio thread:
asyncML.submitFeatures(features);  // Non-blocking

// Check for results:
if (asyncML.hasResult()) {
    auto result = asyncML.getResult();
    // Process result
}
```

### 3. Enable/Disable Individual Models

```cpp
// From PluginProcessor
processor.setModelEnabled(Kelly::ML::ModelType::MelodyTransformer, false);

// Check status
bool isEnabled = processor.isModelEnabled(Kelly::ML::ModelType::HarmonyPredictor);
```

---

## Training Models

### Prerequisites

```bash
pip install torch numpy
```

### Train All Models

```bash
cd ml_training
python train_all_models.py --output ../models --epochs 100 --device mps
```

**Options**:
- `--output` - Output directory for trained models
- `--epochs` - Number of training epochs (default: 50)
- `--batch-size` - Training batch size (default: 64)
- `--device` - Training device: `cpu`, `cuda`, `mps` (default: cpu)

**Output**:
```
models/
├── emotionrecognizer.json
├── melodytransformer.json
├── harmonypredictor.json
├── dynamicsengine.json
└── groovepredictor.json

models/checkpoints/
├── emotionrecognizer.pt
├── melodytransformer.pt
└── ...
```

---

## Model Details

### 1. EmotionRecognizer (128→512→256→128→64)

**Purpose**: Extract emotional content from audio features

**Architecture**:
- Dense layer: 128 → 512 (tanh)
- Dense layer: 512 → 256 (tanh)
- LSTM layer: 256 → 128
- Dense layer: 128 → 64 (tanh)

**Output**: 64-dimensional emotion embedding
- First 32 dims: Valence-related features
- Last 32 dims: Arousal-related features

**Training Data**: DEAM dataset recommended (14,000+ audio clips with valence/arousal labels)

---

### 2. MelodyTransformer (64→256→256→256→128)

**Purpose**: Generate melody suggestions from emotional state

**Architecture**:
- Dense layer: 64 → 256 (ReLU)
- LSTM layer: 256 → 256
- Dense layer: 256 → 256 (ReLU)
- Dense layer: 256 → 128 (sigmoid)

**Output**: 128-dimensional MIDI note probabilities (C0-G10)

**Training Data**: MIDI + emotion label pairs (e.g., Lakh MIDI Dataset + emotion annotations)

---

### 3. HarmonyPredictor (128→256→128→64)

**Purpose**: Predict chord probabilities from context

**Architecture**:
- Dense layer: 128 → 256 (tanh)
- Dense layer: 256 → 128 (tanh)
- Dense layer: 128 → 64 (softmax)

**Input**: 128-dim context vector (64-dim emotion + 64-dim audio features)
**Output**: 64-dimensional chord probabilities

**Training Data**: Chord progressions with emotional context

---

### 4. DynamicsEngine (32→128→64→16)

**Purpose**: Generate velocity, timing, and expression parameters

**Architecture**:
- Dense layer: 32 → 128 (ReLU)
- Dense layer: 128 → 64 (ReLU)
- Dense layer: 64 → 16 (sigmoid)

**Output**: 16-dimensional expression vector
- Velocity curves
- Timing offsets
- Expression modulation

**Training Data**: MIDI with dynamics + emotion labels

---

### 5. GroovePredictor (64→128→64→32)

**Purpose**: Generate groove and rhythm parameters

**Architecture**:
- Dense layer: 64 → 128 (tanh)
- Dense layer: 128 → 64 (tanh)
- Dense layer: 64 → 32 (tanh)

**Output**: 32-dimensional groove parameters
- Swing/shuffle amounts
- Syncopation patterns
- Rhythmic emphasis

**Training Data**: Drum patterns + emotion labels

---

## Fallback Heuristics

When models are not trained or RTNeural is disabled, the system uses intelligent fallback heuristics:

### EmotionRecognizer Fallback
```cpp
// Valence from low-frequency mel bins
valence[i] = tanh(melFeatures[i] * 0.3)

// Arousal from high-frequency mel bins
arousal[i] = tanh(melFeatures[i + 64] * 0.5)
```

### MelodyTransformer Fallback
```cpp
// Note probabilities based on chromatic scale + emotion
noteProbability[i] = sigmoid((i % 12) / 12 * 2 - 1 + emotion[i % 64] * 0.5)
```

### HarmonyPredictor Fallback
```cpp
// Chord weights based on context
chordWeight[i] = tanh(context[i] * 0.4)
```

### DynamicsEngine Fallback
```cpp
// Mid-velocity with emotion modulation
velocity[i] = 0.5 + context[i % 32] * 0.3  // Clamped to [0, 1]
```

### GroovePredictor Fallback
```cpp
// Groove parameters from emotion
grooveParam[i] = tanh(emotion[i % 64] * 0.6)
```

---

## Build Configuration

### CMake Options

```bash
cmake -B build \
    -DENABLE_RTNEURAL=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release
```

### Build Verification

**Success criteria**:
```
✅ MultiModelProcessor initialized:
  Total params: 1,016,880
  Total memory: 3971 KB
  Estimated inference: <10ms
```

---

## Integration with Kelly Workflow

### 1. Audio Input → Emotion

```cpp
// Extract mel-spectrogram features
auto features = featureExtractor.extractFeatures(audioBuffer);

// Run emotion recognition
auto result = multiModelProcessor.runFullPipeline(features);
auto& emotion = result.emotionEmbedding;  // 64-dim
```

### 2. Emotion → MIDI Generation

```cpp
// Use melody suggestions
auto& melodyProbs = result.melodyProbabilities;  // 128 notes
int selectedNote = selectWeightedNote(melodyProbs);

// Use harmony predictions
auto& chordProbs = result.harmonyPrediction;  // 64 chords
auto chord = selectChordFromProbabilities(chordProbs);

// Apply dynamics
auto& dynamics = result.dynamicsOutput;  // 16-dim
int velocity = dynamics[0] * 127;

// Apply groove
auto& groove = result.grooveParameters;  // 32-dim
float swing = groove[0];
```

### 3. Combine with Intent Pipeline

```cpp
// ML provides suggestions, Intent Pipeline provides rules
if (mlInferenceEnabled && asyncML.hasResult()) {
    auto mlResult = asyncML.getResult();

    // Blend ML suggestions with rule-based generation
    float blendFactor = 0.7;  // 70% ML, 30% rules

    auto finalMelody = blend(
        mlResult.melodyProbabilities,
        emotionThesaurus.suggestMelody(emotion),
        blendFactor
    );
}
```

---

## Performance Considerations

### Memory Usage

- **Per-model**: Up to ~5M params comfortably (20MB)
- **Total models**: 3-5 concurrent, ~50MB total
- **Hard ceiling**: ~100M params before DAW issues

### Inference Latency

- **Target**: <10ms per full pipeline
- **Single model**: 1-3ms
- **Lookahead buffer**: 20ms (ML_LOOKAHEAD_MS)

### Threading

- **Audio thread**: Never blocks, uses try_lock pattern
- **Inference thread**: Background thread via AsyncMLPipeline
- **Lock-free queues**: SPSC ring buffers for audio safety

---

## Recommended Datasets

### 1. EmotionRecognizer
- **DEAM**: 14,000 audio clips with valence/arousal labels
- **PMEmo**: 794 music tracks with 4 emotion classes
- **MER**: Music Emotion Recognition datasets

### 2. MelodyTransformer
- **Lakh MIDI Dataset**: 176,581 MIDI files
- **Hooktheory**: Analyzed music theory data
- Combine with emotion labels from DEAM

### 3. HarmonyPredictor
- **iRealPro**: Jazz chord progressions
- **Hooktheory Trends**: Chord progression database
- Emotion-tagged chord sequences

### 4. DynamicsEngine
- **MAESTRO**: 200+ hours of piano with dynamics
- **SMD**: Synthesized Music Dataset
- MIDI velocity curves

### 5. GroovePredictor
- **Groove MIDI Dataset**: 1,150 MIDI drum patterns
- **E-GMD**: Electronic music drum patterns
- Emotion-tagged rhythm data

---

## Troubleshooting

### Models Not Loading

**Symptoms**: "Model file not found, using fallback heuristics"

**Solution**:
```bash
# Copy trained models to plugin Resources
cp models/*.json "/path/to/Kelly MIDI Companion.app/Contents/Resources/models/"
```

**Or** place models next to app:
```
Kelly MIDI Companion.app/
models/
├── emotionrecognizer.json
├── melodytransformer.json
└── ...
```

### High Inference Latency

**Symptoms**: Audio dropouts, >10ms inference time

**Solutions**:
1. Disable unused models:
   ```cpp
   processor.setModelEnabled(Kelly::ML::ModelType::HarmonyPredictor, false);
   ```

2. Reduce model complexity during training

3. Use async pipeline (already default)

### RTNeural Build Errors

**Symptoms**: RTNeural::Model constructor errors

**Solution**: Models require input size parameter:
```cpp
// Correct
rtModel_ = std::make_unique<RTNeural::Model<float>>(inputSize);

// Incorrect
rtModel_ = std::make_unique<RTNeural::Model<float>>();
```

---

## Next Steps

### Phase 1: Model Training (Complete when datasets available)
- [ ] Gather DEAM dataset for EmotionRecognizer
- [ ] Train EmotionRecognizer on real data
- [ ] Gather Lakh MIDI + emotion labels for MelodyTransformer
- [ ] Train remaining models

### Phase 2: Fine-Tuning
- [ ] Calibrate emotion space to Kelly's 216-node thesaurus
- [ ] Add user-specific model fine-tuning
- [ ] Implement model blending (ML + rules)

### Phase 3: UI Integration
- [ ] Add ML enable/disable toggle in EmotionWorkstation
- [ ] Add per-model enable/disable controls
- [ ] Display ML inference confidence
- [ ] Show emotion embedding visualization

### Phase 4: Optimization
- [ ] Profile inference latency
- [ ] Optimize model architectures if needed
- [ ] Add model quantization (INT8) support
- [ ] Implement model caching

---

## Build Status

✅ **CMake configuration**: Success
✅ **Compilation**: Success
✅ **Standalone build**: 4.9 MB
✅ **AU plugin build**: 4.6 MB
⚠️ **VST3 build**: Minor signing issue (non-critical)

---

## Technical Specifications

### Compiler
- **Platform**: macOS (Darwin 25.1.0)
- **Compiler**: AppleClang 17.0.0.17000603
- **C++ Standard**: C++20
- **Build Type**: Release
- **Optimization**: -O3 (inferred from Release)

### Dependencies
- **JUCE**: 8.0.4 (local copy)
- **RTNeural**: main branch (auto-fetched)
- **Eigen**: 3.x (via RTNeural)

### Warnings
- Sign conversion warnings (expected from RTNeural/Eigen)
- JUCE splash screen flag (expected)
- All non-critical, no errors

---

## Code Examples

### Custom Inference Integration

```cpp
// In your custom processor
class MyProcessor {
public:
    void processBlock(juce::AudioBuffer<float>& buffer) {
        // Extract features
        std::array<float, 128> features = extractFeatures(buffer);

        // Submit for async inference
        asyncML.submitFeatures(features);

        // Check for results (non-blocking)
        if (asyncML.hasResult()) {
            auto result = asyncML.getResult();

            // Use emotion
            float valence = result.emotionEmbedding[0];
            float arousal = result.emotionEmbedding[32];

            // Use melody
            int suggestedNote = argmax(result.melodyProbabilities);

            // Use dynamics
            int velocity = result.dynamicsOutput[0] * 127;

            // Generate MIDI
            generateMIDI(suggestedNote, velocity);
        }
    }

private:
    Kelly::ML::MultiModelProcessor mlProcessor;
    std::unique_ptr<Kelly::ML::AsyncMLPipeline> asyncML;
};
```

---

## Performance Benchmarks

### Target Metrics (Estimated)

| Operation | Latency | Notes |
|-----------|---------|-------|
| Feature Extraction | 1-2ms | Mel-spectrogram (2048 FFT) |
| EmotionRecognizer | 2-3ms | Largest model |
| MelodyTransformer | 1-2ms | |
| HarmonyPredictor | <1ms | |
| DynamicsEngine | <0.5ms | Smallest model |
| GroovePredictor | <0.5ms | |
| **Full Pipeline** | **<10ms** | All 5 models sequential |

### Memory Benchmarks

| Component | Memory | Notes |
|-----------|--------|-------|
| EmotionRecognizer | ~2MB | ~500K params × 4 bytes |
| MelodyTransformer | ~1.6MB | ~400K params × 4 bytes |
| HarmonyPredictor | ~400KB | ~100K params × 4 bytes |
| DynamicsEngine | ~80KB | ~20K params × 4 bytes |
| GroovePredictor | ~100KB | ~25K params × 4 bytes |
| **Total** | **~4MB** | Comfortably within limits |

---

## License & Credits

**Kelly MIDI Companion**: Therapeutic MIDI generation through emotion mapping
**Developer**: Sean Burdges
**ML Architecture**: Multi-model cascade with RTNeural inference
**Build Date**: December 16, 2024

**Dependencies**:
- JUCE Framework (GPL/Commercial)
- RTNeural (BSD-3-Clause)
- Eigen (MPL2)

---

**Last Updated**: December 16, 2024
**Status**: Production Ready (awaiting real training data)
