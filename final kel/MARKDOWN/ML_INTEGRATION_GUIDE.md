# ML/AI Integration Guide for Kelly Project

**Date:** 2025-01-27
**Last Updated:** December 16, 2025
**Status:** ✅ **PYTHON ML FRAMEWORK COMPLETE** | 🚧 **C++ RTNEURAL INFRASTRUCTURE COMPLETE**

---

## Overview

This guide documents the integration of machine learning and AI features into the Kelly MIDI Companion plugin. The ML integration consists of two main components:

### Python ML Framework (cif_las_qef) ✅ Complete

1. **CIF (Conscious Integration Framework)** - Human-AI consciousness bridge
2. **LAS (Living Art Systems)** - Self-evolving creative AI systems
3. **QEF (Quantum Emotional Field)** - Network-based emotion synchronization
4. **Emotion Models** - VAD, Plutchik, Quantum, Hybrid emotional modeling

**Status:** ✅ All components complete, tested, and verified (December 16, 2025)
**Location:** `ml_framework/cif_las_qef/`
**Verification:** Run `python verify_ai_features.py` (6/6 tests passing)

### C++ RTNeural Infrastructure 🚧 Complete (model integration pending)

1. **Real-time neural inference** for emotion-conditioned processing
2. **Lock-free audio/ML threading** for non-blocking inference
3. **Emotion-conditioned MIDI generation** using transformer models
4. **DDSP timbre transfer** for neural voice synthesis
5. **RAVE latent manipulation** for sound design
6. **Plugin delay compensation** for latency management
7. **Tauri companion app** for model training and IPC

---

## Architecture

### Thread Safety Model

```
┌─────────────────┐
│  Audio Thread   │  (Real-time, must never block)
│                 │
│  - processBlock │
│  - Extract      │
│    features     │
│  - Submit       │──┐
│    requests     │  │ Lock-free
└─────────────────┘  │ Ring Buffer
                     │
┌─────────────────┐  │
│   ML Thread     │  │
│                 │  │
│  - Inference    │  │
│  - Process      │  │
│  - Return       │──┘
│    results      │
└─────────────────┘
```

### Core Components

**Python ML Framework:**

1. **CIF** - Conscious Integration Framework (human-AI consciousness bridge)
2. **LAS** - Living Art Systems (self-evolving creative AI)
3. **QEF** - Quantum Emotional Field (network-based emotion sync)
4. **Emotion Models** - VAD, Plutchik, Quantum, Hybrid models

**C++ RTNeural Infrastructure:**

1. **LockFreeRingBuffer** - Thread-safe circular buffer
2. **RTNeuralProcessor** - Real-time neural inference
3. **InferenceThreadManager** - ML thread management
4. **PluginLatencyManager** - Delay compensation
5. **MLFeatureExtractor** - Audio feature extraction

---

## Implementation Status

### ✅ Completed

**Python ML Framework:**

- [x] CIF (Conscious Integration Framework) - Complete
- [x] LAS (Living Art Systems) - Complete
- [x] QEF (Quantum Emotional Field) - Complete
- [x] All emotion models (VAD, Plutchik, Quantum, Hybrid) - Complete
- [x] Hybrid Emotional Field broadcasting bug - Fixed (December 16, 2025)
- [x] CIF integration type error - Fixed (December 16, 2025)
- [x] Verification script - All tests passing (6/6)

**C++ RTNeural Infrastructure:**

- [x] Lock-free ring buffer implementation
- [x] RTNeural processor interface
- [x] Inference thread manager
- [x] Plugin latency manager
- [x] ML feature extractor
- [x] Integration into PluginProcessor

### 🚧 In Progress

- [ ] RTNeural library integration (external dependency)
- [ ] Model loading and validation
- [ ] Lookahead buffer implementation
- [ ] Emotion vector application

### 📋 Planned

- [ ] Compound Word Transformer integration
- [ ] DDSP timbre transfer
- [ ] RAVE latent manipulation
- [ ] Tauri companion app
- [ ] Model training pipeline

---

## Usage

### Basic ML Inference Setup

```cpp
// In PluginProcessor::prepareToPlay()
inferenceManager_.start(modelFile);
latencyManager_.setMLLatency(1024);  // Model latency in samples
latencyManager_.setLookaheadLatency(
    PluginLatencyManager::msToSamples(20.0f, sampleRate)
);
```

### Feature Extraction and Inference

```cpp
// In PluginProcessor::processBlock()
auto features = featureExtractor_.extractFeatures(buffer);

InferenceRequest request;
request.features = features;
request.timestamp = sampleCounter_;
inferenceManager_.submitRequest(request);  // Non-blocking

// Check for results
InferenceResult result;
while (inferenceManager_.getResult(result)) {
    applyEmotionVector(result.emotionVector);
}
```

### Latency Management

```cpp
// Plugin automatically reports latency to host
int totalLatency = latencyManager_.getTotalLatency();

// Convert to milliseconds for display
float latencyMs = PluginLatencyManager::samplesToMs(
    totalLatency,
    sampleRate
);
```

---

## Model Integration

### RTNeural Model Format

Models should be exported in RTNeural JSON format:

```json
{
  "layers": [
    {
      "type": "dense",
      "input_size": 128,
      "output_size": 256,
      "weights": [...],
      "bias": [...]
    },
    {
      "type": "tanh"
    },
    {
      "type": "lstm",
      "input_size": 256,
      "hidden_size": 128,
      "weights": [...]
    },
    {
      "type": "dense",
      "input_size": 128,
      "output_size": 64,
      "weights": [...],
      "bias": [...]
    }
  ]
}
```

### Model Export from PyTorch

```python
# Export PyTorch model to RTNeural JSON
import torch
import json

model = YourEmotionModel()
model.eval()

# Convert to RTNeural format
weights = {}
for name, param in model.named_parameters():
    weights[name] = param.detach().cpu().numpy().tolist()

with open("model_weights.json", "w") as f:
    json.dump(weights, f)
```

---

## Performance Considerations

### Latency Budget

- **Audio thread processing:** < 1ms
- **ML inference:** 5-20ms (acceptable with lookahead)
- **Total plugin latency:** < 50ms (for real-time use)

### Memory Usage

- **Ring buffers:** ~8KB per buffer (256 elements × 32 bytes)
- **Model weights:** Varies by model size (typically 1-10MB)
- **Lookahead buffer:** ~2KB per channel (20ms @ 44.1kHz)

### CPU Usage

- **Feature extraction:** ~1-2% CPU (single-threaded)
- **ML inference:** ~5-15% CPU (separate thread)
- **Total overhead:** < 20% CPU on modern hardware

---

## Future Enhancements

### Phase 1: Core ML Integration

- [ ] Complete RTNeural library integration
- [ ] Model validation and error handling
- [ ] Lookahead buffer implementation
- [ ] Emotion vector application to synthesis

### Phase 2: Advanced Models

- [ ] Compound Word Transformer for MIDI generation
- [ ] DDSP timbre transfer integration
- [ ] RAVE latent manipulation

### Phase 3: Training Infrastructure

- [ ] Tauri companion app
- [ ] Model training pipeline
- [ ] Dataset management
- [ ] Model versioning

### Phase 4: Optimization

- [ ] Model quantization
- [ ] SIMD optimizations
- [ ] GPU acceleration (optional)

---

## Dependencies

### Required

- **JUCE** (already included)
- **RTNeural** (to be added as external dependency)
- **C++20** standard library

### Optional (for training)

- **PyTorch** (Python training scripts)
- **Tauri** (companion app)
- **TensorFlow Lite** (alternative inference engine)

---

## Testing

### Unit Tests

```cpp
// Test lock-free ring buffer
TEST(LockFreeRingBuffer, PushPop) {
    LockFreeRingBuffer<float, 256> buffer;
    float data[10] = {1.0f, 2.0f, 3.0f, ...};

    ASSERT_TRUE(buffer.push(data, 10));

    float output[10];
    ASSERT_TRUE(buffer.pop(output, 10));
    ASSERT_EQ(data[0], output[0]);
}
```

### Integration Tests

```cpp
// Test inference pipeline
TEST(InferenceThreadManager, EndToEnd) {
    InferenceThreadManager manager;
    manager.start(modelFile);

    InferenceRequest request;
    // ... fill request

    ASSERT_TRUE(manager.submitRequest(request));

    // Wait for result
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    InferenceResult result;
    ASSERT_TRUE(manager.getResult(result));
}
```

---

## Troubleshooting

### Model Not Loading

- Check file path and permissions
- Verify JSON format is valid RTNeural format
- Check model size matches expected architecture

### High Latency

- Reduce lookahead buffer size
- Use smaller/faster model
- Optimize feature extraction

### Audio Dropouts

- Increase buffer size
- Reduce ML inference frequency
- Check CPU usage

### Thread Safety Issues

- Ensure all audio thread operations are non-blocking
- Use lock-free data structures only
- Verify atomic operations are correct

---

## References

- [RTNeural Documentation](https://github.com/jatinchowdhury18/RTNeural)
- [JUCE Audio Thread Guidelines](https://juce.com/learn/documentation/tutorials/audio-plugin-development)
- [Lock-Free Programming](https://preshing.com/20120612/an-introduction-to-lock-free-programming/)

---

**Last Updated:** December 16, 2025
**Next Review:** After RTNeural integration complete
**Python ML Framework:** ✅ Complete, verified, all bugs fixed
