# ML Inference Integration Guide

This document describes the ML inference infrastructure integrated into the Kelly MIDI Companion plugin.

## Overview

The ML inference system provides real-time emotion analysis from audio input, enabling the plugin to automatically adjust MIDI generation based on detected emotional content. The system is designed to be non-blocking and thread-safe for real-time audio processing.

## Architecture

### Components

1. **LockFreeRingBuffer** (`src/ml/LockFreeRingBuffer.h`)
   - Lock-free ring buffer for thread-safe communication
   - Uses atomic operations and memory ordering
   - Suitable for real-time audio thread communication

2. **InferenceThreadManager** (`src/ml/InferenceThreadManager.h/cpp`)
   - Manages ML inference in a separate thread
   - Non-blocking request/result communication
   - Customizable inference function

3. **PluginLatencyManager** (`src/ml/PluginLatencyManager.h`)
   - Manages plugin latency compensation
   - Accounts for ML inference time and lookahead buffers
   - Reports accurate latency to host DAW

4. **InferenceRequest/Result** (`src/ml/InferenceRequest.h`)
   - Data structures for inference communication
   - 128-dim feature vectors, 64-dim emotion vectors

### Integration in PluginProcessor

The `PluginProcessor` class integrates ML inference with:

- **Lookahead Buffer**: 20ms lookahead for feature extraction
- **Feature Extraction**: Extracts 128-dimensional features from audio
- **Non-blocking Inference**: Submits requests and retrieves results without blocking
- **Emotion Mapping**: Maps 64-dim emotion vectors to valence/arousal

## Usage

### Enabling ML Inference

```cpp
processor.enableMLInference(true);
```

### Setting Custom Inference Function

```cpp
processor.setInferenceFunction([](const std::array<float, 128>& features) -> std::array<float, 64> {
    // Your ML model inference here
    // Return 64-dimensional emotion vector
    std::array<float, 64> result;
    // ... inference code ...
    return result;
});
```

### Accessing ML-Derived Emotion

The plugin automatically updates `mlValence_` and `mlArousal_` atomic variables based on inference results. These can be blended with user-set parameters.

## Latency Management

The system automatically manages latency:

- **Lookahead**: 20ms (configurable via `ML_LOOKAHEAD_MS`)
- **ML Inference**: Configurable (default 1024 samples ≈ 23ms at 44.1kHz)
- **Total Latency**: Automatically reported to host DAW

## Thread Safety

- **Audio Thread**: Never blocks, uses lock-free operations
- **Inference Thread**: Runs ML inference asynchronously
- **Communication**: Lock-free ring buffers for data exchange

## Future Enhancements

1. **RTNeural Integration**: Direct integration with RTNeural for optimized inference
2. **Model Loading**: Support for loading ONNX/TFLite models
3. **Feature Extraction**: More sophisticated spectral analysis
4. **Emotion Mapping**: Learned mapping from emotion vectors to VAD space

## Python Training Scripts

Training scripts for emotion-conditioned models should be placed in:

- `python/training/` - Model training scripts
- `python/models/` - Trained model files

See the provided Python examples for:

- Compound Word Transformer (MIDI generation)
- DDSP Timbre Transfer (audio synthesis)
- RAVE Latent Manipulation (sound design)
