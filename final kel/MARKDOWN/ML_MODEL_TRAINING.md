# ML Model Training Guide

This document provides instructions for training and exporting ML models for use in the Kelly MIDI Companion plugin.

## Overview

The plugin uses three types of ML models:

1. **Emotion Model (RTNeural)** - Maps audio features to emotion vectors (128→64 dimensions)
2. **Compound Word Transformer** - Generates MIDI sequences conditioned on emotion
3. **DDSP Timbre Transfer** - Transfers timbre characteristics while preserving pitch/loudness

## Prerequisites

- Python 3.8+
- PyTorch or TensorFlow (depending on model type)
- RTNeural export tools (for emotion model)
- ONNX Runtime or LibTorch (for transformer model)
- TensorFlow Lite (for DDSP model)

## 1. Emotion Model (RTNeural)

### Architecture

- Input: 128-dimensional audio features
- Architecture: 128→256 (Dense + Tanh) → 128 (LSTM) → 64 (Dense)
- Output: 64-dimensional emotion vector

### Training Steps

1. **Prepare Dataset**
   ```python
   # Collect audio samples with emotion labels (valence, arousal)
   # Extract features using MLFeatureExtractor
   # Create training pairs: (features_128, emotion_vector_64)
   ```

2. **Train Model**
   ```python
   import torch
   import torch.nn as nn

   # Define model architecture matching RTNeural structure
   class EmotionModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.dense1 = nn.Linear(128, 256)
           self.tanh = nn.Tanh()
           self.lstm = nn.LSTM(256, 128, batch_first=True)
           self.dense2 = nn.Linear(128, 64)

       def forward(self, x):
           x = self.tanh(self.dense1(x))
           x, _ = self.lstm(x.unsqueeze(1))
           x = self.dense2(x.squeeze(1))
           return x
   ```

3. **Export to RTNeural JSON**
   ```python
   # Use RTNeural's export tools to convert PyTorch model to JSON
   # The JSON format is compatible with RTNeural::ModelT
   ```

4. **Place Model File**
   - Copy `emotion_model.json` to `data/` directory
   - Plugin will find it via `PathResolver`

## 2. Compound Word Transformer

### Architecture

- Input: Compound word tokens (pitch + velocity + duration)
- Architecture: Transformer encoder-decoder
- Conditioning: Emotion tokens (valence, arousal)
- Output: MIDI note tokens

### Training Steps

1. **Prepare MIDI Dataset**
   ```python
   # Use EMOPIA dataset or custom therapeutic music dataset
   # Tokenize MIDI files using MIDITokenizer
   # Add emotion conditioning tokens
   ```

2. **Train Transformer**
   ```python
   import torch
   from transformers import GPT2LMHeadModel, GPT2Config

   # Use GPT-2 architecture or custom transformer
   config = GPT2Config(
       vocab_size=MIDITokenizer.MAX_TOKEN + 256,  # Include emotion tokens
       n_positions=1024,
       n_ctx=1024,
       n_embd=512,
       n_layer=12,
       n_head=8
   )
   model = GPT2LMHeadModel(config)
   ```

3. **Export to ONNX**
   ```python
   import torch.onnx

   # Export model to ONNX format
   dummy_input = torch.randint(0, config.vocab_size, (1, 128))
   torch.onnx.export(
       model,
       dummy_input,
       "transformer_model.onnx",
       input_names=['tokens'],
       output_names=['logits'],
       dynamic_axes={'tokens': {0: 'batch', 1: 'sequence'}}
   )
   ```

4. **Integrate in Plugin**
   - Load ONNX model using ONNX Runtime
   - Use `MIDITokenizer` to encode/decode tokens
   - Condition generation on emotion vectors from RTNeural model

## 3. DDSP Timbre Transfer

### Architecture

- Input: Pitch (f0) and loudness features
- Architecture: Encoder → Latent → Synthesizer
- Output: Audio with learned timbre

### Training Steps

1. **Prepare Audio Dataset**
   ```python
   # Collect therapeutic instrument samples (soft piano, pads, ambient)
   # Extract pitch (f0) and loudness using DDSP tools
   # Create training pairs: (f0, loudness) → audio
   ```

2. **Train DDSP Model**
   ```python
   import tensorflow as tf
   from ddsp import core, synths, processors

   # Define DDSP architecture
   # See: https://github.com/magenta/ddsp
   ```

3. **Export to TFLite**
   ```python
   # Convert TensorFlow model to TFLite
   converter = tf.lite.TFLiteConverter.from_saved_model('ddsp_model')
   tflite_model = converter.convert()
   with open('ddsp_model.tflite', 'wb') as f:
       f.write(tflite_model)
   ```

4. **Integrate in Plugin**
   - Load TFLite model using TensorFlow Lite C++ API
   - Extract f0 and loudness from input audio
   - Run inference and synthesize output

## Model File Locations

Models should be placed in the `data/` directory or plugin bundle resources:

- `data/emotion_model.json` - RTNeural emotion model
- `data/transformer_model.onnx` - Compound Word Transformer
- `data/ddsp_model.tflite` - DDSP timbre transfer model

The plugin uses `PathResolver` to find these files in:
1. Plugin bundle resources (`.component/Contents/Resources/data/`)
2. Development directory (`./data/`)
3. User data directory (fallback)

## Testing Models

After training, test models using the plugin's ML inference:

1. Enable ML inference in plugin settings
2. Load model file (plugin will search automatically)
3. Process audio input and verify emotion detection
4. Generate MIDI using emotion-conditioned transformer
5. Apply timbre transfer if DDSP model is loaded

## Performance Considerations

- **RTNeural models**: Optimized for real-time, should run in < 1ms per inference
- **Transformer models**: May require lookahead buffer and latency compensation
- **DDSP models**: Real-time synthesis, may need optimized TFLite operations

## Troubleshooting

### Model Not Loading
- Check file path using `PathResolver::findDataFile()`
- Verify JSON/ONNX/TFLite format is correct
- Check plugin logs for error messages

### Inference Too Slow
- Reduce model size (fewer layers/parameters)
- Use quantization (INT8 instead of FP32)
- Optimize model architecture for real-time

### Poor Quality
- Increase training dataset size
- Train for more epochs
- Fine-tune on therapeutic music data
- Adjust hyperparameters

## References

- RTNeural: https://github.com/jatinchowdhury18/RTNeural
- Compound Word Transformer: https://github.com/YatingMusic/compound-word-transformer
- DDSP: https://github.com/magenta/ddsp
- ONNX Runtime: https://onnxruntime.ai/
- TensorFlow Lite: https://www.tensorflow.org/lite

