# Framework-ML Integration Guide

## Overview

This document describes how to integrate trained ML models with the UnifiedFramework (CIF/LAS/QEF).

## Integration Architecture

```
UnifiedFramework
    ↓
LAS Emotion Interface → 64-dim emotion embedding
    ↓
ML Models (5 models in pipeline)
    ↓
MIDI Output
    ↓
Music Brain Validation
```

## Integration Code

### Loading ML Models

```python
import torch
import numpy as np
from pathlib import Path
import sys

# Add ml_training to path
sys.path.insert(0, str(Path(__file__).parent.parent / "ml_training"))

from train_all_models import (
    EmotionRecognizer,
    MelodyTransformer,
    HarmonyPredictor,
    DynamicsEngine,
    GroovePredictor
)

# Load trained models
models_dir = Path("ml_training/trained_models/checkpoints")

emotion_model = EmotionRecognizer()
emotion_model.load_state_dict(
    torch.load(models_dir / "EmotionRecognizer_best.pt", map_location="cpu")["model_state_dict"]
)
emotion_model.eval()

melody_model = MelodyTransformer()
melody_model.load_state_dict(
    torch.load(models_dir / "MelodyTransformer_best.pt", map_location="cpu")["model_state_dict"]
)
melody_model.eval()

# ... load other models similarly
```

### Integration with UnifiedFramework

```python
from ml_framework.cif_las_qef.integration.unified import UnifiedFramework, FrameworkConfig
import torch

# Initialize framework
config = FrameworkConfig(
    enable_cif=True,
    enable_las=True,
    enable_ethics=True,
    enable_qef=False  # Optional
)
framework = UnifiedFramework(config)

# Get emotion embedding from LAS
human_input = {
    "text": "I feel serene and peaceful",
    "valence": 0.7,
    "arousal": -0.3
}

# Process through LAS
las_result = framework.las.ei.process_emotional_input(human_input)
esv = las_result.to_dict()

# Convert ESV to 64-dim embedding (extract first 64 dimensions or map)
emotion_embedding = np.array([
    esv.get("valence", 0.0),
    esv.get("arousal", 0.0),
    esv.get("dominance", 0.5),
    esv.get("tension", 0.5),
    # ... pad or extract to 64 dims
])[:64]

# Ensure exactly 64 dimensions
if len(emotion_embedding) < 64:
    emotion_embedding = np.pad(emotion_embedding, (0, 64 - len(emotion_embedding)))
emotion_embedding = emotion_embedding[:64].astype(np.float32)

# Run ML models
with torch.no_grad():
    emotion_tensor = torch.from_numpy(emotion_embedding).unsqueeze(0)
    
    # Melody generation
    melody_output = melody_model(emotion_tensor)  # (1, 128) note probabilities
    
    # Groove generation
    groove_output = groove_model(emotion_tensor)  # (1, 32) groove parameters
    
    # Harmony (needs context: emotion + melody)
    context = np.concatenate([emotion_embedding, melody_output[0].numpy()[:64]])
    context_tensor = torch.from_numpy(context).unsqueeze(0)
    harmony_output = harmony_model(context_tensor)  # (1, 64) chord probabilities
    
    # Dynamics (needs compact context)
    compact_context = emotion_embedding[:32]
    compact_tensor = torch.from_numpy(compact_context).unsqueeze(0)
    dynamics_output = dynamics_model(compact_tensor)  # (1, 16) expression params

# Convert to MIDI
midi_notes = melody_output[0].numpy()  # 128-dim note probabilities
chords = harmony_output[0].numpy()     # 64-dim chord probabilities
groove = groove_output[0].numpy()      # 32-dim groove parameters
expression = dynamics_output[0].numpy() # 16-dim expression parameters
```

### Complete Integration Function

```python
def generate_music_from_emotion(framework: UnifiedFramework, human_input: Dict, ml_models: Dict) -> Dict:
    """
    Complete integration: Emotion → UnifiedFramework → ML Models → MIDI
    
    Args:
        framework: Initialized UnifiedFramework
        human_input: Human emotional input
        ml_models: Dict with loaded ML models
    
    Returns:
        Dict with MIDI output and metadata
    """
    # Step 1: Process through UnifiedFramework
    result = framework.create_with_consent(
        human_emotional_input=human_input,
        require_consent=True
    )
    
    if not result.get("created"):
        return {"error": "Creation failed", **result}
    
    # Step 2: Extract emotion embedding from LAS output
    las_output = result.get("las_output", {})
    esv = las_output.get("esv", {})
    
    # Convert ESV to 64-dim embedding
    emotion_embedding = np.array([
        esv.get("valence", 0.0),
        esv.get("arousal", 0.0),
        esv.get("dominance", 0.5),
        esv.get("tension", 0.5),
    ])
    
    # Pad or extract to 64 dimensions
    if len(emotion_embedding) < 64:
        # Repeat or interpolate to 64 dims
        emotion_embedding = np.tile(emotion_embedding, 64 // len(emotion_embedding) + 1)[:64]
    emotion_embedding = emotion_embedding[:64].astype(np.float32)
    
    # Step 3: Run ML models
    with torch.no_grad():
        emotion_tensor = torch.from_numpy(emotion_embedding).unsqueeze(0)
        
        melody_output = ml_models["melody"](emotion_tensor)
        groove_output = ml_models["groove"](emotion_tensor)
        
        # Context for harmony: emotion + melody
        context = np.concatenate([emotion_embedding, melody_output[0].numpy()[:64]])
        context_tensor = torch.from_numpy(context).unsqueeze(0)
        harmony_output = ml_models["harmony"](context_tensor)
        
        # Compact context for dynamics
        compact_context = emotion_embedding[:32]
        compact_tensor = torch.from_numpy(compact_context).unsqueeze(0)
        dynamics_output = ml_models["dynamics"](compact_tensor)
    
    # Step 4: Compile MIDI output
    midi_output = {
        "notes": melody_output[0].numpy().tolist(),
        "chords": harmony_output[0].numpy().tolist(),
        "groove": groove_output[0].numpy().tolist(),
        "expression": dynamics_output[0].numpy().tolist(),
        "emotion_embedding": emotion_embedding.tolist(),
        "framework_result": result
    }
    
    return midi_output
```

## Testing Integration

### Manual Test Script

```python
# test_framework_ml_integration.py
import sys
from pathlib import Path
import numpy as np
import torch

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent / "ml_framework"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ml_training"))

from cif_las_qef.integration.unified import UnifiedFramework, FrameworkConfig
from train_all_models import (
    EmotionRecognizer, MelodyTransformer, HarmonyPredictor,
    DynamicsEngine, GroovePredictor
)

# Initialize
framework = UnifiedFramework(FrameworkConfig())

# Load models (simplified - would normally load from checkpoints)
# For testing, we can create dummy models or use RTNeural JSON loader

# Test emotion processing
human_input = {
    "text": "I feel calm and peaceful",
    "valence": 0.7,
    "arousal": -0.3
}

result = framework.create_with_consent(human_emotional_input=human_input)
print(f"Framework result: {result.get('created')}")
print(f"Ethics score: {result.get('overall_ethics')}")

# If models loaded, test ML pipeline
# midi_output = generate_music_from_emotion(framework, human_input, ml_models)
```

## RTNeural Integration (C++)

For plugin integration, use RTNeural JSON format:

```cpp
// In plugin processor
#include "penta/ml/MLInterface.h"

// Load models from Resources/models/
auto modelsDir = juce::File::getSpecialLocation(
    juce::File::currentApplicationFile
).getChildFile("Resources/models");

MLInterface mlInterface;
mlInterface.initialize(modelsDir);

// Run inference
std::vector<float> emotionEmbedding(64);  // From LAS or EmotionRecognizer
std::vector<float> melodyOutput(128);
mlInterface.processMelody(emotionEmbedding, melodyOutput);
```

## Integration Status

### Completed ✅
- UnifiedFramework implementation
- ML model training and validation
- RTNeural JSON export
- Integration architecture documented

### In Progress ⚠️
- Complete integration code implementation
- End-to-end testing
- Plugin integration

### Requirements
- PyTorch for Python inference
- RTNeural for C++ inference
- UnifiedFramework dependencies installed

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-18
