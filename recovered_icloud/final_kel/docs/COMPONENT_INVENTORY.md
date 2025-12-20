# iDAW Component Inventory

## Overview

Complete catalog of all components organized by pipeline type and functionality.

**Total Components Cataloged**:
- Plugins: 5,419 components
- DAW: 491 components
- Standalone: 62 components
- Shared: 145 components

## Plugin Components (VST3/AU/CLAP)

### JUCE Plugin Suite (`iDAW_Core/plugins/`)

**11 Plugins**:

1. **Pencil** (`Pencil/`)
   - Processor: `PencilProcessor.h/cpp`
   - Editor: `PencilEditor.h`
   - Shaders: Graphite fragment/vertex shaders
   - Purpose: Graphite drawing/audio effect

2. **Eraser** (`Eraser/`)
   - Processor: `EraserProcessor.h/cpp`
   - Editor: `EraserEditor.h`
   - Shaders: ChalkDust fragment/vertex shaders
   - Purpose: Erasing/removing audio elements

3. **Palette** (`Palette/`)
   - Processor: `PaletteProcessor.h/cpp`
   - Shaders: Watercolor fragment shader
   - Purpose: Color/tonal processing

4. **Smudge** (`Smudge/`)
   - Processor: `SmudgeProcessor.h/cpp`
   - Shaders: Scrapbook fragment shader
   - Purpose: Audio smudging/blending

5. **Press** (`Press/`)
   - Processor: `PressProcessor.h/cpp`
   - Editor: `PressEditor.h`
   - Shaders: Heartbeat fragment/vertex shaders
   - Purpose: Pressure-sensitive processing

6. **Trace** (`Trace/`)
   - Processor: `TraceProcessor.h/cpp`
   - Shaders: Spirograph fragment shader
   - Purpose: Pattern tracing

7. **Parrot** (`Parrot/`)
   - Processor: `ParrotProcessor.h/cpp`
   - Shaders: Feather fragment/vertex shaders
   - Purpose: Pattern replication/echo

8. **Stencil** (`Stencil/`)
   - Processor: `StencilProcessor.h/cpp`
   - Shaders: Cutout fragment/vertex shaders
   - Purpose: Masking/stencil effects

9. **Chalk** (`Chalk/`)
   - Processor: `ChalkProcessor.h/cpp`
   - Shaders: Dusty fragment/vertex shaders
   - Purpose: Chalk/texture effects

10. **Brush** (`Brush/`)
    - Processor: `BrushProcessor.h/cpp`
    - Shaders: Brushstroke fragment/vertex shaders
    - Purpose: Brush stroke effects

11. **Stamp** (`Stamp/`)
    - Processor: `StampProcessor.h/cpp`
    - Shaders: RubberStamp fragment/vertex shaders
    - Purpose: Stamping/repeating patterns

**Architecture**: Dual-heap (Side A: RT audio, Side B: AI/Python)

## C++ Real-time Engines (Penta-Core)

### Location: `src_penta-core/` and `include/penta/`

### Harmony Engine (`harmony/`)

**Components**:
- `HarmonyEngine.cpp/h` - Main harmony processing engine
- `ChordAnalyzer.cpp/h` - Chord analysis and recognition
- `ChordAnalyzerSIMD.cpp` - SIMD-optimized chord analysis
- `ScaleDetector.cpp/h` - Musical scale detection
- `VoiceLeading.cpp/h` - Voice leading analysis

**Purpose**: Real-time harmony and chord processing, RT-safe

### Groove Engine (`groove/`)

**Components**:
- `GrooveEngine.cpp/h` - Main groove processing engine
- `OnsetDetector.cpp/h` - Audio onset detection
- `RhythmQuantizer.cpp/h` - Rhythm quantization
- `TempoEstimator.cpp/h` - Tempo estimation

**Purpose**: Real-time rhythm and groove processing, RT-safe

### Diagnostics (`diagnostics/`)

**Components**:
- `DiagnosticsEngine.cpp/h` - Main diagnostics engine
- `AudioAnalyzer.cpp/h` - Audio signal analysis
- `PerformanceMonitor.cpp/h` - Performance monitoring

**Purpose**: Real-time diagnostics and performance monitoring

### ML Interface (`ml/`)

**Components**:
- `MLInterface.cpp/h` - Interface for ML model inference

**Purpose**: Bridge between C++ and ML models (RT-safe)

### OSC (`osc/`)

**Components**:
- `OSCServer.cpp/h` - OSC server implementation
- `OSCClient.cpp/h` - OSC client implementation
- `OSCHub.cpp/h` - OSC hub/router
- `OSCMessage.cpp/h` - OSC message handling
- `RTMessageQueue.cpp/h` - Real-time message queue

**Purpose**: Open Sound Control communication (RT-safe)

### Common (`common/`)

**Components**:
- `RTMemoryPool.cpp/h` - Real-time memory pool
- `RTLogger.cpp/h` - Real-time logger
- `RTTypes.h` - Real-time type definitions
- `SIMDKernels.h` - SIMD kernel definitions

**Purpose**: Common RT-safe utilities and types

## Python Bindings

### Location: `python/penta_core/`

### Harmony (`harmony/`)

**Modules**:
- `counterpoint.py` - Counterpoint analysis
- `jazz_voicings.py` - Jazz chord voicings
- `microtonal.py` - Microtonal harmony
- `neo_riemannian.py` - Neo-Riemannian theory
- `tension.py` - Harmonic tension analysis

### Groove (`groove/`)

**Modules**:
- `drum_replacement.py` - Drum replacement
- `groove_dna.py` - Groove DNA encoding
- `humanization.py` - Humanization algorithms
- `performance.py` - Performance analysis
- `polyrhythm.py` - Polyrhythm generation

### ML (`ml/`)

**Modules**:
- `chord_predictor.py` - Chord prediction
- `gpu_utils.py` - GPU utilities
- `inference.py` - Model inference
- `model_registry.py` - Model registry
- `style_transfer.py` - Style transfer

### Rules (`rules/`)

**Modules**:
- `base.py` - Base rule classes
- `context.py` - Rule context
- `counterpoint_rules.py` - Counterpoint rules
- `emotion.py` - Emotion-based rules
- `harmony_rules.py` - Harmony rules
- `rhythm_rules.py` - Rhythm rules
- `severity.py` - Rule severity levels
- `species.py` - Species counterpoint
- `timing.py` - Timing rules
- `voice_leading.py` - Voice leading rules

### Teachers (`teachers/`)

**Modules**:
- `counterpoint_rules.py` - Counterpoint teaching
- `harmony_rules.py` - Harmony teaching
- `rule_breaking_teacher.py` - Rule-breaking teaching
- `rule_reference.py` - Rule reference
- `voice_leading_rules.py` - Voice leading teaching

### DSP (`dsp/`)

**Modules**:
- `parrot_dsp.py` - Parrot DSP algorithms
- `trace_dsp.py` - Trace DSP algorithms

### Collaboration (`collaboration/`)

**Modules**:
- `collab_ui.py` - Collaboration UI
- `intent_versioning.py` - Intent versioning
- `websocket_server.py` - WebSocket server

### Phases (`phases/`)

**Modules**:
- `phase1_infrastructure.py` - Phase 1 infrastructure
- `phase2_python_api.py` - Phase 2 Python API
- `phase3_cpp_engine.py` - Phase 3 C++ engine integration
- `phase4_plugin.py` - Phase 4 plugin integration

### Core Files

- `server.py` - Main Python server
- `utilities.py` - Utility functions
- `__init__.py` - Package initialization

## Music Brain Components

### Location: `music_brain/`

**Key Modules** (from knowledge base):
- Intent-driven composition (3-phase schema)
- Rule-breaking system (harmony, rhythm, arrangement)
- Groove extraction and application
- Chord progression analysis
- Audio feel extraction

## ML Framework Components

### Location: `ml_framework/cif_las_qef/`

### CIF (Conscious Integration Framework) (`cif/`)
- `core.py` - Main CIF implementation
- `sfl.py` - Sensory Fusion Layer
- `crl.py` - Cognitive Resonance Layer
- `asl.py` - Aesthetic Synchronization Layer

### LAS (Living Art Systems) (`las/`)
- `core.py` - Main LAS implementation
- `emotion_interface.py` - Emotion Interface
- `aesthetic_brain.py` - Aesthetic Brain Core
- `generative_body.py` - Generative Body
- `recursive_memory.py` - Recursive Memory
- `reflex_layer.py` - Reflex Layer

### QEF (Quantum Emotional Field) (`qef/`)
- `core.py` - Main QEF implementation
- `len.py` - Local Empathic Node
- `qsl.py` - Quantum Synchronization Layer
- `prl.py` - Planetary Resonance Layer

### Ethics (`ethics/`)
- `core.py` - Resonant Ethics framework
- `consent.py` - Emotional Consent Protocol
- `rights.py` - AI rights framework

### Integration (`integration/`)
- `unified.py` - UnifiedFramework implementation

### Emotion Models (`emotion_models/`)
- `classical.py` - Classical emotion models
- `quantum.py` - Quantum emotion models
- `hybrid.py` - Hybrid emotion models
- `music_generation.py` - Music generation models
- `voice_synthesis.py` - Voice synthesis models
- `field_dynamics.py` - Field dynamics
- `color_mapping.py` - Color mapping
- `simulation.py` - Emotion simulation

## ML Training Components

### Location: `ml_training/`

**Key Files**:
- `train_all_models.py` - Main training script (5 models)
- `training_utils.py` - Training utilities
- `dataset_loaders.py` - Dataset loaders
- `benchmark_inference.py` - Inference benchmarking
- `validate_models.py` - Model validation
- `export_to_onnx.py` - ONNX export
- `deploy_models.py` - Model deployment

**Model Checkpoints**:
- `trained_models/checkpoints/` - PyTorch checkpoints
- `trained_models/*.json` - RTNeural JSON exports
- `trained_models/history/` - Training history

## Pipeline Type Breakdown

### Plugins Pipeline (5,419 components)

**Purpose**: VST3/AU/CLAP plugins for DAWs

**Components**:
- JUCE plugin processors (11 plugins)
- Shader code (GLSL)
- Plugin editors (UI)
- Real-time audio processing

**Shared with**: DAW (UI components, some engines)

### DAW Pipeline (491 components)

**Purpose**: Full standalone DAW application

**Components**:
- Session management
- Project handling
- Timeline/arrangement
- Mixer
- Track management

**Shared with**: Plugins (engines, UI patterns)

### Standalone Pipeline (62 components)

**Purpose**: Desktop/mobile standalone apps

**Components**:
- Tauri/Electron wrapper
- Desktop UI
- Mobile UI (if applicable)
- Standalone runtime

**Shared with**: DAW (core functionality)

## Shared Components (145)

**Location**: Used across multiple pipeline types

**Types**:
- Common utilities
- Emotion processing (CIF/LAS/QEF)
- ML models
- Music theory (Music Brain)
- C++ engines (Penta-Core)
- Python bindings

## File Structure Summary

```
final kel/
├── iDAW_Core/plugins/          # 11 JUCE plugins
├── src_penta-core/             # C++ RT engines
│   ├── harmony/                # Harmony engine
│   ├── groove/                 # Groove engine
│   ├── diagnostics/            # Diagnostics
│   ├── ml/                     # ML interface
│   ├── osc/                    # OSC communication
│   └── common/                 # Common RT utilities
├── include/penta/              # C++ headers
├── python/penta_core/          # Python bindings
│   ├── harmony/                # Harmony Python API
│   ├── groove/                 # Groove Python API
│   ├── ml/                     # ML Python API
│   ├── rules/                  # Rule system
│   ├── teachers/               # Teaching system
│   └── ...
├── ml_framework/cif_las_qef/   # Emotion frameworks
│   ├── cif/                    # CIF
│   ├── las/                    # LAS
│   ├── qef/                    # QEF
│   ├── ethics/                 # Ethics framework
│   └── integration/            # Unified framework
├── ml_training/                # ML training
├── music_brain/                # Music intelligence
└── knowledge_base/             # Generated knowledge catalog
```

## Integration Points

### Python ↔ C++
- Lock-free ring buffer communication
- Dual-heap architecture (Side A: RT, Side B: AI)
- Python bindings via pybind11

### ML Models ↔ Plugins
- RTNeural JSON format
- C++ MLInterface
- Real-time inference

### Emotion Frameworks ↔ ML Models
- UnifiedFramework orchestrates integration
- Emotion embedding (64-dim) as interface

### Music Brain ↔ ML Models
- ML outputs validated by Music Brain
- Rule-breaking system applied
- Intent-driven composition

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-18  
**Total Components**: 6,077+ (plugins + DAW + standalone)
