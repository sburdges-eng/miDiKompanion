# iDAW Project Outline
**Complete File Structure and Organization**

> **Project**: intelligent Digital Audio Workstation (iDAW)  
> **Philosophy**: "Interrogate Before Generate"  
> **Last Updated**: 2025

---

## 📁 Project Structure Overview

```
iDAW/
├── iDAWi/                    # Frontend Interface (Tauri + React)
├── iDAW_Core/                # C++ Core Engine & Plugins
├── DAiW-Music-Brain/         # Python Music Brain (Package)
├── music_brain/              # Python Music Brain (Main)
├── python/                   # Python Penta-Core Modules
├── src/                      # C++ Source Files
├── src_penta-core/           # Penta-Core C++ Source
├── include/                  # C++ Header Files
├── tests/                    # C++ Tests
├── tests_music-brain/        # Python Music Brain Tests
├── tests_penta-core/         # Penta-Core C++ Tests
├── docs/                     # Documentation
├── docs_music-brain/          # Music Brain Documentation
├── docs_penta-core/           # Penta-Core Documentation
├── vault/                    # Knowledge Base (Obsidian)
├── Templates/                # Project Templates
├── data/                     # Data Files
├── examples_music-brain/      # Music Brain Examples
├── examples_penta-core/      # Penta-Core Examples
├── mcp_todo/                 # MCP Todo Server
├── mcp_workstation/          # MCP Workstation Server
├── bindings/                 # Language Bindings
├── plugins/                  # Plugin Implementations
├── benchmarks/               # Performance Benchmarks
├── tools/                    # Utility Scripts
├── mobile/                   # Mobile App Code
├── deployment/               # Deployment Configs
└── [Root Files]              # Configuration & Documentation
```

---

## 🎨 Frontend: iDAWi

**Location**: `/iDAWi/`  
**Tech Stack**: Tauri 2.0 + React 18 + TypeScript + Tailwind CSS

### Core Application Files
- `package.json` - NPM dependencies and scripts
- `tsconfig.json` - TypeScript configuration
- `tsconfig.node.json` - Node TypeScript config
- `vite.config.ts` - Vite build configuration
- `tailwind.config.js` - Tailwind CSS configuration
- `postcss.config.js` - PostCSS configuration
- `.eslintrc.json` - ESLint rules
- `.gitignore` - Git ignore patterns
- `index.html` - HTML entry point
- `README.md` - Project documentation

### Source Code (`/iDAWi/src/`)
- `main.tsx` - React entry point
- `index.tsx` - Application bootstrap
- `App.tsx` - Main application component
- `index.css` - Global styles

### Components (`/iDAWi/src/components/`)

#### SideA (DAW Interface)
- `SideA/index.tsx` - Main DAW side container
- `SideA/Timeline.tsx` - Timeline component
- `SideA/Transport.tsx` - Playback controls
- `SideA/Mixer.tsx` - Audio mixer interface
- `SideA/VUMeter.tsx` - Volume unit meter
- `SideA/Knob.tsx` - Rotatable knob control
- `SideA/Toolbar.tsx` - Toolbar component

#### SideB (Emotion Interface)
- `SideB/index.tsx` - Main emotion side container
- `SideB/EmotionWheel.tsx` - Emotion selection wheel
- `SideB/GhostWriter.tsx` - AI ghost writer component
- `SideB/Interrogator.tsx` - Song interrogation interface
- `SideB/RuleBreaker.tsx` - Rule-breaking suggestions
- `SideB/SideBToolbar.tsx` - Side B toolbar

#### Shared Components
- `shared/FlipIndicator.tsx` - Side flip indicator
- `shared/index.ts` - Shared exports

### Hooks (`/iDAWi/src/hooks/`)
- `useMusicBrain.ts` - Music Brain integration hook
- `useTauriAudio.ts` - Tauri audio engine hook
- `index.ts` - Hook exports

### State Management (`/iDAWi/src/store/`)
- `useStore.ts` - Zustand store configuration

### Tauri Backend (`/iDAWi/src-tauri/`)
- `Cargo.toml` - Rust dependencies
- `build.rs` - Build script
- `tauri.conf.json` - Tauri configuration
- `src/main.rs` - Tauri entry point
- `src/lib.rs` - Library exports
- `src/audio_engine.rs` - Audio processing engine

### Music Brain Bridge (`/iDAWi/music-brain/`)
- `bridge.py` - IPC bridge script
- `requirements.txt` - Python dependencies
- `music_brain/__init__.py` - Package init

### Public Assets (`/iDAWi/public/`)
- `icon.svg` - Application icon
- `index.html` - Public HTML

---

## 🎵 Music Brain: DAiW-Music-Brain

**Location**: `/DAiW-Music-Brain/`  
**Type**: Python Package

### Package Files
- `setup.py` - Package setup
- `pyproject.toml` - Modern Python package config
- `requirements.txt` - Dependencies
- `LICENSE` - MIT License
- `README.md` - Package documentation
- `CLAUDE.md` - AI Assistant guide
- `.gitignore` - Git ignore patterns

### Core Modules (`/DAiW-Music-Brain/music_brain/`)

#### Main Modules
- `__init__.py` - Package initialization
- `api.py` - Public API
- `cli.py` - Command-line interface
- `emotion_api.py` - Emotion analysis API

#### Groove (`/groove/`)
- `__init__.py`
- `extractor.py` - Groove extraction
- `applicator.py` - Groove application
- `templates.py` - Groove templates

#### Structure (`/structure/`)
- `__init__.py`
- `chord.py` - Chord analysis
- `progression.py` - Progression analysis
- `sections.py` - Section detection
- `comprehensive_engine.py` - Full analysis engine

#### Session (`/session/`)
- `__init__.py`
- `generator.py` - Song generation
- `intent_processor.py` - Intent processing
- `intent_schema.py` - Intent schema definitions
- `interrogator.py` - Song interrogation
- `teaching.py` - Teaching module

#### Audio (`/audio/`)
- `__init__.py`
- `feel.py` - Audio feel analysis

#### DAW Integration (`/daw/`)
- `__init__.py`
- `logic.py` - Logic Pro integration
- `mixer_params.py` - Mixer parameter mapping
- `README_LOGIC_PRO.md` - Logic Pro guide

#### Utils (`/utils/`)
- `__init__.py`
- `instruments.py` - Instrument definitions
- `midi_io.py` - MIDI I/O utilities
- `ppq.py` - PPQ (pulses per quarter) utilities

#### Emotion (`/emotion/`)
- `__init__.py`
- `text_analyzer.py` - Text emotion analysis

#### Data (`/data/`)
- `anger.json` - Anger emotion data
- `disgust.json` - Disgust emotion data
- `fear.json` - Fear emotion data
- `joy.json` - Joy emotion data
- `sad.json` - Sadness emotion data
- `surprise.json` - Surprise emotion data
- `chord_progressions.json` - Chord progression database
- `emotional_mapping.py` - Emotion mapping logic
- `genre_pocket_maps.json` - Genre groove maps
- `song_intent_examples.json` - Intent examples
- `song_intent_schema.yaml` - Intent schema YAML

### Documentation (`/DAiW-Music-Brain/docs/`)
- `QUICKSTART.md` - Quick start guide
- `ADVANCED.md` - Advanced usage
- `LOGIC_PRO_INTEGRATION.md` - Logic Pro integration

### Examples (`/DAiW-Music-Brain/examples/`)
- `kelly_song_logic_export.py` - Logic Pro export example

### Tests (`/DAiW-Music-Brain/tests/`)
- `__init__.py`
- `test_api.py` - API tests
- `test_basic.py` - Basic functionality tests
- `test_cli.py` - CLI tests
- `test_mixer_params.py` - Mixer parameter tests
- `test_text_analyzer.py` - Text analyzer tests

### UI (`/DAiW-Music-Brain/ui/`)
- `emotion_to_logic.py` - Emotion to Logic Pro converter
- `README.md` - UI documentation
- `requirements.txt` - UI dependencies

### Vault (`/DAiW-Music-Brain/vault/`)
- `README.md` - Vault documentation
- `Production_Workflows/` - Production workflow guides
- `Songwriting_Guides/` - Songwriting guides
- `Templates/` - Project templates

### Binaries (`/DAiW-Music-Brain/bin/`)
- `daiw-logic` - Logic Pro integration binary
- `README.md` - Binary documentation

---

## 🧠 Music Brain: Main Implementation

**Location**: `/music_brain/`  
**Type**: Python Package (Main Implementation)

### Core Modules
- `__init__.py` - Package initialization
- `api.py` - Public API
- `cli.py` - Command-line interface
- `emotion_api.py` - Emotion API
- `groove_engine.py` - Groove engine
- `harmony.py` - Harmony analysis

### Agents (`/music_brain/agents/`)
- `__init__.py`
- `ableton_bridge.py` - Ableton Live bridge
- `crewai_music_agents.py` - CrewAI agent integration
- `unified_hub.py` - Unified agent hub
- `voice_profiles.py` - Voice profile management

### Arrangement (`/music_brain/arrangement/`)
- `__init__.py`
- `bass_generator.py` - Bass line generation
- `energy_arc.py` - Energy arc analysis
- `generator.py` - Arrangement generation
- `templates.py` - Arrangement templates

### Audio (`/music_brain/audio/`)
- `__init__.py`
- `analyzer.py` - Audio analysis
- `chord_detection.py` - Chord detection from audio
- `feel.py` - Audio feel extraction
- `frequency_analysis.py` - Frequency analysis
- `reference_dna.py` - Reference track DNA extraction

### Collaboration (`/music_brain/collaboration/`)
- `__init__.py`
- `comments.py` - Collaboration comments
- `editing.py` - Collaborative editing
- `session.py` - Collaboration sessions
- `version_control.py` - Version control
- `websocket.py` - WebSocket communication

### Data (`/music_brain/data/`)
- `chord_progression_families.json` - Chord family data
- `chord_progressions.json` - Chord progression database
- `emotional_mapping.py` - Emotion mapping
- `genre_pocket_maps.json` - Genre groove maps
- `humanize_presets.json` - Humanization presets
- `rule_breaking_database.json` - Rule-breaking examples
- `scales_database.json` - Scale database
- `song_intent_examples.json` - Intent examples
- `song_intent_schema.yaml` - Intent schema
- `vernacular_database.json` - Music vernacular database

### DAW Integration (`/music_brain/daw/`)
- `__init__.py`
- `fl_studio.py` - FL Studio integration
- `logic.py` - Logic Pro integration
- `markers.py` - DAW marker management
- `mixer_params.py` - Mixer parameters
- `pro_tools.py` - Pro Tools integration
- `reaper.py` - Reaper integration
- `README_LOGIC_PRO.md` - Logic Pro documentation

### Groove (`/music_brain/groove/`)
- `__init__.py`
- `applicator.py` - Groove application
- `extractor.py` - Groove extraction
- `groove_engine.py` - Groove engine
- `templates.py` - Groove templates

### Integrations (`/music_brain/integrations/`)
- `__init__.py`
- `penta_core.py` - Penta-Core integration

### Orchestrator (`/music_brain/orchestrator/`)
- `__init__.py`
- `bridge_api.py` - Bridge API
- `interfaces.py` - Interface definitions
- `logging_utils.py` - Logging utilities
- `orchestrator.py` - Main orchestrator
- `pipeline.py` - Processing pipeline
- `processors/` - Processing modules
  - `__init__.py`
  - `base.py` - Base processor
  - `groove.py` - Groove processor
  - `harmony.py` - Harmony processor
  - `intent.py` - Intent processor

### Session (`/music_brain/session/`)
- `__init__.py`
- `generator.py` - Song generation
- `intent_processor.py` - Intent processing
- `intent_schema.py` - Intent schema
- `interrogator.py` - Song interrogation
- `teaching.py` - Teaching module

### Structure (`/music_brain/structure/`)
- `__init__.py`
- `chord.py` - Chord analysis
- `comprehensive_engine.py` - Comprehensive analysis
- `progression.py` - Progression analysis
- `sections.py` - Section detection
- `tension_curve.py` - Tension curve analysis

### Text (`/music_brain/text/`)
- `__init__.py`
- `lyrical_mirror.py` - Lyrical mirroring

### Utils (`/music_brain/utils/`)
- `__init__.py`
- `instruments.py` - Instrument definitions
- `midi_io.py` - MIDI I/O
- `ppq.py` - PPQ utilities

### Voice (`/music_brain/voice/`)
- `__init__.py`
- `auto_tune.py` - Auto-tune implementation
- `modulator.py` - Voice modulation
- `synth.py` - Voice synthesis
- `synthesizer.py` - Synthesizer

---

## ⚙️ C++ Core: iDAW_Core

**Location**: `/iDAW_Core/`  
**Type**: C++ Audio Engine

### Headers (`/iDAW_Core/include/`)
- `DreamStateComponent.h` - Dream state component
- `MemoryManager.h` - Memory management
- `PythonBridge.h` - Python bridge interface
- `SafetyUtils.h` - Safety utilities
- `Version.h` - Version information

### Source (`/iDAW_Core/src/`)
- `DreamStateComponent.cpp` - Dream state implementation
- `MemoryManager.cpp` - Memory manager implementation
- `PythonBridge.cpp` - Python bridge implementation

### Plugins (`/iDAW_Core/plugins/`)

#### Brush Plugin
- `include/BrushProcessor.h`
- `src/BrushProcessor.cpp`
- `shaders/Brushstroke.frag`
- `shaders/Brushstroke.vert`

#### Chalk Plugin
- `include/ChalkProcessor.h`
- `src/ChalkProcessor.cpp`
- `shaders/Dusty.frag`
- `shaders/Dusty.vert`

#### Eraser Plugin
- `include/EraserEditor.h`
- `include/EraserProcessor.h`
- `src/EraserProcessor.cpp`
- `shaders/ChalkDust.frag`
- `shaders/ChalkDust.vert`

#### Palette Plugin
- `include/PaletteProcessor.h`
- `src/PaletteProcessor.cpp`
- `shaders/Watercolor.frag`

#### Parrot Plugin
- `include/ParrotProcessor.h`
- `src/ParrotProcessor.cpp`
- `shaders/Feather.frag`
- `shaders/Feather.vert`

#### Pencil Plugin
- `include/PencilEditor.h`
- `include/PencilProcessor.h`
- `src/PencilProcessor.cpp`
- `shaders/Graphite.frag`
- `shaders/Graphite.vert`

#### Press Plugin
- `include/PressEditor.h`
- `include/PressProcessor.h`
- `src/PressProcessor.cpp`
- `shaders/Heartbeat.frag`
- `shaders/Heartbeat.vert`

#### Smudge Plugin
- `include/SmudgeProcessor.h`
- `src/SmudgeProcessor.cpp`
- `shaders/Scrapbook.frag`

#### Stamp Plugin
- `include/StampProcessor.h`
- `src/StampProcessor.cpp`
- `shaders/RubberStamp.frag`
- `shaders/RubberStamp.vert`

#### Stencil Plugin
- `include/StencilProcessor.h`
- `src/StencilProcessor.cpp`
- `shaders/Cutout.frag`
- `shaders/Cutout.vert`

#### Trace Plugin
- `include/TraceProcessor.h`
- `src/TraceProcessor.cpp`
- `shaders/Spirograph.frag`

### Shaders (`/iDAW_Core/shaders/`)
- `HandDrawn.frag` - Hand-drawn fragment shader
- `HandDrawn.vert` - Hand-drawn vertex shader

### Data (`/iDAW_Core/data/`)
- `GenreDefinitions.json` - Genre definitions

### Tests (`/iDAW_Core/tests/`)
- `StressTestSuite.h` - Stress test suite

---

## 🐍 Python Penta-Core

**Location**: `/python/penta_core/`  
**Type**: Python Core Modules

### Core Modules
- `__init__.py` - Package initialization
- `server.py` - Server implementation
- `utilities.py` - Utility functions

### Collaboration (`/python/penta_core/collaboration/`)
- `__init__.py`
- `collab_ui.py` - Collaboration UI
- `intent_versioning.py` - Intent versioning
- `websocket_server.py` - WebSocket server

### DSP (`/python/penta_core/dsp/`)
- `__init__.py`
- `parrot_dsp.py` - Parrot DSP
- `trace_dsp.py` - Trace DSP

### Groove (`/python/penta_core/groove/`)
- `__init__.py`
- `drum_replacement.py` - Drum replacement
- `groove_dna.py` - Groove DNA extraction
- `humanization.py` - Humanization
- `performance.py` - Performance analysis
- `polyrhythm.py` - Polyrhythm generation

### Harmony (`/python/penta_core/harmony/`)
- `__init__.py`
- `counterpoint.py` - Counterpoint rules
- `jazz_voicings.py` - Jazz voicings
- `microtonal.py` - Microtonal support
- `neo_riemannian.py` - Neo-Riemannian theory
- `tension.py` - Tension analysis

### ML (`/python/penta_core/ml/`)
- `__init__.py`
- `chord_predictor.py` - Chord prediction
- `gpu_utils.py` - GPU utilities
- `inference.py` - Model inference
- `model_registry.py` - Model registry
- `style_transfer.py` - Style transfer

### Phases (`/python/penta_core/phases/`)
- `__init__.py`
- `phase1_infrastructure.py` - Phase 1 infrastructure
- `phase2_python_api.py` - Phase 2 Python API
- `phase3_cpp_engine.py` - Phase 3 C++ engine
- `phase4_plugin.py` - Phase 4 plugin

### Rules (`/python/penta_core/rules/`)
- `__init__.py`
- `base.py` - Base rule class
- `context.py` - Rule context
- `counterpoint_rules.py` - Counterpoint rules
- `emotion.py` - Emotion rules
- `harmony_rules.py` - Harmony rules
- `rhythm_rules.py` - Rhythm rules
- `severity.py` - Rule severity
- `species.py` - Species counterpoint
- `timing.py` - Timing rules
- `voice_leading.py` - Voice leading rules

### Teachers (`/python/penta_core/teachers/`)
- `__init__.py`
- `counterpoint_rules.py` - Counterpoint teaching
- `harmony_rules.py` - Harmony teaching
- `rule_breaking_teacher.py` - Rule-breaking teacher
- `rule_reference.py` - Rule reference
- `voice_leading_rules.py` - Voice leading teaching
- `README.md` - Teachers documentation

### Setup
- `setup.py` - Package setup

---

## 💻 C++ Source Files

**Location**: `/src/`

### Audio (`/src/audio/`)
- `AudioFile.cpp` - Audio file I/O

### Core (`/src/core/`)
- `logging.cpp` - Logging system
- `memory.cpp` - Memory management
- `types.cpp` - Type definitions

### DSP (`/src/dsp/`)
- `audio_buffer.cpp` - Audio buffer management
- `filters.cpp` - Audio filters
- `simd_ops.cpp` - SIMD operations

### Export (`/src/export/`)
- `StemExporter.cpp` - Stem export functionality

### Harmony (`/src/harmony/`)
- `chord.cpp` - Chord analysis
- `progression.cpp` - Progression analysis
- `voice_leading.cpp` - Voice leading

### MIDI (`/src/midi/`)
- `groove.cpp` - Groove processing
- `humanizer.cpp` - MIDI humanization
- `midi_engine.cpp` - MIDI engine
- `MidiIO.cpp` - MIDI I/O
- `MidiMessage.cpp` - MIDI message handling
- `MidiSequence.cpp` - MIDI sequence management

### Plugin (`/src/plugin/vst3/`)
- `PluginEditor.cpp` - VST3 plugin editor
- `PluginProcessor.cpp` - VST3 plugin processor

### Project (`/src/project/`)
- `ProjectFile.cpp` - Project file I/O

### Python (`/src/python/`)
- `bindings.cpp` - Python bindings
- `groove_bindings.cpp` - Groove bindings
- `harmony_bindings.cpp` - Harmony bindings

---

## 🔧 Penta-Core C++ Source

**Location**: `/src_penta-core/`

### OSC (`/src_penta-core/osc/`)
- `RTMessageQueue.cpp` - Real-time message queue
- `OSCServer.cpp` - OSC server
- `OSCMessage.cpp` - OSC message handling
- `OSCHub.cpp` - OSC hub

---

## 📚 Documentation

### Main Docs (`/docs/`)

#### Collaboration
- `collaboration/PROTOCOL.md` - Collaboration protocol

#### DAW Integration
- `daw_integration/FL_STUDIO.md` - FL Studio integration
- `daw_integration/PRO_TOOLS.md` - Pro Tools integration
- `daw_integration/REAPER.md` - Reaper integration
- `daw_integration/TEMPLATES_OVERVIEW.md` - Templates overview

#### ML
- `ml/ML_FRAMEWORKS_EVALUATION.md` - ML framework evaluation

#### Mobile
- `mobile/ANDROID_AAP.md` - Android Audio Plugin
- `mobile/IOS_AUDIO_UNIT.md` - iOS Audio Unit
- `mobile/MOBILE_FRAMEWORKS_EVALUATION.md` - Mobile framework evaluation

### Music Brain Docs (`/docs_music-brain/`)
- `AUDIO_ANALYZER_TOOLS.md` - Audio analyzer tools
- `AUTOMATION_GUIDE.md` - Automation guide
- `DAIW_INTEGRATION.md` - DAiW integration
- `DELIVERY_SUMMARY.md` - Delivery summary
- `DELIVERY_SUMMARY_V2.md` - Delivery summary v2
- `FINAL_SESSION_SUMMARY.md` - Final session summary
- `FINAL_SESSION_SUMMARY_V2.md` - Final session summary v2
- `GROOVE_MODULE_GUIDE.md` - Groove module guide
- `GROOVE_MODULE_GUIDE_V2.md` - Groove module guide v2
- `INTEGRATION_GUIDE.md` - Integration guide
- `music_vernacular_database.md` - Music vernacular database
- `rule_breaking_masterpieces_updated.md` - Rule-breaking masterpieces
- `idaw_example_README.md` - iDAW example README
- `downloads_README.md` - Downloads README
- `integrations/penta-core.md` - Penta-Core integration
- `Audio Feel Extractor.md` - Audio feel extractor

### Penta-Core Docs (`/docs_penta-core/`)
- `README.md` - Penta-Core README
- `BUILD.md` - Build instructions
- `PHASE3_DESIGN.md` - Phase 3 design
- `PHASE3_SUMMARY.md` - Phase 3 summary
- `ai-prompting-guide.md` - AI prompting guide
- `audio-interfaces.md` - Audio interfaces
- `comprehensive-system-requirements.md` - System requirements
- `cpp-programming.md` - C++ programming guide
- `daiw-music-brain.md` - DAiW Music Brain integration
- `daw-engine-stability.md` - DAW engine stability
- `daw-programs.md` - DAW programs
- `daw-track-import-methods.md` - Track import methods
- `daw-ui-patterns.md` - DAW UI patterns
- `instrument-learning-research.md` - Instrument learning research
- `low-latency-daw.md` - Low latency DAW
- `media-production.md` - Media production
- `mcp-protocol-debugging.md` - MCP protocol debugging
- `multi-agent-mcp-guide.md` - Multi-agent MCP guide
- `music-generation-research.md` - Music generation research
- `psychoacoustic-sound-design.md` - Psychoacoustic sound design
- `rust-daw-backend.md` - Rust DAW backend
- `sprints-1.md` - Sprint 1 documentation
- `sprints-2.md` - Sprint 2 documentation
- `sprints-3.md` - Sprint 3 documentation
- `swift-sdks.md` - Swift SDKs

---

## 📖 Vault (Knowledge Base)

**Location**: `/vault/`

### Production Guides
- `Production_Guides/Bass Programming Guide.md`
- `Production_Guides/Compression Deep Dive Guide.md`
- `Production_Guides/Drum Programming Guide.md`
- `Production_Guides/Dynamics and Arrangement Guide.md`
- `Production_Guides/EQ Deep Dive Guide.md`
- `Production_Guides/Groove and Rhythm Guide.md`
- `Production_Guides/Guitar Programming Guide.md`

### Songwriting Guides
- `Songwriting_Guides/rule_breaking_masterpieces.md`
- `Songwriting_Guides/rule_breaking_practical.md`
- `Songwriting_Guides/song_intent_schema.md`

### Templates
- `Templates/DAiW_Task_Board.md`

### Songs
- `Songs/when-i-found-you-sleeping/` - Complete song project
  - `README.md`
  - `creative/freeze-expansion.md`
  - `lyrics/v13-with-chords.md`
  - `lyrics/version-history.md`
  - `performance/pitch-correction.md`
  - `performance/timestamped-sheet.md`
  - `performance/vowel-guide.md`
  - `research/genre-research.md`
  - `research/grief-trauma-research.md`
  - `research/reference-songs.md`

### Root
- `README.md` - Vault documentation

---

## 🧪 Tests

### C++ Tests (`/tests/`)
- `AudioFileTest.cpp` - Audio file tests
- `MidiSequenceTest.cpp` - MIDI sequence tests
- `StemExporterTest.cpp` - Stem exporter tests
- `test_groove.cpp` - Groove tests
- `test_harmony.cpp` - Harmony tests
- `test_main.cpp` - Main test suite
- `test_memory.cpp` - Memory tests
- `test_simd.cpp` - SIMD tests

### Music Brain Tests (`/tests_music-brain/`)
- `__init__.py`
- `test_api.py` - API tests
- `test_arrangement_templates.py` - Arrangement template tests
- `test_basic.py` - Basic tests
- `test_bass_generator.py` - Bass generator tests
- `test_bridge_integration.py` - Bridge integration tests
- `test_chord_detection.py` - Chord detection tests
- `test_cli_commands.py` - CLI command tests
- `test_cli_flow.py` - CLI flow tests
- `test_comprehensive_engine.py` - Comprehensive engine tests
- `test_core_modules.py` - Core module tests
- `test_daw_integration.py` - DAW integration tests
- `test_energy_arc.py` - Energy arc tests
- `test_error_paths.py` - Error path tests
- `test_groove_engine.py` - Groove engine tests
- `test_groove_extractor.py` - Groove extractor tests
- `test_harmony_rules.py` - Harmony rule tests
- `test_intent_processor.py` - Intent processor tests
- `test_intent_schema.py` - Intent schema tests
- `test_mcp_todo_models.py` - MCP todo model tests
- `test_mcp_todo_server.py` - MCP todo server tests
- `test_mcp_todo_storage.py` - MCP todo storage tests
- `test_mcp_workstation_models.py` - MCP workstation model tests
- `test_mcp_workstation_phases.py` - MCP workstation phase tests
- `test_mcp_workstation_proposals.py` - MCP workstation proposal tests
- `test_midi_io.py` - MIDI I/O tests
- `test_orchestrator.py` - Orchestrator tests
- `test_penta_core_integration.py` - Penta-Core integration tests
- `test_penta_core_rules.py` - Penta-Core rule tests
- `test_penta_core_server.py` - Penta-Core server tests
- `test_performance.py` - Performance tests
- `test_sprint4_features.py` - Sprint 4 feature tests
- `test_unified_hub.py` - Unified hub tests

### Penta-Core Tests (`/tests_penta-core/`)
- `CMakeLists.txt` - CMake configuration
- `diagnostics_test.cpp` - Diagnostics tests
- `groove_test.cpp` - Groove tests
- `harmony_test.cpp` - Harmony tests
- `osc_test.cpp` - OSC tests
- `performance_test.cpp` - Performance tests
- `plugin_test_harness.cpp` - Plugin test harness
- `rt_memory_test.cpp` - Real-time memory tests

---

## 📦 Templates

**Location**: `/Templates/`

### DAW Templates
- `ableton_live/iDAW_Starter_Template.json`
- `fl_studio/iDAW_Starter_Template.json`
- `logic_pro/iDAW_Starter_Template.json`
- `pro_tools/iDAW_Starter_Template.json`

### Reaper
- `reaper/iDAW_OSC.ReaperOSC`
- `reaper/scripts/iDAW_Analyze_Selection.lua`
- `reaper/effects/penta_core.jsfx`

### Project Templates
- `Project_Template.md`
- `Session_Notes_Template.md`
- `Song_Template.md`
- `Sound_Design_Template.md`
- `Plugin_Template.md`
- `Mix_Notes_Template.md`
- `Reference_Track_Analysis_Template.md`
- `Sample_Pack_Review_Template.md`
- `Weekly_Review_Template.md`

---

## 🔌 MCP Servers

### MCP Todo (`/mcp_todo/`)
- `README.md` - MCP Todo documentation
- `configs/gemini_instructions.md` - Gemini instructions
- Python modules for todo management

### MCP Workstation (`/mcp_workstation/`)
- `configs/setup_guide.md` - Setup guide
- Python modules for workstation management

---

## 🛠️ Tools & Utilities

**Location**: `/tools/`, `/python/`, root scripts

### Python Tools
- `tools/audio_cataloger/audio_cataloger.py` - Audio cataloger
- `tools/audio_cataloger/__init__.py`
- `Python_Tools/audio/analyzer.py` - Audio analyzer
- `Python_Tools/audio/audio_cataloger.py` - Audio cataloger
- `Python_Tools/audio/audio_feel_extractor.py` - Audio feel extractor
- `Python_Tools/groove/generator.py` - Groove generator
- `Python_Tools/groove/groove_applicator.py` - Groove applicator
- `Python_Tools/groove/groove_extractor.py` - Groove extractor
- `Python_Tools/structure/structure_analyzer.py` - Structure analyzer
- `Python_Tools/structure/structure_extractor.py` - Structure extractor
- `Python_Tools/utils/instruments.py` - Instrument utilities
- `Python_Tools/utils/orchestral.py` - Orchestral utilities
- `Python_Tools/utils/ppq.py` - PPQ utilities

### Root Scripts
- `attrib_fromdict.py` - Attribute from dict utility
- `build_industrial_kit.py` - Industrial kit builder
- `build_standalone.py` - Standalone builder
- `daiw_listener_public.py` - DAiW listener
- `intent.py` - Intent processing
- `ppq.py` - PPQ utility
- `progression.py` - Progression utility
- `synth.py` - Synthesizer utility
- `teaching.py` - Teaching utility
- `teaching_tools.py` - Teaching tools
- `template_storage.py` - Template storage
- `templates.py` - Template utilities
- `therapy.py` - Therapy utility
- `todo_app.py` - Todo application
- `validate_merge.py` - Merge validation
- `vernacular.py` - Vernacular utility

---

## 📊 Data Files

**Location**: `/data/`, `/Data_Files/`

### Data Directory
- JSON files for emotion data, chord progressions, etc.
- Python modules for data processing

### Data Files Directory
- `Data_Files/*.json` - Various JSON data files

---

## 🎯 Examples

### Music Brain Examples (`/examples_music-brain/`)
- MIDI files (`.mid`)
- Python examples
- JSON configuration files
- `intents/README.md` - Intent examples documentation

### Penta-Core Examples (`/examples_penta-core/`)
- Python example scripts

---

## 📱 Mobile

**Location**: `/mobile/`
- Python scripts for mobile app development

---

## 🚀 Deployment

**Location**: `/deployment/`
- `streamlit_cloud.md` - Streamlit Cloud deployment
- `pwa_wrapper.md` - PWA wrapper documentation

---

## 🔗 Bindings

**Location**: `/bindings/`
- C++ binding files
- Text documentation

---

## ⚡ Benchmarks

**Location**: `/benchmarks/`
- C++ benchmark files

---

## 📄 Root Configuration Files

### Build & Configuration
- `.gitignore_music-brain` - Music Brain gitignore
- `.gitignore_penta-core` - Penta-Core gitignore
- `build_fileio/*.txt` - Build file I/O
- `modules/*.txt` - Module definitions
- `PKG-INFO` - Package information
- `valgrind.supp` - Valgrind suppression file
- `top_level.txt` - Top-level definitions

### Documentation (Root)
- `PROJECT_RULES.md` - Project rules (Lariat Banquet System - appears to be unrelated)
- `AUTOMATION_GUIDE.md` - Automation guide
- `ChatGPT_Custom_GPT_Instructions.md` - ChatGPT instructions
- `DAiW_Task_Board.md` - DAiW task board
- `PLATFORM_QUICK_REFERENCE.md` - Platform quick reference
- `PROJECT_RULES.md` - Project rules
- `TEST_README.md` - Test README
- Various production guides (Rock, Indie Alternative, Ambient Atmospheric, etc.)
- Various songwriting guides
- Various workflow guides

### JSON Data (Root)
- `angry.json` - Anger emotion data
- `surprise.json` - Surprise emotion data
- `song_intent_examples.json` - Intent examples
- `vernacular_database.json` - Vernacular database

### MIDI Files
- `i_feel_broken.mid` - Example MIDI file

### Text Files
- `MPK_Mini3_Knob_Assignments.txt` - MIDI controller assignments

---

## 🎨 Super-Spork

**Location**: `/super-spork/`
- Node.js application
- `package.json` - NPM configuration
- `index.js` - Main application
- `README.md` - Documentation
- `LICENSE` - License file
- `haikus.json` - Haiku data
- `process.json` - Process configuration
- `web.config` - Web configuration
- `public/` - Public assets
- `views/` - View templates

---

## 🔧 External Dependencies

**Location**: `/external/`
- Header files (`.h`)
- Text documentation

---

## 📋 Include Headers

**Location**: `/include/`
- C++ header files (`.h`, `.hpp`)
- Core system headers

---

## 🏗️ Build Files

- `build_industrial_kit.py` - Industrial kit build script
- `build_standalone.py` - Standalone build script
- Various CMakeLists.txt files
- Build configuration files

---

## 📝 Summary

### Project Components
1. **iDAWi** - Frontend Tauri application (React + TypeScript)
2. **iDAW_Core** - C++ audio engine with plugins
3. **DAiW-Music-Brain** - Python music brain package
4. **music_brain** - Main Python music brain implementation
5. **python/penta_core** - Python core modules
6. **src/** - C++ source files
7. **src_penta-core/** - Penta-Core C++ source
8. **Tests** - Comprehensive test suites
9. **Documentation** - Extensive documentation
10. **Vault** - Knowledge base and guides

### Key Technologies
- **Frontend**: Tauri, React, TypeScript, Tailwind CSS
- **Backend**: Rust (Tauri), C++ (Audio Engine), Python (Music Brain)
- **Audio**: Real-time audio processing, MIDI, VST3 plugins
- **AI/ML**: Music generation, emotion analysis, rule-breaking engine
- **DAW Integration**: Logic Pro, Pro Tools, Reaper, FL Studio, Ableton Live

### Philosophy
**"Interrogate Before Generate"** - The system helps musicians explore emotional intent before making technical decisions, making them braver in their creative process.

---

**Generated**: 2025  
**Project**: iDAW (intelligent Digital Audio Workstation)  
**Total Files**: 1000+ files across multiple languages and frameworks
