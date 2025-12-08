# iDAW Project Outline - Complete File Structure

**Intelligent Digital Audio Workstation**  
A comprehensive music production system combining C++ core, Python intelligence, and modern web UI.

---

## Table of Contents

1. [Core C++ Components](#core-c-components)
2. [Web Frontend (iDAWi)](#web-frontend-idawi)
3. [Python Music Brain](#python-music-brain)
4. [Penta Core](#penta-core)
5. [Documentation](#documentation)
6. [Templates & Guides](#templates--guides)
7. [Build & Configuration](#build--configuration)
8. [Tests](#tests)
9. [Tools & Utilities](#tools--utilities)
10. [Data Files](#data-files)
11. [External Dependencies](#external-dependencies)

---

## Core C++ Components

### Main Source Files (`src/`)

#### Audio Processing
- `src/audio/AudioFile.cpp` - Audio file I/O operations
- `src/dsp/audio_buffer.cpp` - Audio buffer management
- `src/dsp/filters.cpp` - DSP filter implementations
- `src/dsp/simd_ops.cpp` - SIMD-optimized operations

#### MIDI Processing
- `src/midi/groove.cpp` - Groove extraction and application
- `src/midi/humanizer.cpp` - MIDI humanization algorithms
- `src/midi/midi_engine.cpp` - Core MIDI engine
- `src/midi/MidiIO.cpp` - MIDI I/O operations
- `src/midi/MidiMessage.cpp` - MIDI message handling
- `src/midi/MidiSequence.cpp` - MIDI sequence management

#### Harmony & Music Theory
- `src/harmony/chord.cpp` - Chord analysis and generation
- `src/harmony/progression.cpp` - Chord progression analysis
- `src/harmony/voice_leading.cpp` - Voice leading algorithms

#### Core Systems
- `src/core/logging.cpp` - Logging system
- `src/core/memory.cpp` - Memory management
- `src/core/types.cpp` - Core type definitions

#### Export & Project Management
- `src/export/StemExporter.cpp` - Audio stem export functionality
- `src/project/ProjectFile.cpp` - Project file I/O

#### Plugin System
- `src/plugin/vst3/PluginEditor.cpp` - VST3 plugin editor
- `src/plugin/vst3/PluginProcessor.cpp` - VST3 plugin processor

#### Python Bindings
- `src/python/bindings.cpp` - Python C API bindings
- `src/python/groove_bindings.cpp` - Groove Python bindings
- `src/python/harmony_bindings.cpp` - Harmony Python bindings

### Header Files (`include/`)

#### Audio
- `include/daiw/audio/AudioFile.h`
- `include/daiw/export/StemExporter.h`

#### MIDI
- `include/daiw/midi/` (3 header files)

#### Project
- `include/daiw/project/` (1 header file)

#### Core Types
- `include/daiw/memory.hpp`
- `include/daiw/types.hpp`

#### Penta Core Headers
- `include/penta/` (21 header files)

### iDAW Core (`iDAW_Core/`)

#### Core Components
- `iDAW_Core/src/DreamStateComponent.cpp` - Dream state visualization
- `iDAW_Core/src/MemoryManager.cpp` - Memory management
- `iDAW_Core/src/PythonBridge.cpp` - Python integration bridge

#### Headers
- `iDAW_Core/include/DreamStateComponent.h`
- `iDAW_Core/include/MemoryManager.h`
- `iDAW_Core/include/PythonBridge.h`
- `iDAW_Core/include/SafetyUtils.h`
- `iDAW_Core/include/Version.h`

#### Plugin System (10 Audio Plugins)

**Brush Plugin**
- `iDAW_Core/plugins/Brush/include/BrushProcessor.h`
- `iDAW_Core/plugins/Brush/src/BrushProcessor.cpp`
- `iDAW_Core/plugins/Brush/shaders/Brushstroke.frag`
- `iDAW_Core/plugins/Brush/shaders/Brushstroke.vert`

**Chalk Plugin**
- `iDAW_Core/plugins/Chalk/include/ChalkProcessor.h`
- `iDAW_Core/plugins/Chalk/src/ChalkProcessor.cpp`
- `iDAW_Core/plugins/Chalk/shaders/Dusty.frag`
- `iDAW_Core/plugins/Chalk/shaders/Dusty.vert`

**Eraser Plugin**
- `iDAW_Core/plugins/Eraser/include/EraserEditor.h`
- `iDAW_Core/plugins/Eraser/include/EraserProcessor.h`
- `iDAW_Core/plugins/Eraser/src/EraserProcessor.cpp`
- `iDAW_Core/plugins/Eraser/shaders/ChalkDust.frag`
- `iDAW_Core/plugins/Eraser/shaders/ChalkDust.vert`

**Palette Plugin**
- `iDAW_Core/plugins/Palette/include/PaletteProcessor.h`
- `iDAW_Core/plugins/Palette/src/PaletteProcessor.cpp`
- `iDAW_Core/plugins/Palette/shaders/Watercolor.frag`

**Parrot Plugin**
- `iDAW_Core/plugins/Parrot/include/ParrotProcessor.h`
- `iDAW_Core/plugins/Parrot/src/ParrotProcessor.cpp`
- `iDAW_Core/plugins/Parrot/shaders/Feather.frag`
- `iDAW_Core/plugins/Parrot/shaders/Feather.vert`

**Pencil Plugin**
- `iDAW_Core/plugins/Pencil/include/PencilEditor.h`
- `iDAW_Core/plugins/Pencil/include/PencilProcessor.h`
- `iDAW_Core/plugins/Pencil/src/PencilProcessor.cpp`
- `iDAW_Core/plugins/Pencil/shaders/Graphite.frag`
- `iDAW_Core/plugins/Pencil/shaders/Graphite.vert`

**Press Plugin**
- `iDAW_Core/plugins/Press/include/PressEditor.h`
- `iDAW_Core/plugins/Press/include/PressProcessor.h`
- `iDAW_Core/plugins/Press/src/PressProcessor.cpp`
- `iDAW_Core/plugins/Press/shaders/Heartbeat.frag`
- `iDAW_Core/plugins/Press/shaders/Heartbeat.vert`

**Smudge Plugin**
- `iDAW_Core/plugins/Smudge/include/SmudgeProcessor.h`
- `iDAW_Core/plugins/Smudge/src/SmudgeProcessor.cpp`
- `iDAW_Core/plugins/Smudge/shaders/Scrapbook.frag`

**Stamp Plugin**
- `iDAW_Core/plugins/Stamp/include/StampProcessor.h`
- `iDAW_Core/plugins/Stamp/src/StampProcessor.cpp`
- `iDAW_Core/plugins/Stamp/shaders/RubberStamp.frag`
- `iDAW_Core/plugins/Stamp/shaders/RubberStamp.vert`

**Stencil Plugin**
- `iDAW_Core/plugins/Stencil/include/StencilProcessor.h`
- `iDAW_Core/plugins/Stencil/src/StencilProcessor.cpp`
- `iDAW_Core/plugins/Stencil/shaders/Cutout.frag`
- `iDAW_Core/plugins/Stencil/shaders/Cutout.vert`

**Trace Plugin**
- `iDAW_Core/plugins/Trace/include/TraceProcessor.h`
- `iDAW_Core/plugins/Trace/src/TraceProcessor.cpp`
- `iDAW_Core/plugins/Trace/shaders/Spirograph.frag`

#### Shared Shaders
- `iDAW_Core/shaders/HandDrawn.frag`
- `iDAW_Core/shaders/HandDrawn.vert`

#### Data
- `iDAW_Core/data/GenreDefinitions.json`

#### Tests
- `iDAW_Core/tests/StressTestSuite.h`

### Legacy Plugin Files (`plugins/`)
- `plugins/CMakeLists.txt` - Plugin CMake configuration
- `plugins/` (5 files: 2 *.cpp, 2 *.h, 1 *.txt)

### Legacy Plugin Headers (Root)
- `PluginProcessor.h` - Legacy plugin processor header
- `PluginProcessor.cpp` - Legacy plugin processor implementation
- `PluginEditor.h` - Legacy plugin editor header
- `PluginEditor.cpp` - Legacy plugin editor implementation
- `PluginProcessorTest.cpp` - Plugin processor tests
- `PluginEditorTest.cpp` - Plugin editor tests

### External Headers (`external/`)
- `external/` (3 files: 2 *.h, 1 *.txt)

---

## Web Frontend (iDAWi)

### Configuration Files
- `iDAWi/package.json` - NPM dependencies and scripts
- `iDAWi/tsconfig.json` - TypeScript configuration
- `iDAWi/tsconfig.node.json` - Node TypeScript config
- `iDAWi/vite.config.ts` - Vite build configuration
- `iDAWi/tailwind.config.js` - Tailwind CSS configuration
- `iDAWi/postcss.config.js` - PostCSS configuration
- `iDAWi/.eslintrc.json` - ESLint configuration
- `iDAWi/.gitignore` - Git ignore rules
- `iDAWi/index.html` - Main HTML entry point
- `iDAWi/README.md` - Frontend documentation

### Source Files (`iDAWi/src/`)

#### Main Entry Points
- `iDAWi/src/main.tsx` - React application entry
- `iDAWi/src/index.tsx` - DOM mounting point
- `iDAWi/src/App.tsx` - Main application component
- `iDAWi/src/index.css` - Global styles

#### Components

**SideA (Traditional DAW Interface)**
- `iDAWi/src/components/SideA/SideA.tsx` - Main SideA container
- `iDAWi/src/components/SideA/Timeline.tsx` - Timeline component
- `iDAWi/src/components/SideA/Transport.tsx` - Transport controls
- `iDAWi/src/components/SideA/VUMeter.tsx` - VU meter display
- `iDAWi/src/components/SideA/Mixer.tsx` - Mixer interface
- `iDAWi/src/components/SideA/Knob.tsx` - Knob control component
- `iDAWi/src/components/SideA/Toolbar.tsx` - Toolbar component
- `iDAWi/src/components/SideA/index.tsx` - SideA exports

**SideB (Emotion-Driven Interface)**
- `iDAWi/src/components/SideB/SideB.tsx` - Main SideB container
- `iDAWi/src/components/SideB/EmotionWheel.tsx` - Emotion wheel interface
- `iDAWi/src/components/SideB/GhostWriter.tsx` - AI lyric writing assistant
- `iDAWi/src/components/SideB/Interrogator.tsx` - Song intent interrogator
- `iDAWi/src/components/SideB/RuleBreaker.tsx` - Music theory rule breaker
- `iDAWi/src/components/SideB/SideBToolbar.tsx` - SideB toolbar
- `iDAWi/src/components/SideB/index.tsx` - SideB exports

**Shared Components**
- `iDAWi/src/components/shared/FlipIndicator.tsx` - Side flip indicator
- `iDAWi/src/components/shared/index.ts` - Shared exports

#### Hooks
- `iDAWi/src/hooks/useMusicBrain.ts` - Music Brain integration hook
- `iDAWi/src/hooks/useTauriAudio.ts` - Tauri audio engine hook
- `iDAWi/src/hooks/index.ts` - Hooks exports

#### State Management
- `iDAWi/src/store/useStore.ts` - Zustand store configuration

### Tauri Backend (`iDAWi/src-tauri/`)
- `iDAWi/src-tauri/Cargo.toml` - Rust dependencies
- `iDAWi/src-tauri/build.rs` - Build script
- `iDAWi/src-tauri/tauri.conf.json` - Tauri configuration
- `iDAWi/src-tauri/src/main.rs` - Tauri entry point
- `iDAWi/src-tauri/src/lib.rs` - Tauri library
- `iDAWi/src-tauri/src/audio_engine.rs` - Audio engine implementation

### Public Assets
- `iDAWi/public/icon.svg` - Application icon
- `iDAWi/public/index.html` - Public HTML

### Music Brain Bridge
- `iDAWi/music-brain/bridge.py` - Python bridge for Music Brain
- `iDAWi/music-brain/requirements.txt` - Python dependencies
- `iDAWi/music-brain/music_brain/__init__.py` - Package init

---

## Python Music Brain

### Main Package (`music_brain/`)

#### Core Modules
- `music_brain/__init__.py` - Package initialization
- `music_brain/api.py` - REST API server
- `music_brain/emotion_api.py` - Emotion analysis API
- `music_brain/harmony.py` - Harmony analysis module
- `music_brain/groove_engine.py` - Groove engine
- `music_brain/cli.py` - Command-line interface

#### Agents (`music_brain/agents/`)
- `music_brain/agents/__init__.py`
- `music_brain/agents/ableton_bridge.py` - Ableton Live integration
- `music_brain/agents/crewai_music_agents.py` - CrewAI agent system
- `music_brain/agents/unified_hub.py` - Unified agent hub
- `music_brain/agents/voice_profiles.py` - Voice profile management

#### Arrangement (`music_brain/arrangement/`)
- `music_brain/arrangement/__init__.py`
- `music_brain/arrangement/bass_generator.py` - Bass line generation
- `music_brain/arrangement/energy_arc.py` - Energy arc analysis
- `music_brain/arrangement/generator.py` - Arrangement generator
- `music_brain/arrangement/templates.py` - Arrangement templates

#### Audio Analysis (`music_brain/audio/`)
- `music_brain/audio/__init__.py`
- `music_brain/audio/analyzer.py` - Audio analysis engine
- `music_brain/audio/chord_detection.py` - Chord detection from audio
- `music_brain/audio/feel.py` - Audio feel extraction
- `music_brain/audio/frequency_analysis.py` - Frequency analysis
- `music_brain/audio/reference_dna.py` - Reference track analysis

#### Collaboration (`music_brain/collaboration/`)
- `music_brain/collaboration/__init__.py`
- `music_brain/collaboration/comments.py` - Comment system
- `music_brain/collaboration/editing.py` - Collaborative editing
- `music_brain/collaboration/session.py` - Session management
- `music_brain/collaboration/version_control.py` - Version control
- `music_brain/collaboration/websocket.py` - WebSocket server

#### Data (`music_brain/data/`)
- `music_brain/data/chord_progression_families.json`
- `music_brain/data/chord_progressions.json`
- `music_brain/data/emotional_mapping.py`
- `music_brain/data/genre_pocket_maps.json`
- `music_brain/data/humanize_presets.json`
- `music_brain/data/rule_breaking_database.json`
- `music_brain/data/scales_database.json`
- `music_brain/data/song_intent_examples.json`
- `music_brain/data/song_intent_schema.yaml`
- `music_brain/data/vernacular_database.json`

#### DAW Integration (`music_brain/daw/`)
- `music_brain/daw/__init__.py`
- `music_brain/daw/fl_studio.py` - FL Studio integration
- `music_brain/daw/logic.py` - Logic Pro integration
- `music_brain/daw/markers.py` - DAW marker management
- `music_brain/daw/mixer_params.py` - Mixer parameter control
- `music_brain/daw/pro_tools.py` - Pro Tools integration
- `music_brain/daw/reaper.py` - Reaper integration
- `music_brain/daw/README_LOGIC_PRO.md` - Logic Pro documentation

#### Groove (`music_brain/groove/`)
- `music_brain/groove/__init__.py`
- `music_brain/groove/applicator.py` - Groove application
- `music_brain/groove/extractor.py` - Groove extraction
- `music_brain/groove/groove_engine.py` - Groove engine
- `music_brain/groove/templates.py` - Groove templates

#### Integrations (`music_brain/integrations/`)
- `music_brain/integrations/__init__.py`
- `music_brain/integrations/penta_core.py` - Penta Core integration

#### Orchestrator (`music_brain/orchestrator/`)
- `music_brain/orchestrator/__init__.py`
- `music_brain/orchestrator/bridge_api.py` - Bridge API
- `music_brain/orchestrator/interfaces.py` - Interface definitions
- `music_brain/orchestrator/logging_utils.py` - Logging utilities
- `music_brain/orchestrator/orchestrator.py` - Main orchestrator
- `music_brain/orchestrator/pipeline.py` - Processing pipeline

**Orchestrator Processors**
- `music_brain/orchestrator/processors/__init__.py`
- `music_brain/orchestrator/processors/base.py` - Base processor
- `music_brain/orchestrator/processors/groove.py` - Groove processor
- `music_brain/orchestrator/processors/harmony.py` - Harmony processor
- `music_brain/orchestrator/processors/intent.py` - Intent processor

#### Session Management (`music_brain/session/`)
- `music_brain/session/__init__.py`
- `music_brain/session/generator.py` - Session generator
- `music_brain/session/intent_processor.py` - Intent processing
- `music_brain/session/intent_schema.py` - Intent schema
- `music_brain/session/interrogator.py` - Song intent interrogator
- `music_brain/session/teaching.py` - Teaching/tutorial system

#### Structure (`music_brain/structure/`)
- `music_brain/structure/__init__.py`
- `music_brain/structure/chord.py` - Chord structure analysis
- `music_brain/structure/comprehensive_engine.py` - Comprehensive analysis engine
- `music_brain/structure/progression.py` - Progression analysis
- `music_brain/structure/sections.py` - Song section analysis
- `music_brain/structure/tension_curve.py` - Tension curve analysis

#### Text Processing (`music_brain/text/`)
- `music_brain/text/__init__.py`
- `music_brain/text/lyrical_mirror.py` - Lyrical analysis

#### Utilities (`music_brain/utils/`)
- `music_brain/utils/__init__.py`
- `music_brain/utils/instruments.py` - Instrument definitions
- `music_brain/utils/midi_io.py` - MIDI I/O utilities
- `music_brain/utils/ppq.py` - PPQ (pulses per quarter) utilities

#### Voice (`music_brain/voice/`)
- `music_brain/voice/__init__.py`
- `music_brain/voice/auto_tune.py` - Auto-tune implementation
- `music_brain/voice/modulator.py` - Voice modulation
- `music_brain/voice/synth.py` - Voice synthesis
- `music_brain/voice/synthesizer.py` - Synthesizer engine

### DAiW-Music-Brain Package (`DAiW-Music-Brain/`)

#### Core Package
- `DAiW-Music-Brain/pyproject.toml` - Package configuration
- `DAiW-Music-Brain/setup.py` - Setup script
- `DAiW-Music-Brain/requirements.txt` - Dependencies
- `DAiW-Music-Brain/README.md` - Package documentation
- `DAiW-Music-Brain/LICENSE` - License file
- `DAiW-Music-Brain/CLAUDE.md` - AI assistant guide
- `DAiW-Music-Brain/.gitignore` - Git ignore rules

#### Music Brain Module (`DAiW-Music-Brain/music_brain/`)
- `DAiW-Music-Brain/music_brain/__init__.py`
- `DAiW-Music-Brain/music_brain/api.py`
- `DAiW-Music-Brain/music_brain/cli.py`

**Audio Module**
- `DAiW-Music-Brain/music_brain/audio/__init__.py`
- `DAiW-Music-Brain/music_brain/audio/feel.py`

**Data Files**
- `DAiW-Music-Brain/music_brain/data/anger.json`
- `DAiW-Music-Brain/music_brain/data/chord_progressions.json`
- `DAiW-Music-Brain/music_brain/data/disgust.json`
- `DAiW-Music-Brain/music_brain/data/emotional_mapping.py`
- `DAiW-Music-Brain/music_brain/data/fear.json`
- `DAiW-Music-Brain/music_brain/data/genre_pocket_maps.json`
- `DAiW-Music-Brain/music_brain/data/joy.json`
- `DAiW-Music-Brain/music_brain/data/sad.json`
- `DAiW-Music-Brain/music_brain/data/song_intent_examples.json`
- `DAiW-Music-Brain/music_brain/data/song_intent_schema.yaml`
- `DAiW-Music-Brain/music_brain/data/surprise.json`

**DAW Integration**
- `DAiW-Music-Brain/music_brain/daw/__init__.py`
- `DAiW-Music-Brain/music_brain/daw/logic.py`
- `DAiW-Music-Brain/music_brain/daw/mixer_params.py`

**Emotion Analysis**
- `DAiW-Music-Brain/music_brain/emotion/__init__.py`
- `DAiW-Music-Brain/music_brain/emotion/text_analyzer.py`

**Groove Module**
- `DAiW-Music-Brain/music_brain/groove/__init__.py`
- `DAiW-Music-Brain/music_brain/groove/applicator.py`
- `DAiW-Music-Brain/music_brain/groove/extractor.py`
- `DAiW-Music-Brain/music_brain/groove/templates.py`

**Session Management**
- `DAiW-Music-Brain/music_brain/session/__init__.py`
- `DAiW-Music-Brain/music_brain/session/generator.py`
- `DAiW-Music-Brain/music_brain/session/intent_processor.py`
- `DAiW-Music-Brain/music_brain/session/intent_schema.py`
- `DAiW-Music-Brain/music_brain/session/interrogator.py`
- `DAiW-Music-Brain/music_brain/session/teaching.py`

**Structure Analysis**
- `DAiW-Music-Brain/music_brain/structure/__init__.py`
- `DAiW-Music-Brain/music_brain/structure/chord.py`
- `DAiW-Music-Brain/music_brain/structure/comprehensive_engine.py`
- `DAiW-Music-Brain/music_brain/structure/progression.py`
- `DAiW-Music-Brain/music_brain/structure/sections.py`

**Utilities**
- `DAiW-Music-Brain/music_brain/utils/__init__.py`
- `DAiW-Music-Brain/music_brain/utils/instruments.py`
- `DAiW-Music-Brain/music_brain/utils/midi_io.py`
- `DAiW-Music-Brain/music_brain/utils/ppq.py`

#### Documentation (`DAiW-Music-Brain/docs/`)
- `DAiW-Music-Brain/docs/QUICKSTART.md`
- `DAiW-Music-Brain/docs/LOGIC_PRO_INTEGRATION.md`
- `DAiW-Music-Brain/docs/ADVANCED.md`

#### Examples (`DAiW-Music-Brain/examples/`)
- `DAiW-Music-Brain/examples/kelly_song_logic_export.py`

#### Tests (`DAiW-Music-Brain/tests/`)
- `DAiW-Music-Brain/tests/__init__.py`
- `DAiW-Music-Brain/tests/test_api.py`
- `DAiW-Music-Brain/tests/test_basic.py`
- `DAiW-Music-Brain/tests/test_cli.py`
- `DAiW-Music-Brain/tests/test_mixer_params.py`
- `DAiW-Music-Brain/tests/test_text_analyzer.py`

#### UI (`DAiW-Music-Brain/ui/`)
- `DAiW-Music-Brain/ui/emotion_to_logic.py`
- `DAiW-Music-Brain/ui/README.md`
- `DAiW-Music-Brain/ui/requirements.txt`

#### Vault (`DAiW-Music-Brain/vault/`)
- `DAiW-Music-Brain/vault/README.md`
- `DAiW-Music-Brain/vault/Production_Workflows/cpp_audio_architecture.md`
- `DAiW-Music-Brain/vault/Production_Workflows/hybrid_development_roadmap.md`
- `DAiW-Music-Brain/vault/Production_Workflows/juce_getting_started.md`
- `DAiW-Music-Brain/vault/Production_Workflows/juce_survival_kit.md`
- `DAiW-Music-Brain/vault/Production_Workflows/osc_bridge_python_cpp.md`
- `DAiW-Music-Brain/vault/Songwriting_Guides/rule_breaking_masterpieces.md`
- `DAiW-Music-Brain/vault/Songwriting_Guides/rule_breaking_practical.md`
- `DAiW-Music-Brain/vault/Songwriting_Guides/song_intent_schema.md`
- `DAiW-Music-Brain/vault/Templates/DAiW_Task_Board.md`

#### Binaries (`DAiW-Music-Brain/bin/`)
- `DAiW-Music-Brain/bin/daiw-logic` - Logic Pro integration binary
- `DAiW-Music-Brain/bin/README.md`

---

## Penta Core

### Source Files (`src_penta-core/`)

#### Common
- `src_penta-core/common/` (2 *.cpp files)

#### Diagnostics
- `src_penta-core/diagnostics/AudioAnalyzer.cpp`
- `src_penta-core/diagnostics/DiagnosticsEngine.cpp`
- `src_penta-core/diagnostics/PerformanceMonitor.cpp`

#### Groove
- `src_penta-core/groove/GrooveEngine.cpp`
- `src_penta-core/groove/OnsetDetector.cpp`
- `src_penta-core/groove/RhythmQuantizer.cpp`
- `src_penta-core/groove/TempoEstimator.cpp`

#### Harmony
- `src_penta-core/harmony/` (5 *.cpp files)

#### Machine Learning
- `src_penta-core/ml/` (1 *.cpp file)

#### OSC (Open Sound Control)
- `src_penta-core/osc/RTMessageQueue.cpp`
- `src_penta-core/osc/OSCServer.cpp`
- `src_penta-core/osc/OSCMessage.cpp`
- `src_penta-core/osc/OSCHub.cpp`
- `src_penta-core/osc/` (5 total files)

#### Super-Spork
- `src_penta-core/super-spork/` (14 files: 3 *.jpg, 3 *.json, 2 *.jpeg, etc.)

### Python Integration (`penta_core_music-brain/`)
- `penta_core_music-brain/__init__.py`
- `penta_core_music-brain/__main__.py`
- `penta_core_music-brain/.env.example`
- `penta_core_music-brain/requirements.txt`
- `penta_core_music-brain/server.py`

---

## Documentation

### Main Documentation (`docs/`)
- 9 markdown files covering various aspects

### Music Brain Documentation (`docs_music-brain/`)
- 17 files: 16 *.md, 1 *.txt

### Penta Core Documentation (`docs_penta-core/`)
- 25 markdown files

### Obsidian Documentation (`Obsidian_Documentation/`)
- 15 markdown files

### Production Workflows (`Production_Workflows/`)
- 15 markdown guides:
  - `Mastering_Checklist.md`
  - `Metal_Production_Guide.md`
  - `Mixing_Workflow_Checklist.md`
  - `Piano_and_Keys_Humanization_Guide.md`
  - `Pop_Production_Guide.md`
  - `R_B_and_Soul_Production_Guide.md`
  - `Reference_Track_Analysis_Guide.md`
  - `Reverb_and_Delay_Guide.md`
  - `Rock_Production_Guide.md`
  - `Sampling_Guide.md`
  - `Sound_Design_From_Scratch.md`
  - `Strings_and_Orchestral_Guide.md`
  - `Synth_Humanization_Guide.md`
  - `Vocal_Production_Guide.md`
  - `Vocal_Recording_Workflow.md`

### Songwriting Guides (`Songwriting_Guides/`)
- 11 markdown guides

### Theory Reference (`Theory_Reference/`)
- 2 markdown files

### Root Level Documentation

#### Core Documentation
- `SYSTEM_OVERVIEW.md` - System architecture overview
- `PROJECT_RULES.md` - Project rules and guidelines
- `PROJECT_ROADMAP.md` - Project roadmap
- `PROJECT_TIMELINE.md` - Project timeline
- `PROPOSAL_SUMMARY.md` - Proposal summary
- `AUTOMATION_GUIDE.md` - Automation guide
- `DAW_INTEGRATION.md` - DAW integration documentation
- `DAIW_INTEGRATION.md` - DAiW integration guide
- `INTEGRATION_GUIDE.md` - Integration guide
- `INTEGRATION_COMPLETE.md` - Integration completion notes
- `Integration_Architecture.md` - Integration architecture
- `BUILD_STANDALONE.md` - Standalone build instructions
- `BUILD.md` - Build documentation
- `BUILD_COMPLETE.md` - Build completion notes
- `TROUBLESHOOTING.md` - Troubleshooting guide
- `CODE_REVIEW_FOLLOWUP.md` - Code review follow-up
- `BREAKING_CHANGES.md` - Breaking changes log
- `MERGE_SUMMARY.md` - Merge summary
- `MERGE_COMPLETE.md` - Merge completion notes
- `MVP_COMPLETE.md` - MVP completion notes
- `MVP_COMPLETE copy.md` - MVP completion (backup)
- `PHASE_2_QUICKSTART.md` - Phase 2 quickstart guide
- `PHASE_2_PLAN.md` - Phase 2 plan
- `PERFORMANCE_SUMMARY.md` - Performance summary
- `PERFORMANCE_IMPROVEMENTS.md` - Performance improvements
- `PERFORMANCE_OPTIMIZATIONS.md` - Performance optimizations
- `OPTIMIZATION_SUMMARY.md` - Optimization summary
- `FINAL_SESSION_SUMMARY.md` - Final session summary
- `FINAL_SESSION_SUMMARY_V2.md` - Final session summary v2
- `next_steps.md` - Next steps document
- `TEST_README.md` - Test documentation
- `QA_CUSTOMIZATION_GUIDE.md` - QA customization guide

#### Setup & Installation
- `START_HERE.txt` - Getting started guide
- `INSTALL.md` - Installation instructions
- `QUICKSTART_penta-core.md` - Penta Core quickstart
- `README_penta-core.md` - Penta Core README
- `README_music-brain.md` - Music Brain README
- `ROADMAP_penta-core.md` - Penta Core roadmap
- `PLATFORM_QUICK_REFERENCE.md` - Platform reference
- `Gemini_Setup.md` - Gemini AI setup

#### Development Guides
- `CLAUDE.md` - Claude AI assistant guide
- `CLAUDE_AGENT_GUIDE.md` - Claude agent guide
- `ChatGPT_Custom_GPT_Instructions.md` - ChatGPT instructions
- `GPT_Upload_Checklist.md` - GPT upload checklist
- `DEVELOPMENT_ROADMAP_music-brain.md` - Music Brain roadmap
- `hybrid_development_roadmap.md` - Hybrid development roadmap

#### Task & Project Management
- `DAiW_Task_Board.md` - Task board
- `COMPREHENSIVE_TODO_IDAW.md` - Comprehensive TODO list
- `TODO_IMPLEMENTATION_SUMMARY.md` - TODO implementation summary
- `TODO_COMPLETION_SUMMARY.md` - TODO completion summary
- `TRACK_2_SUMMARY.md` - Track 2 summary

#### Technical Documentation
- `VST_PLUGIN_IMPLEMENTATION_PLAN.md` - VST plugin plan
- `dependencies.md` - Dependency documentation
- `SOURCES.txt` - Source file listing
- `Doxyfile` - Doxygen configuration
- `MAIN_DOCUMENTATION.md` - Main documentation index

#### Completion & Summary Documents
- `DELIVERY_SUMMARY.md` - Delivery summary
- `DELIVERY_SUMMARY_V2.md` - Delivery summary v2
- `CLEANUP_SUMMARY.md` - Cleanup summary
- `FILE_CLEANUP_SUMMARY.md` - File cleanup summary
- `music_brain_vault_fixes_complete.md` - Music Brain vault fixes
- `music_brain_vault_analysis_report.md` - Music Brain vault analysis
- `SPRINT_5_COMPLETION_SUMMARY.md` - Sprint 5 completion
- `SAMPLE_LIBRARY_COMPLETE.md` - Sample library completion

#### Reference Documents
- `REFINED_PRIORITY_PLANS.md` - Refined priority plans
- `progress.md` - Progress tracking

### Production Guides (Root Level)
- `Ambient Atmospheric Production Guide.md`
- `Country Production Guide.md`
- `EQ Deep Dive Guide.md`
- `Guitar Recording Workflow.md`
- `Hook Writing Guide.md`
- `Indie Alternative Production Guide.md`
- `Jazz Production Guide.md`
- `Logic Pro Settings.md`
- `Lyric Writing Guide.md`
- `Metal Production Guide.md`
- `Music Release Strategy Guide.md`
- `Overcoming Writers Block.md`
- `Piano and Keys Humanization Guide.md`
- `PreSonus AudioBox iTwo.md`
- `R&B and Soul Production Guide.md`
- `Rewriting and Editing Guide.md`
- `Sampling Guide.md`
- `Social Media for Musicians.md`
- `Sound Design From Scratch.md`
- `Synth Humanization Guide.md`

### Sprint Documentation
- `Sprint_1_Core_Testing_and_Quality.md`
- `Sprint_4_Audio_and_MIDI_Enhancements.md`
- `Sprint_5_Platform_and_Environment_Support.md`
- `Sprint_6_Advanced_Music_Theory_and_AI.md`
- `Sprint_7_Mobile_Web_Companion.md`

### Vault Documentation (`vault/`)
- `vault/README.md`
- `vault/Production_Guides/` (7 guides)
- `vault/Songs/when-i-found-you-sleeping/` (complete song project)
- `vault/Songwriting_Guides/` (3 guides + MIDI examples)
- `vault/Templates/` (1 template)

---

## Templates & Guides

### DAW Templates (`Templates/`)

#### Ableton Live
- `Templates/ableton_live/iDAW_Starter_Template.json`

#### FL Studio
- `Templates/fl_studio/iDAW_Starter_Template.json`

#### Logic Pro
- `Templates/logic_pro/iDAW_Starter_Template.json`

#### Pro Tools
- `Templates/pro_tools/iDAW_Starter_Template.json`

#### Reaper
- `Templates/reaper/iDAW_OSC.ReaperOSC`
- `Templates/reaper/scripts/iDAW_Analyze_Selection.lua`
- `Templates/reaper/effects/penta_core.jsfx`

### Other Templates
- `Templates/` (20 total files: 13 *.md, 4 *.json, 1 *.jsfx)

---

## Build & Configuration

### CMake
- `CMakeLists.txt` - Main CMake configuration

### Python Configuration
- `pyproject.toml` - Main Python project configuration
- `pyproject_penta-core.toml` - Penta Core Python config
- `requirements.txt` - Python dependencies
- `environment.yml` - Conda environment

### Build Scripts
- `build_industrial_kit.py` - Industrial kit build script
- `build_standalone.py` - Standalone build script
- `build_quick.sh` - Quick build script (Unix)
- `build_quick.bat` - Quick build script (Windows)

### Platform-Specific
- `macos/` (3 files: 1 *.entitlements, 1 *.plist, 1 *.sh)
- `install_windows.ps1` - Windows installation script
- `web.config` - Web server configuration

### Docker
- `Dockerfile` - Main Dockerfile
- `.cursor/Dockerfile` - Cursor-specific Dockerfile

### GitHub
- `.github/workflows/ci.yml` - CI/CD workflow
- `.github/copilot-instructions.md` - GitHub Copilot instructions
- 8 total files: 5 *.yml, 3 *.md

---

## Tests

### C++ Tests (`tests/`)
- `tests/test_main.cpp`
- `tests/test_memory.cpp`
- `tests/test_simd.cpp`
- `tests/test_harmony.cpp`
- `tests/test_groove.cpp`
- `tests/AudioFileTest.cpp`
- `tests/MidiSequenceTest.cpp`
- `tests/StemExporterTest.cpp`

### Penta Core Tests (`tests_penta-core/`)
- `tests_penta-core/CMakeLists.txt`
- `tests_penta-core/diagnostics_test.cpp`
- `tests_penta-core/groove_test.cpp`
- `tests_penta-core/harmony_test.cpp`
- `tests_penta-core/osc_test.cpp`
- `tests_penta-core/performance_test.cpp`
- `tests_penta-core/plugin_test_harness.cpp`
- `tests_penta-core/rt_memory_test.cpp`

### Music Brain Tests (`tests_music-brain/`)
- `tests_music-brain/__init__.py`
- `tests_music-brain/test_api.py`
- `tests_music-brain/test_arrangement_templates.py`
- `tests_music-brain/test_basic.py`
- `tests_music-brain/test_bass_generator.py`
- `tests_music-brain/test_bridge_integration.py`
- `tests_music-brain/test_chord_detection.py`
- `tests_music-brain/test_cli_commands.py`
- `tests_music-brain/test_cli_flow.py`
- `tests_music-brain/test_comprehensive_engine.py`
- `tests_music-brain/test_core_modules.py`
- `tests_music-brain/test_daw_integration.py`
- `tests_music-brain/test_energy_arc.py`
- `tests_music-brain/test_error_paths.py`
- `tests_music-brain/test_groove_engine.py`
- `tests_music-brain/test_groove_extractor.py`
- `tests_music-brain/test_harmony_rules.py`
- `tests_music-brain/test_intent_processor.py`
- `tests_music-brain/test_intent_schema.py`
- `tests_music-brain/test_mcp_todo_models.py`
- `tests_music-brain/test_mcp_todo_server.py`
- `tests_music-brain/test_mcp_todo_storage.py`
- `tests_music-brain/test_mcp_workstation_models.py`
- `tests_music-brain/test_mcp_workstation_phases.py`
- `tests_music-brain/test_mcp_workstation_proposals.py`
- `tests_music-brain/test_midi_io.py`
- `tests_music-brain/test_orchestrator.py`
- `tests_music-brain/test_penta_core_integration.py`
- `tests_music-brain/test_penta_core_rules.py`
- `tests_music-brain/test_penta_core_server.py`
- `tests_music-brain/test_performance.py`
- `tests_music-brain/test_sprint4_features.py`
- `tests_music-brain/test_unified_hub.py`

### Benchmarks (`benchmarks/`)
- 3 C++ benchmark files

---

## Tools & Utilities

### Python Tools (`python/`)
- `python/penta_core/teachers/README.md` - Penta Core teachers documentation
- 51 total files: 50 *.py, 1 *.md

### Python Tools Directory (`Python_Tools/`)
- `Python_Tools/groove/groove_extractor.py` - Groove extraction tool
- `Python_Tools/groove/groove_applicator.py` - Groove application tool
- 11 total Python files

### Root Level Python Scripts
- `app.py` - Main application entry
- `audio_analyzer_starter.py` - Audio analyzer launcher
- `audio_refinery.py` - Audio refinement tools
- `audio_tools.py` - Audio utility functions
- `auto_tune.py` - Auto-tune implementation
- `base.py` - Base classes and utilities
- `chord.py` - Chord analysis
- `comparator.py` - Comparison utilities
- `comprehensive_engine.py` - Comprehensive analysis engine
- `daiw_listener_public.py` - Public listener interface
- `daiw_menubar.py` - Menu bar application
- `emotional_mapping.py` - Emotional mapping utilities
- `feel_matching.py` - Feel matching algorithms
- `generate_scales_db.py` - Scale database generator
- `groove_tools.py` - Groove analysis tools
- `harmony_generator.py` - Harmony generation
- `idaw_ableton_ui.py` - Ableton UI integration
- `idaw_complete_pipeline.py` - Complete processing pipeline
- `idaw_library_integration.py` - Library integration
- `intent.py` - Intent processing
- `intent_schema.py` - Intent schema definitions
- `kelly_song_example.py` - Example song processing
- `launcher.py` - Application launcher
- `models.py` - Data models
- `progression_analysis.py` - Progression analysis
- `sample_downloader.py` - Sample downloader
- `sections.py` - Section analysis
- `song_intent_schema.md` - Song intent schema documentation
- `synth.py` - Synthesizer implementation
- `teaching.py` - Teaching system
- `teaching_tools.py` - Teaching utilities
- `template_storage.py` - Template storage
- `templates.py` - Template management
- `therapy.py` - Therapy/analysis tools
- `todo_app.py` - Todo application
- `validate_merge.py` - Merge validation
- `vernacular.py` - Vernacular processing

### Tools Directory (`tools/`)
- `tools/audio_cataloger/audio_cataloger.py`
- `tools/audio_cataloger/__init__.py`
- 2 total Python files

### MCP (Model Context Protocol) Tools

#### MCP Todo (`mcp_todo/`)
- 21 files: 9 *.py, 6 *.json, 3 *no-ext, and others

#### MCP Workstation (`mcp_workstation/`)
- `mcp_workstation/__main__.py`
- `mcp_workstation/proposals.py`
- `mcp_workstation/configs/` (3 files: 2 *.json, 1 *.md)
- 14 total files: 11 *.py, 2 *.json, 1 *.md

### Mobile Tools (`mobile/`)
- `mobile/android_aap.py` - Android Audio Application Protocol
- 5 total Python files

### Build FileIO (`build_fileio/`)
- `build_fileio/CMakeLists.txt` - FileIO CMake configuration
- 1 total text file

---

## Data Files

### Root Level JSON
- `disgust.json`
- `fear.json`
- `genre_pocket_maps.json`
- `rule_breaking_database.json`
- `rule_breaking_masterpieces.md`
- `song_intent_examples.json`
- `surprise.json`
- `vernacular_database.json`

### Data Directory (`data/`)
- 18 files: 10 *.json, 7 *.py, 1 *.md

### Data Files Directory (`Data_Files/`)
- 5 JSON files

### MIDI Files
- `i_feel_broken.mid`
- `examples_music-brain/` (10 *.mid files)
- `vault/Songs/when-i-found-you-sleeping/midi/kelly_song_complete_arrangement.mid`
- `vault/Songwriting_Guides/midi_examples/` (4 *.mid files)

### Project Files
- `iDAW_20251127_182312_neutral.idaw.json` - iDAW project file

---

## External Dependencies

### Bindings (`bindings/`)
- `bindings/CMakeLists.txt` - Bindings CMake configuration
- 6 total files: 5 *.cpp, 1 *.txt

### Modules (`modules/`)
- 1 text file

### Super-Spork (`super-spork/`)
- 14 files including:
  - `super-spork/index.js`
  - `super-spork/package.json`
  - `super-spork/package-lock.json`
  - `super-spork/process.json`
  - `super-spork/README.md`
  - `super-spork/LICENSE`
  - `super-spork/web.config`
  - `super-spork/views/index.ejs`
  - `super-spork/public/css/main.css`
  - `super-spork/haikus.json`
  - Image files (3 *.jpg, 3 *.json, 2 *.jpeg)

---

## Additional Files

### Header Files (Root)
- `abstract.h`
- `availability.h`

### Configuration Files
- `.gitignore` - Main git ignore rules
- `.gitignore_music-brain` - Music Brain git ignore
- `.gitignore_penta-core` - Penta Core git ignore
- `top_level.txt` - Top level configuration
- `valgrind.supp` - Valgrind suppression file
- `CMakeLists_fileio.txt` - FileIO CMake configuration (alternative)

### Documentation Files
- `ChatGPT_Custom_GPT_Instructions.md` - ChatGPT instructions
- `DAiW_Task_Board.md` - Task board template
- `CODE_OF_CONDUCT.md` - Code of conduct
- `FREESOUND_PACK_LIST.md` - Freesound pack listing
- `AUDIO_ANALYZER_TOOLS.md` - Audio analyzer tools documentation
- `Audio Feel Extractor.md` - Audio feel extraction guide
- `COMPLETE_DAW_DOCUMENTATION_WITH_AUDIO.md` - Complete DAW documentation
- `Project Template.md` - Project template
- `MPK_Mini3_Knob_Assignments.txt` - MIDI controller assignments
- `Custom_GPT_Build_Script.md` - Custom GPT build script

### Deployment
- `deployment/streamlit_cloud.md` - Streamlit cloud deployment
- `deployment/pwa_wrapper.md` - PWA wrapper documentation
- 2 total markdown files

### Examples

#### Music Brain Examples (`examples_music-brain/`)
- `examples_music-brain/intents/README.md` - Intent examples documentation
- 14 total files: 10 *.mid, 2 *.py, 1 *.json, and others

#### Penta Core Examples (`examples_penta-core/`)
- 5 Python files

### Run Tests
- `RunTests.cpp` - Test runner

### Test Files (Root)
- `OSCCommunicationTest.cpp` - OSC communication tests
- `test_basic.py` - Basic Python tests
- `test_performance.py` - Performance tests
- `test_performance_optimizations.py` - Performance optimization tests

---

## Project Statistics

**Total Files (Approximate):**
- C++ Source Files: ~150+
- Python Files: ~250+
- TypeScript/React Files: ~50+
- Rust Files (Tauri): ~5+
- Documentation Files: ~150+
- Configuration Files: ~40+
- Test Files: ~60+
- Data Files: ~40+
- Build Scripts: ~10+
- Templates: ~25+

**Key Technologies:**
- **Frontend:** React, TypeScript, Tauri, Tailwind CSS, Vite
- **Backend:** C++ (JUCE framework), Rust (Tauri), Python
- **Audio:** JUCE audio framework, custom DSP algorithms
- **AI/ML:** Python-based music intelligence, CrewAI agents
- **Build:** CMake, Python setuptools, npm/yarn

**Architecture:**
- **Core:** C++ audio engine with JUCE
- **Intelligence:** Python Music Brain for AI-driven features
- **UI:** React/TypeScript web frontend wrapped in Tauri
- **Integration:** OSC, MIDI, DAW bridges (Logic, Pro Tools, Reaper, FL Studio, Ableton)

---

## File Organization Principles

1. **Separation of Concerns:**
   - Core audio processing in C++
   - Intelligence and AI in Python
   - UI in TypeScript/React

2. **Modular Design:**
   - Each plugin is self-contained
   - Music Brain modules are independent
   - Frontend components are modular

3. **Documentation:**
   - Comprehensive guides for each feature
   - Production workflows
   - Integration documentation

4. **Testing:**
   - Unit tests for core functionality
   - Integration tests for DAW bridges
   - Performance benchmarks

5. **Templates:**
   - DAW-specific starter templates
   - Production workflow templates
   - Song project templates

---

**Last Updated:** 2025-01-27  
**Project Version:** 0.2.0  
**Maintainer:** Sean Burdges
