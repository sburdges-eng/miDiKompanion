# Kelly MIDI Companion

**Version:** v3.0.00 (KELLYMIDI-V3.0.00)

## Therapeutic MIDI Generation Plugin

Kelly translates emotions into music through a 216-node emotion thesaurus and intentional rule-breaking system.

> *"Kelly doesn't need to BE the DAW. Kelly needs to be IN the DAW."*

---

## Project Structure

```
KELLY MIDI VERSION 3.0.00/
├── CMakeLists.txt              # Build configuration
├── VERSION                     # Version info
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/                        # C++ Source Code
│   ├── Kelly.h                 # Main header
│   ├── core/                   # Core emotion engine
│   │   ├── EmotionThesaurus.*  # 216-node emotion graph
│   │   ├── EmotionThesaurusLoader.*
│   │   ├── WoundProcessor.*    # Wound processing
│   │   ├── RuleBreakEngine.*   # Intentional rule-breaking
│   │   └── IntentPipeline.*    # 3-phase processor
│   │
│   ├── engines/                # Music Generation Engines (v3.0)
│   │   ├── BassEngine.*        # Bass line generation
│   │   ├── MelodyEngine.*      # Melody generation
│   │   ├── RhythmEngine.*      # Rhythm patterns
│   │   ├── PadEngine.*         # Pad/sustained sounds
│   │   ├── StringEngine.*      # String arrangements
│   │   ├── CounterMelodyEngine.*
│   │   ├── GrooveEngine.*      # Groove/swing
│   │   ├── ArrangementEngine.* # Song arrangement
│   │   ├── TensionEngine.*     # Tension/release
│   │   ├── TransitionEngine.*  # Section transitions
│   │   ├── DynamicsEngine.*    # Dynamic control
│   │   ├── FillEngine.*        # Musical fills
│   │   ├── VariationEngine.*   # Pattern variations
│   │   └── VoiceLeading.*      # Voice leading rules
│   │
│   ├── midi/                   # MIDI Generation
│   │   ├── MidiBuilder.*       # JUCE MIDI export
│   │   ├── MidiGenerator.*     # Main MIDI generator
│   │   ├── ChordGenerator.*    # Chord progressions
│   │   ├── GrooveEngine.*      # MIDI groove
│   │   ├── InstrumentSelector.* # GM instrument mapping
│   │   └── midi_pipeline.cpp   # MIDI pipeline
│   │
│   ├── ui/                     # User Interface
│   │   ├── EmotionWheel.*      # Emotion selector
│   │   ├── CassetteView.*      # Cassette visual
│   │   ├── SidePanel.*         # Side controls
│   │   ├── GenerateButton.*    # Generate button
│   │   └── KellyLookAndFeel.*  # UI theme
│   │
│   ├── plugin/                 # JUCE Plugin
│   │   ├── PluginProcessor.*   # Audio processor
│   │   ├── PluginEditor.*      # Plugin UI
│   │   └── PluginState.*       # State management
│   │
│   ├── voice/                  # Voice Synthesis (v2.0)
│   │   └── VoiceSynthesizer.*
│   │
│   ├── biometric/              # Biometric Input (v2.0)
│   │   └── BiometricInput.*
│   │
│   └── common/                 # Shared Types
│       ├── Types.h
│       ├── KellyTypes.h
│       └── DataLoader.h
│
├── python/                     # Python Implementation
│   ├── emotion_thesaurus.py    # Emotion thesaurus
│   ├── harmony_system.py       # Harmony system
│   ├── engines/                # Python engines (mirrors C++)
│   │   ├── kellymidicompanion_bass_engine.py
│   │   ├── kellymidicompanion_melody_engine.py
│   │   ├── kellymidicompanion_rhythm_engine.py
│   │   └── ... (all engines)
│   └── kellymidicompanion/     # Full Python package
│       ├── kellymidicompanion_session/
│       ├── kellymidicompanion_groove/
│       └── kellymidicompanion_data/
│
├── data/                       # Data Files
│   ├── emotions/               # Emotion definitions (JSON)
│   │   ├── anger.json
│   │   ├── joy.json
│   │   ├── sad.json
│   │   └── ... (all emotions)
│   ├── progressions/           # Chord progressions
│   │   ├── common_progressions.json
│   │   ├── chord_progression_families.json
│   │   └── chord_progressions_db.json
│   └── grooves/                # Groove patterns
│       ├── genre_pocket_maps.json
│       └── genre_mix_fingerprints.json
│
├── include/                    # External Headers
│   └── daiw/                   # DAiW Music Brain headers
│       ├── harmony.hpp
│       ├── midi.hpp
│       ├── memory.hpp
│       └── ... (all headers)
│
├── logic_pro/                  # Logic Pro Integration
│   └── Logic_Pro_Scripter_Kelly.js
│
├── scripts/                    # Build Scripts
│   └── build_and_install.sh
│
├── tests/                      # Test Suite
│   └── test_emotion_engine.cpp
│
└── docs/                       # Documentation
    ├── README.md
    ├── CHANGELOG.md
    ├── WORKSPACE_SETUP.md
    └── ... (all docs)
```

---

## Quick Start

### Prerequisites

- CMake 3.22+
- C++20 compiler (Clang 14+, GCC 11+, or MSVC 2022)
- macOS 11+ / Windows 10+ / Ubuntu 22.04+

### Build

```bash
# Configure (JUCE will be downloaded automatically)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release

# Plugins will be in:
# - build/KellyMidiCompanion_artefacts/Release/VST3/
# - build/KellyMidiCompanion_artefacts/Release/AU/    (macOS only)
```

### Install (macOS)

```bash
./scripts/build_and_install.sh Release
```

---

## What's New in v3.0.00

- **Complete Engine Suite**: 14 specialized music generation engines
- **Consolidated Codebase**: All sources organized in clean directory structure
- **Python + C++ Parity**: Full Python implementation mirrors C++ engines
- **DAiW Integration**: Music Brain headers included for advanced harmony
- **Logic Pro Scripter**: Direct Logic Pro MIDI Scripter integration

---

## Philosophy

> *"Interrogate Before Generate"* — The tool shouldn't finish art for people; it should make them braver.

Kelly exists because technical perfection is not the same as emotional truth. Sometimes the "wrong" note is exactly right.

---

*Built with love, grief, and JUCE.*
