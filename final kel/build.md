# Kelly MIDI Companion - Complete Build Guide

**Version:** v3.0.00+
**Focus:** AI/ML Features and Learning Systems
**Last Updated:** 2024

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Build Components](#build-components)
4. [Phase 1: Python Environment Setup](#phase-1-python-environment-setup)
5. [Phase 2: C++ Build with AI Integration](#phase-2-c-build-with-ai-integration)
6. [Phase 3: Verification](#phase-3-verification)
7. [AI/ML Features Verification](#aiml-features-verification)
8. [Integration Testing](#integration-testing)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Configuration](#advanced-configuration)

---

## Overview

This guide covers the complete build process for Kelly MIDI Companion with emphasis on AI/ML features and learning systems. The build includes:

- **Python ML Framework**: CIF/LAS/QEF conscious AI systems
- **C++ Plugin**: JUCE-based audio plugin (VST3, AU, Standalone)
- **Python Bridge**: C++ to Python integration for AI features
- **Python Utilities**: MIDI processing and wrapper tools
- **Test Suite**: Comprehensive unit and integration tests

### Build Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Build Components                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐      ┌──────────────┐                │
│  │ Python ML    │◄────►│ Python       │                │
│  │ Framework    │      │ Bridge       │                │
│  │ (CIF/LAS/QEF)│      │ (kelly_bridge)│                │
│  └──────────────┘      └──────┬───────┘                │
│                               │                         │
│                               ▼                         │
│                        ┌──────────────┐                │
│                        │ C++ Plugin   │                │
│                        │ (JUCE)       │                │
│                        └──────────────┘                │
│                                                          │
│  ┌──────────────┐      ┌──────────────┐                │
│  │ Python       │      │ Test Suite   │                │
│  │ Utilities    │      │ (C++/Python) │                │
│  └──────────────┘      └──────────────┘                │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### System Requirements

**macOS:**

- macOS 11.0 (Big Sur) or later
- Xcode Command Line Tools: `xcode-select --install`
- Homebrew (recommended): `brew install cmake`

**Linux (Ubuntu/Debian):**

- Ubuntu 22.04+ or Debian 11+
- Build essentials: `sudo apt-get install build-essential cmake`

**Windows:**

- Windows 10/11
- Visual Studio 2022 with C++ workload
- CMake 3.22+

### Required Tools

| Tool | Version | Installation |
|------|---------|--------------|
| **CMake** | 3.22+ | `brew install cmake` (macOS) or `apt-get install cmake` (Linux) |
| **C++ Compiler** | C++20 support | Clang 14+, GCC 11+, or MSVC 2022 |
| **Python** | 3.8+ | `python3 --version` |
| **pip** | Latest | `python3 -m pip install --upgrade pip` |

### Optional Tools

- **pybind11**: Automatically fetched by CMake when `BUILD_PYTHON_BRIDGE=ON`
- **Google Test**: Automatically fetched by CMake when `BUILD_TESTS=ON`
- **JUCE**: Automatically fetched by CMake (version 8.0.4)

---

## Build Components

### 1. Python ML Framework (Primary AI Focus)

The ML Framework implements advanced conscious AI systems:

- **CIF (Conscious Integration Framework)**: Human-AI consciousness bridge
  - Sensory Fusion Layer (SFL)
  - Cognitive Resonance Layer (CRL)
  - Aesthetic Synchronization Layer (ASL)
  - Five-stage integration process

- **LAS (Living Art Systems)**: Self-evolving creative AI systems
  - Emotion Interface (EI)
  - Aesthetic Brain Core (ABC)
  - Generative Body (GB)
  - Recursive Memory (RM)
  - Reflex Layer (RL)
  - Aesthetic DNA (aDNA)

- **QEF (Quantum Emotional Field)**: Network-based collective emotion synchronization
  - Local Empathic Nodes (LENs)
  - Quantum Synchronization Layer (QSL)
  - Planetary Resonance Layer (PRL)

- **Emotion Models**: Multiple emotion processing models
  - Classical emotion models
  - Quantum emotion models
  - Hybrid emotion models
  - Voice synthesis models

- **Resonant Ethics**: Ethical framework for conscious AI
  - Five Pillars of ethical principles
  - Resonant Rights Doctrine (RRD)
  - Emotional Consent Protocol (ECP)

**Location:** `ml_framework/`

### 2. C++ Plugin with Python Bridge

The C++ plugin provides real-time MIDI generation:

- **JUCE Plugin**: VST3, AU (macOS), Standalone formats
- **Emotion Engine**: 216-node emotion thesaurus
- **MIDI Generation**: 14 specialized engines (melody, bass, rhythm, etc.)
- **Python Bridge**: Optional C++ to Python integration via pybind11

**Location:** `src/`

**Python Bridge Module:** `kelly_bridge` (built to `python/` directory)

### 3. Python Utilities

MIDI processing tools and bridge wrapper:

- MIDI file I/O (`mido`)
- Real-time MIDI (`python-rtmidi`)
- Bridge wrapper functions
- Example scripts

**Location:** `python/`

### 4. Test Suite

Comprehensive testing:

- C++ unit tests (Google Test)
- Integration tests
- Python framework tests
- Bridge integration tests

**Location:** `tests/`

---

## Phase 1: Python Environment Setup

### Step 1.1: Automated Setup (Recommended)

Use the provided setup script:

```bash
cd "/Users/seanburdges/Desktop/final kel"
chmod +x setup_workspace.sh
./setup_workspace.sh
```

This script will:

- Check Python, CMake, and C++ compiler
- Create virtual environments for ML framework and Python utilities
- Install all dependencies

### Step 1.2: Manual Setup

#### ML Framework Environment

```bash
cd ml_framework

# Create virtual environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy, scipy, matplotlib; print('✓ ML Framework dependencies OK')"

# Deactivate
deactivate
```

**Dependencies installed:**

- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `matplotlib>=3.5.0` - Visualization

#### Python Utilities Environment

```bash
cd ../python

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import mido; print('✓ Python utilities dependencies OK')"

# Deactivate
deactivate
```

**Dependencies installed:**

- `mido>=1.2.10` - MIDI file I/O
- `python-rtmidi>=1.4.9` - Real-time MIDI (optional)
- `pytest>=7.0.0` - Testing (optional)

### Step 1.3: Verify Python Environments

```bash
# Check ML framework
cd ml_framework
source venv/bin/activate
python -c "from cif_las_qef import UnifiedFramework; print('✓ ML Framework import OK')"
deactivate

# Check Python utilities
cd ../python
source venv/bin/activate
python -c "import mido; print('✓ Python utilities OK')"
deactivate
```

---

## Phase 2: C++ Build with AI Integration

### Step 2.1: Configure CMake with Python Bridge

**Important:** Enable `BUILD_PYTHON_BRIDGE=ON` to integrate AI features.

```bash
cd "/Users/seanburdges/Desktop/final kel"

# Clean previous build (optional)
rm -rf build

# Configure with Python bridge enabled
cmake -B build \
    -DBUILD_PYTHON_BRIDGE=ON \
    -DBUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release

# Verify configuration
cmake -B build -L | grep BUILD_PYTHON_BRIDGE
# Should show: BUILD_PYTHON_BRIDGE:BOOL=ON
```

**CMake Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_PYTHON_BRIDGE` | `OFF` | **Enable for AI features** - Builds `kelly_bridge` Python module |
| `BUILD_TESTS` | `ON` | Build unit tests |
| `CMAKE_BUILD_TYPE` | `Debug` | `Release` or `Debug` |

**Expected CMake Output:**

```
-- Python-C++ bridge enabled
--   Python executable: /usr/bin/python3
--   Python version: 3.11.0
--   Module will be built to: /path/to/final kel/python
-- Unit tests enabled
--   Run tests with: ctest or ./tests/KellyTests
```

### Step 2.2: Build Release Version

```bash
# Build all targets (plugin + bridge + tests)
cmake --build build --config Release -j$(sysctl -n hw.ncpu)

# macOS: Use all CPU cores
# Linux: Use $(nproc) instead of sysctl
# Windows: Omit -j flag or use -j%NUMBER_OF_PROCESSORS%
```

**Build Targets:**

1. **KellyMidiCompanion** - Main plugin (VST3, AU, Standalone)
2. **kelly_bridge** - Python module (if `BUILD_PYTHON_BRIDGE=ON`)
3. **KellyTests** - Test suite (if `BUILD_TESTS=ON`)

**Build Output Locations:**

```
build/
├── KellyMidiCompanion_artefacts/
│   └── Release/
│       ├── VST3/
│       │   └── Kelly MIDI Companion.vst3
│       ├── AU/
│       │   └── Kelly MIDI Companion.component  (macOS only)
│       └── Standalone/
│           └── Kelly MIDI Companion.app        (macOS)
│
├── python/
│   └── kelly_bridge.so                         (Linux/macOS)
│   └── kelly_bridge.pyd                        (Windows)
│
└── tests/
    └── KellyTests                              (executable)
```

### Step 2.3: Verify Python Bridge Module

```bash
# Check if bridge module was created
ls -lh build/python/kelly_bridge.*
# Should show: kelly_bridge.so (Linux/macOS) or kelly_bridge.pyd (Windows)

# Copy to python/ directory (if not already there)
cp build/python/kelly_bridge.* python/ 2>/dev/null || true

# Test import (from python/ directory)
cd python
source venv/bin/activate
python -c "import sys; sys.path.insert(0, '.'); import kelly_bridge; print('✓ Python bridge import OK')"
deactivate
cd ..
```

**Note:** The bridge module is automatically placed in `python/` directory by CMake (see `CMakeLists.txt` line 208).

### Step 2.4: Build Tests

Tests are built automatically when `BUILD_TESTS=ON`. Verify:

```bash
# Check test executable exists
ls -lh build/tests/KellyTests

# Or run tests directly
cd build
ctest --output-on-failure
cd ..
```

---

## Phase 3: Verification

### Step 3.1: Test ML Framework Examples

**Important:** Set PYTHONPATH before running examples, or install the package in development mode.

#### Basic Usage Example

```bash
cd ml_framework
source venv/bin/activate

# Option 1: Set PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"
python examples/basic_usage.py

# Option 2: Install in development mode (one-time)
# pip install -e .
# python examples/basic_usage.py

# Expected output:
# Creating with ethical consent protocol...
# === Creation Result ===
# Created: True
# Consent Granted: True
# Overall Ethics Score: 0.709...
# LAS Output: audio
# ...
deactivate
cd ..
```

#### Emotion Models Demo

```bash
cd ml_framework
source venv/bin/activate

# Run emotion models demo
python examples/emotion_models_demo.py

# This demonstrates:
# - Classical emotion models
# - Quantum emotion models
# - Hybrid emotion models
# - Voice synthesis models
deactivate
cd ..
```

#### Advanced Field Demo

```bash
cd ml_framework
source venv/bin/activate

# Run advanced field demo
python examples/advanced_field_demo.py

# Demonstrates QEF network capabilities
deactivate
cd ..
```

#### Music Voice Demo

```bash
cd ml_framework
source venv/bin/activate

# Run music voice demo
python examples/music_voice_demo.py

# Demonstrates voice synthesis integration
deactivate
cd ..
```

### Step 3.2: Verify Python Bridge

```bash
cd python
source venv/bin/activate

# Test basic import
python -c "import sys; sys.path.insert(0, '.'); import kelly_bridge; print('Bridge OK')"

# Run bridge examples
python examples/basic_usage.py
python examples/advanced_features.py

deactivate
cd ..
```

**Expected Bridge Functions:**

The `kelly_bridge` module provides C++ functions accessible from Python:

- Emotion thesaurus operations
- MIDI generation functions
- Emotion-to-music mapping
- Wound processing
- Rule-breaking engine

### Step 3.3: Run Test Suite

```bash
cd build

# Run all tests
ctest --output-on-failure

# Run specific test category
ctest -R core --output-on-failure        # Core tests
ctest -R engines --output-on-failure     # Engine tests
ctest -R integration --output-on-failure # Integration tests

# Run with verbose output
ctest --output-on-failure --verbose

cd ..
```

**Test Categories:**

- `core/` - Core emotion engine tests
- `engines/` - Music generation engine tests
- `integration/` - Integration tests
- `midi/` - MIDI processing tests
- `voice/` - Voice synthesis tests
- `utils/` - Utility function tests

### Step 3.4: Verify Plugin Installation

#### macOS

```bash
# VST3 location
ls -lh ~/Library/Audio/Plug-Ins/VST3/Kelly\ MIDI\ Companion.vst3

# AU location
ls -lh ~/Library/Audio/Plug-Ins/Components/Kelly\ MIDI\ Companion.component

# Standalone app
ls -lh build/KellyMidiCompanion_artefacts/Release/Standalone/Kelly\ MIDI\ Companion.app
```

#### Linux

```bash
# VST3 location (user)
ls -lh ~/.vst3/Kelly\ MIDI\ Companion.vst3

# Or system-wide
ls -lh /usr/local/lib/vst3/Kelly\ MIDI\ Companion.vst3
```

#### Windows

```bash
# VST3 location
dir "%LOCALAPPDATA%\Programs\Common\VST3\Kelly MIDI Companion.vst3"

# Or system-wide
dir "C:\Program Files\Common Files\VST3\Kelly MIDI Companion.vst3"
```

---

## AI/ML Features Verification

### 1. CIF Integration (Conscious Integration Framework)

**Test Human-AI Consciousness Bridge:**

```python
# ml_framework/examples/basic_usage.py demonstrates CIF
from cif_las_qef import CIF

cif = CIF()

# Test integration
result = cif.integrate(
    human_bio_data={"heart_rate": 75, "eeg_alpha": 0.6},
    las_emotional_state={"esv": {"valence": 0.3, "arousal": 0.5}}
)

print(f"CIF Status: {cif.get_status()}")
print(f"Integration Stage: {result.get('stage', 'N/A')}")
```

**Expected:** Integration progresses through 5 stages (calibration → resonance → synchronization → co-creation → emergence)

### 2. LAS Systems (Living Art Systems)

**Test Self-Evolving Creative AI:**

```python
from cif_las_qef import LAS

las = LAS()

# Generate creative output
result = las.generate(
    emotional_input={
        "biofeedback": {"heart_rate": 75},
        "voice": {"tone": 0.3}
    },
    creative_goal={"style": "ambient"}
)

# Evolve from feedback
evolution = las.evolve({
    "aesthetic_rating": 0.8,
    "emotional_resonance": 0.7
})

print(f"Evolution Result: {evolution}")
print(f"aDNA Updated: {evolution.get('adna_updated', False)}")
```

**Expected:** LAS generates creative output and evolves based on feedback, updating its Aesthetic DNA (aDNA)

### 3. QEF Network (Quantum Emotional Field)

**Test Emotion Synchronization:**

```python
from cif_las_qef import QEF

qef = QEF(node_id="test_node")

# Activate node
qef.activate()

# Emit emotional state
qas = qef.emit_emotional_state(
    esv={"valence": 0.3, "arousal": 0.5}
)

# Get collective resonance
collective = qef.receive_collective_resonance()

print(f"QAS Emitted: {qas}")
print(f"Collective Resonance: {collective.get('resonance_level', 'N/A')}")
```

**Expected:** QEF node activates and can emit/receive emotional states (network features require additional infrastructure for full deployment)

### 4. Emotion Models

**Test All Emotion Model Types:**

```python
# Run emotion models demo
cd ml_framework
source venv/bin/activate
python examples/emotion_models_demo.py
```

**Models Verified:**

- ✅ Classical emotion models
- ✅ Quantum emotion models
- ✅ Hybrid emotion models
- ✅ Voice synthesis models

### 5. Python Bridge Integration

**Test C++ to Python Integration:**

```python
import kelly_bridge

# Test emotion thesaurus
emotion = kelly_bridge.get_emotion_by_name("joy")
print(f"Emotion: {emotion}")

# Test MIDI generation
midi_data = kelly_bridge.generate_midi(
    valence=0.5,
    arousal=0.6,
    intensity=0.4
)
print(f"MIDI Generated: {len(midi_data)} bytes")
```

**Expected:** Bridge functions are accessible and return expected data types

### 6. Learning Systems

**Test Recursive Memory and Evolution:**

```python
from cif_las_qef import UnifiedFramework, FrameworkConfig

config = FrameworkConfig(
    enable_las=True,
    enable_ethics=True
)
framework = UnifiedFramework(config)

# Create with feedback loop
result1 = framework.create_with_consent(
    human_emotional_input={"biofeedback": {"heart_rate": 70}},
    creative_goal={"style": "ambient"}
)

# Provide feedback
framework.evolve_from_feedback({
    "aesthetic_rating": 0.8,
    "emotional_resonance": 0.7
})

# Create again (should show learning)
result2 = framework.create_with_consent(
    human_emotional_input={"biofeedback": {"heart_rate": 70}},
    creative_goal={"style": "ambient"}
)

# Compare results (should show evolution)
print(f"First creation: {result1.get('las_output', {})}")
print(f"Second creation: {result2.get('las_output', {})}")
```

**Expected:** Second creation shows influence from previous feedback (recursive memory)

---

## Integration Testing

### Full Integration Test

Test complete pipeline: Python ML Framework → Python Bridge → C++ Plugin

```bash
# 1. Start with ML framework
cd ml_framework
source venv/bin/activate
python examples/basic_usage.py
deactivate
cd ..

# 2. Test Python bridge
cd python
source venv/bin/activate
python examples/basic_usage.py
deactivate
cd ..

# 3. Test C++ plugin (load in DAW or standalone)
# Open build/KellyMidiCompanion_artefacts/Release/Standalone/Kelly MIDI Companion.app
```

### Bridge Integration Test

```python
# python/examples/advanced_features.py
import kelly_bridge
from cif_las_qef import UnifiedFramework

# Use ML framework to generate emotion
framework = UnifiedFramework()
emotion_result = framework.create_with_consent(...)

# Pass to C++ bridge
midi_output = kelly_bridge.generate_midi(
    valence=emotion_result['esv']['valence'],
    arousal=emotion_result['esv']['arousal'],
    intensity=emotion_result['esv']['intensity']
)
```

---

## Troubleshooting

### Python Environment Issues

**Problem:** `ModuleNotFoundError: No module named 'cif_las_qef'`

**Solution:**

```bash
cd ml_framework
source venv/bin/activate
pip install -e .  # Install in development mode
# Or ensure PYTHONPATH includes ml_framework/
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Problem:** Virtual environment not activating

**Solution:**

```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# Verify
which python  # Should point to venv/bin/python
```

### CMake Build Issues

**Problem:** `BUILD_PYTHON_BRIDGE` not found

**Solution:**

```bash
# Ensure CMake version is 3.22+
cmake --version

# Clean and reconfigure
rm -rf build
cmake -B build -DBUILD_PYTHON_BRIDGE=ON
```

**Problem:** Python not found by CMake

**Solution:**

```bash
# Specify Python executable
cmake -B build \
    -DBUILD_PYTHON_BRIDGE=ON \
    -DPython3_EXECUTABLE=$(which python3)
```

**Problem:** pybind11 download fails

**Solution:**

```bash
# Manual pybind11 installation
git clone https://github.com/pybind/pybind11.git
cmake -B build \
    -DBUILD_PYTHON_BRIDGE=ON \
    -Dpybind11_DIR=$(pwd)/pybind11
```

**Problem:** Path with spaces causes FetchContent failures (JUCE/pybind11 download errors)

**Solution (FIXED):**

The CMakeLists.txt has been updated to automatically use local dependencies if available in the `external/` directory. This resolves path-with-spaces issues.

To use local dependencies (recommended for paths with spaces):

```bash
# Manually clone dependencies once
mkdir -p external
cd external
git clone --depth 1 --branch 8.0.4 https://github.com/juce-framework/JUCE.git
git clone --depth 1 --branch v2.11.1 https://github.com/pybind/pybind11.git
cd ..

# Build normally - CMake will detect and use local dependencies
cmake -B build -DBUILD_PYTHON_BRIDGE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

If local dependencies aren't available, CMake will automatically fetch them (may fail on paths with spaces).

**Alternative workaround:** Use a symlink without spaces:

```bash
ln -s "/Users/seanburdges/Desktop/final kel" ~/kelly-build
cd ~/kelly-build
cmake -B build -DBUILD_PYTHON_BRIDGE=ON
```

**Problem:** RTNeural tag v1.2.0 not found

**Solution:**

RTNeural is an optional dependency. Disable it if the tag doesn't exist:

```bash
cmake -B build -DBUILD_PYTHON_BRIDGE=ON -DENABLE_RTNEURAL=OFF
```

Or use a local copy:

```bash
mkdir -p external
cd external
git clone https://github.com/jatinchowdhury18/RTNeural.git
cd RTNeural
git checkout <valid-tag>  # Check available tags with: git tag
cd ../..
cmake -B build -DBUILD_PYTHON_BRIDGE=ON
```

**Problem:** ML framework examples fail with `ModuleNotFoundError: No module named 'cif_las_qef'`

**Solution:**

Set PYTHONPATH to include the ml_framework directory:

```bash
cd ml_framework
source venv/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"
python examples/basic_usage.py
```

Or install the package in development mode:

```bash
cd ml_framework
source venv/bin/activate
pip install -e .
python examples/basic_usage.py
```

**Problem:** Compilation errors - duplicate function definitions

**Solution:**

If you encounter errors like:

- `redefinition of 'enableMLInference'`
- `redefinition of 'extractFeatures'`
- `redefinition of 'applyEmotionVector'`

These indicate duplicate function definitions in `src/plugin/PluginProcessor.cpp`. Remove the duplicate definitions or consolidate them into a single implementation.

**Problem:** Compilation errors - missing struct members

**Solution:**

If you encounter errors about missing members in `Chord` or `MidiNote` structures, ensure the header files match the implementation. Check:

- `src/engine/MidiGenerator.h` - struct definitions
- `src/midi/MidiBuilder.h` - struct definitions
- Ensure all struct members are properly defined

### Python Bridge Import Issues

**Problem:** `ImportError: cannot import name 'kelly_bridge'`

**Solution:**

```bash
# Check module exists
ls -lh python/kelly_bridge.*

# Check Python path
cd python
source venv/bin/activate
python -c "import sys; print(sys.path)"
python -c "import sys; sys.path.insert(0, '.'); import kelly_bridge"
```

**Problem:** `Symbol not found` or `Undefined symbol` errors

**Solution:**

```bash
# Rebuild bridge
rm -rf build
cmake -B build -DBUILD_PYTHON_BRIDGE=ON
cmake --build build --target kelly_bridge

# Verify dependencies
otool -L python/kelly_bridge.so  # macOS
ldd python/kelly_bridge.so       # Linux
```

### Test Failures

**Problem:** Tests fail with linking errors

**Solution:**

```bash
# Rebuild tests
cmake --build build --target KellyTests

# Run with verbose output
cd build
ctest --output-on-failure --verbose
```

**Problem:** Python tests fail

**Solution:**

```bash
# Ensure virtual environment is activated
cd ml_framework
source venv/bin/activate
pytest tests/  # If pytest tests exist
```

### Plugin Loading Issues

**Problem:** Plugin doesn't appear in DAW

**Solution (macOS):**

```bash
# Check plugin location
ls -lh ~/Library/Audio/Plug-Ins/VST3/Kelly\ MIDI\ Companion.vst3

# Remove quarantine (if needed)
xattr -d com.apple.quarantine ~/Library/Audio/Plug-Ins/VST3/Kelly\ MIDI\ Companion.vst3

# Rebuild and reinstall
./build_and_install.sh Release
```

**Problem:** Plugin crashes on load

**Solution:**

```bash
# Build debug version
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Run standalone with debugger
lldb build/KellyMidiCompanion_artefacts/Debug/Standalone/Kelly\ MIDI\ Companion.app
```

---

## Advanced Configuration

### Custom Python Installation

If using a custom Python installation:

```bash
cmake -B build \
    -DBUILD_PYTHON_BRIDGE=ON \
    -DPython3_EXECUTABLE=/path/to/custom/python3 \
    -DPython3_INCLUDE_DIR=/path/to/custom/python3/include \
    -DPython3_LIBRARY=/path/to/custom/python3/lib/libpython3.11.dylib
```

### Build Only Specific Targets

```bash
# Build only plugin (no bridge, no tests)
cmake --build build --target KellyMidiCompanion

# Build only bridge
cmake --build build --target kelly_bridge

# Build only tests
cmake --build build --target KellyTests
```

### Debug Build with Symbols

```bash
# Configure debug build
cmake -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_PYTHON_BRIDGE=ON

# Build with debug symbols
cmake --build build --config Debug

# Run with debugger
lldb build/KellyMidiCompanion_artefacts/Debug/Standalone/Kelly\ MIDI\ Companion.app
```

### Cross-Platform Build

**macOS Universal Binary:**

```bash
cmake -B build \
    -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
    -DBUILD_PYTHON_BRIDGE=ON
```

**Linux Static Linking:**

```bash
cmake -B build \
    -DBUILD_PYTHON_BRIDGE=ON \
    -DCMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++"
```

---

## Quick Reference

### Complete Build Commands

```bash
# 1. Setup Python environments
./setup_workspace.sh

# 2. Build plugin with Python bridge (disable RTNeural if needed)
cmake -B build \
    -DBUILD_PYTHON_BRIDGE=ON \
    -DBUILD_TESTS=ON \
    -DENABLE_RTNEURAL=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# 3. Verify ML framework
cd ml_framework
source venv/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"
python examples/basic_usage.py
python examples/emotion_models_demo.py
deactivate
cd ..

# 4. Verify Python bridge (after successful build)
cd python
source venv/bin/activate
python -c "import sys; sys.path.insert(0, '.'); import kelly_bridge; print('Bridge OK')"
deactivate
cd ..

# 5. Run tests (if build succeeded)
cd build && ctest --output-on-failure && cd ..
```

### Verified Working Components

✅ **ML Framework** - All components verified:

- CIF (Conscious Integration Framework) - ✓ Working
- LAS (Living Art Systems) - ✓ Working
- QEF (Quantum Emotional Field) - ✓ Working
- ResonantEthics - ✓ Working
- Emotion Models (Classical, Quantum, Hybrid, Voice) - ✓ Working
- Dependencies (NumPy, SciPy, Matplotlib) - ✓ Installed

✅ **Python Environment Setup** - ✓ Complete

- ML framework virtual environment - ✓ Created
- Python utilities virtual environment - ✓ Created
- All dependencies installed - ✓ Complete

✅ **Python Bridge** - Successfully built and verified:

- CMake configuration - ✓ Successful
- Bridge compilation - ✓ Successful (December 2024)
- Bridge module created - ✓ `python/kelly_bridge.cpython-314-darwin.so`
- Module import - ✓ Working
- Basic functionality - ✓ Verified (IntentPipeline, EmotionThesaurus, enums, utilities)
- Note: KellyBrain bindings temporarily disabled due to MidiGenerator.h conflicts

⚠️ **C++ Build** - Requires code fixes:

- CMake configuration - ✓ Successful
- Plugin compilation - ⚠️ Has duplicate function definitions
- Test compilation - ⚠️ Has duplicate test definitions

### File Locations

| Component | Location |
|-----------|----------|
| ML Framework | `ml_framework/` |
| Python Utilities | `python/` |
| C++ Source | `src/` |
| Tests | `tests/` |
| Build Output | `build/` |
| Plugin (VST3) | `build/KellyMidiCompanion_artefacts/Release/VST3/` |
| Plugin (AU) | `build/KellyMidiCompanion_artefacts/Release/AU/` |
| Bridge Module | `python/kelly_bridge.so` (or `.pyd` on Windows) |

### Key Environment Variables

```bash
# Python path for ML framework (required for examples)
export PYTHONPATH="${PYTHONPATH}:$(pwd)/ml_framework"

# Python path for bridge
export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"

# CMake build type
export CMAKE_BUILD_TYPE=Release
```

### Build Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Python Environments | ✅ Complete | Both venvs created and dependencies installed |
| ML Framework | ✅ Verified | All 6/6 verification tests passing |
| ML Framework Examples | ✅ Working | Requires PYTHONPATH or `pip install -e .` |
| CMake Configuration | ✅ Success | Python bridge detected, pybind11 found |
| Python Bridge Build | ✅ Success | Built successfully (December 2024) |
| Python Bridge Functionality | ✅ Verified | IntentPipeline, EmotionThesaurus, enums working |
| C++ Plugin Build | ⚠️ Errors | Duplicate function definitions need fixing |
| Test Suite Build | ⚠️ Errors | Duplicate test definitions need fixing |

**Python Bridge Status (December 2024):**

✅ **Built and Functional:**

- Module compiled successfully: `python/kelly_bridge.cpython-314-darwin.so`
- All core APIs working: IntentPipeline, EmotionThesaurus, enums, utility functions
- Basic functionality verified with test suite
- Access via: `python/test_bridge_basic.py`

**Available APIs:**

- `IntentPipeline` - Process wounds to musical parameters
- `EmotionThesaurus` - Query 216-node emotion system (accessed via `pipeline.thesaurus()`)
- `Wound`, `EmotionNode`, `IntentResult`, `RuleBreak` - Core data types
- `EmotionCategory`, `RuleBreakType` - Enum types
- Utility functions: `midi_note_to_name()`, `note_name_to_midi()`, `category_to_string()`

**Note:** `KellyBrain` bindings are temporarily disabled due to MidiGenerator.h type conflicts. Use `IntentPipeline` directly instead.

**Next Steps for Full Build:**

1. Fix duplicate function definitions in `PluginProcessor.cpp`
2. Fix duplicate test definitions in `test_ui_processor_integration.cpp`
3. Rebuild and verify all components
4. Consider re-enabling `KellyBrain` bindings once MidiGenerator.h conflicts are resolved

---

## Next Steps

After successful build:

1. **Load Plugin in DAW**: Test VST3/AU plugin in your DAW
2. **Run Examples**: Explore `ml_framework/examples/` and `python/examples/`
3. **Read Documentation**: See `MARKDOWN/` for detailed guides
4. **Experiment**: Try different emotion inputs and observe outputs
5. **Contribute**: Expand tests, improve documentation, add features

---

## Support

For issues or questions:

1. Check `MARKDOWN/` documentation
2. Review `tests/` for usage examples
3. Check `ml_framework/README.md` for ML framework details
4. Review CMake output for build errors

---

**Build Status Checklist:**

- [x] Python environments created and dependencies installed
- [x] CMake configured with `BUILD_PYTHON_BRIDGE=ON`
- [x] ML framework verified (6/6 tests passing)
- [x] ML framework examples run successfully (with PYTHONPATH)
- [x] Python bridge module built successfully (December 2024)
- [x] Python bridge module importable and functional
- [x] Python bridge basic functionality tested and verified
- [ ] C++ plugin built successfully (has compilation errors to fix)
- [ ] Test suite passes (has compilation errors to fix)
- [ ] Plugin loads in DAW or standalone app (requires successful build)

### Verification Results (December 2024)

**ML Framework Verification:**

```
✓ PASS: Core Components (CIF, LAS, QEF, ResonantEthics)
✓ PASS: Emotion Models (VADModel, PlutchikWheel, QuantumEmotionalField)
✓ PASS: CIF Functionality
✓ PASS: LAS Functionality
✓ PASS: QEF Functionality
✓ PASS: Dependencies (NumPy 2.3.5, SciPy 1.16.3, Matplotlib 3.10.8)

Results: 6/6 tests passed
🎉 All AI/ML features verified successfully!
```

**ML Framework Example Output:**

```
Creating with ethical consent protocol...
=== Creation Result ===
Created: True
Consent Granted: True
Overall Ethics Score: 0.709...
LAS Output: audio
=== Providing Feedback ===
Evolution Result: {'las_evolution': {'evolution_count': 1, ...}}
=== Collective Resonance ===
Resonance Level: 0.01
Active Nodes: 1
```

**CMake Configuration:**

```
✓ Python-C++ bridge enabled
✓ Python executable: /Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14
✓ Python version: 3.14.2
✓ Module will be built to: /Users/seanburdges/Desktop/final kel/python
✓ Unit tests enabled
```

**Python Bridge Build Status (December 2024):**

✅ **Successfully Built:**

- Module compiled: `python/kelly_bridge.cpython-314-darwin.so` (422KB)
- All core APIs functional: IntentPipeline, EmotionThesaurus, enums, utilities
- Basic functionality verified with test suite (all tests passing)

**Known Issues to Fix (Other Components):**

1. Duplicate function definitions in `PluginProcessor.cpp`:
   - `enableMLInference()` defined twice (lines 593 and 738)
   - `extractFeatures()` defined twice (lines 624 and 782)
   - `applyEmotionVector()` defined twice (lines 631 and 809)

2. Duplicate test definitions:
   - `TEST_F(UIProcessorIntegrationTest, ParameterSynchronization)` defined twice in `test_ui_processor_integration.cpp`

---

*Built with love, grief, and JUCE.*
*AI/ML features powered by CIF/LAS/QEF framework.*
