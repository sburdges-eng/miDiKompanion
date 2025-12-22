# Multi-Component Build Guide

> **Comprehensive documentation for building all components of the miDiKompanion project**

This repository contains three major components that can be built independently or together:

1. **Git Multi-Repository Updater** - Modular build system for Git batch operations
2. **Music Brain (DAiW/iDAW)** - Python toolkit for music analysis and composition
3. **Penta Core** - C++/Python hybrid real-time music analysis engine

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start: Build Everything](#quick-start-build-everything)
- [Component 1: Git Multi-Repository Updater](#component-1-git-multi-repository-updater)
- [Component 2: Music Brain](#component-2-music-brain)
- [Component 3: Penta Core](#component-3-penta-core)
- [Build Order and Dependencies](#build-order-and-dependencies)
- [Platform-Specific Notes](#platform-specific-notes)
- [Troubleshooting](#troubleshooting)
- [Testing](#testing)

---

## Prerequisites

### All Components

- **Git** 2.30 or higher
- **Bash** (for build scripts)
- **Make** (optional, for Makefile-based builds)

### Music Brain (Python Component)

- **Python** 3.9 or higher
- **pip** (Python package installer)
- **virtualenv** (recommended)

#### Python Dependencies
```bash
# Core dependencies
mido>=1.2.10
numpy>=1.21.0

# Optional: Audio analysis
librosa>=0.9.0
soundfile>=0.10.0

# Optional: Music theory
music21>=7.0.0

# Development tools
pytest>=7.0.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.900
```

### Penta Core (C++/Python Hybrid)

- **CMake** 3.20 or higher
- **C++20 compatible compiler**
  - GCC 10+ (Linux)
  - Clang 12+ (macOS)
  - MSVC 2019+ (Windows)
- **Python** 3.8 or higher with development headers
- **pybind11** 2.10.0 or higher

#### Optional Dependencies
- **JUCE** (for plugin builds, auto-fetched by CMake)
- **libasound2-dev** (Linux only, for audio I/O)
- **libfreetype6-dev, libx11-dev** (Linux only, for GUI)

### Platform-Specific Requirements

#### macOS
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake python@3.11 make
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  python3-dev \
  python3-pip \
  python3-venv \
  git \
  libasound2-dev \
  libfreetype6-dev \
  libx11-dev \
  libxrandr-dev \
  libxinerama-dev \
  libxcursor-dev
```

#### Windows
1. Install **Visual Studio 2019 or later** with C++ tools
2. Install **CMake** from https://cmake.org/download/
3. Install **Python** from https://python.org/downloads/
4. Install **Git for Windows** from https://git-scm.com/download/win

---

## Quick Start: Build Everything

### Option 1: Automated Build (Recommended)

```bash
# Clone the repository
git clone https://github.com/sburdges-eng/miDiKompanion.git
cd miDiKompanion

# Build all components in order
./build_all.sh

# Or build specific components
./build_all.sh --git-updater
./build_all.sh --music-brain
./build_all.sh --penta-core
```

### Option 2: Manual Build (Step-by-Step)

```bash
# 1. Build Git Multi-Repository Updater
make standard

# 2. Build Music Brain (Python)
pip install -e .

# 3. Build Penta Core (C++/Python)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DPENTA_BUILD_PYTHON_BINDINGS=ON \
         -DPENTA_BUILD_JUCE_PLUGIN=ON
cmake --build . -j$(nproc)
cd ..
```

---

## Component 1: Git Multi-Repository Updater

A modular build system for creating customized Git repository batch update scripts.

### Build Profiles

| Profile | Description | Modules Included |
|---------|-------------|------------------|
| **minimal** | Core functionality only | None |
| **standard** | Recommended for most users | colors, config |
| **full** | All features | colors, config, verbose, summary |
| **custom** | User-defined | Editable in `profiles/custom.profile` |

### Quick Build

```bash
# Build standard version (recommended)
make

# Build specific profile
make minimal
make standard
make full
make custom

# Install to ~/bin
make install

# Install to /usr/local/bin (requires sudo)
make install-system
```

### Manual Build

```bash
# Using the build script directly
./build.sh profiles/standard.profile

# Output will be in dist/git-update.sh
```

### Usage

```bash
# Run the generated script
cd /path/to/your/repos
./dist/git-update.sh

# Or if installed
git-update
```

### Configuration

Create a `.git-update-config` file in your repositories directory:

```bash
BRANCHES=("main" "dev" "staging")
ROOT="/path/to/repos"
EXCLUDE=("old-repo" "archived-project")
```

### File Structure
```
.
├── Makefile           # Build system
├── build.sh           # Build script
├── modules/           # Feature modules
│   ├── colors.sh      # Color output
│   ├── config.sh      # Config file support
│   ├── verbose.sh     # Verbose mode
│   └── summary.sh     # Summary reports
├── core/              # Core components
│   ├── header.sh      # Script header
│   ├── main-loop.sh   # Main update loop
│   └── footer.sh      # Script footer
├── profiles/          # Build profiles
│   ├── minimal.profile
│   ├── standard.profile
│   ├── full.profile
│   └── custom.profile
└── dist/              # Generated scripts
```

---

## Component 2: Music Brain

Python toolkit for music production intelligence: groove extraction, chord analysis, and AI-assisted songwriting.

### Philosophy

**"Interrogate Before Generate"** — The tool shouldn't finish art for people. It should make them braver.

### Quick Build

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install as editable package
pip install -e .

# With all optional dependencies
pip install -e ".[audio,theory,dev]"
```

### Alternative: System-wide Installation

```bash
# Install globally (not recommended for development)
pip install .
```

### Verify Installation

```bash
# Check CLI is available
daiw --help

# Run basic test
python -c "from music_brain.groove import extract_groove; print('Success!')"
```

### Quick Test

```bash
# Extract groove from MIDI
daiw extract examples/midi/drums.mid

# Analyze chord progression
daiw diagnose "F-C-Am-Dm"

# Create song intent template
daiw intent new --title "My Song"
```

### Key Features

- **Groove Engine**: Extract and apply timing feel
- **Harmony Analysis**: Chord detection and progression analysis
- **Intent Schema**: Three-phase emotional interrogation
- **Rule-Breaking Engine**: Intentional theory violations
- **CLI Tools**: Command-line interface for all features

### File Structure
```
music_brain/
├── __init__.py
├── cli.py                 # CLI entry point
├── data/                  # JSON/YAML data files
│   ├── chord_progressions.json
│   ├── emotional_mapping.py
│   └── song_intent_schema.yaml
├── groove/                # Groove extraction
│   ├── extractor.py
│   ├── applicator.py
│   └── templates.py
├── structure/             # Harmonic analysis
│   ├── chord.py
│   ├── progression.py
│   └── sections.py
├── session/               # Intent schema
│   ├── intent_schema.py
│   ├── intent_processor.py
│   ├── teaching.py
│   └── interrogator.py
└── audio/                 # Audio analysis
    └── feel.py
```

---

## Component 3: Penta Core

Professional-grade music analysis engine with hybrid C++/Python architecture for real-time performance.

### Architecture

```
Python "Brain"  (Flexibility, AI, Experimentation)
      ↕ pybind11
C++ "Engine"    (Real-time performance, DSP, Analysis)
      ↕ JUCE
DAW Integration (VST3, AU, Standalone)
```

### Quick Build

```bash
# Create build directory
mkdir -p build && cd build

# Configure (Release build with all features)
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DPENTA_BUILD_PYTHON_BINDINGS=ON \
  -DPENTA_BUILD_JUCE_PLUGIN=ON \
  -DPENTA_ENABLE_SIMD=ON

# Build (use -j for parallel compilation)
cmake --build . --config Release -j8

# Run tests
ctest --output-on-failure

# Return to root
cd ..
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Debug | Build type: Debug, Release, RelWithDebInfo |
| `PENTA_BUILD_PYTHON_BINDINGS` | ON | Build pybind11 Python module |
| `PENTA_BUILD_JUCE_PLUGIN` | ON | Build JUCE VST3/AU plugins |
| `PENTA_BUILD_TESTS` | ON | Build unit tests |
| `PENTA_ENABLE_SIMD` | ON | Enable SIMD optimizations (AVX2) |
| `PENTA_ENABLE_LTO` | OFF | Enable link-time optimization |

### Build Configurations

#### Development Build
```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPENTA_BUILD_TESTS=ON \
  -DPENTA_ENABLE_SIMD=OFF
cmake --build build
```

#### Release Build (Maximum Performance)
```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPENTA_ENABLE_SIMD=ON \
  -DPENTA_ENABLE_LTO=ON \
  -DCMAKE_CXX_FLAGS="-march=native"
cmake --build build
```

#### Python-Only Build
```bash
cmake -B build \
  -DPENTA_BUILD_PYTHON_BINDINGS=ON \
  -DPENTA_BUILD_JUCE_PLUGIN=OFF \
  -DPENTA_BUILD_TESTS=OFF
cmake --build build
```

#### Plugin-Only Build
```bash
cmake -B build \
  -DPENTA_BUILD_PYTHON_BINDINGS=OFF \
  -DPENTA_BUILD_JUCE_PLUGIN=ON \
  -DPENTA_BUILD_TESTS=OFF
cmake --build build
```

### Install Python Module

```bash
# Install to user site-packages
cmake --install build --prefix ~/.local

# Or install in development mode
pip install -e .
```

### Install Plugins

#### macOS
```bash
# VST3
cp -r build/plugins/PentaCorePlugin_artefacts/Release/VST3/PentaCorePlugin.vst3 \
      ~/Library/Audio/Plug-Ins/VST3/

# AU
cp -r build/plugins/PentaCorePlugin_artefacts/Release/AU/PentaCorePlugin.component \
      ~/Library/Audio/Plug-Ins/Components/
```

#### Linux
```bash
# VST3
cp -r build/plugins/PentaCorePlugin_artefacts/Release/VST3/PentaCorePlugin.vst3 \
      ~/.vst3/
```

#### Windows
```powershell
# VST3
copy build\plugins\PentaCorePlugin_artefacts\Release\VST3\PentaCorePlugin.vst3 ^
     %CommonProgramFiles%\VST3\
```

### Verify Installation

```bash
# Test C++ library
./build/tests/penta_tests

# Test Python bindings
python -c "from penta_core import PentaCore; print('Success!')"

# Run examples
python examples/harmony_example.py
python examples/groove_example.py
```

### Key Features

- **Harmony Analysis**: Real-time chord detection, scale detection, voice leading
- **Groove Analysis**: Onset detection, tempo estimation, rhythm quantization
- **Performance Monitoring**: CPU usage, latency measurement, audio monitoring
- **OSC Communication**: Real-time safe messaging to DAWs

### File Structure
```
penta_core/
├── CMakeLists.txt
├── include/                # C++ headers
│   ├── HarmonyEngine.h
│   ├── GrooveEngine.h
│   └── PerformanceMonitor.h
├── src/                    # C++ implementation
│   ├── HarmonyEngine.cpp
│   ├── GrooveEngine.cpp
│   └── PerformanceMonitor.cpp
├── src_penta-core/         # Additional C++ code
├── python/                 # Python bindings
│   └── penta_core/
│       └── __init__.py
├── bindings/               # pybind11 bindings
│   └── penta_core_bindings.cpp
├── plugins/                # JUCE plugin code
│   ├── PluginProcessor.cpp
│   └── PluginEditor.cpp
├── tests/                  # C++ tests
├── tests_penta-core/       # Additional tests
└── examples/               # Usage examples
```

---

## Build Order and Dependencies

### Dependency Graph

```
Git Multi-Repository Updater (standalone)
│
├─ Music Brain (Python only)
│  │
│  └─ Uses: Python 3.9+, mido, numpy
│
└─ Penta Core (C++ + Python)
   │
   ├─ Depends on: CMake, C++20 compiler
   ├─ Optional: JUCE (auto-fetched)
   └─ Python bindings: pybind11
```

### Recommended Build Order

1. **Git Multi-Repository Updater** (independent, builds first)
2. **Music Brain** (Python-only, can build anytime)
3. **Penta Core** (C++ component, builds last due to compilation time)

### Building All Components

```bash
# Build in recommended order
make standard                    # Git Updater
pip install -e ".[audio,theory]" # Music Brain
cmake -B build && cmake --build build  # Penta Core
```

---

## Platform-Specific Notes

### macOS

#### Git Updater
- Built-in bash works fine
- Make available via Xcode Command Line Tools

#### Music Brain
- Use Homebrew Python: `brew install python@3.11`
- Avoid system Python for development

#### Penta Core
- Universal binary support:
  ```bash
  cmake -B build -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
  ```
- JUCE auto-fetched, no manual installation needed
- Audio plugins install to `~/Library/Audio/Plug-Ins/`

### Linux

#### Git Updater
- Bash usually pre-installed
- Make available via `build-essential` package

#### Music Brain
- Install Python dev headers: `python3-dev`
- Consider system packages for heavy dependencies:
  ```bash
  sudo apt install python3-numpy python3-scipy
  ```

#### Penta Core
- Install ALSA dev libraries: `libasound2-dev`
- Install X11 dependencies for GUI
- VST3 plugins install to `~/.vst3/`

### Windows

#### Git Updater
- Use Git Bash or WSL
- Windows batch scripts not provided (use WSL recommended)

#### Music Brain
- Use official Python installer from python.org
- Consider Windows Terminal for better CLI experience

#### Penta Core
- Requires Visual Studio 2019+ with C++ tools
- Use CMake GUI or command line with proper generator:
  ```powershell
  cmake -B build -G "Visual Studio 16 2019" -A x64
  cmake --build build --config Release
  ```
- VST3 plugins install to `C:\Program Files\Common Files\VST3\`

---

## Troubleshooting

### Git Multi-Repository Updater

#### Profile not found
```
Error: Profile not found: profiles/standard.profile
```
**Solution**: Ensure you're in the repository root directory.

#### Permission denied
```bash
# Make build script executable
chmod +x build.sh
```

### Music Brain

#### Module not found after installation
```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall in editable mode
pip install -e . -c pyproject_music-brain.toml
```

#### CLI command not found
```bash
# Check installation
pip show idaw

# Try running as module
python -m music_brain.cli --help
```

#### Import errors with audio dependencies
```bash
# Install audio extras
pip install ".[audio]" -c pyproject_music-brain.toml

# On some systems, you may need system packages first:
# Ubuntu/Debian:
sudo apt install libsndfile1 ffmpeg

# macOS:
brew install libsndfile ffmpeg
```

### Penta Core

#### CMake can't find Python
```bash
# Specify Python explicitly
cmake -B build -DPython3_EXECUTABLE=/usr/bin/python3.11
```

#### pybind11 not found
```bash
# Install via pip
pip install pybind11

# Or let CMake fetch it (default behavior)
cmake -B build  # Will auto-download pybind11
```

#### SIMD compilation errors
```bash
# Disable SIMD if your CPU doesn't support AVX2
cmake -B build -DPENTA_ENABLE_SIMD=OFF
```

#### JUCE build errors
```bash
# Clean and re-fetch JUCE
rm -rf build/_deps/juce-*
cmake -B build  # Will re-fetch JUCE
```

#### Link errors on Linux
```bash
# Install missing libraries
sudo apt install \
  libasound2-dev \
  libfreetype6-dev \
  libx11-dev \
  libxrandr-dev \
  libxinerama-dev \
  libxcursor-dev
```

#### Windows: Missing MSVC compiler
- Install Visual Studio 2019 or later
- During installation, select "Desktop development with C++"
- Restart terminal after installation

---

## Testing

### Git Multi-Repository Updater

```bash
# Build all profiles
make all-profiles

# Test the standard build
cd /tmp/test-repos
git init repo1 && cd repo1 && git commit --allow-empty -m "test" && cd ..
git init repo2 && cd repo2 && git commit --allow-empty -m "test" && cd ..
cd ..
/path/to/dist/git-update.sh
```

### Music Brain

```bash
# Run pytest suite
pytest tests_music-brain/

# Run specific test
pytest tests_music-brain/test_groove_extractor.py -v

# Run with coverage
pytest tests_music-brain/ --cov=music_brain
```

### Penta Core

```bash
# C++ tests
cd build
ctest --output-on-failure

# Or run test binary directly
./tests/penta_tests

# Python tests
cd ..
pytest tests_penta-core/ -v
```

### Integration Testing

```bash
# Test all components work together
# 1. Build git updater
make standard

# 2. Install Music Brain
pip install -e ".[audio,theory]" -c pyproject_music-brain.toml

# 3. Build Penta Core
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 4. Run integration test
python -c "
from music_brain.groove import extract_groove
from penta_core import PentaCore
print('All components working!')
"
```

---

## Additional Resources

### Documentation

- **Git Updater**: [README.md](README.md)
- **Music Brain**: [README_music-brain.md](README_music-brain.md)
- **Penta Core**: [README_penta-core.md](README_penta-core.md), [BUILD.md](BUILD.md)

### Build Scripts

- **Git Updater**: `build.sh`, `Makefile`
- **Music Brain**: `pyproject_music-brain.toml`
- **Penta Core**: `CMakeLists.txt`, `pyproject_penta-core.toml`

### Platform-Specific Guides

- **macOS**: [install_macos.sh](install_macos.sh)
- **Linux**: [install_linux.sh](install_linux.sh)
- **Windows**: [install_windows.ps1](install_windows.ps1)

### Quick References

- [QUICKSTART.md](QUICKSTART.md) - Penta Core quick start
- [QUICK_START.md](QUICK_START.md) - General quick start
- [BUILD.md](BUILD.md) - Detailed Penta Core build instructions

---

## Support

For build issues:

1. **Check this documentation first**
2. **Search existing GitHub issues**: https://github.com/sburdges-eng/miDiKompanion/issues
3. **Create a new issue** with:
   - Component name (Git Updater / Music Brain / Penta Core)
   - OS and version
   - Build command used
   - Full error output
   - Relevant system information (CMake/Python/compiler versions)

---

## License

All components are licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Version Information

- **Git Multi-Repository Updater**: 1.0.0
- **Music Brain (iDAW)**: 1.0.0
- **Penta Core**: 0.3.0

Last Updated: December 22, 2024
