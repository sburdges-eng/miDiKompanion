# Kelly - Therapeutic iDAW

**Kelly** is a therapeutic interactive Digital Audio Workstation (iDAW) that translates emotions into music.

## Overview

Kelly helps users express and process emotions through music generation, using a unique three-phase intent system:

1. **Wound** → Identify the emotional trigger
2. **Emotion** → Map to the 216-node emotion thesaurus
3. **Rule-breaks** → Express through intentional musical violations

## Architecture

### Brain (Python 3.11)
- **music21**: Music theory and analysis
- **librosa**: Audio analysis
- **mido**: MIDI processing
- **Typer**: Command-line interface

### Body (C++20)
- **JUCE 7**: Audio framework
- **Qt 6**: GUI framework
- **CMake 3.27+**: Build system

### Plugins
- **CLAP 1.2**: Cross-platform plugin format
- **VST3 3.7**: Steinberg plugin format

### Audio Support
- ASIO (Windows)
- CoreAudio (macOS)
- JACK (Linux)

### Testing
- **pytest**: Python tests
- **Catch2**: C++ tests
- **GoogleTest**: Additional C++ testing

### CI/CD
- GitHub Actions for continuous integration
- Multi-platform builds (Linux, macOS, Windows)

### Profiling
- Tracy
- Valgrind
- Perfetto

## Features

- **216-node Emotion Thesaurus**: Comprehensive emotional mapping system
- **Groove Templates**: Rhythmic patterns for different emotional expressions
- **Chord Diagnostics**: Analyze and generate emotionally-appropriate harmonies
- **MIDI Pipeline**: Real-time MIDI generation based on emotional state
- **3-Phase Intent Processing**: Wound → Emotion → Musical Rule-breaks

## Quick Start

### Python CLI

```bash
# Install Python dependencies
pip install -e ".[dev]"

# List available emotions
kelly list-emotions

# Process an emotional wound and generate music
kelly process "feeling of loss" --intensity 0.8 --output output.mid --tempo 90

# Show version
kelly version
```

### Building C++ Components

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PLUGINS=ON

# Build
cmake --build build --config Release

# Run tests
cd build && ctest --output-on-failure
```

## Project Structure

```
Kelly/
├── src/
│   ├── kelly/          # Python package
│   │   ├── core/       # Core emotion processing
│   │   ├── cli.py      # CLI interface
│   │   └── __init__.py
│   ├── core/           # C++ core library
│   ├── gui/            # Qt GUI application
│   └── plugin/         # JUCE audio plugins
├── tests/
│   ├── python/         # pytest tests
│   └── cpp/            # Catch2 tests
├── docs/               # Documentation
├── .github/
│   └── workflows/      # CI configuration
├── CMakeLists.txt      # C++ build configuration
├── pyproject.toml      # Python package configuration
└── README.md
```

## Development

### Python Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run linting
ruff check src/kelly tests/python

# Format code
black src/kelly tests/python

# Type checking
mypy src/kelly

# Run tests with coverage
pytest tests/python -v --cov=kelly
```

### C++ Development

```bash
# Configure with tests
cmake -B build -DBUILD_TESTS=ON -DENABLE_TRACY=ON

# Build
cmake --build build

# Run tests
cd build && ctest -V
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.