# Building the Python-C++ Bridge

Quick guide for building and using the Python-C++ bridge.

## Quick Start

```bash
# 1. Configure with Python bridge enabled
cmake -B build -DBUILD_PYTHON_BRIDGE=ON

# 2. Build
cmake --build build

# 3. Install Python package (optional)
cd python
pip install -e .
```

## Prerequisites

### macOS

```bash
# Install Python development headers
brew install python3

# Verify Python version
python3 --version  # Should be 3.8+
```

### Linux (Ubuntu/Debian)

```bash
# Install Python development headers
sudo apt-get update
sudo apt-get install python3-dev python3-pip

# Verify
python3 --version
```

### Windows

1. Install Python from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Check "Install for all users" and "Install development headers"

## Build Options

### Debug Build

```bash
cmake -B build -DBUILD_PYTHON_BRIDGE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

### Release Build (Recommended)

```bash
cmake -B build -DBUILD_PYTHON_BRIDGE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Specify Python Version

```bash
# Use specific Python executable
cmake -B build -DBUILD_PYTHON_BRIDGE=ON \
    -DPython3_EXECUTABLE=/usr/bin/python3.10
```

## Verification

After building, verify the module exists:

```bash
# macOS/Linux
ls python/kelly_bridge*.so

# Windows
ls python/kelly_bridge*.pyd
```

Test import:

```bash
cd python
python3 -c "import kelly_bridge; print('Success!')"
```

## Running Examples

```bash
cd python/examples
python3 basic_usage.py
python3 advanced_features.py
```

## Troubleshooting

### "Python not found"

**Solution**: Install Python development headers (see Prerequisites)

### "pybind11 not found"

**Solution**: CMake should fetch automatically. If not:
```bash
pip install pybind11
```

### "ImportError: No module named 'kelly_bridge'"

**Solution**:
1. Check module exists: `ls python/kelly_bridge*`
2. Add to PYTHONPATH: `export PYTHONPATH=$PWD/python:$PYTHONPATH`
3. Or install package: `cd python && pip install -e .`

### Build errors with JUCE

**Note**: The Python bridge doesn't require JUCE. If you see JUCE-related errors, they shouldn't affect the bridge build.

## Next Steps

- See `python/README.md` for Python API documentation
- See `python/examples/` for usage examples
- See `PYTHON_BRIDGE.md` for architecture details
