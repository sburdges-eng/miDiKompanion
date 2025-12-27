# Build Status

This document describes the current build status and requirements for the miDiKompanion project.

## ✅ Working Builds

### Python Package (Kelly)

The Python package builds and tests successfully.

**Requirements:**
- Python 3.11+
- pip

**Quick Start:**
```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/python -v

# Use the CLI
kelly --help
```

**Status:** ✅ **All 17 tests passing** (70% coverage)

### Web Application (iDAW)

The TypeScript/React web application builds successfully.

**Requirements:**
- Node.js 18+
- npm

**Quick Start:**
```bash
# Install dependencies
npm install

# Build production bundle
npm run build

# Run development server
npm run dev
```

**Status:** ✅ **Build completes successfully**

### MCP Workstation

The MCP (Model Context Protocol) workstation package for multi-AI orchestration.

**Requirements:**
- Python 3.11+
- Dependencies installed via pip

**Quick Start:**
```bash
# Import and use
python -c "from mcp_workstation import Workstation; print('MCP Workstation ready!')"

# Run MCP server
python -m mcp_workstation.server
```

**Status:** ✅ **Imports work correctly**

## ⚠️ Optional/Advanced Builds

### C++ Components (Kelly Core)

The C++ components require additional system dependencies and are **optional** for most use cases.

**Requirements:**
- CMake 3.27+
- C++20 compatible compiler
- Qt6 (for GUI components)
- JUCE framework (auto-fetched by CMake)
- Catch2 (for tests, auto-fetched by CMake)

**Platform-Specific Dependencies:**

**macOS:**
```bash
brew install cmake qt@6
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build qt6-base-dev libasound2-dev
```

**Windows:**
```bash
choco install cmake ninja
# Qt6 must be installed separately
```

**Build Instructions:**
```bash
# Configure
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DBUILD_PLUGINS=OFF

# Build
cmake --build build --config Release

# Run tests
cd build && ctest --output-on-failure
```

**Status:** ⚠️ **Requires Qt6 installation** (not included in default setup)

### Audio Plugins (VST3/CLAP)

Audio plugin builds require JUCE and are **optional**.

**Requirements:**
- All C++ component requirements
- JUCE 7.0+ (auto-fetched by CMake)

**Build Instructions:**
```bash
# Configure with plugins enabled
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_PLUGINS=ON

# Build plugins
cmake --build build --config Release --target KellyPlugin

# Install (macOS example)
cp -r build/**/*.vst3 ~/Library/Audio/Plug-Ins/VST3/
cp -r build/**/*.clap ~/Library/Audio/Plug-Ins/CLAP/
```

**Status:** ⚠️ **Requires full C++ build environment**

## 🔧 Recent Fixes

### Fixed: Python Import Errors

**Issue:** Tests were failing with:
```
ImportError: cannot import name 'get_mcp_tools' from 'miDiKompanion.server'
```

**Root Cause:** Duplicate `__init__.py` and `__main__.py` files in the root directory were making Python treat the root as a package, causing import conflicts.

**Fix:** Removed duplicate files from root. The correct package structure is:
- `src/kelly/` - Kelly Python package
- `mcp_workstation/` - MCP Workstation package

## 📊 Build Matrix

| Component | Python 3.11 | Node 18+ | CMake/C++ | Status |
|-----------|-------------|----------|-----------|--------|
| Kelly (Python) | ✅ | - | - | Working |
| iDAW (Web) | - | ✅ | - | Working |
| MCP Workstation | ✅ | - | - | Working |
| C++ Core | - | - | ⚠️ | Requires Qt6 |
| Audio Plugins | - | - | ⚠️ | Requires Qt6 + JUCE |

## 🎯 Recommended Setup

For most users, the **Python and Web builds are sufficient**:

1. Install Python dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Install Node dependencies:
   ```bash
   npm install
   ```

3. Run tests:
   ```bash
   pytest tests/python -v
   npm run build
   ```

The C++ components are only needed for:
- Audio plugin development
- Advanced performance optimization
- Native GUI application

## 📝 CI/CD Status

GitHub Actions workflows test:
- ✅ Python package on Linux, macOS, Windows
- ✅ Web application build
- ⚠️ C++ components (optional, requires system dependencies)

See `.github/workflows/ci.yml` for details.

## 🆘 Troubleshooting

### "Cannot import kelly"

Make sure you've installed the package:
```bash
pip install -e ".[dev]"
```

### "Qt6 not found"

The C++ build requires Qt6. Install it:
- macOS: `brew install qt@6`
- Ubuntu: `sudo apt-get install qt6-base-dev`
- Windows: Download from https://www.qt.io/

Or skip the C++ build (it's optional).

### "npm install fails"

Make sure you have Node.js 18+ installed:
```bash
node --version  # Should be 18.x or higher
```

## 📚 Additional Documentation

- [BUILD.md](BUILD.md) - Detailed C++ build instructions
- [README.md](README.md) - Project overview
- [pyproject.toml](pyproject.toml) - Python package configuration
- [package.json](package.json) - Node.js package configuration

---

**Last Updated:** December 2025
**Build System Version:** Kelly 0.1.0
