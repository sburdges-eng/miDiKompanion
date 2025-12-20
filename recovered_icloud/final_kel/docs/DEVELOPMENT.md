# Development Guide - Kelly MIDI Companion

Guide for setting up and working with the Kelly MIDI Companion development environment.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Debugging](#debugging)
- [Common Tasks](#common-tasks)

## Development Setup

### Initial Setup

1. **Clone and setup**:

   ```bash
   git clone <repository-url>
   cd "final kel"
   ./scripts/setup_environment.sh
   ```

2. **Activate environment**:

   ```bash
   source venv/bin/activate  # Main project
   # or
   cd ml_framework && source venv/bin/activate  # ML framework
   ```

3. **Install development dependencies**:

   ```bash
   pip install -e ".[dev]"
   ```

### IDE Setup

#### Visual Studio Code

Recommended extensions:

- C/C++ (Microsoft)
- CMake Tools
- Python
- clang-format

Create `.vscode/settings.json`:

```json
{
  "cmake.buildDirectory": "${workspaceFolder}/build",
  "cmake.configureSettings": {
    "CMAKE_BUILD_TYPE": "Debug"
  },
  "C_Cpp.default.cppStandard": "c++20",
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python"
}
```

#### CLion

1. Open project root directory
2. CLion will detect CMakeLists.txt automatically
3. Configure CMake: File → Settings → Build, Execution, Deployment → CMake
4. Set build directory: `build`
5. Set CMake options: `-DCMAKE_BUILD_TYPE=Debug`

#### Xcode (macOS)

```bash
cmake -B build -G Xcode -DCMAKE_BUILD_TYPE=Debug
open build/KellyMidiCompanion.xcodeproj
```

## Project Structure

```
final kel/
├── src/                    # C++ source code
│   ├── plugin/            # JUCE plugin (AudioProcessor/Editor)
│   ├── engine/            # Emotion processing engines
│   ├── engines/           # Music generation engines
│   ├── midi/              # MIDI generation
│   ├── ui/                 # Plugin UI components
│   ├── ml/                 # ML inference
│   └── voice/              # Voice synthesis
│
├── src_penta-core/         # Penta-Core C++ library
│   ├── common/            # RT-safe utilities
│   ├── harmony/            # Harmony analysis
│   ├── groove/             # Groove analysis
│   └── osc/                # OSC communication
│
├── include/                # C++ headers
│   └── penta/             # Penta-Core headers
│
├── python/                 # Python utilities and bindings
├── ml_framework/          # ML framework (CIF/LAS/QEF)
├── ml_training/            # ML model training
├── tests/                  # C++ unit tests
├── tests_music-brain/      # Python integration tests
│
├── CMakeLists.txt         # Main build configuration
├── pyproject.toml         # Python package configuration
└── build_all.sh           # Unified build script
```

## Development Workflow

### Making Changes

1. **Create a branch**:

   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes** and test locally:

   ```bash
   # Build
   ./build_all.sh --debug

   # Run tests
   ./build_all.sh --test
   ```

3. **Commit changes**:

   ```bash
   git add .
   git commit -m "Description of changes"
   ```

4. **Push and create PR**:

   ```bash
   git push origin feature/my-feature
   ```

### Build Workflow

```bash
# 1. Check dependencies
./scripts/check_dependencies.sh

# 2. Setup environment (if needed)
./scripts/setup_environment.sh

# 3. Build in debug mode
./build_all.sh --debug

# 4. Test changes
./build_all.sh --test

# 5. Build release for testing
./build_all.sh --release
```

### Incremental Builds

After initial build, only rebuild what changed:

```bash
# Rebuild only changed files
cmake --build build --config Debug

# Rebuild specific target
cmake --build build --target KellyMidiCompanion

# Clean and rebuild
./build_all.sh --clean --debug
```

## Code Style

### C++ Style

- **Standard**: C++20
- **Naming**:
  - Classes: `PascalCase`
  - Functions: `camelCase`
  - Variables: `camelCase`
  - Constants: `UPPER_SNAKE_CASE`
- **Formatting**: Use clang-format (see `.clang-format`)
- **Line length**: 100 characters

Format code:

```bash
clang-format -i src/**/*.{cpp,h}
```

### Python Style

- **Standard**: PEP 8
- **Line length**: 100 characters
- **Formatter**: black
- **Linter**: ruff
- **Type hints**: Required for public APIs

Format code:

```bash
black .
ruff check --fix .
```

### CMake Style

- Use `cmake-format` for consistent formatting
- Follow CMake best practices
- Document all options

Format CMake:

```bash
cmake-format -i CMakeLists.txt **/CMakeLists.txt
```

## Testing

### Running C++ Tests

```bash
# Build tests
cmake -B build -DBUILD_TESTS=ON
cmake --build build --target KellyTests

# Run all tests
cd build && ctest --output-on-failure

# Run specific test
./build/KellyTests --gtest_filter="TestName*"
```

### Running Python Tests

```bash
# Activate environment
source venv/bin/activate

# Run all tests
pytest tests_music-brain/ -v

# Run with coverage
pytest tests_music-brain/ -v --cov=music_brain --cov-report=html

# Run specific test
pytest tests_music-brain/test_specific.py -v
```

### Writing Tests

#### C++ Tests (Google Test)

```cpp
#include <gtest/gtest.h>
#include "MyClass.h"

TEST(MyClassTest, BasicFunctionality) {
    MyClass obj;
    EXPECT_EQ(obj.getValue(), 42);
}
```

#### Python Tests (pytest)

```python
import pytest
from music_brain import MyModule

def test_basic_functionality():
    obj = MyModule()
    assert obj.get_value() == 42
```

## Debugging

### Debugging C++ Code

#### Using GDB/LLDB

```bash
# Build debug
./build_all.sh --debug

# Debug standalone app
lldb ./build/KellyMidiCompanion_artefacts/Debug/Standalone/Kelly\ MIDI\ Companion

# Or debug in IDE (VS Code, CLion, Xcode)
```

#### Debugging in DAW

1. Build debug version
2. Load plugin in DAW
3. Attach debugger to DAW process:

   ```bash
   lldb -p $(pgrep "Logic Pro X")
   # or
   gdb -p $(pgrep "reaper")
   ```

### Debugging Python Code

```bash
# Use pdb
python -m pdb script.py

# Use IDE debugger (VS Code, PyCharm)
# Set breakpoints and run in debug mode
```

### Common Debugging Scenarios

**Plugin crashes on load**:

- Check JUCE initialization
- Verify all dependencies linked
- Check for memory issues (use AddressSanitizer)

**Audio glitches**:

- Check real-time safety (no allocations in audio thread)
- Verify buffer sizes
- Profile with Instruments (macOS) or perf (Linux)

**Python bridge fails**:

- Check Python version matches
- Verify module is built
- Check import paths

## Common Tasks

### Adding a New Engine

1. Create header: `src/engines/MyEngine.h`
2. Create implementation: `src/engines/MyEngine.cpp`
3. Add to CMakeLists.txt:

   ```cmake
   target_sources(KellyMidiCompanion PRIVATE
       src/engines/MyEngine.cpp
   )
   ```

4. Add tests: `tests/engines/test_my_engine.cpp`

### Adding a Python Module

1. Create module: `python/my_module.py`
2. Add to package: Update `pyproject.toml` or `setup.py`
3. Add tests: `tests_music-brain/test_my_module.py`
4. Install: `pip install -e .`

### Updating Dependencies

1. Update version in `requirements.txt` or `pyproject.toml`
2. Update CMakeLists.txt for C++ dependencies
3. Test build: `./build_all.sh --clean`
4. Update documentation if needed

### Profiling

#### C++ Profiling

```bash
# macOS - Instruments
instruments -t "Time Profiler" ./build/Standalone/Kelly\ MIDI\ Companion

# Linux - perf
perf record ./build/Standalone/Kelly\ MIDI\ Companion
perf report

# Valgrind (memory)
valgrind --leak-check=full ./build/KellyTests
```

#### Python Profiling

```bash
python -m cProfile -o profile.stats script.py
python -m pstats profile.stats
```

## Best Practices

1. **Always test locally** before pushing
2. **Run tests** before committing: `./build_all.sh --test`
3. **Format code** before committing
4. **Write tests** for new features
5. **Update documentation** when adding features
6. **Keep commits focused** - one feature per commit
7. **Write clear commit messages**

## Getting Help

- Check `BUILD.md` for build issues
- Check `CONTRIBUTING.md` for contribution guidelines
- Review existing code for patterns
- Ask in project discussions/issues

## Next Steps

- Read [BUILD.md](BUILD.md) for build instructions
- Read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
- Explore the codebase to understand architecture
- Start with small changes to get familiar
