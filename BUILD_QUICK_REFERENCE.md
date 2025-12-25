# Quick Build Reference

> **One-page cheat sheet for building all components**

## 🚀 Build All at Once

```bash
./build_all.sh
```

## 📦 Individual Components

### Git Multi-Repository Updater
```bash
make                    # Standard build
make install            # Install to ~/bin
./dist/git-update.sh    # Run it
```

### Music Brain (Python/DAiW)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
kelly --help            # Test CLI
```

### Penta Core (C++/Python)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
cd build && ctest       # Run tests
```

## 🔧 Prerequisites Quick Check

```bash
# Check all prerequisites
git --version           # 2.30+
python3 --version       # 3.9+
cmake --version         # 3.20+
c++ --version          # C++20 support
```

## 📖 Full Documentation

- **[MULTI_BUILD.md](MULTI_BUILD.md)** - Complete build guide
- **[README.md](README.md)** - Project overview

## 🆘 Quick Troubleshooting

### Git Updater won't build
```bash
chmod +x build.sh
make clean && make
```

### Music Brain import error
```bash
pip install -e .
python3 -c "import music_brain"
```

### Penta Core CMake fails
```bash
rm -rf build
cmake -B build -DPENTA_ENABLE_SIMD=OFF
cmake --build build
```

## 🎯 Common Tasks

| Task | Command |
|------|---------|
| Build everything | `./build_all.sh` |
| Build Git Updater only | `./build_all.sh --git-updater` |
| Build Music Brain only | `./build_all.sh --music-brain` |
| Build Penta Core only | `./build_all.sh --penta-core` |
| Clean all builds | `make clean && rm -rf build` |
| Get help | `./build_all.sh --help` |

## 📂 Output Locations

- **Git Updater**: `dist/git-update.sh`
- **Music Brain**: Installed in Python environment
- **Penta Core**: `build/` directory
  - C++ library: `build/src/libpenta_core.a`
  - Python bindings: `build/bindings/penta_core*.so`
  - Plugins: `build/plugins/PentaCorePlugin_artefacts/`
  - Tests: `build/tests/penta_tests`

## ⚡ Platform-Specific

### macOS
```bash
brew install cmake python@3.11
xcode-select --install
```

### Linux (Ubuntu/Debian)
```bash
sudo apt install build-essential cmake python3-dev python3-pip
```

### Windows
- Install Visual Studio 2019+ with C++ tools
- Install CMake from cmake.org
- Install Python from python.org
- Use Git Bash or WSL

---

**Last Updated**: December 22, 2024
