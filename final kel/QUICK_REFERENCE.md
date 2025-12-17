# Kelly MIDI Companion - Quick Reference

## 🚀 Quick Commands

### Build C++ Plugin
```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build

# Build with Python bridge
cmake -B build -DBUILD_PYTHON_BRIDGE=ON
cmake --build build
```

### Run Tests
```bash
# C++ tests
cmake -B build -DBUILD_TESTING=ON
cmake --build build
cd build && ctest --output-on-failure

# Python tests
cd python && pytest
```

### Python Development
```bash
# Activate ML Framework environment
cd ml_framework && source venv/bin/activate

# Activate Python utilities environment
cd python && source venv/bin/activate

# Run examples
cd ml_framework/examples
python basic_usage.py
python emotion_models_demo.py
```

### Visualization
```bash
cd "CODE/PYTHON CODE"
python visualize_3d_emotion_wheel.py
python visualize_quantum_emotional_field.py
```

---

## 📁 Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/` | C++/JUCE plugin source code |
| `ml_framework/` | CIF/LAS/QEF ML framework |
| `python/` | Python utilities and bridge |
| `data/` | Emotion mappings, music theory data |
| `tests/` | Test suite |
| `MARKDOWN/` | Documentation |
| `reference/` | Reference implementations |

---

## 🎯 Core Concepts

### Emotion Thesaurus (216 nodes)
- **Valence**: -1.0 (sad) to +1.0 (happy) → Mode (minor/major)
- **Arousal**: 0.0 (calm) to 1.0 (intense) → Tempo, rhythm
- **Intensity**: 0.0 (subtle) to 1.0 (extreme) → Dynamics, rule-breaking

### Three-Phase Intent System
1. **Wound** → Identify emotional trigger
2. **Emotion** → Map to thesaurus (VAI coordinates)
3. **Rule-Breaks** → Intentional music theory violations

### ML Framework Components
- **CIF**: Conscious Integration Framework
- **LAS**: Living Art Systems
- **QEF**: Quantum Emotional Field
- **Ethics**: Resonant Ethics protocols

---

## 🔧 Development Workflow

### Adding a New Engine
1. Create `src/engines/YourEngine.h` and `.cpp`
2. Add to `CMakeLists.txt` target_sources
3. Integrate in `src/plugin/PluginProcessor.cpp`
4. Add tests in `tests/`

### Adding Emotion Data
1. Add JSON to `data/emotions/`
2. Follow existing structure (valence, arousal, intensity)
3. Update `src/engine/EmotionThesaurus.cpp` if needed

### Python-C++ Bridge
```bash
# Build bridge
cmake -B build -DBUILD_PYTHON_BRIDGE=ON
cmake --build build

# Test import
cd python
python3 -c "import kelly_bridge; print('Success!')"
```

---

## 📚 Documentation Quick Links

- **Main README**: `README.md`
- **Project Status**: `MARKDOWN/MASTER_STATUS.md`
- **Build Guide**: `MARKDOWN/BUILD_PYTHON_BRIDGE.md`
- **Integration**: `MARKDOWN/INTEGRATION_COMPLETE.md`
- **ML Framework**: `ml_framework/README.md`
- **Phase 2**: `phase2/KELLY_PHASE2_IMPLEMENTATION_GUIDE.md`

---

## 🐛 Common Issues

### Build Fails
- Check CMake version: `cmake --version` (need 3.22+)
- Check C++ compiler: `clang++ --version` (need C++20 support)
- Clean build: `rm -rf build && cmake -B build`

### Python Import Errors
- Activate virtual environment: `source venv/bin/activate`
- Check PYTHONPATH: `export PYTHONPATH=$PWD:$PYTHONPATH`
- Reinstall: `pip install -r requirements.txt`

### Plugin Not Loading in DAW
- **macOS**: Check Gatekeeper, codesign plugin
- **Windows**: Check VST3 path
- **Linux**: Check permissions, library paths

---

## 🎵 Philosophy

> *"Interrogate Before Generate"* — The tool shouldn't finish art for people; it should make them braver.

---

## 📞 Workspace Files

- **`.cursorrules`**: AI assistant guidelines
- **`.vscode/settings.json`**: VS Code workspace settings
- **`kelly-midi-companion.code-workspace`**: Multi-root workspace
- **`setup_workspace.sh`**: Initial setup script

---

*Last updated: Workspace setup complete*
