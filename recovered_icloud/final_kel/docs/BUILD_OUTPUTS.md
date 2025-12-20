# Build Outputs Directory Structure

All build outputs are organized in the following directory structure:

```
final kel/
├── build/                          # C++ build artifacts
│   ├── KellyMidiCompanion_artefacts/
│   │   └── Release/                # Release builds
│   │       ├── VST3/               # VST3 plugins
│   │       ├── AU/                 # AU plugins (macOS)
│   │       └── Standalone/          # Standalone apps
│   └── Debug/                      # Debug builds (if built)
│
├── dist/                           # Distribution packages
│   ├── plugins/                    # Final plugin packages
│   ├── apps/                       # App bundles
│   └── installers/                 # Installer packages
│
├── output/                         # General build outputs
│   ├── logs/                       # Build logs
│   ├── reports/                    # Test reports
│   └── artifacts/                  # Build artifacts
│
├── python/                         # Python bridge output
│   └── kelly_bridge.*              # Compiled Python module
│
├── ml_training/
│   └── models/                     # Trained ML models
│       ├── *.pth                  # PyTorch checkpoints
│       └── *.json                  # RTNeural exports
│
└── datasets/                       # Dataset storage
    └── audio/                      # Audio datasets
```

## Output Paths

### C++ Plugin Builds

- **VST3**: `build/KellyMidiCompanion_artefacts/Release/VST3/`
- **AU**: `build/KellyMidiCompanion_artefacts/Release/AU/`
- **Standalone**: `build/KellyMidiCompanion_artefacts/Release/Standalone/`

### Python Bridge

- **Module**: `python/kelly_bridge.so` (Linux/macOS) or `python/kelly_bridge.pyd` (Windows)

### ML Models

- **Training Output**: `ml_training/models/`
- **RTNeural JSON**: `ml_training/models/*.json`

### Distribution

- **macOS App**: `dist/iDAW.app` (if built with --macos-app)
- **Plugins**: `dist/plugins/` (after packaging)

## Environment Variables

You can override output paths using:

```bash
export KELLY_BUILD_DIR="${PWD}/build"
export KELLY_DIST_DIR="${PWD}/dist"
export KELLY_OUTPUT_DIR="${PWD}/output"
```

## Notes

- All paths are relative to the project root
- Build outputs are excluded from git (see .gitignore)
- Use `./build_complete.sh --clean` to remove all build outputs
