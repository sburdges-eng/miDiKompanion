# Build Output Paths Configuration

## Problem Solved

Previously, build outputs were scattered and sometimes written to Desktop. All outputs are now organized in dedicated directories within the project.

## Directory Structure

```
final kel/
├── build/                    # C++ build artifacts (CMake output)
│   └── KellyMidiCompanion_artefacts/
│       └── Release/          # or Debug/
│           ├── VST3/         # VST3 plugins
│           ├── AU/          # AU plugins (macOS)
│           └── Standalone/ # Standalone apps
│
├── output/                   # Organized build outputs
│   ├── logs/                # Build logs (cmake, ctest, etc.)
│   ├── reports/             # Test reports and analysis
│   └── artifacts/          # Additional build artifacts
│
├── dist/                     # Distribution packages
│   ├── plugins/            # Final plugin packages
│   ├── apps/               # App bundles
│   └── installers/         # Installer packages
│
├── python/                   # Python bridge output
│   └── kelly_bridge.*      # Compiled Python module
│
└── ml_training/
    └── models/              # Trained ML models
        ├── *.pth           # PyTorch checkpoints
        └── *.json          # RTNeural exports
```

## Environment Variables

You can override default paths using environment variables:

```bash
# Set custom build directory
export KELLY_BUILD_DIR="/path/to/custom/build"

# Set custom distribution directory
export KELLY_DIST_DIR="/path/to/custom/dist"

# Set custom output directory
export KELLY_OUTPUT_DIR="/path/to/custom/output"
```

## Updated Scripts

All build scripts now use the organized output structure:

- `build_complete.sh` - Complete build with all phases
- `build_all.sh` - Full build with tests
- `build_quick.sh` - Quick build without tests

## Log Files

All build logs are saved to `output/logs/`:

- `cmake_configure.log` - CMake configuration output
- `cmake_build.log` - CMake build output
- `ctest.log` - C++ test output
- `penta_cmake_configure.log` - Penta-Core configuration
- `penta_cmake_build.log` - Penta-Core build

## Test Reports

Test reports are saved to `output/reports/`:

- `ctest_report.log` - C++ test results

## Benefits

1. **No Desktop clutter** - All outputs in project directories
2. **Organized structure** - Easy to find build artifacts
3. **Log preservation** - All build logs saved for debugging
4. **Git-friendly** - Output directories in .gitignore
5. **Customizable** - Environment variables for custom paths

## Clean Build

To remove all build outputs:

```bash
./build_complete.sh --clean
# or
rm -rf build/ output/ dist/
```

## Notes

- All paths are **relative to project root**
- Build outputs are **excluded from git** (see .gitignore)
- Logs are **appended** on each build (not overwritten)
- Use `--clean` flag to start fresh
