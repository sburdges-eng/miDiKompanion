# Building iDAW Standalone macOS Application

This guide explains how to build a standalone macOS application bundle for iDAW that includes all components.

## Quick Start

```bash
./build_macos_standalone.sh
```

This will create `dist/iDAW.app` with:
- React frontend (Tauri)
- Python backend (embedded)
- C++ audio engines
- All dependencies

## Prerequisites

### Required Tools

- **macOS 10.15+** (Catalina or later)
- **Xcode Command Line Tools**: `xcode-select --install`
- **Python 3.9+**: `python3 --version`
- **Node.js 18+**: `node --version`
- **Rust/Cargo**: `cargo --version` (install from https://rustup.rs)
- **CMake 3.22+**: `cmake --version`

### Optional Tools

- **Ninja** (faster builds): `brew install ninja`
- **create-dmg** (for DMG creation): `brew install create-dmg`

## Build Options

### Basic Build

```bash
./build_macos_standalone.sh
```

### Release Build (Optimized)

```bash
./build_macos_standalone.sh --release
```

### Code Signing

```bash
export DEVELOPER_ID="Developer ID Application: Your Name"
./build_macos_standalone.sh --sign
```

### Notarization (Requires Signing)

```bash
export DEVELOPER_ID="Developer ID Application: Your Name"
export APPLE_ID="your@email.com"
export APPLE_TEAM_ID="YOUR_TEAM_ID"
./build_macos_standalone.sh --sign --notarize
```

## Build Process

The build script performs these steps:

1. **Check Requirements** - Verifies all tools are installed
2. **Install Python Dependencies** - Installs music_brain and dependencies
3. **Install Node Dependencies** - Installs React/Tauri dependencies
4. **Build C++ Libraries** - Compiles penta_core and iDAW_Core
5. **Build React Frontend** - Compiles TypeScript/React to static files
6. **Build Tauri Application** - Creates native macOS app bundle
7. **Embed Python Runtime** - Copies Python packages and scripts
8. **Copy C++ Libraries** - Includes compiled audio engines
9. **Code Signing** - Signs the app (if requested)
10. **Notarization** - Notarizes with Apple (if requested)

## Output

After building, you'll find:

```
dist/
└── iDAW.app/              # Application bundle
    ├── Contents/
    │   ├── MacOS/         # Executable
    │   ├── Resources/     # Frontend assets, Python runtime
    │   └── Frameworks/    # C++ libraries
    └── ...
```

## Python Embedding

The app embeds Python in two ways:

### Development Mode

- Uses system Python (`python3`)
- Requires Python packages to be installed
- Faster iteration

### Production Mode

- Bundles Python packages in `Resources/python/`
- Includes `start_api.py` launcher
- Self-contained (no system Python required)

## Running the App

### From Finder

Double-click `dist/iDAW.app`

### From Terminal

```bash
open dist/iDAW.app
```

### Debug Mode

```bash
./dist/iDAW.app/Contents/MacOS/idaw
```

## Troubleshooting

### "Python interpreter not found"

- Ensure Python 3.9+ is installed: `python3 --version`
- For production builds, Python packages are bundled automatically

### "API script not found"

- Check that `music_brain/api.py` exists
- Verify Python packages are installed: `pip3 install -e .`

### "C++ build failed"

- Ensure CMake is installed: `cmake --version`
- Check that required C++ libraries are available
- Some components (JUCE) may be optional

### "Tauri build failed"

- Ensure Rust is installed: `rustup install stable`
- Check Node dependencies: `npm install`
- Verify frontend builds: `npm run build`

### App crashes on launch

- Check console logs: `Console.app` → Search for "iDAW"
- Verify Python server starts: Check `Resources/python/start_api.py`
- Test API manually: `curl http://127.0.0.1:8000/health`

## Architecture

```
┌─────────────────────────────────────┐
│         iDAW.app Bundle            │
├─────────────────────────────────────┤
│  ┌──────────────────────────────┐  │
│  │   Tauri (Rust Backend)        │  │
│  │   - Manages Python server     │  │
│  │   - Handles IPC               │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │   React Frontend              │  │
│  │   - UI Components             │  │
│  │   - Emotion Wheel             │  │
│  │   - DAW Interface             │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │   Python Backend              │  │
│  │   - Music Brain API           │  │
│  │   - Music Generation          │  │
│  │   - Emotion Mapping           │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │   C++ Audio Engines          │  │
│  │   - penta_core                │  │
│  │   - iDAW_Core                 │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

## Distribution

### Create DMG

```bash
create-dmg \
  --volname "iDAW" \
  --window-pos 200 120 \
  --window-size 600 400 \
  dist/iDAW-1.0.0.dmg \
  dist/iDAW.app
```

### App Store Distribution

1. Code sign with App Store certificate
2. Notarize the app
3. Create App Store package: `productbuild --component dist/iDAW.app /Applications iDAW.pkg`

## Development vs Production

### Development

- Uses system Python
- Hot-reloads frontend changes
- Debug symbols included
- Faster build times

### Production

- Bundles Python runtime
- Optimized builds
- Code signed (optional)
- Notarized (optional)
- Smaller bundle size

## Next Steps

1. **Test the app**: Run `dist/iDAW.app` and verify all features work
2. **Code sign**: Set up Developer ID for distribution
3. **Create DMG**: Package for easy distribution
4. **Test on clean system**: Verify no external dependencies required

## Support

For build issues:
1. Check error messages in terminal output
2. Verify all prerequisites are installed
3. Review `build/` directory for build artifacts
4. Check Tauri logs: `~/.tauri/logs/`

For runtime issues:
1. Check Console.app for crash logs
2. Verify Python server is running: `curl http://127.0.0.1:8000/health`
3. Test components individually
