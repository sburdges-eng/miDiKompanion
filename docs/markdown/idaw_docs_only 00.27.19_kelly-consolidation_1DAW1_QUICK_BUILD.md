# Quick Build Reference

## One-Command Build

```bash
./build_macos_standalone.sh
```

Output: `dist/iDAW.app`


## Prerequisites Check

```bash
# Check all required tools
python3 --version  # Need 3.9+
node --version     # Need 18+
cargo --version    # Need latest
cmake --version    # Need 3.22+
```

## Build Options

| Command | Description |
|---------|-------------|
| `./build_macos_standalone.sh` | Basic build (debug) |
| `./build_macos_standalone.sh --release` | Optimized build |
| `./build_macos_standalone.sh --sign` | Code sign app |
| `./build_macos_standalone.sh --sign --notarize` | Sign + notarize |

## Environment Variables (for signing)

```bash
export DEVELOPER_ID="Developer ID Application: Your Name"
export APPLE_ID="your@email.com"
export APPLE_TEAM_ID="YOUR_TEAM_ID"
```

## Run the App

```bash
open dist/iDAW.app
```

## Verify Build

```bash
# Check app bundle exists
ls -la dist/iDAW.app

# Check Python is embedded
ls -la dist/iDAW.app/Contents/Resources/python/

# Check frontend is built
ls -la dist/iDAW.app/Contents/Resources/dist/
```

## Common Commands

```bash
# Clean build
rm -rf build dist
./build_macos_standalone.sh

# Rebuild just frontend
npm run build

# Rebuild just Tauri
cd src-tauri && cargo tauri build

# Test Python server manually
python3 -m music_brain.api
```

## Troubleshooting Quick Fixes

**Build fails at Python step:**
```bash
pip3 install -e ".[all]"
```

**Build fails at Node step:**
```bash
npm install
```

**Build fails at Tauri step:**
```bash
cd src-tauri && cargo build
```

**App won't start:**
```bash
# Check logs
./dist/iDAW.app/Contents/MacOS/idaw
```

## File Locations

- **Build script**: `build_macos_standalone.sh`
- **Output**: `dist/iDAW.app`
- **Build artifacts**: `build/`
- **Tauri config**: `src-tauri/tauri.conf.json`
- **Python launcher**: `music_brain/start_api_embedded.py`
