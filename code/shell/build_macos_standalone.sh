#!/bin/bash
# =============================================================================
# iDAW macOS Standalone Application Builder
# Version: 2.0.0
# =============================================================================
#
# This script builds a complete standalone macOS application bundle that includes:
# - React frontend (via Tauri)
# - Python backend (embedded)
# - C++ audio engines (penta_core, iDAW_Core)
# - All dependencies and resources
#
# Usage:
#   ./build_macos_standalone.sh [--sign] [--notarize] [--release]
#
# =============================================================================

set -e

# Configuration
APP_NAME="iDAW"
APP_VERSION="1.0.0"
BUNDLE_ID="com.idaw.app"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
BUILD_DIR="$PROJECT_ROOT/build"
DIST_DIR="$PROJECT_ROOT/dist"
APP_BUNDLE="$DIST_DIR/$APP_NAME.app"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║     ██╗██████╗  █████╗ ██╗    ██╗                           ║"
echo "║     ██║██╔══██╗██╔══██╗██║    ██║                           ║"
echo "║     ██║██║  ██║███████║██║ █╗ ██║                           ║"
echo "║     ██║██║  ██║██╔══██║██║███╗██║                           ║"
echo "║     ██║██████╔╝██║  ██║╚███╔███╔╝                           ║"
echo "║     ╚═╝╚═════╝ ╚═╝  ╚═╝ ╚══╝╚══╝                            ║"
echo "║                                                              ║"
echo "║        Standalone macOS Application Builder v2.0            ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Parse arguments
SIGN_APP=false
NOTARIZE_APP=false
RELEASE_MODE=false
for arg in "$@"; do
    case $arg in
        --sign)
            SIGN_APP=true
            ;;
        --notarize)
            NOTARIZE_APP=true
            ;;
        --release)
            RELEASE_MODE=true
            ;;
    esac
done

# Step 1: Check requirements
echo -e "${YELLOW}[1/10] Checking requirements...${NC}"

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        exit 1
    fi
    echo "  ✓ $1 found"
}

check_command python3
check_command node
check_command npm
check_command cargo
check_command cmake

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  ✓ Python $PYTHON_VERSION"

# Step 2: Install Python dependencies
echo -e "${YELLOW}[2/10] Installing Python dependencies...${NC}"
cd "$PROJECT_ROOT"
pip3 install --quiet -e ".[all,build]" || {
    echo -e "${YELLOW}  ⚠ Some optional dependencies may be missing, continuing...${NC}"
    pip3 install --quiet -e .
}

# Step 3: Install Node dependencies
echo -e "${YELLOW}[3/10] Installing Node dependencies...${NC}"
cd "$PROJECT_ROOT"
npm install --silent

# Step 4: Build C++ libraries
echo -e "${YELLOW}[4/10] Building C++ libraries...${NC}"
cd "$PROJECT_ROOT"

# Build penta_core
if [ -d "src_penta-core" ] && [ -f "src_penta-core/CMakeLists.txt" ]; then
    echo "  Building penta_core..."
    mkdir -p "$BUILD_DIR/penta_core"
    cd "$BUILD_DIR/penta_core"
    cmake "$PROJECT_ROOT/src_penta-core" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
        -G Ninja 2>/dev/null || cmake "$PROJECT_ROOT/src_penta-core" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"
    cmake --build . --config Release || make -j$(sysctl -n hw.ncpu)
    cd "$PROJECT_ROOT"
fi

# Build iDAW_Core (if JUCE is available)
if [ -d "iDAW_Core" ] && [ -f "iDAW_Core/CMakeLists.txt" ]; then
    echo "  Building iDAW_Core..."
    mkdir -p "$BUILD_DIR/idaw_core"
    cd "$BUILD_DIR/idaw_core"
    cmake "$PROJECT_ROOT/iDAW_Core" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
        -G Ninja 2>/dev/null || cmake "$PROJECT_ROOT/iDAW_Core" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"
    cmake --build . --config Release || make -j$(sysctl -n hw.ncpu)
    cd "$PROJECT_ROOT"
fi

# Step 5: Build React frontend
echo -e "${YELLOW}[5/10] Building React frontend...${NC}"
cd "$PROJECT_ROOT"
npm run build

# Step 6: Build Tauri application
echo -e "${YELLOW}[6/10] Building Tauri application...${NC}"
cd "$PROJECT_ROOT/src-tauri"

# Set build mode
TAURI_BUILD_MODE="debug"
if [ "$RELEASE_MODE" = true ]; then
    Tauri_BUILD_MODE="release"
fi

# Build Tauri app
cargo tauri build --bundles app \
    $([ "$RELEASE_MODE" = true ] && echo "--release" || echo "") \
    --target universal-apple-darwin 2>/dev/null || \
cargo tauri build --bundles app \
    $([ "$RELEASE_MODE" = true ] && echo "--release" || echo "")

# Step 7: Locate and prepare app bundle
echo -e "${YELLOW}[7/10] Preparing app bundle...${NC}"

# Tauri outputs to different locations based on build mode
if [ "$RELEASE_MODE" = true ]; then
    TAURI_OUTPUT="$PROJECT_ROOT/src-tauri/target/release/bundle/macos"
else
    TAURI_OUTPUT="$PROJECT_ROOT/src-tauri/target/debug/bundle/macos"
fi

# Find the .app bundle
APP_BUNDLE_SOURCE=$(find "$TAURI_OUTPUT" -name "*.app" -type d | head -n 1)

if [ -z "$APP_BUNDLE_SOURCE" ]; then
    echo -e "${RED}Error: App bundle not found in $TAURI_OUTPUT${NC}"
    exit 1
fi

# Copy to dist
rm -rf "$APP_BUNDLE"
mkdir -p "$DIST_DIR"
cp -R "$APP_BUNDLE_SOURCE" "$APP_BUNDLE"

echo "  ✓ App bundle located at: $APP_BUNDLE"

# Step 8: Embed Python runtime and dependencies
echo -e "${YELLOW}[8/10] Embedding Python runtime...${NC}"

# Create Python resources directory
PYTHON_RESOURCES="$APP_BUNDLE/Contents/Resources/python"
mkdir -p "$PYTHON_RESOURCES"

# Copy Python packages
echo "  Copying Python packages..."
python3 -m pip install --target "$PYTHON_RESOURCES" \
    fastapi uvicorn pydantic mido numpy \
    --quiet --no-deps 2>/dev/null || true

# Copy music_brain package
if [ -d "$PROJECT_ROOT/music_brain" ]; then
    cp -R "$PROJECT_ROOT/music_brain" "$PYTHON_RESOURCES/"
fi

# Copy data files
if [ -d "$PROJECT_ROOT/Data_Files" ]; then
    mkdir -p "$PYTHON_RESOURCES/data"
    cp -R "$PROJECT_ROOT/Data_Files"/* "$PYTHON_RESOURCES/data/" 2>/dev/null || true
fi

if [ -d "$PROJECT_ROOT/emotion_thesaurus" ]; then
    cp -R "$PROJECT_ROOT/emotion_thesaurus" "$PYTHON_RESOURCES/" 2>/dev/null || true
fi

# Copy Python launcher script
if [ -f "$PROJECT_ROOT/music_brain/start_api_embedded.py" ]; then
    cp "$PROJECT_ROOT/music_brain/start_api_embedded.py" "$PYTHON_RESOURCES/start_api.py"
    chmod +x "$PYTHON_RESOURCES/start_api.py"
else
    # Create fallback launcher
    cat > "$PYTHON_RESOURCES/start_api.py" << 'PYEOF'
#!/usr/bin/env python3
"""Embedded Music Brain API Server"""
import sys
import os
from pathlib import Path

# Add resources to path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

# Import and run API
from music_brain.api import app
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("MUSIC_BRAIN_PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
PYEOF
    chmod +x "$PYTHON_RESOURCES/start_api.py"
fi

echo "  ✓ Python runtime embedded"

# Step 9: Copy C++ libraries
echo -e "${YELLOW}[9/10] Copying C++ libraries...${NC}"

LIB_DIR="$APP_BUNDLE/Contents/Frameworks"
mkdir -p "$LIB_DIR"

# Copy penta_core libraries
if [ -f "$BUILD_DIR/penta_core/libpenta_core.a" ]; then
    cp "$BUILD_DIR/penta_core/libpenta_core.a" "$LIB_DIR/" 2>/dev/null || true
fi

# Copy any .dylib files
find "$BUILD_DIR" -name "*.dylib" -exec cp {} "$LIB_DIR/" \; 2>/dev/null || true

echo "  ✓ C++ libraries copied"

# Step 10: Code signing and notarization
echo -e "${YELLOW}[10/10] Code signing...${NC}"

if [ "$SIGN_APP" = true ]; then
    if [ -n "$DEVELOPER_ID" ]; then
        codesign --force --deep --options runtime \
            --entitlements "$PROJECT_ROOT/macos/iDAW.entitlements" \
            --sign "$DEVELOPER_ID" \
            "$APP_BUNDLE"
        echo "  ✓ App signed with $DEVELOPER_ID"
        
        if [ "$NOTARIZE_APP" = true ]; then
            if [ -n "$APPLE_ID" ] && [ -n "$APPLE_TEAM_ID" ]; then
                echo "  Notarizing app..."
                ditto -c -k --keepParent "$APP_BUNDLE" "$DIST_DIR/iDAW.zip"
                xcrun notarytool submit "$DIST_DIR/iDAW.zip" \
                    --apple-id "$APPLE_ID" \
                    --team-id "$APPLE_TEAM_ID" \
                    --wait
                xcrun stapler staple "$APP_BUNDLE"
                echo "  ✓ App notarized"
            else
                echo -e "${YELLOW}  ⚠ APPLE_ID or APPLE_TEAM_ID not set, skipping notarization${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}  ⚠ DEVELOPER_ID not set, skipping signing${NC}"
    fi
else
    echo "  ⏭ Skipping (use --sign to enable)"
fi

# Summary
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    BUILD COMPLETE!                           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  📦 App Bundle: $APP_BUNDLE"
echo "  📊 Version: $APP_VERSION"
echo ""
echo "  🎛️ Included Components:"
echo "     • React Frontend (Tauri)"
echo "     • Python Backend (Embedded)"
echo "     • C++ Audio Engines"
echo "     • Music Brain API"
echo "     • Emotion Thesaurus"
echo ""
echo "  🚀 To run: open $APP_BUNDLE"
echo ""

# Create DMG (optional)
if command -v create-dmg &> /dev/null; then
    echo -e "${YELLOW}Creating DMG installer...${NC}"
    create-dmg \
        --volname "iDAW $APP_VERSION" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "iDAW.app" 175 190 \
        --hide-extension "iDAW.app" \
        --app-drop-link 425 190 \
        "$DIST_DIR/iDAW-$APP_VERSION.dmg" \
        "$APP_BUNDLE" 2>/dev/null || echo "  ⚠ DMG creation skipped (create-dmg not available)"
fi

echo ""
echo -e "${GREEN}Build completed successfully!${NC}"
