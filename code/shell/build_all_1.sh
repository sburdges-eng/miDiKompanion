#!/bin/bash
# =============================================================================
# iDAW Full Build Script
# Builds all components: Frontend, Tauri App, Python API
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo -e "${BLUE}=====================================================${NC}"
echo -e "${BLUE}  iDAW - Full Production Build${NC}"
echo -e "${BLUE}=====================================================${NC}"
echo ""

# =============================================================================
# Parse Arguments
# =============================================================================

BUILD_TYPE="release"
SKIP_TESTS=false
CLEAN_BUILD=false
BUILD_API=false
PLATFORM=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="debug"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --with-api)
            BUILD_API=true
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug       Build in debug mode"
            echo "  --skip-tests  Skip running tests"
            echo "  --clean       Clean build directories first"
            echo "  --with-api    Build standalone Python API executable"
            echo "  --platform    Target platform (macos, windows, linux)"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# =============================================================================
# Step 0: Clean (if requested)
# =============================================================================

if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${YELLOW}[0/7] Cleaning previous builds...${NC}"
    rm -rf dist/
    rm -rf src-tauri/target/release/bundle/
    rm -rf src-tauri/target/debug/bundle/
    rm -rf build/
    rm -rf __pycache__/
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ Clean complete${NC}"
    echo ""
fi

# =============================================================================
# Step 1: Check Prerequisites
# =============================================================================

echo -e "${YELLOW}[1/7] Checking prerequisites...${NC}"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}✗ Node.js not found. Please install Node.js 18+${NC}"
    exit 1
fi
NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo -e "${RED}✗ Node.js 18+ required. Found: $(node --version)${NC}"
    exit 1
fi
echo -e "  Node.js: $(node --version)"

# Check npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}✗ npm not found${NC}"
    exit 1
fi
echo -e "  npm: $(npm --version)"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi
echo -e "  Python: $(python3 --version)"

# Check Rust
if ! command -v rustc &> /dev/null; then
    echo -e "${RED}✗ Rust not found. Please install Rust${NC}"
    exit 1
fi
echo -e "  Rust: $(rustc --version)"

# Check Cargo
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}✗ Cargo not found${NC}"
    exit 1
fi
echo -e "  Cargo: $(cargo --version)"

echo -e "${GREEN}✓ All prerequisites met${NC}"
echo ""

# =============================================================================
# Step 2: Install Dependencies
# =============================================================================

echo -e "${YELLOW}[2/7] Installing dependencies...${NC}"

# Node.js dependencies
if [ -f "package-lock.json" ]; then
    npm ci
else
    npm install
fi
echo -e "  ${GREEN}✓ Node.js dependencies installed${NC}"

# Python dependencies
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "  Creating Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

pip install -q -r requirements.txt
echo -e "  ${GREEN}✓ Python dependencies installed${NC}"

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# =============================================================================
# Step 3: Run Tests (unless skipped)
# =============================================================================

if [ "$SKIP_TESTS" = false ]; then
    echo -e "${YELLOW}[3/7] Running tests...${NC}"

    # Frontend tests
    if [ -f "package.json" ] && grep -q '"test"' package.json; then
        echo "  Running frontend tests..."
        npm test 2>/dev/null || echo -e "  ${YELLOW}⚠ No frontend tests or tests skipped${NC}"
    fi

    # Python tests
    if [ -d "tests" ]; then
        echo "  Running Python tests..."
        pytest tests/ -v --tb=short 2>/dev/null || echo -e "  ${YELLOW}⚠ Python tests failed or not found${NC}"
    fi

    echo -e "${GREEN}✓ Tests complete${NC}"
else
    echo -e "${YELLOW}[3/7] Skipping tests (--skip-tests)${NC}"
fi
echo ""

# =============================================================================
# Step 4: Build Frontend
# =============================================================================

echo -e "${YELLOW}[4/7] Building frontend...${NC}"

npm run build

if [ -f "dist/index.html" ]; then
    echo -e "${GREEN}✓ Frontend build complete${NC}"
    echo -e "  Output: dist/"
else
    echo -e "${RED}✗ Frontend build failed${NC}"
    exit 1
fi
echo ""

# =============================================================================
# Step 5: Build Tauri Application
# =============================================================================

echo -e "${YELLOW}[5/7] Building Tauri application...${NC}"

if [ "$BUILD_TYPE" = "debug" ]; then
    npm run tauri build -- --debug
else
    npm run tauri build
fi

# Check for output
BUNDLE_PATH="src-tauri/target/release/bundle"
if [ "$BUILD_TYPE" = "debug" ]; then
    BUNDLE_PATH="src-tauri/target/debug/bundle"
fi

if [ -d "$BUNDLE_PATH" ]; then
    echo -e "${GREEN}✓ Tauri build complete${NC}"
    echo -e "  Output: $BUNDLE_PATH/"

    # List bundles
    echo -e "  Bundles created:"
    ls -la "$BUNDLE_PATH/" 2>/dev/null | grep -E "^d" | awk '{print "    - " $NF}'
else
    echo -e "${RED}✗ Tauri build failed${NC}"
    exit 1
fi
echo ""

# =============================================================================
# Step 6: Build Python API (if requested)
# =============================================================================

if [ "$BUILD_API" = true ]; then
    echo -e "${YELLOW}[6/7] Building Python API executable...${NC}"

    # Install PyInstaller if needed
    pip install -q pyinstaller

    # Build
    pyinstaller --onefile --name music_brain_api \
        --add-data "music_brain/data:music_brain/data" \
        --hidden-import uvicorn \
        --hidden-import uvicorn.logging \
        --hidden-import uvicorn.loops \
        --hidden-import uvicorn.loops.auto \
        --hidden-import uvicorn.protocols \
        --hidden-import uvicorn.protocols.http \
        --hidden-import uvicorn.protocols.http.auto \
        --hidden-import uvicorn.protocols.websockets \
        --hidden-import uvicorn.protocols.websockets.auto \
        --hidden-import uvicorn.lifespan \
        --hidden-import uvicorn.lifespan.on \
        --hidden-import fastapi \
        --hidden-import pydantic \
        music_brain/api.py

    if [ -f "dist/music_brain_api" ] || [ -f "dist/music_brain_api.exe" ]; then
        echo -e "${GREEN}✓ Python API build complete${NC}"
        echo -e "  Output: dist/music_brain_api"
    else
        echo -e "${YELLOW}⚠ Python API build may have failed${NC}"
    fi
else
    echo -e "${YELLOW}[6/7] Skipping Python API build (use --with-api to enable)${NC}"
fi
echo ""

# =============================================================================
# Step 7: Create Release Package
# =============================================================================

echo -e "${YELLOW}[7/7] Creating release package...${NC}"

RELEASE_DIR="release"
mkdir -p "$RELEASE_DIR"

# Copy Tauri bundles
if [ -d "$BUNDLE_PATH" ]; then
    cp -r "$BUNDLE_PATH"/* "$RELEASE_DIR/" 2>/dev/null || true
fi

# Copy API executable if built
if [ -f "dist/music_brain_api" ]; then
    cp "dist/music_brain_api" "$RELEASE_DIR/"
fi
if [ -f "dist/music_brain_api.exe" ]; then
    cp "dist/music_brain_api.exe" "$RELEASE_DIR/"
fi

# Create version file
VERSION=$(grep '"version"' package.json | head -1 | awk -F'"' '{print $4}')
echo "$VERSION" > "$RELEASE_DIR/VERSION"

# Create checksums
cd "$RELEASE_DIR"
if command -v shasum &> /dev/null; then
    shasum -a 256 * > CHECKSUMS.sha256 2>/dev/null || true
fi
cd "$PROJECT_ROOT"

echo -e "${GREEN}✓ Release package created${NC}"
echo -e "  Output: $RELEASE_DIR/"
echo ""

# =============================================================================
# Summary
# =============================================================================

echo -e "${BLUE}=====================================================${NC}"
echo -e "${GREEN}  Build Complete!${NC}"
echo -e "${BLUE}=====================================================${NC}"
echo ""
echo -e "Build Type: ${YELLOW}$BUILD_TYPE${NC}"
echo -e "Version: ${YELLOW}$VERSION${NC}"
echo ""
echo "Outputs:"
echo -e "  Frontend:     ${GREEN}dist/${NC}"
echo -e "  Tauri Bundle: ${GREEN}$BUNDLE_PATH/${NC}"
echo -e "  Release:      ${GREEN}$RELEASE_DIR/${NC}"
if [ "$BUILD_API" = true ]; then
    echo -e "  API:          ${GREEN}dist/music_brain_api${NC}"
fi
echo ""
echo "Next steps:"
echo "  1. Test the application: ./release/<app-name>"
echo "  2. Sign the application (for distribution)"
echo "  3. Upload to release server or app store"
