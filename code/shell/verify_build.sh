#!/bin/bash
# =============================================================================
# iDAW Build Verification Script
# Verifies all build outputs are present and functional
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo -e "${BLUE}=====================================================${NC}"
echo -e "${BLUE}  iDAW Build Verification${NC}"
echo -e "${BLUE}=====================================================${NC}"
echo ""

ERRORS=0

# =============================================================================
# Check Prerequisites
# =============================================================================

echo -e "${YELLOW}Checking prerequisites...${NC}"

check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} $1: $(command -v "$1")"
    else
        echo -e "  ${RED}✗${NC} $1: NOT FOUND"
        ((ERRORS++))
    fi
}

check_command node
check_command npm
check_command python3
check_command rustc
check_command cargo
echo ""

# =============================================================================
# Check Project Structure
# =============================================================================

echo -e "${YELLOW}Checking project structure...${NC}"

check_file() {
    if [ -f "$1" ]; then
        echo -e "  ${GREEN}✓${NC} $1"
    else
        echo -e "  ${RED}✗${NC} $1: NOT FOUND"
        ((ERRORS++))
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "  ${GREEN}✓${NC} $1/"
    else
        echo -e "  ${RED}✗${NC} $1/: NOT FOUND"
        ((ERRORS++))
    fi
}

check_file "package.json"
check_file "requirements.txt"
check_file "src-tauri/Cargo.toml"
check_file "src-tauri/tauri.conf.json"
check_dir "src"
check_dir "music_brain"
check_dir "src-tauri/src"
echo ""

# =============================================================================
# Check Dependencies
# =============================================================================

echo -e "${YELLOW}Checking dependencies...${NC}"

# Node.js dependencies
if [ -d "node_modules" ]; then
    NODE_DEPS=$(ls node_modules | wc -l | tr -d ' ')
    echo -e "  ${GREEN}✓${NC} node_modules: $NODE_DEPS packages"
else
    echo -e "  ${RED}✗${NC} node_modules: NOT FOUND (run npm install)"
    ((ERRORS++))
fi

# Python virtual environment
if [ -d "venv" ]; then
    echo -e "  ${GREEN}✓${NC} Python venv: present"
else
    echo -e "  ${YELLOW}⚠${NC} Python venv: NOT FOUND (optional)"
fi

# Rust dependencies (check Cargo.lock)
if [ -f "src-tauri/Cargo.lock" ]; then
    echo -e "  ${GREEN}✓${NC} Cargo.lock: present"
else
    echo -e "  ${YELLOW}⚠${NC} Cargo.lock: NOT FOUND (will be created on build)"
fi
echo ""

# =============================================================================
# Check Build Outputs
# =============================================================================

echo -e "${YELLOW}Checking build outputs...${NC}"

# Frontend build
if [ -f "dist/index.html" ]; then
    DIST_SIZE=$(du -sh dist 2>/dev/null | cut -f1)
    echo -e "  ${GREEN}✓${NC} Frontend build: dist/ ($DIST_SIZE)"
else
    echo -e "  ${YELLOW}⚠${NC} Frontend build: NOT FOUND (run npm run build)"
fi

# Tauri build
BUNDLE_PATH="src-tauri/target/release/bundle"
if [ -d "$BUNDLE_PATH" ]; then
    echo -e "  ${GREEN}✓${NC} Tauri build: $BUNDLE_PATH/"

    # Check for specific bundles
    for bundle_type in dmg msi deb AppImage; do
        BUNDLE_FILE=$(find "$BUNDLE_PATH" -name "*.$bundle_type" 2>/dev/null | head -1)
        if [ -n "$BUNDLE_FILE" ]; then
            BUNDLE_SIZE=$(du -h "$BUNDLE_FILE" 2>/dev/null | cut -f1)
            echo -e "    ${GREEN}✓${NC} $bundle_type: $(basename "$BUNDLE_FILE") ($BUNDLE_SIZE)"
        fi
    done
else
    echo -e "  ${YELLOW}⚠${NC} Tauri build: NOT FOUND (run npm run tauri build)"
fi

# Release directory
if [ -d "release" ]; then
    RELEASE_FILES=$(ls release 2>/dev/null | wc -l | tr -d ' ')
    echo -e "  ${GREEN}✓${NC} Release directory: $RELEASE_FILES files"
else
    echo -e "  ${YELLOW}⚠${NC} Release directory: NOT FOUND"
fi
echo ""

# =============================================================================
# Check API
# =============================================================================

echo -e "${YELLOW}Checking API...${NC}"

API_URL="http://127.0.0.1:8000"

if curl -s --connect-timeout 2 "$API_URL/health" > /dev/null 2>&1; then
    HEALTH=$(curl -s "$API_URL/health")
    echo -e "  ${GREEN}✓${NC} API is running at $API_URL"
    echo -e "    Health: $HEALTH"
else
    echo -e "  ${YELLOW}⚠${NC} API is not running at $API_URL"
fi
echo ""

# =============================================================================
# Check Configuration
# =============================================================================

echo -e "${YELLOW}Checking configuration...${NC}"

# Package.json version
VERSION=$(grep '"version"' package.json | head -1 | awk -F'"' '{print $4}')
echo -e "  Version (package.json): ${BLUE}$VERSION${NC}"

# Tauri version
TAURI_VERSION=$(grep '^version' src-tauri/Cargo.toml | head -1 | awk -F'"' '{print $2}')
echo -e "  Version (Cargo.toml): ${BLUE}$TAURI_VERSION${NC}"

# Check version consistency
if [ "$VERSION" != "$TAURI_VERSION" ]; then
    echo -e "  ${YELLOW}⚠${NC} Version mismatch between package.json and Cargo.toml"
fi
echo ""

# =============================================================================
# Summary
# =============================================================================

echo -e "${BLUE}=====================================================${NC}"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}  Verification Complete - All checks passed!${NC}"
else
    echo -e "${RED}  Verification Complete - $ERRORS error(s) found${NC}"
fi
echo -e "${BLUE}=====================================================${NC}"
echo ""

# Quick commands
echo "Quick Commands:"
echo "  Development:  ./start.sh"
echo "  Build:        ./scripts/build_all.sh"
echo "  Deploy:       ./scripts/deploy.sh [target]"
echo ""

exit $ERRORS
