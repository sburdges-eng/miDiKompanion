#!/bin/bash
# =============================================================================
# iDAW macOS Standalone Application Builder
# Version: 2.1.0 (Consolidated & Improved)
# =============================================================================
#
# This script builds a complete standalone macOS application bundle that includes:
# - React frontend (via Tauri)
# - Python backend (embedded)
# - C++ audio engines (penta_core, iDAW_Core)
# - All dependencies and resources
#
# Usage:
#   ./scripts/build_macos.sh [--sign] [--notarize] [--release] [--clean]
#
# Options:
#   --sign      Sign the app with your Developer ID
#   --notarize  Notarize the app with Apple (requires --sign)
#   --release   Build in release mode (optimized)
#   --clean     Clean build directories before building
#
# =============================================================================

set -euo pipefail

# Configuration
readonly APP_NAME="iDAW"
readonly APP_VERSION="1.0.0"
readonly BUNDLE_ID="com.idaw.app"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly BUILD_DIR="$PROJECT_ROOT/build"
readonly DIST_DIR="$PROJECT_ROOT/dist"
readonly APP_BUNDLE="$DIST_DIR/$APP_NAME.app"

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly CYAN='\033[0;36m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Build options
SIGN_APP=false
NOTARIZE_APP=false
RELEASE_MODE=false
CLEAN_BUILD=false

# Logging functions
log_info() { echo -e "${BLUE}ℹ${NC} $1"; }
log_success() { echo -e "${GREEN}✓${NC} $1"; }
log_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }
log_step() { echo -e "${CYAN}[$1]${NC} $2"; }

# Parse arguments
parse_args() {
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
            --clean)
                CLEAN_BUILD=true
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_warning "Unknown option: $arg"
                ;;
        esac
    done
}

show_help() {
    cat << EOF
iDAW macOS Standalone Application Builder

Usage: $0 [OPTIONS]

Options:
  --sign       Sign the app with Developer ID
  --notarize   Notarize the app (requires --sign)
  --release    Build in release mode
  --clean      Clean build directories first
  --help, -h   Show this help message

Environment Variables:
  DEVELOPER_ID    Developer ID for code signing
  APPLE_ID        Apple ID for notarization
  APPLE_TEAM_ID   Apple Team ID for notarization

Examples:
  $0                    # Basic debug build
  $0 --release          # Optimized release build
  $0 --sign --notarize  # Signed and notarized build
EOF
}

# Header
show_header() {
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
    echo "║        Standalone macOS Application Builder v2.1            ║"
    echo "║                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check requirements
check_requirements() {
    log_step "1/10" "Checking requirements..."
    
    local missing_tools=()
    
    check_tool() {
        if ! command -v "$1" &> /dev/null; then
            missing_tools+=("$1")
            return 1
        fi
        log_success "$1 found"
        return 0
    }
    
    check_tool python3 || true
    check_tool node || true
    check_tool npm || true
    check_tool cargo || true
    check_tool cmake || true
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install missing tools and try again"
        exit 1
    fi
    
    # Check Python version
    local python_version
    python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    log_info "Python version: $python_version"
}

# Clean build directories
clean_build() {
    if [ "$CLEAN_BUILD" = true ]; then
        log_step "0/10" "Cleaning build directories..."
        rm -rf "$BUILD_DIR" "$DIST_DIR"
        log_success "Build directories cleaned"
    fi
}

# Install dependencies
install_dependencies() {
    log_step "2/10" "Installing Python dependencies..."
    cd "$PROJECT_ROOT"
    if ! pip3 install --quiet -e ".[all,build]" 2>/dev/null; then
        log_warning "Some optional dependencies may be missing, continuing..."
        pip3 install --quiet -e . || {
            log_error "Failed to install Python dependencies"
            exit 1
        }
    fi
    
    log_step "3/10" "Installing Node dependencies..."
    if ! npm install --silent; then
        log_error "Failed to install Node dependencies"
        exit 1
    fi
}

# Build C++ libraries
build_cpp_libraries() {
    log_step "4/10" "Building C++ libraries..."
    cd "$PROJECT_ROOT"
    
    # Build penta_core
    if [ -d "src_penta-core" ] && [ -f "src_penta-core/CMakeLists.txt" ]; then
        log_info "Building penta_core..."
        mkdir -p "$BUILD_DIR/penta_core"
        cd "$BUILD_DIR/penta_core"
        
        if cmake "$PROJECT_ROOT/src_penta-core" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
            -G Ninja 2>/dev/null || \
           cmake "$PROJECT_ROOT/src_penta-core" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"; then
            if cmake --build . --config Release 2>/dev/null || \
               make -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"; then
                log_success "penta_core built"
            else
                log_warning "penta_core build failed (optional)"
            fi
        else
            log_warning "penta_core CMake configuration failed (optional)"
        fi
        cd "$PROJECT_ROOT"
    fi
}

# Build frontend and Tauri
build_tauri_app() {
    log_step "5/10" "Building React frontend..."
    cd "$PROJECT_ROOT"
    if ! npm run build; then
        log_error "Frontend build failed"
        exit 1
    fi
    
    log_step "6/10" "Building Tauri application..."
    cd "$PROJECT_ROOT/src-tauri"
    
    local build_mode="debug"
    [ "$RELEASE_MODE" = true ] && build_mode="release"
    
    if cargo tauri build --bundles app \
        $([ "$RELEASE_MODE" = true ] && echo "--release" || echo "") \
        --target universal-apple-darwin 2>/dev/null || \
       cargo tauri build --bundles app \
        $([ "$RELEASE_MODE" = true ] && echo "--release" || echo ""); then
        log_success "Tauri app built"
    else
        log_error "Tauri build failed"
        exit 1
    fi
}

# Prepare app bundle
prepare_app_bundle() {
    log_step "7/10" "Preparing app bundle..."
    
    local tauri_output
    if [ "$RELEASE_MODE" = true ]; then
        tauri_output="$PROJECT_ROOT/src-tauri/target/release/bundle/macos"
    else
        tauri_output="$PROJECT_ROOT/src-tauri/target/debug/bundle/macos"
    fi
    
    local app_bundle_source
    app_bundle_source=$(find "$tauri_output" -name "*.app" -type d | head -n 1)
    
    if [ -z "$app_bundle_source" ]; then
        log_error "App bundle not found in $tauri_output"
        exit 1
    fi
    
    rm -rf "$APP_BUNDLE"
    mkdir -p "$DIST_DIR"
    cp -R "$app_bundle_source" "$APP_BUNDLE"
    log_success "App bundle prepared"
}

# Embed Python runtime
embed_python() {
    log_step "8/10" "Embedding Python runtime..."
    
    local python_resources="$APP_BUNDLE/Contents/Resources/python"
    mkdir -p "$python_resources"
    
    # Copy Python packages
    log_info "Copying Python packages..."
    if python3 -m pip install --target "$python_resources" \
        fastapi uvicorn pydantic mido numpy \
        --quiet --no-deps 2>/dev/null; then
        log_success "Python packages copied"
    else
        log_warning "Some Python packages may be missing"
    fi
    
    # Copy music_brain
    if [ -d "$PROJECT_ROOT/music_brain" ]; then
        cp -R "$PROJECT_ROOT/music_brain" "$python_resources/"
    fi
    
    # Copy data files
    if [ -d "$PROJECT_ROOT/Data_Files" ]; then
        mkdir -p "$python_resources/data"
        cp -R "$PROJECT_ROOT/Data_Files"/* "$python_resources/data/" 2>/dev/null || true
    fi
    
    if [ -d "$PROJECT_ROOT/emotion_thesaurus" ]; then
        cp -R "$PROJECT_ROOT/emotion_thesaurus" "$python_resources/" 2>/dev/null || true
    fi
    
    # Copy launcher
    if [ -f "$PROJECT_ROOT/music_brain/start_api_embedded.py" ]; then
        cp "$PROJECT_ROOT/music_brain/start_api_embedded.py" "$python_resources/start_api.py"
        chmod +x "$python_resources/start_api.py"
    fi
    
    log_success "Python runtime embedded"
}

# Copy C++ libraries
copy_cpp_libraries() {
    log_step "9/10" "Copying C++ libraries..."
    
    local lib_dir="$APP_BUNDLE/Contents/Frameworks"
    mkdir -p "$lib_dir"
    
    if [ -f "$BUILD_DIR/penta_core/libpenta_core.a" ]; then
        cp "$BUILD_DIR/penta_core/libpenta_core.a" "$lib_dir/" 2>/dev/null || true
    fi
    
    find "$BUILD_DIR" -name "*.dylib" -exec cp {} "$lib_dir/" \; 2>/dev/null || true
    log_success "C++ libraries copied"
}

# Code signing
code_sign() {
    log_step "10/10" "Code signing..."
    
    if [ "$SIGN_APP" = true ]; then
        if [ -n "${DEVELOPER_ID:-}" ]; then
            if codesign --force --deep --options runtime \
                --entitlements "$PROJECT_ROOT/macos/iDAW.entitlements" \
                --sign "$DEVELOPER_ID" \
                "$APP_BUNDLE" 2>/dev/null; then
                log_success "App signed with $DEVELOPER_ID"
                
                if [ "$NOTARIZE_APP" = true ]; then
                    notarize_app
                fi
            else
                log_error "Code signing failed"
                exit 1
            fi
        else
            log_warning "DEVELOPER_ID not set, skipping signing"
        fi
    else
        log_info "Skipping code signing (use --sign to enable)"
    fi
}

# Notarize app
notarize_app() {
    if [ -n "${APPLE_ID:-}" ] && [ -n "${APPLE_TEAM_ID:-}" ]; then
        log_info "Notarizing app..."
        ditto -c -k --keepParent "$APP_BUNDLE" "$DIST_DIR/iDAW.zip"
        
        if xcrun notarytool submit "$DIST_DIR/iDAW.zip" \
            --apple-id "$APPLE_ID" \
            --team-id "$APPLE_TEAM_ID" \
            --wait; then
            xcrun stapler staple "$APP_BUNDLE"
            log_success "App notarized"
        else
            log_error "Notarization failed"
            exit 1
        fi
    else
        log_warning "APPLE_ID or APPLE_TEAM_ID not set, skipping notarization"
    fi
}

# Show summary
show_summary() {
    echo ""
    log_success "╔══════════════════════════════════════════════════════════════╗"
    log_success "║                    BUILD COMPLETE!                           ║"
    log_success "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  📦 App Bundle: $APP_BUNDLE"
    echo "  📊 Version: $APP_VERSION"
    echo ""
    echo "  🚀 To run: open $APP_BUNDLE"
    echo ""
}

# Main function
main() {
    show_header
    parse_args "$@"
    clean_build
    check_requirements
    install_dependencies
    build_cpp_libraries
    build_tauri_app
    prepare_app_bundle
    embed_python
    copy_cpp_libraries
    code_sign
    show_summary
}

# Run main
main "$@"
