#!/usr/bin/env bash
set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Component flags
BUILD_GIT_UPDATER=false
BUILD_MUSIC_BRAIN=false
BUILD_PENTA_CORE=false
BUILD_ALL=false

# Parse command line arguments
if [ $# -eq 0 ]; then
    BUILD_ALL=true
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --git-updater)
            BUILD_GIT_UPDATER=true
            ;;
        --music-brain)
            BUILD_MUSIC_BRAIN=true
            ;;
        --penta-core)
            BUILD_PENTA_CORE=true
            ;;
        --all)
            BUILD_ALL=true
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Build all or specific components of miDiKompanion"
            echo ""
            echo "Options:"
            echo "  --git-updater    Build Git Multi-Repository Updater only"
            echo "  --music-brain    Build Music Brain (DAiW/iDAW) only"
            echo "  --penta-core     Build Penta Core only"
            echo "  --all            Build all components (default if no options)"
            echo "  --help, -h       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                        # Build all components"
            echo "  $0 --git-updater          # Build Git Updater only"
            echo "  $0 --music-brain --penta-core  # Build Music Brain and Penta Core"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
    shift
done

# If BUILD_ALL is true, enable all components
if [ "$BUILD_ALL" = true ]; then
    BUILD_GIT_UPDATER=true
    BUILD_MUSIC_BRAIN=true
    BUILD_PENTA_CORE=true
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}miDiKompanion Multi-Component Build${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print status messages
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing_deps=()
    
    # Check for git
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    fi
    
    # Check for make (if building git updater)
    if [ "$BUILD_GIT_UPDATER" = true ] && ! command -v make &> /dev/null; then
        print_warning "make not found. Will use build.sh directly."
    fi
    
    # Check for python (if building Music Brain or Penta Core)
    if [ "$BUILD_MUSIC_BRAIN" = true ] || [ "$BUILD_PENTA_CORE" = true ]; then
        if ! command -v python3 &> /dev/null; then
            missing_deps+=("python3")
        else
            python_version=$(python3 --version | awk '{print $2}')
            print_status "Found Python $python_version"
        fi
        
        if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
            missing_deps+=("pip")
        fi
    fi
    
    # Check for cmake (if building Penta Core)
    if [ "$BUILD_PENTA_CORE" = true ]; then
        if ! command -v cmake &> /dev/null; then
            missing_deps+=("cmake")
        else
            cmake_version=$(cmake --version | head -n1 | awk '{print $3}')
            print_status "Found CMake $cmake_version"
        fi
        
        # Check for C++ compiler
        if ! command -v c++ &> /dev/null && ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
            missing_deps+=("C++ compiler (g++/clang++)")
        fi
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_error "Missing required dependencies:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        echo ""
        echo "Please install missing dependencies and try again."
        echo "See MULTI_BUILD.md for installation instructions."
        exit 1
    fi
    
    print_success "All prerequisites met"
    echo ""
}

# Build Git Multi-Repository Updater
build_git_updater() {
    print_status "Building Git Multi-Repository Updater..."
    
    if [ -f "Makefile" ]; then
        if make standard; then
            print_success "Git Updater built successfully"
            print_status "Output: dist/git-update.sh"
        else
            print_error "Git Updater build failed"
            return 1
        fi
    elif [ -f "build.sh" ]; then
        if ./build.sh profiles/standard.profile; then
            print_success "Git Updater built successfully"
            print_status "Output: dist/git-update.sh"
        else
            print_error "Git Updater build failed"
            return 1
        fi
    else
        print_error "Neither Makefile nor build.sh found"
        return 1
    fi
    
    echo ""
}

# Build Music Brain
build_music_brain() {
    print_status "Building Music Brain (DAiW/iDAW)..."
    
    # Check if pyproject_music-brain.toml exists
    if [ ! -f "pyproject_music-brain.toml" ]; then
        print_error "pyproject_music-brain.toml not found"
        return 1
    fi
    
    # Try pip3 first, fall back to pip
    local pip_cmd="pip3"
    if ! command -v pip3 &> /dev/null; then
        pip_cmd="pip"
    fi
    
    print_status "Installing Music Brain in editable mode..."
    if $pip_cmd install -e . -c pyproject_music-brain.toml; then
        print_success "Music Brain installed successfully"
        
        # Check if CLI is available
        if command -v daiw &> /dev/null; then
            print_status "CLI command 'daiw' is available"
        else
            print_warning "CLI command 'daiw' not found in PATH"
            print_warning "You may need to add Python scripts directory to PATH"
        fi
    else
        print_error "Music Brain build failed"
        return 1
    fi
    
    echo ""
}

# Build Penta Core
build_penta_core() {
    print_status "Building Penta Core (C++/Python hybrid)..."
    
    # Check if CMakeLists.txt exists
    if [ ! -f "CMakeLists.txt" ]; then
        print_error "CMakeLists.txt not found"
        return 1
    fi
    
    # Create build directory
    print_status "Creating build directory..."
    mkdir -p build
    
    # Configure
    print_status "Configuring with CMake..."
    if ! cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DPENTA_BUILD_PYTHON_BINDINGS=ON \
        -DPENTA_BUILD_JUCE_PLUGIN=ON \
        -DPENTA_ENABLE_SIMD=ON; then
        print_error "CMake configuration failed"
        return 1
    fi
    
    # Build
    print_status "Building (this may take a few minutes)..."
    local num_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    if cmake --build build --config Release -j"$num_cores"; then
        print_success "Penta Core built successfully"
        
        # Run tests if available
        if [ -d "build/tests" ] || [ -d "build/tests_penta-core" ]; then
            print_status "Running tests..."
            if cd build && ctest --output-on-failure && cd ..; then
                print_success "All tests passed"
            else
                print_warning "Some tests failed"
            fi
        fi
    else
        print_error "Penta Core build failed"
        return 1
    fi
    
    echo ""
}

# Main build process
main() {
    check_prerequisites
    
    local build_failed=false
    
    # Build components in recommended order
    if [ "$BUILD_GIT_UPDATER" = true ]; then
        if ! build_git_updater; then
            build_failed=true
        fi
    fi
    
    if [ "$BUILD_MUSIC_BRAIN" = true ]; then
        if ! build_music_brain; then
            build_failed=true
        fi
    fi
    
    if [ "$BUILD_PENTA_CORE" = true ]; then
        if ! build_penta_core; then
            build_failed=true
        fi
    fi
    
    # Summary
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Build Summary${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    if [ "$BUILD_GIT_UPDATER" = true ]; then
        if [ -f "dist/git-update.sh" ]; then
            echo -e "${GREEN}✓${NC} Git Multi-Repository Updater: dist/git-update.sh"
        else
            echo -e "${RED}✗${NC} Git Multi-Repository Updater: Build failed"
        fi
    fi
    
    if [ "$BUILD_MUSIC_BRAIN" = true ]; then
        if python3 -c "import music_brain" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} Music Brain (DAiW): Installed"
        else
            echo -e "${RED}✗${NC} Music Brain (DAiW): Build failed"
        fi
    fi
    
    if [ "$BUILD_PENTA_CORE" = true ]; then
        if [ -d "build" ] && [ -f "build/CMakeCache.txt" ]; then
            echo -e "${GREEN}✓${NC} Penta Core: build/ directory"
        else
            echo -e "${RED}✗${NC} Penta Core: Build failed"
        fi
    fi
    
    echo ""
    
    if [ "$build_failed" = true ]; then
        print_error "Some components failed to build"
        echo "See error messages above for details"
        echo "For help, see MULTI_BUILD.md"
        exit 1
    else
        print_success "All requested components built successfully!"
        echo ""
        echo "Next steps:"
        if [ "$BUILD_GIT_UPDATER" = true ]; then
            echo "  - Run Git Updater: ./dist/git-update.sh"
        fi
        if [ "$BUILD_MUSIC_BRAIN" = true ]; then
            echo "  - Try Music Brain CLI: daiw --help"
        fi
        if [ "$BUILD_PENTA_CORE" = true ]; then
            echo "  - Test Penta Core: python examples/harmony_example.py"
        fi
        echo ""
        echo "For more information, see MULTI_BUILD.md"
    fi
}

# Run main
main
