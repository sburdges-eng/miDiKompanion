#!/bin/bash
# =============================================================================
# Latency Test Runner
# =============================================================================
# Runs comprehensive latency tests for the iDAW system
# =============================================================================

set -euo pipefail

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly CYAN='\033[0;36m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ${NC} $1"; }
log_success() { echo -e "${GREEN}✓${NC} $1"; }
log_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║              iDAW Latency Test Suite                         ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

cd "$PROJECT_ROOT"

# Check requirements
log_info "Checking requirements..."

check_requirement() {
    if command -v "$1" &> /dev/null; then
        log_success "$1 found"
        return 0
    else
        log_error "$1 not found"
        return 1
    fi
}

MISSING=0
check_requirement python3 || MISSING=1
# Check pytest via Python module
if ! python3 -m pytest --version &>/dev/null; then
    log_warning "pytest not found, installing..."
    pip3 install pytest requests --quiet
fi

# Check requests via Python import
if ! python3 -c "import requests" 2>/dev/null; then
    log_warning "requests not found, installing..."
    pip3 install requests --quiet
fi

if [ $MISSING -eq 1 ]; then
    log_error "Missing required tools"
    exit 1
fi

# Check if Python API dependencies are installed
log_info "Checking Python dependencies..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    log_warning "FastAPI not installed, installing dependencies..."
    pip3 install fastapi uvicorn pydantic --quiet || {
        log_error "Failed to install dependencies"
        exit 1
    }
fi

# Run latency tests
log_info "Running latency tests..."
echo ""

if python3 -m pytest "$PROJECT_ROOT/tests_music-brain/test_integration_latency.py" -v -s; then
    log_success "All latency tests passed!"
    echo ""
    exit 0
else
    log_error "Some latency tests failed"
    echo ""
    exit 1
fi
