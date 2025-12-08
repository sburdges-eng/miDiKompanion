#!/bin/bash
# Test runner script for iDAW standalone app tests

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  iDAW Test Suite Runner                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Test results
RUST_TESTS_PASSED=0
PYTHON_TESTS_PASSED=0
TOTAL_TESTS=0

# Run Rust tests
echo -e "${YELLOW}[1/2] Running Rust tests...${NC}"
cd "$(dirname "$0")/src-tauri"

if cargo test --lib 2>&1 | tee /tmp/rust_tests.log; then
    echo -e "${GREEN}✓ Rust tests passed${NC}"
    RUST_TESTS_PASSED=1
else
    echo -e "${RED}✗ Rust tests failed${NC}"
    RUST_TESTS_PASSED=0
fi

# Run Python tests
echo ""
echo -e "${YELLOW}[2/2] Running Python tests...${NC}"
cd "$(dirname "$0")"

# Test embedded launcher
if python3 -m pytest tests_music-brain/test_embedded_launcher.py -v 2>&1 | tee /tmp/python_launcher_tests.log; then
    echo -e "${GREEN}✓ Embedded launcher tests passed${NC}"
    PYTHON_TESTS_PASSED=1
else
    echo -e "${RED}✗ Embedded launcher tests failed${NC}"
    PYTHON_TESTS_PASSED=0
fi

# Test build script
if python3 -m pytest tests_music-brain/test_build_script.py -v 2>&1 | tee /tmp/python_build_tests.log; then
    echo -e "${GREEN}✓ Build script tests passed${NC}"
    PYTHON_TESTS_PASSED=$((PYTHON_TESTS_PASSED + 1))
else
    echo -e "${RED}✗ Build script tests failed${NC}"
fi

# Summary
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                      Test Summary                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

if [ $RUST_TESTS_PASSED -eq 1 ]; then
    echo -e "${GREEN}✓ Rust tests: PASSED${NC}"
else
    echo -e "${RED}✗ Rust tests: FAILED${NC}"
fi

if [ $PYTHON_TESTS_PASSED -ge 1 ]; then
    echo -e "${GREEN}✓ Python tests: PASSED${NC}"
else
    echo -e "${RED}✗ Python tests: FAILED${NC}"
fi

echo ""
if [ $RUST_TESTS_PASSED -eq 1 ] && [ $PYTHON_TESTS_PASSED -ge 1 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Check logs above.${NC}"
    exit 1
fi
