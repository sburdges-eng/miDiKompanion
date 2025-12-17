#!/bin/bash

# Kelly MIDI Companion - Test Runner Script
# Usage: ./run_tests.sh [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default options
BUILD_DIR="build"
BUILD_TESTS=ON
VERBOSE=OFF
FILTER=""
REPEAT=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --no-build)
            BUILD_TESTS=OFF
            shift
            ;;
        --verbose|-v)
            VERBOSE=ON
            shift
            ;;
        --filter)
            FILTER="$2"
            shift 2
            ;;
        --repeat)
            REPEAT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --build-dir DIR     Build directory (default: build)"
            echo "  --no-build          Skip building tests"
            echo "  --verbose, -v       Verbose output"
            echo "  --filter PATTERN    Run only matching tests"
            echo "  --repeat N          Run tests N times"
            echo "  --help, -h          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}Kelly MIDI Companion - Test Runner${NC}"
echo "=========================================="
echo ""

# Build tests if requested
if [ "$BUILD_TESTS" = "ON" ]; then
    echo -e "${YELLOW}Building tests...${NC}"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    cmake .. -DBUILD_TESTS=ON
    make -j$(nproc 2>/dev/null || echo 4)
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Build failed!${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Build successful!${NC}"
    echo ""
else
    cd "$BUILD_DIR"
fi

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
echo ""

CTEST_ARGS=""
if [ "$VERBOSE" = "ON" ]; then
    CTEST_ARGS="$CTEST_ARGS --verbose"
fi

GTEST_ARGS=""
if [ -n "$FILTER" ]; then
    GTEST_ARGS="$GTEST_ARGS --gtest_filter=$FILTER"
fi
if [ "$REPEAT" -gt 1 ]; then
    GTEST_ARGS="$GTEST_ARGS --gtest_repeat=$REPEAT"
fi

# Run with ctest or directly
if [ -z "$FILTER" ] && [ "$REPEAT" -eq 1 ]; then
    ctest $CTEST_ARGS --output-on-failure
    RESULT=$?
else
    ./tests/KellyTests $GTEST_ARGS
    RESULT=$?
fi

echo ""
if [ $RESULT -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Some tests failed!${NC}"
fi

exit $RESULT
