#!/bin/bash
# Kelly MIDI Companion - Workspace Setup Script
# Sets up Python virtual environments and installs dependencies for ML framework and Python utilities

set -e  # Exit on error

echo "=========================================="
echo "Kelly MIDI Companion - Workspace Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.8 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Found Python ${PYTHON_VERSION}"

# Check CMake version
echo ""
echo "Checking CMake installation..."
if ! command -v cmake &> /dev/null; then
    echo "WARNING: cmake not found. C++ build will fail without CMake 3.22+"
    echo "Install with: brew install cmake (macOS) or sudo apt-get install cmake (Linux)"
else
    CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
    echo "Found CMake ${CMAKE_VERSION}"
fi

# Check C++ compiler
echo ""
echo "Checking C++ compiler..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v clang++ &> /dev/null; then
        echo "WARNING: clang++ not found. Install Xcode Command Line Tools: xcode-select --install"
    else
        COMPILER_VERSION=$(clang++ --version | head -n1)
        echo "Found: ${COMPILER_VERSION}"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
        echo "WARNING: No C++ compiler found. Install with: sudo apt-get install build-essential"
    else
        if command -v g++ &> /dev/null; then
            COMPILER_VERSION=$(g++ --version | head -n1)
            echo "Found: ${COMPILER_VERSION}"
        fi
    fi
fi

echo ""
echo "=========================================="
echo "Setting up Python environments..."
echo "=========================================="

# ML Framework setup
echo ""
echo "1. Setting up ML Framework environment..."
cd ml_framework

if [ ! -d "venv" ]; then
    echo "   Creating virtual environment..."
    python3 -m venv venv
else
    echo "   Virtual environment already exists"
fi

echo "   Activating virtual environment..."
source venv/bin/activate

echo "   Upgrading pip..."
pip install --upgrade pip --quiet

echo "   Installing ML framework dependencies..."
pip install -r requirements.txt

echo "   ML Framework setup complete!"
deactivate
cd ..

# Python utilities setup
echo ""
echo "2. Setting up Python Utilities environment..."
cd python

if [ ! -d "venv" ]; then
    echo "   Creating virtual environment..."
    python3 -m venv venv
else
    echo "   Virtual environment already exists"
fi

echo "   Activating virtual environment..."
source venv/bin/activate

echo "   Upgrading pip..."
pip install --upgrade pip --quiet

echo "   Installing Python utilities dependencies..."
pip install -r requirements.txt

echo "   Python Utilities setup complete!"
deactivate
cd ..

echo ""
echo "=========================================="
echo "Workspace setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Build the plugin:"
echo "   cmake -B build -DBUILD_PYTHON_BRIDGE=ON -DCMAKE_BUILD_TYPE=Release"
echo "   cmake --build build --config Release"
echo ""
echo "2. Test ML framework:"
echo "   cd ml_framework && source venv/bin/activate && python examples/basic_usage.py"
echo ""
echo "3. Test Python bridge (after building):"
echo "   cd python && source venv/bin/activate && python -c \"import kelly_bridge; print('Bridge OK')\""
echo ""
