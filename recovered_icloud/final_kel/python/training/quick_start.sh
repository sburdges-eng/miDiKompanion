#!/bin/bash
# Quick start script for training emotion model

set -e  # Exit on error

echo "=========================================="
echo "Kelly ML Model Training - Quick Start"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if we're in the right directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Step 1: Creating dummy dataset..."
python3 train_emotion_model.py \
    --data dummy_dataset.json \
    --create-dummy \
    --dummy-samples 1000

echo ""
echo "Step 2: Training model (this may take a few minutes)..."
python3 train_emotion_model.py \
    --data dummy_dataset.json \
    --output emotion_model.pt \
    --epochs 10 \
    --batch-size 32 \
    --learning-rate 0.001

echo ""
echo "Step 3: Testing model..."
python3 test_emotion_model.py --model emotion_model.pt

echo ""
echo "Step 4: Exporting to RTNeural format..."
python3 export_to_rtneural.py \
    --model emotion_model.pt \
    --output emotion_model.json

echo ""
echo "=========================================="
echo "✓ Training complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy emotion_model.json to plugin data directory:"
echo "   cp emotion_model.json ../../data/emotion_model.json"
echo ""
echo "2. Build plugin with RTNeural enabled:"
echo "   cmake -B build -DENABLE_RTNEURAL=ON"
echo "   cmake --build build"
echo ""
echo "3. Enable ML inference in plugin settings"
echo ""

