#!/bin/bash
# Setup script for downloading and organizing datasets for Kelly ML training
# This script helps automate dataset preparation

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATASETS_DIR="$PROJECT_ROOT/datasets"

echo "=========================================="
echo "Kelly ML Dataset Setup"
echo "=========================================="
echo ""
echo "This script will help you set up datasets for training."
echo "Project root: $PROJECT_ROOT"
echo "Datasets directory: $DATASETS_DIR"
echo ""

# Create datasets directory structure
mkdir -p "$DATASETS_DIR"/{audio,midi,chords,dynamics_midi,drums}
mkdir -p "$DATASETS_DIR/training"/{audio,midi,chords,dynamics_midi,drums}

echo "✓ Created dataset directory structure"
echo ""

# Check for existing datasets
echo "Checking for existing datasets..."
echo ""

# Check DEAM
if [ -d "$DATASETS_DIR/deam" ] && [ "$(find "$DATASETS_DIR/deam" -name "*.wav" -o -name "*.mp3" | wc -l)" -gt 0 ]; then
    echo "✓ DEAM dataset found"
else
    echo "✗ DEAM dataset not found"
    echo "  Download from: https://cvml.unige.ch/databases/DEAM/"
    echo "  Requires registration"
fi

# Check Lakh MIDI
if [ -d "$DATASETS_DIR/lakh_midi" ] && [ "$(find "$DATASETS_DIR/lakh_midi" -name "*.mid" -o -name "*.midi" | wc -l)" -gt 100 ]; then
    echo "✓ Lakh MIDI dataset found"
else
    echo "✗ Lakh MIDI dataset not found"
    echo "  Download from: https://colinraffel.com/projects/lmd/"
    echo "  Recommended: lmd_clean.tar.gz (~1.7GB)"
fi

# Check MAESTRO
if [ -d "$DATASETS_DIR/maestro" ] && [ "$(find "$DATASETS_DIR/maestro" -name "*.mid" -o -name "*.midi" | wc -l)" -gt 0 ]; then
    echo "✓ MAESTRO dataset found"
else
    echo "✗ MAESTRO dataset not found"
    echo "  Download from: https://magenta.tensorflow.org/datasets/maestro"
    echo "  Or use: pip install tensorflow-datasets && python -c 'import tensorflow_datasets as tfds; tfds.load(\"maestro\")'"
fi

# Check Groove MIDI
if [ -d "$DATASETS_DIR/groove" ] && [ "$(find "$DATASETS_DIR/groove" -name "*.mid" -o -name "*.midi" | wc -l)" -gt 0 ]; then
    echo "✓ Groove MIDI dataset found"
else
    echo "✗ Groove MIDI dataset not found"
    echo "  Download from: https://magenta.tensorflow.org/datasets/groove"
    echo "  Or use: pip install tensorflow-datasets && python -c 'import tensorflow_datasets as tfds; tfds.load(\"groove\")'"
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Download missing datasets (see links above)"
echo "2. Extract datasets to: $DATASETS_DIR"
echo "3. Run dataset preparation:"
echo "   python $SCRIPT_DIR/prepare_datasets.py --datasets-dir $DATASETS_DIR"
echo "4. Organize datasets for training:"
echo "   python $SCRIPT_DIR/download_datasets.py --datasets-dir $DATASETS_DIR --organize"
echo "5. Start training:"
echo "   python $SCRIPT_DIR/train_all_models.py --datasets-dir $DATASETS_DIR"
echo ""
echo "Or use synthetic data for testing:"
echo "   python $SCRIPT_DIR/train_all_models.py --synthetic"
echo ""
