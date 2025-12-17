#!/bin/bash
# Kelly MIDI Companion - Training Pipeline Setup Script

echo "=========================================="
echo "Kelly MIDI Companion ML Training Pipeline"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To train all models, run:"
echo "  python scripts/train_all_models.py --output ../models"
echo ""
echo "For help:"
echo "  python scripts/train_all_models.py --help"
echo ""
