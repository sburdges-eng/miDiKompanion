#!/bin/bash
# Install PyTorch and dependencies for model training

set -e

echo "Installing PyTorch training dependencies..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install PyTorch (CPU version)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio

# Install other dependencies
echo "Installing NumPy, tqdm..."
pip install numpy>=1.24.0 tqdm>=4.65.0

# Install type stubs for mypy
echo "Installing type stubs for mypy..."
pip install types-torch types-numpy || echo "Warning: Type stubs installation failed (optional)"

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} installed')"
python3 -c "import numpy; print(f'✓ NumPy {numpy.__version__} installed')"
python3 -c "import tqdm; print(f'✓ tqdm installed')"

echo ""
echo "✓ All dependencies installed successfully!"
echo ""
echo "Next steps:"
echo "  1. Create test model: python3 create_test_model.py"
echo "  2. Train model: python3 train_emotion_model.py --data dataset.json"
echo "  3. Export to RTNeural: python3 export_to_rtneural.py --model model.pt"
