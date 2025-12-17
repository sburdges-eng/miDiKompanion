# Installing PyTorch for Model Training

PyTorch is required for training and exporting emotion models. Here's how to install it:

## Quick Install

```bash
# For CPU only (recommended for testing)
pip install torch torchvision torchaudio

# For GPU support (if you have CUDA)
# Visit https://pytorch.org/get-started/locally/ for GPU installation

# Install type stubs for mypy (optional but recommended)
pip install types-torch types-numpy
```

## Install All Training Dependencies

```bash
# Install from requirements file
pip install -r requirements.txt

# Or install with type stubs
pip install torch numpy tqdm types-torch types-numpy
```

## Verify Installation

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

## After Installation

Once PyTorch is installed, you can:

1. **Create test model**:

   ```bash
   python3 create_test_model.py --output test_emotion_model.pt
   ```

2. **Export to RTNeural**:

   ```bash
   python3 export_to_rtneural.py --model test_emotion_model.pt --output emotion_model.json
   ```

3. **Deploy to plugin**:

   ```bash
   cp emotion_model.json ../../data/emotion_model.json
   ```

## Alternative: Use Pre-trained Model

If you don't want to install PyTorch, you can:

- Use a pre-trained model from another source
- Manually create the RTNeural JSON format
- Wait for a pre-trained model to be provided

## System Requirements

- Python 3.8+
- pip (Python package manager)
- ~2GB disk space for PyTorch installation
