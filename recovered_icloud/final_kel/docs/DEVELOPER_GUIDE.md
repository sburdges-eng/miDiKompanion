# iDAW Developer Guide

## Architecture Overview

See `docs/SYSTEM_ARCHITECTURE.md` for complete architecture documentation.

## Key Components

### Emotion Frameworks
- **CIF**: Conscious Integration Framework (`ml_framework/cif_las_qef/cif/`)
- **LAS**: Living Art Systems (`ml_framework/cif_las_qef/las/`)
- **QEF**: Quantum Emotional Field (`ml_framework/cif_las_qef/qef/`)
- **Ethics**: Resonant Ethics Framework (`ml_framework/cif_las_qef/ethics/`)

### ML Models
- **Location**: `ml_training/trained_models/`
- **Format**: RTNeural JSON, PyTorch checkpoints
- **Documentation**: `docs/ML_MODELS_ARCHITECTURE.md`

### Music Brain
- **Location**: `music_brain/`
- **Features**: Intent-driven composition, rule-breaking, groove extraction

### Plugins
- **Location**: `iDAW_Core/plugins/`
- **Framework**: JUCE 8
- **Formats**: VST3, AU, CLAP

## API Reference

### UnifiedFramework

```python
from ml_framework.cif_las_qef.integration.unified import UnifiedFramework, FrameworkConfig

config = FrameworkConfig(
    enable_cif=True,
    enable_las=True,
    enable_ethics=True,
    enable_qef=True
)
framework = UnifiedFramework(config)

result = framework.create_with_consent(
    human_emotional_input={
        "text": "I feel serene",
        "valence": 0.7,
        "arousal": -0.3
    }
)
```

### ML Models

```python
import torch
from ml_training.train_all_models import EmotionRecognizer, MelodyTransformer

# Load model
model = EmotionRecognizer()
model.load_state_dict(torch.load("trained_models/checkpoints/EmotionRecognizer_best.pt")["model_state_dict"])
model.eval()

# Inference
input_tensor = torch.randn(1, 128)  # Audio features
output = model(input_tensor)  # 64-dim emotion embedding
```

## Contributing

### Code Style

- Python: PEP 8, max line length 127
- C++: Follow JUCE style guidelines
- Use type hints in Python

### Testing

- Run unit tests: `pytest tests/`
- Validate models: `python ml_training/validate_models.py`
- Benchmark performance: `python ml_training/benchmark_inference.py`

### Pull Requests

1. Fork repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit PR

## Build Instructions

### Python Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Plugin Build

```bash
cd iDAW_Core
cmake -B build
cmake --build build
```

## Performance Guidelines

### RT-Safety

- **No allocations in audio thread**
- **Use lock-free data structures**
- **Pre-allocate memory**
- **Keep inference <10ms**

### Memory Management

- **Models <4MB each**
- **Total memory <50MB per plugin**
- **Use memory pools**

## Documentation

- Architecture: `docs/SYSTEM_ARCHITECTURE.md`
- ML Models: `docs/ML_MODELS_ARCHITECTURE.md`
- Integration: `docs/INTEGRATION_POINTS.md`
- Component Inventory: `docs/COMPONENT_INVENTORY.md`

---

**Version**: 1.0  
**Last Updated**: 2025-12-18
