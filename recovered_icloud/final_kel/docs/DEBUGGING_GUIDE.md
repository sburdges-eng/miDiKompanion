# Debugging Guide

## Common Issues and Solutions

### 1. Module Import Errors

**Issue**: `ModuleNotFoundError: No module named 'matplotlib'`

**Solution**: Matplotlib is now optional. The framework will work without it, but plotting features will be disabled.

**Fixed**: ✅ Made matplotlib import optional in `simulation.py`

### 2. ML Model Import Errors

**Issue**: `ModuleNotFoundError: No module named 'ml_framework'`

**Solution**: Ensure Python path is set correctly:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "ml_framework"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ml_training"))
```

### 3. Model Loading Issues

**Issue**: Models not loading from checkpoints

**Solution**: 
- Verify checkpoints exist: `ls ml_training/trained_models/checkpoints/*.pt`
- Check model paths in `MLModelIntegration.__init__()`
- Ensure PyTorch is installed: `pip install torch`

### 4. Test Failures

**Issue**: Tests failing due to missing dependencies

**Solution**:
- Install required packages: `pip install -r requirements.txt`
- Check Python version (3.11+ recommended)
- Verify paths are set correctly in test files

### 5. Docker Build Failures

**Issue**: Docker build failing

**Solution**:
- Check `requirements.txt` exists (created during deployment)
- Verify Dockerfile syntax
- Check Docker daemon is running

## Debugging Commands

### Verify Installation

```bash
# Check Python version
python3 --version

# Check installed packages
pip list | grep -E "(torch|numpy|matplotlib)"

# Verify models exist
ls -lh ml_training/trained_models/*.json

# Check deployment
ls -lh ml_training/deployment/models/*.json
```

### Test Imports

```bash
# Test framework import
cd ml_framework
python3 -c "from cif_las_qef.integration.unified import UnifiedFramework; print('OK')"

# Test ML integration import
python3 -c "from cif_las_qef.integration.ml_model_integration import MLModelIntegration; print('OK')"
```

### Run Tests

```bash
# Run end-to-end test
cd tests
python3 test_end_to_end_integration.py

# Run model validation
cd ml_training
python3 validate_models.py trained_models/

# Benchmark performance
python3 benchmark_inference.py trained_models/emotionrecognizer.json
```

## Known Issues

### Matplotlib Optional

Matplotlib is now optional. Plotting methods will return `None` if matplotlib is not available, but the framework will function normally.

### ML Models Optional

ML model integration is optional. If models aren't available, the integration will work in limited mode.

## Logging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Debugging

### Check Latency

```bash
cd ml_training
python3 benchmark_inference.py trained_models/emotionrecognizer.json --iterations 1000
```

### Check Memory

```python
import tracemalloc
tracemalloc.start()
# ... your code ...
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.2f} MB, Peak: {peak / 1024 / 1024:.2f} MB")
```

## Fixes Applied

### 2025-12-18

1. ✅ Made matplotlib import optional in `simulation.py`
2. ✅ Fixed import paths in test files
3. ✅ Added error handling for missing dependencies
4. ✅ Created debugging guide

---

**Last Updated**: 2025-12-18
