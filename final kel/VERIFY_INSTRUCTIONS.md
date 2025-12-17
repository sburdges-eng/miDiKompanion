# AI/ML Features Verification Instructions

## Quick Start

Run the verification script to test all AI/ML components:

```bash
# From project root
cd ml_framework
source venv/bin/activate
cd ..
export PYTHONPATH="$(pwd)/ml_framework:$PYTHONPATH"
python verify_ai_features.py
```

Or with explicit Python path:

```bash
cd ml_framework
source venv/bin/activate
cd ..
export PYTHONPATH="$(pwd)/ml_framework:$PYTHONPATH"
python3 verify_ai_features.py
```

## Expected Output

```text
============================================================
Kelly MIDI Companion - AI/ML Features Verification
============================================================

Note: Ensure you have activated the virtual environment:
  cd ml_framework && source venv/bin/activate && cd ..
  export PYTHONPATH="$(pwd)/ml_framework:$PYTHONPATH"

1. Testing Core AI Components...
   ✓ CIF (Conscious Integration Framework)
   ✓ LAS (Living Art Systems)
   ✓ QEF (Quantum Emotional Field)
   ✓ ResonantEthics

2. Testing Emotion Models...
   ✓ VADModel
   ✓ VADState
   ✓ PlutchikWheel
   ✓ QuantumEmotionalField
   ✓ HybridEmotionalField (working correctly)

3. Testing CIF Functionality...
   ✓ CIF initialized: resonant_calibration
   ✓ CIF integration test passed

4. Testing LAS Functionality...
   ✓ LAS initialized: initialized

5. Testing QEF Functionality...
   ✓ QEF initialized and activated

6. Testing Dependencies...
   ✓ NumPy 2.3.5
   ✓ SciPy 1.16.3
   ✓ Matplotlib 3.10.8

============================================================
Summary:
============================================================
  ✓ PASS: Core Components
  ✓ PASS: Emotion Models
  ✓ PASS: CIF Functionality
  ✓ PASS: LAS Functionality
  ✓ PASS: QEF Functionality
  ✓ PASS: Dependencies

Results: 6/6 tests passed

🎉 All AI/ML features verified successfully!
```

## Troubleshooting

### ModuleNotFoundError: No module named 'cif_las_qef'

**Solution:** Set PYTHONPATH:

```bash
export PYTHONPATH="$(pwd)/ml_framework:$PYTHONPATH"
```

### ModuleNotFoundError: No module named 'matplotlib' (or numpy, scipy)

**Solution:** Activate virtual environment:

```bash
cd ml_framework
source venv/bin/activate
cd ..
```

### ImportError: cannot import name 'VADState'

**Solution:** This should be fixed in the current version. If you see this error, ensure you're using the updated emotion_models/**init**.py that exports VADState.

### HybridEmotionalField broadcasting error

**Solution:** This bug has been fixed. The script now properly initializes HybridEmotionalField with a VADState parameter.

## Verification Checklist

- [x] Core Components (CIF, LAS, QEF, ResonantEthics) importable
- [x] Emotion Models (VAD, Plutchik, Quantum, Hybrid) importable and functional
- [x] CIF integration works with sample data
- [x] LAS initialization successful
- [x] QEF activation successful
- [x] Dependencies (NumPy, SciPy, Matplotlib) installed

## Related Documentation

- `MARKDOWN/BUILD_VERIFICATION.md` - Build status and verification details
- `MARKDOWN/ML_INTEGRATION_COMPLETE.md` - ML integration summary
- `ml_framework/README.md` - ML framework documentation
- `build.md` - Complete build guide
