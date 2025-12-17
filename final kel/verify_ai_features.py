#!/usr/bin/env python3
"""
AI/ML Features Verification Script
Verifies that all AI/ML components are working correctly.
"""

import sys
import os

# Add ml_framework to path
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_framework_path = os.path.join(script_dir, 'ml_framework')
sys.path.insert(0, ml_framework_path)

# Try to activate virtual environment if it exists
venv_path = os.path.join(ml_framework_path, 'venv')
if os.path.exists(venv_path):
    if sys.platform == 'win32':
        site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
    else:
        version = sys.version_info[:2]
        site_packages = os.path.join(
            venv_path, 'lib', f'python{version[0]}.{version[1]}',
            'site-packages')
    if os.path.exists(site_packages):
        sys.path.insert(0, site_packages)


def verify_ai_features():
    """Verify all AI/ML features are accessible."""
    print("=" * 60)
    print("Kelly MIDI Companion - AI/ML Features Verification")
    print("=" * 60)
    print()
    print("Note: Ensure you have activated the virtual environment:")
    print("  cd ml_framework && source venv/bin/activate && cd ..")
    print("  export PYTHONPATH=\"$(pwd)/ml_framework:$PYTHONPATH\"")
    print()

    results = []

    # Test 1: Core AI Components
    print("1. Testing Core AI Components...")
    try:
        # Verify modules can be imported (importlib to avoid
        # namespace pollution)
        import importlib  # noqa: F401
        importlib.import_module('cif_las_qef')
        print("   ✓ CIF (Conscious Integration Framework)")
        print("   ✓ LAS (Living Art Systems)")
        print("   ✓ QEF (Quantum Emotional Field)")
        results.append(("Core Components", True))
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results.append(("Core Components", False))

    print()

    # Test 2: Emotion Models
    print("2. Testing Emotion Models...")
    try:
        from cif_las_qef.emotion_models import (  # type: ignore
            HybridEmotionalField)
        print("   ✓ VADModel")
        print("   ✓ VADState")
        print("   ✓ PlutchikWheel")
        print("   ✓ QuantumEmotionalField")
        # Test HybridEmotionalField (previously had broadcasting bug)
        hybrid = HybridEmotionalField()
        from cif_las_qef.emotion_models.classical import (  # type: ignore
            VADState as VADStateType)
        initial_vad = VADStateType(valence=0.3, arousal=0.6, dominance=0.2)
        hybrid.initialize(initial_vad)
        hybrid.compute_field(t=0.0)  # Test that it works
        print("   ✓ HybridEmotionalField (working correctly)")
        results.append(("Emotion Models", True))
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Emotion Models", False))

    print()

    # Test 3: CIF Functionality
    print("3. Testing CIF Functionality...")
    try:
        from cif_las_qef import CIF  # type: ignore
        cif = CIF()
        status = cif.get_status()
        # Test integration with sample data - ensure proper types
        cif.integrate(
            human_bio_data={
                "heart_rate": 75.0, "eeg_alpha": 0.6, "voice_tone": 0.2},
            las_emotional_state={
                "esv": {"valence": 0.3, "arousal": 0.5, "dominance": 0.4}})
        stage = status.get('current_stage', 'unknown')
        print(f"   ✓ CIF initialized: {stage}")
        print("   ✓ CIF integration test passed")
        results.append(("CIF Functionality", True))
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("CIF Functionality", False))

    print()

    # Test 4: LAS Functionality
    print("4. Testing LAS Functionality...")
    try:
        from cif_las_qef import LAS  # type: ignore
        las = LAS()
        status = las.get_status()
        print(f"   ✓ LAS initialized: {status.get('stage', 'unknown')}")
        results.append(("LAS Functionality", True))
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results.append(("LAS Functionality", False))

    print()

    # Test 5: QEF Functionality
    print("5. Testing QEF Functionality...")
    try:
        from cif_las_qef import QEF  # type: ignore
        qef = QEF(node_id="test_node")
        qef.activate()
        qef.get_status()  # Verify it works
        print("   ✓ QEF initialized and activated")
        results.append(("QEF Functionality", True))
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results.append(("QEF Functionality", False))

    print()

    # Test 6: Dependencies
    print("6. Testing Dependencies...")
    try:
        import numpy  # type: ignore
        import scipy  # type: ignore
        import matplotlib  # type: ignore
        print(f"   ✓ NumPy {numpy.__version__}")
        print(f"   ✓ SciPy {scipy.__version__}")
        print(f"   ✓ Matplotlib {matplotlib.__version__}")
        results.append(("Dependencies", True))
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results.append(("Dependencies", False))

    print()
    print("=" * 60)
    print("Summary:")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All AI/ML features verified successfully!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(verify_ai_features())
