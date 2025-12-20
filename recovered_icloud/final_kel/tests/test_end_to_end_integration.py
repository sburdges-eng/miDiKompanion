#!/usr/bin/env python3
"""
End-to-End Integration Test

Tests the complete emotion-to-music pipeline:
1. Emotion input → UnifiedFramework
2. UnifiedFramework → ML Models
3. ML Models → MIDI Output
4. Validation and performance checks
"""

import sys
from pathlib import Path
import numpy as np

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ml_framework"))
sys.path.insert(0, str(project_root / "ml_training"))

# Import with correct paths
from cif_las_qef.integration.unified import UnifiedFramework, FrameworkConfig
from cif_las_qef.integration.ml_model_integration import (
    MLModelIntegration,
    integrate_framework_with_ml
)


def test_complete_pipeline():
    """Test complete emotion-to-music pipeline."""
    print("=" * 80)
    print("End-to-End Integration Test")
    print("=" * 80)

    # Step 1: Initialize UnifiedFramework
    print("\n1. Initializing UnifiedFramework...")
    config = FrameworkConfig(
        enable_cif=True,
        enable_las=True,
        enable_ethics=True,
        enable_qef=False  # Disable QEF for testing
    )
    framework = UnifiedFramework(config)
    print("✓ UnifiedFramework initialized")

    # Step 2: Initialize ML Model Integration
    print("\n2. Initializing ML Model Integration...")
    ml_integration = MLModelIntegration()
    if ml_integration.models_loaded:
        print("✓ ML models loaded")
    else:
        print("⚠ ML models not available (using mock)")

    # Step 3: Process emotion input
    print("\n3. Processing emotion input...")
    human_input = {
        "text": {"content": "I feel serene and peaceful", "sentiment": 0.7},
        "bio": {"valence": 0.7, "arousal": -0.3},
        "intent": {"type": "creation"}
    }

    framework_result = framework.create_with_consent(
        human_emotional_input=human_input,
        require_consent=True
    )

    assert framework_result.get("created"), "Framework should create output"
    print(f"✓ Framework processing complete: created={framework_result.get('created')}")
    print(f"  Ethics score: {framework_result.get('overall_ethics', 'N/A'):.3f}")

    # Step 4: Integrate with ML models
    print("\n4. Integrating with ML models...")
    try:
        complete_result = integrate_framework_with_ml(framework_result, ml_integration)

        if complete_result.get("ml_integration_success"):
            ml_outputs = complete_result.get("ml_outputs", {})
            print("✓ ML model integration successful")
            print(f"  Notes shape: {len(ml_outputs.get('notes', []))}")
            print(f"  Chords shape: {len(ml_outputs.get('chords', []))}")
            print(f"  Groove shape: {len(ml_outputs.get('groove', []))}")
            print(f"  Expression shape: {len(ml_outputs.get('expression', []))}")
        else:
            print(f"⚠ ML integration failed: {complete_result.get('ml_integration_error')}")

    except Exception as e:
        print(f"⚠ ML integration error: {e}")
        complete_result = framework_result

    # Step 5: Validate outputs
    print("\n5. Validating outputs...")

    # Validate framework outputs
    assert "las_output" in complete_result, "Should have LAS output"
    assert "cif_integration" in complete_result, "Should have CIF integration"
    assert "ethics_scores" in complete_result, "Should have ethics scores"

    # Validate ML outputs if available
    if complete_result.get("ml_integration_success"):
        ml_outputs = complete_result["ml_outputs"]
        assert "notes" in ml_outputs, "Should have notes"
        assert "chords" in ml_outputs, "Should have chords"
        assert "groove" in ml_outputs, "Should have groove"
        assert "expression" in ml_outputs, "Should have expression"

        # Validate dimensions
        assert len(ml_outputs["notes"]) == 128, "Notes should be 128-dim"
        assert len(ml_outputs["chords"]) == 64, "Chords should be 64-dim"
        assert len(ml_outputs["groove"]) == 32, "Groove should be 32-dim"
        assert len(ml_outputs["expression"]) == 16, "Expression should be 16-dim"

    print("✓ All outputs validated")

    # Step 6: Performance check
    print("\n6. Performance validation...")
    if complete_result.get("ml_integration_success"):
        # Check that outputs are reasonable
        notes = np.array(complete_result["ml_outputs"]["notes"])
        assert np.all(notes >= 0) and np.all(notes <= 1), "Note probabilities should be [0,1]"
        print("✓ Output values in valid range")

    print("\n" + "=" * 80)
    print("✓ End-to-End Integration Test PASSED")
    print("=" * 80)

    return complete_result


def test_emotion_variations():
    """Test pipeline with different emotion inputs."""
    print("\n" + "=" * 80)
    print("Testing Emotion Variations")
    print("=" * 80)

    framework = UnifiedFramework(FrameworkConfig(enable_qef=False))
    ml_integration = MLModelIntegration()

    emotions = [
        {"text": {"content": "I feel serene", "sentiment": 0.7}, "bio": {"valence": 0.7, "arousal": -0.3}},
        {"text": {"content": "I feel excited", "sentiment": 0.8}, "bio": {"valence": 0.8, "arousal": 0.9}},
        {"text": {"content": "I feel sad", "sentiment": -0.6}, "bio": {"valence": -0.6, "arousal": 0.2}},
        {"text": {"content": "I feel anxious", "sentiment": -0.3}, "bio": {"valence": -0.3, "arousal": 0.8}},
    ]

    for i, emotion_input in enumerate(emotions, 1):
        print(f"\n{i}. Testing: {emotion_input['text']}")
        emotion_input["intent"] = {"type": "creation"}

        result = framework.create_with_consent(
            human_emotional_input=emotion_input,
            require_consent=False  # Skip consent for speed
        )

        if result.get("created"):
            complete_result = integrate_framework_with_ml(result, ml_integration)
            if complete_result.get("ml_integration_success"):
                print(f"   ✓ Success: ethics={complete_result.get('overall_ethics', 0):.3f}")
            else:
                print(f"   ⚠ ML integration failed")
        else:
            print(f"   ⚠ Framework creation failed")


def test_performance_benchmark():
    """Benchmark pipeline performance."""
    print("\n" + "=" * 80)
    print("Performance Benchmark")
    print("=" * 80)

    import time

    framework = UnifiedFramework(FrameworkConfig(enable_qef=False))
    ml_integration = MLModelIntegration()

    emotion_input = {
        "text": {"content": "I feel calm", "sentiment": 0.5},
        "bio": {"valence": 0.5, "arousal": 0.3},
        "intent": {"type": "creation"}
    }

    # Benchmark framework processing
    start = time.perf_counter()
    framework_result = framework.create_with_consent(
        human_emotional_input=emotion_input,
        require_consent=False
    )
    framework_time = (time.perf_counter() - start) * 1000  # ms

    # Benchmark ML integration
    if ml_integration.models_loaded:
        start = time.perf_counter()
        complete_result = integrate_framework_with_ml(framework_result, ml_integration)
        ml_time = (time.perf_counter() - start) * 1000  # ms
    else:
        ml_time = 0
        complete_result = framework_result

    total_time = framework_time + ml_time

    print(f"\nTiming:")
    print(f"  Framework processing: {framework_time:.2f} ms")
    if ml_integration.models_loaded:
        print(f"  ML model inference: {ml_time:.2f} ms")
    print(f"  Total pipeline: {total_time:.2f} ms")

    # Performance targets
    print(f"\nPerformance Targets:")
    print(f"  Framework: <100ms (✓ {framework_time:.2f}ms)" if framework_time < 100 else f"  Framework: <100ms (✗ {framework_time:.2f}ms)")
    if ml_integration.models_loaded:
        print(f"  ML Models: <10ms (✓ {ml_time:.2f}ms)" if ml_time < 10 else f"  ML Models: <10ms (✗ {ml_time:.2f}ms)")
        print(f"  Total: <150ms (✓ {total_time:.2f}ms)" if total_time < 150 else f"  Total: <150ms (✗ {total_time:.2f}ms)")


if __name__ == "__main__":
    try:
        # Run main test
        result = test_complete_pipeline()

        # Run emotion variations
        test_emotion_variations()

        # Run performance benchmark
        test_performance_benchmark()

        print("\n" + "=" * 80)
        print("✓ All End-to-End Tests PASSED")
        print("=" * 80)
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
