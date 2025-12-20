#!/usr/bin/env python3
"""
End-to-End Test: Emotion Input → ML Models → Music Brain → MIDI Output

Tests the complete pipeline:
1. Emotion input (text/valence/arousal)
2. UnifiedFramework emotion processing
3. ML model inference (all 5 models)
4. Music Brain intent generation
5. MIDI output generation
"""

import sys
from pathlib import Path
import numpy as np
import json
from typing import Dict, Any

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_framework.cif_las_qef.integration.ml_music_brain_bridge import (
    MLMusicBrainBridge,
    generate_music_from_emotion
)


def test_emotion_to_music_pipeline():
    """Test complete emotion-to-music pipeline."""
    print("=" * 70)
    print("END-TO-END TEST: Emotion → ML Models → Music Brain → MIDI")
    print("=" * 70)

    # Test cases
    test_cases = [
        {
            "name": "Calm and Peaceful",
            "input": {
                "text": "I feel calm and peaceful",
                "valence": 0.7,
                "arousal": -0.3
            }
        },
        {
            "name": "Grief and Loss",
            "input": {
                "text": "I'm processing grief and loss",
                "valence": -0.5,
                "arousal": -0.2
            }
        },
        {
            "name": "Energetic Joy",
            "input": {
                "text": "I feel energetic and joyful",
                "valence": 0.8,
                "arousal": 0.7
            }
        },
        {
            "name": "Tense Anxiety",
            "input": {
                "text": "I'm feeling tense and anxious",
                "valence": -0.3,
                "arousal": 0.8
            }
        }
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'='*70}")

        try:
            # Create bridge
            bridge = MLMusicBrainBridge()

            # Run pipeline
            output_path = f"test_output_{i}.mid"
            result = bridge.emotion_to_music(
                emotion_input=test_case["input"],
                output_path=output_path,
                require_consent=True
            )

            # Validate result
            assert result is not None, "Result should not be None"
            assert hasattr(result, "emotional_state"), "Should have emotional_state"
            assert hasattr(result, "musical_params"), "Should have musical_params"
            assert hasattr(result, "mixer_params"), "Should have mixer_params"

            # Print summary
            print("\n✓ Pipeline completed successfully")
            print(f"\n{result.summary()}")

            results.append({
                "test": test_case["name"],
                "status": "PASS",
                "result": result.to_dict()
            })

        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "test": test_case["name"],
                "status": "FAIL",
                "error": str(e)
            })

    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    print(f"Total tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    for result in results:
        status_symbol = "✓" if result["status"] == "PASS" else "✗"
        print(f"  {status_symbol} {result['test']}: {result['status']}")

    # Save results
    results_path = Path(__file__).parent / "e2e_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")

    return passed == len(results)


def test_ml_model_inference():
    """Test ML model inference separately."""
    print("\n" + "=" * 70)
    print("ML MODEL INFERENCE TEST")
    print("=" * 70)

    from ml_framework.cif_las_qef.integration.ml_music_brain_bridge import MLModelLoader

    loader = MLModelLoader()
    loader.load_all_models()

    # Test EmotionRecognizer
    print("\n1. Testing EmotionRecognizer...")
    audio_features = np.random.randn(128).astype(np.float32)
    emotion_embedding = loader.run_inference("EmotionRecognizer", audio_features)
    assert emotion_embedding.shape == (64,), f"Expected (64,), got {emotion_embedding.shape}"
    print(f"   ✓ Output shape: {emotion_embedding.shape}")

    # Test MelodyTransformer
    print("\n2. Testing MelodyTransformer...")
    melody_probs = loader.run_inference("MelodyTransformer", emotion_embedding)
    assert melody_probs.shape == (128,), f"Expected (128,), got {melody_probs.shape}"
    print(f"   ✓ Output shape: {melody_probs.shape}")

    # Test HarmonyPredictor
    print("\n3. Testing HarmonyPredictor...")
    context_vec = np.random.randn(128).astype(np.float32)
    harmony_probs = loader.run_inference("HarmonyPredictor", context_vec)
    assert harmony_probs.shape == (64,), f"Expected (64,), got {harmony_probs.shape}"
    print(f"   ✓ Output shape: {harmony_probs.shape}")

    # Test DynamicsEngine
    print("\n4. Testing DynamicsEngine...")
    dynamics_context = np.random.randn(32).astype(np.float32)
    dynamics = loader.run_inference("DynamicsEngine", dynamics_context)
    assert dynamics.shape == (16,), f"Expected (16,), got {dynamics.shape}"
    print(f"   ✓ Output shape: {dynamics.shape}")

    # Test GroovePredictor
    print("\n5. Testing GroovePredictor...")
    groove = loader.run_inference("GroovePredictor", emotion_embedding)
    assert groove.shape == (32,), f"Expected (32,), got {groove.shape}"
    print(f"   ✓ Output shape: {groove.shape}")

    print("\n✓ All ML models working correctly")


def test_unified_framework_integration():
    """Test UnifiedFramework integration."""
    print("\n" + "=" * 70)
    print("UNIFIED FRAMEWORK INTEGRATION TEST")
    print("=" * 70)

    from ml_framework.cif_las_qef.integration.unified import UnifiedFramework, FrameworkConfig

    config = FrameworkConfig(
        enable_cif=True,
        enable_las=True,
        enable_ethics=True,
        enable_qef=True
    )

    framework = UnifiedFramework(config)

    emotion_input = {
        "text": "I feel serene",
        "valence": 0.7,
        "arousal": -0.3
    }

    result = framework.create_with_consent(
        human_emotional_input=emotion_input,
        require_consent=True
    )

    assert result is not None, "Result should not be None"
    assert "created" in result, "Should have 'created' field"

    print(f"✓ Framework integration: created={result.get('created')}")
    if result.get("created"):
        print(f"  Ethics score: {result.get('overall_ethics', 'N/A')}")


def test_music_brain_integration():
    """Test Music Brain integration."""
    print("\n" + "=" * 70)
    print("MUSIC BRAIN INTEGRATION TEST")
    print("=" * 70)

    from music_brain.emotion_api import MusicBrain

    brain = MusicBrain()

    # Test text-to-emotion
    emotion_text = "I feel calm and peaceful"
    result = brain.generate_from_text(emotion_text)

    assert result is not None, "Result should not be None"
    assert hasattr(result, "emotional_state"), "Should have emotional_state"
    assert hasattr(result, "musical_params"), "Should have musical_params"

    print(f"✓ Music Brain integration successful")
    print(f"  Primary emotion: {result.emotional_state.primary_emotion}")
    print(f"  Tempo: {result.musical_params.tempo_suggested} BPM")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RUNNING ALL END-TO-END TESTS")
    print("=" * 70)

    all_passed = True

    # Test 1: ML Model Inference
    try:
        test_ml_model_inference()
    except Exception as e:
        print(f"\n✗ ML Model Inference test failed: {e}")
        all_passed = False

    # Test 2: Unified Framework Integration
    try:
        test_unified_framework_integration()
    except Exception as e:
        print(f"\n✗ Unified Framework test failed: {e}")
        all_passed = False

    # Test 3: Music Brain Integration
    try:
        test_music_brain_integration()
    except Exception as e:
        print(f"\n✗ Music Brain test failed: {e}")
        all_passed = False

    # Test 4: Full Pipeline
    try:
        pipeline_passed = test_emotion_to_music_pipeline()
        all_passed = all_passed and pipeline_passed
    except Exception as e:
        print(f"\n✗ Full pipeline test failed: {e}")
        all_passed = False

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)

    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
