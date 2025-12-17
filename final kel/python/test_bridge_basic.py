#!/usr/bin/env python3
"""
Basic test for kelly_bridge module.
Tests core functionality of the Python bridge.
"""

import sys
sys.path.insert(0, '.')

import kelly_bridge

def test_emotion_thesaurus():
    """Test EmotionThesaurus functionality."""
    print("Testing EmotionThesaurus...")

    # Get thesaurus instance from IntentPipeline (no public constructor)
    pipeline = kelly_bridge.IntentPipeline()
    thesaurus = pipeline.thesaurus()
    print(f"  ✓ Thesaurus accessed, size: {thesaurus.size()}")

    # Test find_by_id
    emotion = thesaurus.find_by_id(1)
    if emotion:
        print(f"  ✓ find_by_id(1): {emotion.name} (valence={emotion.valence:.2f})")
    else:
        print("  ✗ find_by_id(1) returned None")

    # Test find_by_name
    emotion = thesaurus.find_by_name("joy")
    if emotion:
        print(f"  ✓ find_by_name('joy'): {emotion.name} (id={emotion.id})")
    else:
        print("  ✗ find_by_name('joy') returned None")

    # Test find_nearest
    emotion = thesaurus.find_nearest(0.5, 0.7, 0.8)
    print(f"  ✓ find_nearest(0.5, 0.7, 0.8): {emotion.name}")

    # Test find_by_category
    joy_emotions = thesaurus.find_by_category(kelly_bridge.EmotionCategory.Joy)
    print(f"  ✓ find_by_category(Joy): {len(joy_emotions)} emotions")

    return True

def test_intent_pipeline():
    """Test IntentPipeline functionality."""
    print("\nTesting IntentPipeline...")

    # Create pipeline
    pipeline = kelly_bridge.IntentPipeline()
    print("  ✓ IntentPipeline created")

    # Create a wound
    wound = kelly_bridge.Wound("feeling of loss and grief", 0.8, "test")
    print(f"  ✓ Wound created: '{wound.description}' (intensity={wound.intensity})")

    # Process the wound
    result = pipeline.process(wound)
    print(f"  ✓ Processed wound")
    print(f"    Emotion: {result.emotion.name} (id={result.emotion.id})")
    print(f"    Valence: {result.emotion.valence:.2f}")
    print(f"    Arousal: {result.emotion.arousal:.2f}")
    print(f"    Intensity: {result.emotion.intensity:.2f}")
    print(f"    Rule breaks: {len(result.ruleBreaks)}")
    print(f"    Mode: {result.mode}")
    print(f"    Tempo: {result.tempo:.2f}")

    # Test thesaurus access
    thesaurus_ref = pipeline.thesaurus()
    print(f"  ✓ Accessed thesaurus via pipeline: size={thesaurus_ref.size()}")

    return True

def test_enums():
    """Test enum types."""
    print("\nTesting Enums...")

    # Test EmotionCategory
    categories = [
        kelly_bridge.EmotionCategory.Joy,
        kelly_bridge.EmotionCategory.Sadness,
        kelly_bridge.EmotionCategory.Anger,
    ]
    print(f"  ✓ EmotionCategory enum: {len(categories)} values tested")

    # Test RuleBreakType
    rule_types = [
        kelly_bridge.RuleBreakType.Harmony,
        kelly_bridge.RuleBreakType.Rhythm,
        kelly_bridge.RuleBreakType.Dynamics,
    ]
    print(f"  ✓ RuleBreakType enum: {len(rule_types)} values tested")

    return True

def test_utility_functions():
    """Test utility functions."""
    print("\nTesting Utility Functions...")

    # Test midi_note_to_name
    note_name = kelly_bridge.midi_note_to_name(60)
    print(f"  ✓ midi_note_to_name(60) = '{note_name}'")

    # Test note_name_to_midi
    note_num = kelly_bridge.note_name_to_midi("C4")
    print(f"  ✓ note_name_to_midi('C4') = {note_num}")

    # Test category_to_string
    cat_str = kelly_bridge.category_to_string(kelly_bridge.EmotionCategory.Joy)
    print(f"  ✓ category_to_string(Joy) = '{cat_str}'")

    return True

def main():
    print("=" * 60)
    print("Kelly Bridge - Basic Functionality Test")
    print("=" * 60)
    print()

    try:
        test_enums()
        test_emotion_thesaurus()
        test_intent_pipeline()
        test_utility_functions()

        print()
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
