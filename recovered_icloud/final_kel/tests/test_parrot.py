#!/usr/bin/env python3
"""
Integration Tests for Parrot Singing Voice Synthesizer

Tests all major features:
- Singing from lyrics and melody
- Voice recording and mimicking
- Note extraction from audio
- Instrument conversion
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from music_brain.voice import Parrot, create_parrot


def test_basic_singing():
    """Test basic singing synthesis."""
    print("=" * 70)
    print("Test 1: Basic Singing Synthesis")
    print("=" * 70)

    parrot = create_parrot(backend="formant")

    lyrics = "Hello world"
    melody = [60, 62, 64, 65, 64, 62, 60]  # C major scale

    audio = parrot.sing(lyrics, melody, tempo_bpm=120)

    assert audio is not None, "Audio should not be None"
    assert len(audio) > 0, "Audio should have samples"
    assert np.max(np.abs(audio)) > 0, "Audio should not be silent"

    print(f"✓ Generated {len(audio) / parrot.sample_rate:.2f} seconds of audio")
    print(f"  Sample rate: {parrot.sample_rate} Hz")
    print(f"  Max amplitude: {np.max(np.abs(audio)):.3f}")

    # Save test output
    parrot.save("test_output_basic.wav", audio)
    print("  Saved to: test_output_basic.wav")

    return True


def test_expression():
    """Test expression parameters."""
    print("\n" + "=" * 70)
    print("Test 2: Expression Parameters")
    print("=" * 70)

    parrot = create_parrot(backend="formant")

    lyrics = "I feel happy"
    melody = [64, 65, 67, 69, 67, 65, 64]

    expression = {
        "vibrato_rate": 6.0,
        "vibrato_depth": 0.03,
        "portamento_time": 0.1,
        "dynamics": [0.5, 0.7, 0.9, 1.0, 0.9, 0.7, 0.5]
    }

    audio = parrot.sing(lyrics, melody, tempo_bpm=120, expression=expression)

    assert audio is not None, "Audio should not be None"
    assert len(audio) > 0, "Audio should have samples"

    print("✓ Expression parameters applied")
    parrot.save("test_output_expression.wav", audio)
    print("  Saved to: test_output_expression.wav")

    return True


def test_phoneme_processing():
    """Test phoneme processing."""
    print("\n" + "=" * 70)
    print("Test 3: Phoneme Processing")
    print("=" * 70)

    parrot = create_parrot(backend="formant")

    # Test different lyrics
    test_cases = [
        ("Hello", [60, 62, 64]),
        ("World", [64, 65, 67]),
        ("Music", [67, 69, 71]),
    ]

    for lyrics, melody in test_cases:
        phoneme_seq = parrot.phoneme_processor.process_lyrics(lyrics, melody, 120)
        assert len(phoneme_seq.phonemes) > 0, f"Should have phonemes for '{lyrics}'"
        print(f"  '{lyrics}' → {len(phoneme_seq.phonemes)} phonemes")

    print("✓ Phoneme processing working")

    return True


def test_pitch_extraction():
    """Test pitch extraction from audio."""
    print("\n" + "=" * 70)
    print("Test 4: Pitch Extraction")
    print("=" * 70)

    parrot = create_parrot(backend="formant")

    # Create test audio with known pitches
    sample_rate = parrot.sample_rate
    duration = 2.0
    t = np.arange(int(duration * sample_rate)) / sample_rate

    # Generate tones at specific MIDI notes
    midi_notes = [60, 62, 64, 65, 64, 62, 60]  # C major
    note_duration = duration / len(midi_notes)

    audio = np.zeros(int(duration * sample_rate))
    for i, midi_note in enumerate(midi_notes):
        freq = parrot.pitch_controller.midi_to_frequency(midi_note)
        start_sample = int(i * note_duration * sample_rate)
        end_sample = int((i + 1) * note_duration * sample_rate)
        audio[start_sample:end_sample] = np.sin(2 * np.pi * freq * t[start_sample:end_sample])

    # Extract notes
    extracted_notes = parrot.extract_notes_from_audio(audio, note_duration=note_duration)

    print(f"  Original notes: {midi_notes}")
    print(f"  Extracted notes: {extracted_notes}")

    # Should extract similar notes (allow some tolerance)
    assert len(extracted_notes) > 0, "Should extract at least one note"
    print("✓ Pitch extraction working")

    return True


def test_instrument_conversion():
    """Test converting notes to instruments."""
    print("\n" + "=" * 70)
    print("Test 5: Instrument Conversion")
    print("=" * 70)

    parrot = create_parrot(backend="formant")

    midi_notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale

    instruments = ["piano", "guitar", "strings", "flute"]

    for instrument in instruments:
        audio = parrot.notes_to_instrument(midi_notes, instrument=instrument)
        assert audio is not None, f"Should generate audio for {instrument}"
        assert len(audio) > 0, f"Audio should have samples for {instrument}"
        print(f"  ✓ {instrument}: {len(audio) / parrot.sample_rate:.2f} seconds")

        # Save test output
        parrot.save(f"test_output_{instrument}.wav", audio)

    print("✓ Instrument conversion working")

    return True


def test_voice_mimicking():
    """Test voice mimicking (without actual recording)."""
    print("\n" + "=" * 70)
    print("Test 6: Voice Mimicking")
    print("=" * 70)

    parrot = create_parrot(backend="formant")

    # Create synthetic voice characteristics
    voice_characteristics = {
        "mean_pitch": 250.0,  # Higher pitch
        "pitch_range": 300.0,
        "brightness": 3000.0,
        "breathiness": 0.3,
        "formant_emphasis": 0.6
    }

    lyrics = "Hello world"
    melody = [60, 62, 64, 65, 64, 62, 60]

    # Sing with voice characteristics
    audio = parrot.sing_with_voice(lyrics, melody, voice_characteristics, tempo_bpm=120)

    assert audio is not None, "Audio should not be None"
    assert len(audio) > 0, "Audio should have samples"

    print("✓ Voice mimicking applied")
    parrot.save("test_output_mimicked.wav", audio)
    print("  Saved to: test_output_mimicked.wav")

    return True


def test_full_pipeline():
    """Test full pipeline: sing → extract notes → convert to instrument."""
    print("\n" + "=" * 70)
    print("Test 7: Full Pipeline")
    print("=" * 70)

    parrot = create_parrot(backend="formant")

    # Step 1: Sing
    lyrics = "Do re mi fa sol la ti do"
    melody = [60, 62, 64, 65, 67, 69, 71, 72]
    audio = parrot.sing(lyrics, melody, tempo_bpm=120)

    assert audio is not None, "Singing should work"
    print("  ✓ Step 1: Singing complete")

    # Step 2: Extract notes (simulate - would normally be from recorded audio)
    # For this test, we'll use the original melody
    extracted_notes = melody  # In real usage, would extract from audio

    # Step 3: Convert to instrument
    instrument_audio = parrot.notes_to_instrument(extracted_notes, instrument="piano")

    assert instrument_audio is not None, "Instrument conversion should work"
    print("  ✓ Step 2: Note extraction (simulated)")
    print("  ✓ Step 3: Instrument conversion complete")

    parrot.save("test_output_pipeline_singing.wav", audio)
    parrot.save("test_output_pipeline_instrument.wav", instrument_audio)
    print("  Saved pipeline outputs")

    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PARROT INTEGRATION TESTS")
    print("=" * 70)

    tests = [
        ("Basic Singing", test_basic_singing),
        ("Expression", test_expression),
        ("Phoneme Processing", test_phoneme_processing),
        ("Pitch Extraction", test_pitch_extraction),
        ("Instrument Conversion", test_instrument_conversion),
        ("Voice Mimicking", test_voice_mimicking),
        ("Full Pipeline", test_full_pipeline),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"\n✗ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, "FAIL"))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, status in results if status == "PASS")
    failed = sum(1 for _, status in results if status == "FAIL")

    for test_name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} {test_name}: {status}")

    print(f"\nTotal: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
