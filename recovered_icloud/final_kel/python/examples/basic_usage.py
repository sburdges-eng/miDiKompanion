#!/usr/bin/env python3
"""
Basic usage example for Kelly MIDI Companion Python bridge.

This demonstrates the core functionality:
1. Processing emotional wounds
2. Generating MIDI from emotions
3. Exporting MIDI files
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kelly import Kelly, EmotionCategory


def main():
    print("Kelly MIDI Companion - Basic Usage Example\n")
    
    # Initialize Kelly
    kelly = Kelly(tempo=120)
    print(f"Initialized Kelly with tempo: {kelly.tempo} BPM\n")
    
    # Example 1: Generate from emotional description
    print("=" * 60)
    print("Example 1: Generate from emotional description")
    print("=" * 60)
    
    description = "feeling of loss and grief"
    intensity = 0.8
    
    print(f"Processing: '{description}' (intensity: {intensity})")
    result, midi = kelly.generate(description, intensity=intensity, bars=4)
    
    print(f"\nResult:")
    print(f"  Emotion: {result.emotion.name} (ID: {result.emotion.id})")
    print(f"  Valence: {result.emotion.valence:.2f}")
    print(f"  Arousal: {result.emotion.arousal:.2f}")
    print(f"  Intensity: {result.emotion.intensity:.2f}")
    print(f"  Rule breaks: {len(result.ruleBreaks)}")
    print(f"  Musical params:")
    print(f"    Tempo: {result.musicalParams.tempoSuggested} BPM")
    print(f"    Key: {result.musicalParams.keySuggested}")
    print(f"    Mode: {result.musicalParams.modeSuggested}")
    print(f"  Generated {len(midi)} MIDI notes\n")
    
    # Example 2: Generate from emotion name
    print("=" * 60)
    print("Example 2: Generate from emotion name")
    print("=" * 60)
    
    emotion_name = "joy"
    result2, midi2 = kelly.generate_from_emotion(emotion_name, bars=4)
    
    if result2:
        print(f"Found emotion: {result2.emotion.name}")
        print(f"  Valence: {result2.emotion.valence:.2f}")
        print(f"  Arousal: {result2.emotion.arousal:.2f}")
        print(f"  Generated {len(midi2)} MIDI notes\n")
    
    # Example 3: Generate from VAI values
    print("=" * 60)
    print("Example 3: Generate from valence/arousal/intensity")
    print("=" * 60)
    
    valence = -0.5  # Negative emotion
    arousal = 0.7   # Moderate excitement
    intensity = 0.6  # Moderate intensity
    
    print(f"VAI: valence={valence}, arousal={arousal}, intensity={intensity}")
    midi3 = kelly.generate_from_vai(valence, arousal, intensity, bars=4)
    print(f"Generated {len(midi3)} MIDI notes\n")
    
    # Example 4: Find emotion
    print("=" * 60)
    print("Example 4: Find emotion by VAI")
    print("=" * 60)
    
    emotion = kelly.find_emotion(valence, arousal, intensity)
    if emotion:
        print(f"Nearest emotion: {emotion.name}")
        print(f"  Category: {emotion.category}")
        print(f"  VAI: ({emotion.valence:.2f}, {emotion.arousal:.2f}, {emotion.intensity:.2f})\n")
    
    # Example 5: Get emotions by category
    print("=" * 60)
    print("Example 5: Get emotions by category")
    print("=" * 60)
    
    sadness_emotions = kelly.get_emotions_by_category(EmotionCategory.Sadness)
    print(f"Found {len(sadness_emotions)} sadness-related emotions:")
    for emo in sadness_emotions[:5]:  # Show first 5
        print(f"  - {emo.name} (ID: {emo.id})")
    print()
    
    # Example 6: Export MIDI (if mido is available)
    print("=" * 60)
    print("Example 6: Export MIDI")
    print("=" * 60)
    
    output_file = "kelly_output.mid"
    if kelly.export_midi(midi, output_file):
        print(f"Exported MIDI to: {output_file}")
    else:
        print("MIDI export requires 'mido' library: pip install mido")
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
