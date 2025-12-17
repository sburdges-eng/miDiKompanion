#!/usr/bin/env python3
"""
Advanced features example for Kelly MIDI Companion Python bridge.

Demonstrates:
1. IntentPipeline for journey processing
2. EmotionThesaurus queries
3. Custom wound processing
4. Batch generation
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kelly import (
    Kelly,
    IntentPipeline,
    EmotionThesaurus,
    Wound,
    SideA,
    SideB,
    EmotionCategory,
)


def main():
    print("Kelly MIDI Companion - Advanced Features Example\n")
    
    # Initialize
    kelly = Kelly(tempo=120)
    
    # Example 1: IntentPipeline for journey processing
    print("=" * 60)
    print("Example 1: Emotional Journey (Side A → Side B)")
    print("=" * 60)
    
    pipeline = IntentPipeline()
    
    # Current state (Side A)
    side_a = SideA()
    side_a.description = "feeling lost and empty"
    side_a.intensity = 0.8
    
    # Desired state (Side B)
    side_b = SideB()
    side_b.description = "finding peace and acceptance"
    side_b.intensity = 0.6
    
    print(f"Side A: {side_a.description} (intensity: {side_a.intensity})")
    print(f"Side B: {side_b.description} (intensity: {side_b.intensity})")
    
    journey_result = pipeline.process_journey(side_a, side_b)
    print(f"\nJourney Result:")
    print(f"  Emotion: {journey_result.emotion.name}")
    print(f"  Rule breaks: {len(journey_result.ruleBreaks)}")
    print(f"  Tempo: {journey_result.musicalParams.tempoSuggested} BPM")
    print(f"  Mode: {journey_result.musicalParams.modeSuggested}\n")
    
    # Example 2: EmotionThesaurus queries
    print("=" * 60)
    print("Example 2: EmotionThesaurus Queries")
    print("=" * 60)
    
    thesaurus = pipeline.thesaurus()
    print(f"Thesaurus contains {thesaurus.size()} emotions\n")
    
    # Find by name
    grief = thesaurus.find_by_name("grief")
    if grief:
        print(f"Found 'grief':")
        print(f"  ID: {grief.id}")
        print(f"  VAI: ({grief.valence:.2f}, {grief.arousal:.2f}, {grief.intensity:.2f})")
        print(f"  Category: {grief.category}")
        print(f"  Nearby emotions: {grief.nearbyEmotionIds}\n")
    
    # Find nearest
    nearest = thesaurus.find_nearest(-0.7, 0.3, 0.8)
    if nearest:
        print(f"Nearest to VAI(-0.7, 0.3, 0.8): {nearest.name}\n")
    
    # Get nearby emotions
    nearby = thesaurus.get_nearby(-0.5, 0.4, 0.6, threshold=0.5)
    print(f"Emotions near VAI(-0.5, 0.4, 0.6):")
    for emo in nearby[:5]:
        print(f"  - {emo.name} (distance: {abs(emo.valence + 0.5) + abs(emo.arousal - 0.4) + abs(emo.intensity - 0.6):.2f})")
    print()
    
    # Example 3: Custom wound processing
    print("=" * 60)
    print("Example 3: Custom Wound Processing")
    print("=" * 60)
    
    custom_wound = Wound()
    custom_wound.description = "complex mix of nostalgia and hope"
    custom_wound.intensity = 0.7
    custom_wound.source = "internal"
    custom_wound.context = "thinking about the past but looking forward"
    custom_wound.triggers = ["memory", "future", "transition"]
    
    print(f"Custom wound: {custom_wound.description}")
    print(f"  Context: {custom_wound.context}")
    print(f"  Triggers: {', '.join(custom_wound.triggers)}")
    
    custom_result = pipeline.process(custom_wound)
    print(f"\nProcessed result:")
    print(f"  Emotion: {custom_result.emotion.name}")
    print(f"  Rule breaks: {len(custom_result.ruleBreaks)}")
    for rb in custom_result.ruleBreaks:
        print(f"    - {rb.description} (severity: {rb.severity:.2f})")
    print()
    
    # Example 4: Batch generation
    print("=" * 60)
    print("Example 4: Batch Generation")
    print("=" * 60)
    
    emotions_to_generate = ["grief", "joy", "anger", "fear", "hope"]
    print(f"Generating MIDI for {len(emotions_to_generate)} emotions...")
    
    results = []
    for emotion_name in emotions_to_generate:
        result, midi = kelly.generate_from_emotion(emotion_name, bars=2)
        if result:
            results.append((emotion_name, len(midi)))
            print(f"  ✓ {emotion_name}: {len(midi)} notes")
    
    print(f"\nGenerated {len(results)} MIDI sequences\n")
    
    # Example 5: Category exploration
    print("=" * 60)
    print("Example 5: Category Exploration")
    print("=" * 60)
    
    categories = [
        EmotionCategory.Sadness,
        EmotionCategory.Joy,
        EmotionCategory.Anger,
    ]
    
    for category in categories:
        emotions = thesaurus.get_by_category(category)
        print(f"{category}: {len(emotions)} emotions")
        for emo in emotions[:3]:  # Show first 3
            print(f"  - {emo.name}")
    print()
    
    print("=" * 60)
    print("Advanced examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
