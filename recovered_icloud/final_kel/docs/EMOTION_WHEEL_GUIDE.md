# Emotion Wheel Guide

## Overview

The Emotion Wheel is miDiKompanion's primary interface for emotion selection. It displays a 216-node emotion thesaurus organized in a hierarchical structure.

## Understanding the Structure

### 6 Base Emotions

The wheel is organized around 6 fundamental emotions:

1. **Happy** - Positive valence, moderate to high arousal
2. **Sad** - Negative valence, low to moderate arousal
3. **Angry** - Negative valence, high arousal
4. **Fear** - Negative valence, high arousal
5. **Surprise** - Mixed valence, high arousal
6. **Disgust** - Negative valence, moderate arousal

### 36 Sub-Emotions

Each base emotion has 6 sub-emotions:

- **Happy**: Joy, Contentment, Euphoria, Pride, Relief, Hope
- **Sad**: Grief, Melancholy, Loneliness, Despair, Regret, Nostalgia
- **Angry**: Rage, Frustration, Annoyance, Resentment, Contempt, Irritation
- **Fear**: Terror, Anxiety, Worry, Dread, Panic, Apprehension
- **Surprise**: Amazement, Astonishment, Shock, Confusion, Bewilderment, Awe
- **Disgust**: Revulsion, Loathing, Distaste, Aversion, Repugnance, Contempt

### 216 Total Nodes

Each sub-emotion has 6 sub-sub-emotions, creating 216 unique emotion nodes (6 × 6 × 6).

## Using the Emotion Wheel

### Visual Selection

1. **Click** on any emotion node in the wheel
2. The **Emotion Radar** updates to show the emotion's VAD coordinates:
   - **Valence** (X-axis): Negative (-1.0) to Positive (+1.0)
   - **Arousal** (Y-axis): Calm (0.0) to Excited (1.0)
   - **Intensity** (visual size): Subtle to Extreme

3. **Parameter sliders** automatically adjust to match the emotion

### Emotion Relationships

Emotions near each other on the wheel are related:
- **Adjacent emotions** share similar VAD coordinates
- **Opposite emotions** are on opposite sides of the wheel
- The system uses these relationships for context-aware generation

### Intensity Tiers

Each emotion has 6 intensity levels:
1. **Subtle** - Barely perceptible
2. **Mild** - Noticeable but manageable
3. **Moderate** - Clearly felt, influences behavior
4. **Strong** - Powerful, hard to ignore
5. **Intense** - Overwhelming, dominates experience
6. **Overwhelming** - All-consuming, transcendent

## Emotion Categories

### Positive Emotions (Valence > 0)

- **High Arousal**: Euphoria, Excitement, Elation
- **Moderate Arousal**: Contentment, Satisfaction, Peace
- **Low Arousal**: Serenity, Calm, Relaxation

### Negative Emotions (Valence < 0)

- **High Arousal**: Rage, Terror, Panic
- **Moderate Arousal**: Anxiety, Frustration, Worry
- **Low Arousal**: Grief, Melancholy, Despair

### Mixed Emotions (Valence ≈ 0)

- **Surprise** - Can be positive or negative depending on context
- **Confusion** - Uncertain emotional state
- **Awe** - Complex emotion with mixed valence

## Musical Mapping

Each emotion maps to musical attributes:

### Valence → Mode
- **Positive** (valence > 0): Major modes, Lydian
- **Negative** (valence < 0): Minor modes, Phrygian, Locrian
- **Neutral** (valence ≈ 0): Modal mixture, borrowed chords

### Arousal → Tempo
- **Low Arousal** (0.0-0.3): 40-70 BPM (slow, contemplative)
- **Moderate Arousal** (0.3-0.7): 70-120 BPM (moderate, balanced)
- **High Arousal** (0.7-1.0): 120-180+ BPM (fast, energetic)

### Intensity → Dynamics
- **Subtle** (0.0-0.2): pp-p (very quiet)
- **Mild** (0.2-0.4): p-mp (quiet)
- **Moderate** (0.4-0.6): mp-mf (medium)
- **Strong** (0.6-0.8): mf-f (loud)
- **Intense** (0.8-0.9): f-ff (very loud)
- **Overwhelming** (0.9-1.0): ff-fff with dynamic contrast

## Tips for Emotion Selection

1. **Start Broad**: Click a base emotion (Happy, Sad, etc.)
2. **Refine**: Use the sub-emotions for more specific feelings
3. **Check the Radar**: The Emotion Radar shows the exact VAD coordinates
4. **Adjust Parameters**: Fine-tune after selection if needed
5. **Combine with Text**: Type a description AND select from the wheel for nuanced results

## Emotion Transitions

The system supports emotional journeys:

1. Select an **initial emotion** (e.g., "Anxiety")
2. Generate MIDI
3. Select a **target emotion** (e.g., "Calm")
4. The system can morph between emotions (advanced feature in v1.1)

## Common Emotion Selections

### Therapeutic Use Cases

- **Processing Grief**: Select "Grief" → Generate → Listen
- **Managing Anxiety**: Select "Anxiety" → Adjust to "Calm" → Generate
- **Celebrating Joy**: Select "Euphoria" → Generate upbeat music
- **Exploring Anger**: Select "Rage" → Generate → Process emotions

### Creative Use Cases

- **Film Scoring**: Select emotions matching scene mood
- **Songwriting**: Use emotions as starting points for compositions
- **Music Therapy**: Guide clients through emotion wheel exploration

## Advanced Features (v1.1+)

- **Emotion Blending**: Select multiple emotions for complex states
- **Emotion Transitions**: Morph between emotions over time
- **ML Enhancement**: Toggle ML models for enhanced generation (optional)

---

**Version**: 1.0
**Last Updated**: 2025-01-XX
