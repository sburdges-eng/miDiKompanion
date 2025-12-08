# Emotion System Documentation

## Overview

iDAW uses a hierarchical 6×6×6 emotion thesaurus containing 216 emotion nodes. This system allows precise emotional expression and maps emotions to musical parameters.

## Structure

### Base Emotions (6)

1. **Sad** - Negative emotional states characterized by loss, disappointment, and low energy
2. **Happy** - Positive emotional states with high energy and pleasure
3. **Angry** - Negative states with high arousal and hostility
4. **Fear** - Negative states with anxiety and apprehension
5. **Disgust** - Negative states with revulsion and rejection
6. **Surprise** - Neutral states with unexpectedness

### Intensity Levels (6)

1. **Low** - Subtle, barely noticeable
2. **Moderate** - Noticeable but manageable
3. **High** - Strong and prominent
4. **Intense** - Very strong, consuming
5. **Extreme** - Overwhelming, all-encompassing
6. **Overwhelming** - Beyond control, consuming

### Specific Emotions (6 per base)

Each base emotion has 6 specific variants. For example, **Sad** includes:

- Grief
- Melancholy
- Heartbroken
- Anguished
- Despair
- Sorrow

## Emotion-to-Music Mapping

### Mode Mapping

- **Sad** → Aeolian (Natural Minor)
- **Happy** → Ionian (Major)
- **Fear** → Phrygian (Dark, tense)
- **Angry** → Locrian (Most dissonant)
- **Disgust** → Dorian
- **Surprise** → Lydian

### Tempo Mapping

- **Low** → 65 BPM
- **Moderate** → 90 BPM
- **High** → 115 BPM
- **Intense** → 130 BPM
- **Extreme** → 145 BPM
- **Overwhelming** → 160 BPM

### Progression Examples

- **Grief** → i-VI-III-VII (F-C-Am-Dm in F minor)
- **Joy** → I-V-vi-IV (Classic pop progression)
- **Rage** → i-bII-bVII-i (Tense, dissonant)

### Dynamics Mapping

- **Low** → p (Piano - soft)
- **Moderate** → mf (Mezzo forte)
- **High** → f (Forte - loud)
- **Intense** → ff (Fortissimo - very loud)

## Psychology References

The emotion system is based on:

- Plutchik's Wheel of Emotions
- Ekman's Basic Emotions
- Russell's Circumplex Model
- Emotional granularity research

## Usage Examples

### Example 1: Grief

- Base: Sad
- Intensity: Intense
- Specific: Grief
- **Result**: F Aeolian, 130 BPM, i-VI-III-VII progression, legato articulation

### Example 2: Joy

- Base: Happy
- Intensity: High
- Specific: Joy
- **Result**: C Ionian, 115 BPM, I-V-vi-IV progression, legato articulation

## Implementation

See `music_brain/emotion_mapper.py` for the complete mapping implementation.
