"""
Emotion to Music Mapper
Maps emotional states to musical parameters (mode, tempo, progression, dynamics)
"""

from typing import Dict, Any, Optional
import random


def emotion_to_mode(base_emotion: str) -> str:
    """
    Map base emotion to musical mode.

    Args:
        base_emotion: Base emotion (sad, happy, fear, angry, etc.)

    Returns:
        Mode name (Aeolian, Ionian, Phrygian, Locrian, etc.)
    """
    emotion_lower = base_emotion.lower()

    mode_map = {
        "sad": "Aeolian",           # Natural minor
        "sadness": "Aeolian",
        "grief": "Aeolian",
        "melancholy": "Aeolian",
        "depression": "Aeolian",
        "happy": "Ionian",           # Major
        "happiness": "Ionian",
        "joy": "Ionian",
        "joyful": "Ionian",
        "elation": "Ionian",
        "fear": "Phrygian",          # Dark, tense
        "fearful": "Phrygian",
        "anxiety": "Phrygian",
        "terror": "Phrygian",
        "angry": "Locrian",          # Most dissonant
        "anger": "Locrian",
        "rage": "Locrian",
        "fury": "Locrian",
        "neutral": "Ionian",         # Default to major
        "calm": "Ionian",
        "peaceful": "Ionian",
    }

    # Try direct match first
    if emotion_lower in mode_map:
        return mode_map[emotion_lower]

    # Try partial matches
    for key, mode in mode_map.items():
        if key in emotion_lower or emotion_lower in key:
            return mode

    # Default to Ionian (major)
    return "Ionian"


def emotion_to_tempo(intensity: str) -> int:
    """
    Map intensity level to tempo range.

    Args:
        intensity: Intensity level (low, moderate, high, intense)

    Returns:
        Tempo in BPM (random within range)
    """
    intensity_lower = intensity.lower()

    tempo_ranges = {
        "low": (60, 80),
        "moderate": (80, 100),
        "medium": (80, 100),
        "high": (100, 120),
        "intense": (120, 140),
    }

    if intensity_lower in tempo_ranges:
        min_bpm, max_bpm = tempo_ranges[intensity_lower]
        return random.randint(min_bpm, max_bpm)

    # Default to moderate
    return random.randint(80, 100)


def emotion_to_progression(base_emotion: str, intensity: str, specific_emotion: Optional[str] = None) -> str:
    """
    Map emotion to chord progression (Roman numerals).

    Args:
        base_emotion: Base emotion
        intensity: Intensity level
        specific_emotion: Specific emotion variant (optional)

    Returns:
        Progression string (e.g., "i-VI-III-VII" or "I-V-vi-IV")
    """
    emotion_lower = base_emotion.lower()
    specific = specific_emotion.lower() if specific_emotion else ""

    # Grief-specific progression
    if "grief" in emotion_lower or "grief" in specific:
        return "i-VI-III-VII"  # Minor progression with borrowed chords

    # Joy-specific progression
    if "joy" in emotion_lower or "joy" in specific or "happy" in emotion_lower:
        return "I-V-vi-IV"  # Classic pop progression

    # Sad/melancholy progressions
    if "sad" in emotion_lower or "melancholy" in emotion_lower:
        progressions = [
            "i-bVI-bIII-bVII",  # Minor with borrowed chords
            "i-iv-bVI-V",        # Minor with iv
            "vi-IV-I-V",         # Relative minor progression
        ]
        return random.choice(progressions)

    # Fear/anxiety progressions
    if "fear" in emotion_lower or "anxiety" in emotion_lower:
        return "i-bII-bVII-i"  # Phrygian progression

    # Anger/rage progressions
    if "angry" in emotion_lower or "rage" in emotion_lower:
        return "i-bII-bVII-i"  # Locrian/Phrygian, tense

    # Default progression
    return "I-V-vi-IV"


def map_emotion_to_music(
    base: str,
    intensity: str,
    sub: Optional[str] = None
) -> Dict[str, Any]:
    """
    Map emotional parameters to complete musical configuration.

    Args:
        base: Base emotion (sad, happy, fear, angry, etc.)
        intensity: Intensity level (low, moderate, high, intense)
        sub: Specific emotion variant (grief, joy, etc.)

    Returns:
        Dictionary with:
        - key: Musical key (e.g., "F", "C")
        - mode: Mode name (e.g., "Aeolian", "Ionian")
        - tempo: Tempo in BPM
        - progression: Chord progression (Roman numerals)
        - dynamics: Dynamic level (pp, p, mf, f, ff)
    """
    # Select key (prefer F for Kelly example, but allow variation)
    keys = ["C", "D", "E", "F", "G", "A", "B"]
    # Weight F higher for Kelly example
    key = random.choices(keys, weights=[1, 1, 1, 3, 1, 1, 1])[0]

    # Map emotion to mode
    mode = emotion_to_mode(base)

    # Map intensity to tempo
    tempo = emotion_to_tempo(intensity)

    # Map emotion to progression
    progression = emotion_to_progression(base, intensity, sub)

    # Map intensity to dynamics
    intensity_lower = intensity.lower()
    if intensity_lower == "low":
        dynamics = "p"  # Piano (soft)
    elif intensity_lower == "moderate" or intensity_lower == "medium":
        dynamics = "mf"  # Mezzo forte (moderate)
    elif intensity_lower == "high":
        dynamics = "f"  # Forte (loud)
    elif intensity_lower == "intense":
        dynamics = "ff"  # Fortissimo (very loud)
    else:
        dynamics = "mf"

    return {
        "key": key,
        "mode": mode,
        "tempo": tempo,
        "progression": progression,
        "dynamics": dynamics,
    }
