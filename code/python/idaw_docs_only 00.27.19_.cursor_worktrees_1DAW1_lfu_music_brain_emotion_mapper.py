"""
Emotion to Music Mapper
Maps emotional states to musical parameters (key, mode, tempo, progression)
"""

from typing import Dict, Any, Optional
import random


def emotion_to_mode(base_emotion: str, intensity: str = "moderate", specific_emotion: Optional[str] = None) -> str:
    """
    Map emotion to musical mode.

    Args:
        base_emotion: Base emotion (sad, happy, fear, angry, etc.)
        intensity: Intensity level (low, moderate, high, intense)
        specific_emotion: Specific emotion variant (grief, joy, etc.)

    Returns:
        Mode name (Ionian, Aeolian, Phrygian, Locrian, etc.)
    """
    # Base emotion mapping
    mode_map = {
        "sad": "Aeolian",      # Natural minor
        "happy": "Ionian",     # Major
        "fear": "Phrygian",    # Dark, tense
        "angry": "Locrian",    # Most dissonant
        "neutral": "Ionian",   # Default to major
    }

    # Specific emotion overrides
    if specific_emotion:
        specific_map = {
            "grief": "Aeolian",
            "joy": "Ionian",
            "anxiety": "Phrygian",
            "rage": "Locrian",
            "melancholy": "Aeolian",
            "euphoria": "Ionian",
            "dread": "Phrygian",
            "fury": "Locrian",
        }
        if specific_emotion.lower() in specific_map:
            return specific_map[specific_emotion.lower()]

    # Return mapped mode or default
    return mode_map.get(base_emotion.lower(), "Ionian")


def emotion_to_tempo(intensity: str, base_emotion: Optional[str] = None) -> int:
    """
    Map intensity to tempo range and return a value.

    Args:
        intensity: Intensity level (low, moderate, high, intense)
        base_emotion: Optional base emotion for fine-tuning

    Returns:
        Tempo in BPM
    """
    # Base tempo ranges by intensity
    tempo_ranges = {
        "low": (60, 80),
        "moderate": (80, 100),
        "high": (100, 120),
        "intense": (120, 140),
    }

    # Get range
    min_tempo, max_tempo = tempo_ranges.get(intensity.lower(), (80, 100))

    # Fine-tune based on emotion
    if base_emotion:
        if base_emotion.lower() == "sad":
            # Sad emotions tend to be slower
            min_tempo = max(50, min_tempo - 10)
            max_tempo = max(70, max_tempo - 10)
        elif base_emotion.lower() == "angry":
            # Angry emotions can be faster
            min_tempo = min(100, min_tempo + 10)
            max_tempo = min(160, max_tempo + 10)

    # Return random tempo in range
    return random.randint(min_tempo, max_tempo)


def emotion_to_progression(base_emotion: str, intensity: str = "moderate", specific_emotion: Optional[str] = None) -> str:
    """
    Map emotion to chord progression pattern.

    Args:
        base_emotion: Base emotion
        intensity: Intensity level
        specific_emotion: Specific emotion variant

    Returns:
        Progression string in Roman numerals (e.g., "i-VI-III-VII")
    """
    # Specific emotion progressions
    if specific_emotion:
        specific_progressions = {
            "grief": "i-VI-III-VII",      # Minor progression, emotional
            "joy": "I-V-vi-IV",            # Pop progression, uplifting
            "anxiety": "i-bII-bVII-i",     # Phrygian, tense
            "rage": "i-bII-i-bII",         # Locrian, unstable
            "melancholy": "i-iv-bVI-V",    # Minor with borrowed chords
            "euphoria": "I-IV-V-I",        # Simple major, strong
            "dread": "i-bVI-bVII-i",       # Dark minor
            "fury": "i-bII-i-bII",         # Unstable
        }
        if specific_emotion.lower() in specific_progressions:
            return specific_progressions[specific_emotion.lower()]

    # Base emotion progressions
    base_progressions = {
        "sad": "i-VI-III-VII",
        "happy": "I-V-vi-IV",
        "fear": "i-bII-bVII-i",
        "angry": "i-bII-i-bII",
        "neutral": "I-V-vi-IV",
    }

    return base_progressions.get(base_emotion.lower(), "I-V-vi-IV")


def emotion_to_key(base_emotion: str, intensity: str = "moderate") -> str:
    """
    Map emotion to musical key.

    Args:
        base_emotion: Base emotion
        intensity: Intensity level

    Returns:
        Key name (e.g., "F", "C", "D") - just the note name, not chord
    """
    # Keys associated with emotions (note names only, mode is separate)
    key_map = {
        "sad": ["F", "D", "A", "E"],
        "happy": ["C", "G", "F", "D"],
        "fear": ["D", "E", "A"],
        "angry": ["B", "F#", "C#"],
        "neutral": ["C", "F", "G"],
    }

    keys = key_map.get(base_emotion.lower(), ["C"])
    return random.choice(keys)


def map_emotion_to_music(
    base_emotion: str,
    intensity: str = "moderate",
    specific_emotion: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete mapping from emotion to musical parameters.

    Args:
        base_emotion: Base emotion (sad, happy, fear, angry)
        intensity: Intensity level (low, moderate, high, intense)
        specific_emotion: Specific emotion variant (grief, joy, etc.)

    Returns:
        Dictionary with key, mode, tempo, progression, dynamics

    Example:
        >>> result = map_emotion_to_music("sad", "intense", "grief")
        >>> print(result["key"])  # "F"
        >>> print(result["mode"])  # "Aeolian"
        >>> print(result["tempo"])  # 82
        >>> print(result["progression"])  # "i-VI-III-VII"
    """
    # Get mode
    mode = emotion_to_mode(base_emotion, intensity, specific_emotion)

    # Get tempo
    tempo = emotion_to_tempo(intensity, base_emotion)

    # Get progression
    progression = emotion_to_progression(base_emotion, intensity, specific_emotion)

    # Get key
    key = emotion_to_key(base_emotion, intensity)

    # Map mode name to standard mode format
    mode_standard = mode.lower()
    if mode_standard == "ionian":
        mode_standard = "major"
    elif mode_standard == "aeolian":
        mode_standard = "minor"
    # Keep other modes as-is (phrygian, locrian, etc.)

    # Calculate dynamics based on intensity
    dynamics = {
        "low": 0.3,
        "moderate": 0.6,
        "high": 0.8,
        "intense": 1.0,
    }.get(intensity.lower(), 0.6)

    return {
        "key": key,
        "mode": mode_standard,
        "tempo": tempo,
        "progression": progression,
        "dynamics": dynamics,
        "mode_name": mode,  # Keep original mode name
    }
