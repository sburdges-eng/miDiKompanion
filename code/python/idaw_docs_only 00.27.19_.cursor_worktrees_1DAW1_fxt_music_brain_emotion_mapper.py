"""
Emotion to Music Mapper
Maps emotional states to musical parameters (mode, tempo, progression, dynamics)
"""

from typing import Dict, Any, Optional, List
import random

# Emotion to mode mapping
EMOTION_TO_MODE = {
    "sad": "aeolian",
    "happy": "ionian",
    "fear": "phrygian",
    "angry": "locrian",
    "disgust": "dorian",
    "surprise": "lydian",
    "neutral": "ionian",
}

# Intensity to tempo mapping (BPM)
INTENSITY_TO_TEMPO = {
    "low": 65,
    "moderate": 90,
    "high": 115,
    "intense": 130,
    "extreme": 145,
    "overwhelming": 160,
}

# Sub-emotion to progression mapping
SUB_EMOTION_PROGRESSIONS = {
    "grief": ["i", "VI", "III", "VII"],  # F minor: F-C-Am-Dm
    "joy": ["I", "V", "vi", "IV"],
    "rage": ["i", "bII", "bVII", "i"],
    "fear": ["i", "bII", "bVII", "i"],
    "melancholy": ["i", "bVI", "bIII", "bVII"],
    "sadness": ["i", "iv", "bVI", "V"],
    "anger": ["i", "bII", "bVII", "i"],
    "anxiety": ["i", "bII", "bVII", "i"],
    "happiness": ["I", "V", "vi", "IV"],
    "elation": ["I", "IV", "V", "I"],
    "calm": ["I", "IV", "I", "V"],
    "peaceful": ["I", "iii", "IV", "V"],
}


def mode_to_major_minor(mode_name: str) -> str:
    """
    Convert mode name to "major" or "minor" for SongGenerator compatibility.

    Args:
        mode_name: Mode name (Aeolian, Ionian, Phrygian, Locrian, etc.)

    Returns:
        "major" or "minor"
    """
    mode_lower = mode_name.lower()

    # Major modes
    if mode_lower in ["ionian", "lydian", "mixolydian"]:
        return "major"

    # Minor modes
    if mode_lower in ["aeolian", "dorian", "phrygian", "locrian"]:
        return "minor"

    # Default to major
    return "major"


def emotion_to_mode(base_emotion: str) -> str:
    """
    Map base emotion to musical mode.

    Args:
        base_emotion: Base emotion (sad, happy, fear, angry, etc.)

    Returns:
        Mode name (Aeolian, Ionian, Phrygian, Locrian, etc.)
    """
    emotion_lower = base_emotion.lower()

    # Check direct mapping first
    if emotion_lower in EMOTION_TO_MODE:
        mode_name = EMOTION_TO_MODE[emotion_lower]
        return mode_name.capitalize()

    # Fallback mappings
    mode_map = {
        "sad": "Aeolian",
        "sadness": "Aeolian",
        "grief": "Aeolian",
        "melancholy": "Aeolian",
        "depression": "Aeolian",
        "happy": "Ionian",
        "happiness": "Ionian",
        "joy": "Ionian",
        "joyful": "Ionian",
        "elation": "Ionian",
        "fear": "Phrygian",
        "fearful": "Phrygian",
        "anxiety": "Phrygian",
        "terror": "Phrygian",
        "angry": "Locrian",
        "anger": "Locrian",
        "rage": "Locrian",
        "fury": "Locrian",
        "neutral": "Ionian",
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
    Map intensity level to tempo.

    Args:
        intensity: Intensity level (low, moderate, high, intense, extreme, overwhelming)

    Returns:
        Tempo in BPM
    """
    intensity_lower = intensity.lower()
    return INTENSITY_TO_TEMPO.get(intensity_lower, 90)


def emotion_to_progression(base_emotion: str, intensity: str, specific_emotion: Optional[str] = None) -> List[str]:
    """
    Map emotion to chord progression (Roman numerals).

    Args:
        base_emotion: Base emotion
        intensity: Intensity level
        specific_emotion: Specific emotion variant (optional)

    Returns:
        Progression as list of Roman numerals (e.g., ["i", "VI", "III", "VII"])
    """
    emotion_lower = base_emotion.lower()
    specific = specific_emotion.lower() if specific_emotion else ""

    # Check specific emotion first
    if specific and specific in SUB_EMOTION_PROGRESSIONS:
        return SUB_EMOTION_PROGRESSIONS[specific]

    # Grief-specific progression
    if "grief" in emotion_lower or "grief" in specific:
        return ["i", "VI", "III", "VII"]  # Minor progression with borrowed chords

    # Joy-specific progression
    if "joy" in emotion_lower or "joy" in specific or "happy" in emotion_lower:
        return ["I", "V", "vi", "IV"]  # Classic pop progression

    # Sad/melancholy progressions
    if "sad" in emotion_lower or "melancholy" in emotion_lower:
        progressions = [
            ["i", "bVI", "bIII", "bVII"],  # Minor with borrowed chords
            ["i", "iv", "bVI", "V"],        # Minor with iv
            ["vi", "IV", "I", "V"],         # Relative minor progression
        ]
        return random.choice(progressions)

    # Fear/anxiety progressions
    if "fear" in emotion_lower or "anxiety" in emotion_lower:
        return ["i", "bII", "bVII", "i"]  # Phrygian progression

    # Anger/rage progressions
    if "angry" in emotion_lower or "rage" in emotion_lower:
        return ["i", "bII", "bVII", "i"]  # Locrian/Phrygian, tense

    # Default progression
    return ["I", "V", "vi", "IV"]


def get_key_from_emotion(base: str, sub: Optional[str] = None) -> str:
    """
    Get appropriate key for emotion.

    Args:
        base: Base emotion
        sub: Specific emotion variant

    Returns:
        Key name (e.g., "F", "C")
    """
    # For grief/sad emotions, prefer F (Kelly example)
    if base.lower() in ["sad", "grief", "melancholy"]:
        return "F"

    # For happy/joy, prefer C or G
    if base.lower() in ["happy", "joy", "elation"]:
        return random.choice(["C", "G"])

    # Default
    return random.choice(["C", "D", "E", "F", "G", "A", "B"])


def get_dynamics_from_intensity(intensity: str) -> str:
    """
    Map intensity to dynamic level.

    Args:
        intensity: Intensity level

    Returns:
        Dynamic marking (pp, p, mf, f, ff)
    """
    intensity_lower = intensity.lower()

    dynamics_map = {
        "low": "p",           # Piano (soft)
        "moderate": "mf",     # Mezzo forte (moderate)
        "medium": "mf",
        "high": "f",          # Forte (loud)
        "intense": "ff",      # Fortissimo (very loud)
        "extreme": "ff",
        "overwhelming": "ff",
    }

    return dynamics_map.get(intensity_lower, "mf")


def get_articulation_from_emotion(base: str, intensity: str) -> str:
    """
    Get articulation style from emotion.

    Args:
        base: Base emotion
        intensity: Intensity level

    Returns:
        Articulation style (legato, staccato, etc.)
    """
    if base.lower() in ["sad", "grief", "melancholy"]:
        return "legato"  # Smooth, connected

    if base.lower() in ["angry", "rage"]:
        return "staccato"  # Short, detached

    if base.lower() in ["fear", "anxiety"]:
        return "marcato"  # Accented

    return "legato"  # Default


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
        - progression: Chord progression (Roman numerals as list)
        - dynamics: Dynamic level (pp, p, mf, f, ff)
        - articulation: Articulation style
    """
    # Get key from emotion
    key = get_key_from_emotion(base, sub)

    # Map emotion to mode
    mode = emotion_to_mode(base)

    # Map intensity to tempo
    tempo = emotion_to_tempo(intensity)

    # Map emotion to progression (returns list)
    progression = emotion_to_progression(base, intensity, sub)

    # Map intensity to dynamics
    dynamics = get_dynamics_from_intensity(intensity)

    # Get articulation
    articulation = get_articulation_from_emotion(base, intensity)

    return {
        "key": key,
        "mode": mode,
        "tempo": tempo,
        "progression": progression,
        "dynamics": dynamics,
        "articulation": articulation,
    }
