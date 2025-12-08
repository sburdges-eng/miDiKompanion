"""
Emotion to Music Mapper
Maps emotional states to musical parameters (mode, tempo, progression, dynamics)
Supports 6×6×6 emotion thesaurus (216 emotion nodes)
"""

from typing import Dict, Any, Optional, List
import random

# Emotion to Mode mapping
EMOTION_TO_MODE = {
    "sad": "aeolian",
    "sadness": "aeolian",
    "happy": "ionian",
    "happiness": "ionian",
    "fear": "phrygian",
    "angry": "locrian",
    "anger": "locrian",
    "disgust": "dorian",
    "surprise": "lydian",
    "neutral": "ionian",
}

# Intensity to Tempo mapping (exact BPM values)
INTENSITY_TO_TEMPO = {
    "low": 65,
    "moderate": 90,
    "high": 115,
    "intense": 130,
    "extreme": 145,
    "overwhelming": 160,
}

# Sub-emotion progressions (Roman numerals)
SUB_EMOTION_PROGRESSIONS = {
    "grief": ["i", "VI", "III", "VII"],
    "joy": ["I", "V", "vi", "IV"],
    "rage": ["i", "bII", "bVII", "i"],
    "terror": ["i", "bII", "i", "bVII"],
    "ecstasy": ["I", "V", "I", "IV"],
    "melancholy": ["i", "iv", "bVI", "V"],
    "anxiety": ["i", "bII", "bVII", "i"],
    "elation": ["I", "V", "vi", "IV"],
    "despair": ["i", "bVI", "bIII", "bVII"],
    "euphoria": ["I", "V", "I", "vi"],
    "dread": ["i", "bII", "i", "bVII"],
    "bliss": ["I", "V", "vi", "IV"],
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
        Mode name (aeolian, ionian, phrygian, locrian, etc.) - lowercase
    """
    emotion_lower = base_emotion.lower()

    # Try direct match first
    if emotion_lower in EMOTION_TO_MODE:
        return EMOTION_TO_MODE[emotion_lower]

    # Try partial matches
    for key, mode in EMOTION_TO_MODE.items():
        if key in emotion_lower or emotion_lower in key:
            return mode

    # Default to Ionian (major)
    return "ionian"


def emotion_to_tempo(intensity: str) -> int:
    """
    Map intensity level to tempo (exact BPM values).

    Args:
        intensity: Intensity level (low, moderate, high, intense, extreme, overwhelming)

    Returns:
        Tempo in BPM (exact value from mapping)
    """
    intensity_lower = intensity.lower()

    if intensity_lower in INTENSITY_TO_TEMPO:
        return INTENSITY_TO_TEMPO[intensity_lower]

    # Default to moderate
    return INTENSITY_TO_TEMPO.get("moderate", 90)


def emotion_to_progression(base_emotion: str, intensity: str, specific_emotion: Optional[str] = None) -> List[str]:
    """
    Map emotion to chord progression (Roman numerals as list).

    Args:
        base_emotion: Base emotion
        intensity: Intensity level
        specific_emotion: Specific emotion variant (optional)

    Returns:
        Progression as list of Roman numerals (e.g., ["i", "VI", "III", "VII"])
    """
    emotion_lower = base_emotion.lower()
    specific = specific_emotion.lower() if specific_emotion else ""

    # Check sub-emotion progressions first
    for sub_emotion, progression in SUB_EMOTION_PROGRESSIONS.items():
        if sub_emotion in specific or sub_emotion in emotion_lower:
            return progression

    # Fallback to base emotion patterns
    if "grief" in emotion_lower or "grief" in specific:
        return ["i", "VI", "III", "VII"]
    if "joy" in emotion_lower or "joy" in specific or "happy" in emotion_lower:
        return ["I", "V", "vi", "IV"]
    if "sad" in emotion_lower or "melancholy" in emotion_lower:
        return ["i", "iv", "bVI", "V"]
    if "fear" in emotion_lower or "anxiety" in emotion_lower:
        return ["i", "bII", "bVII", "i"]
    if "angry" in emotion_lower or "rage" in emotion_lower:
        return ["i", "bII", "bVII", "i"]

    # Default progression
    return ["I", "V", "vi", "IV"]


def get_key_from_emotion(base: str, sub: Optional[str] = None) -> str:
    """
    Get root note key from emotion.

    Args:
        base: Base emotion
        sub: Specific emotion variant

    Returns:
        Root note (e.g., "F", "C")
    """
    # Kelly example: sad/intense/grief → F
    if "grief" in (sub or "").lower() or ("sad" in base.lower() and "intense" in str(sub or "").lower()):
        return "F"

    # Prefer F for emotional music, but allow variation
    keys = ["C", "D", "E", "F", "G", "A", "B"]
    return random.choices(keys, weights=[1, 1, 1, 3, 1, 1, 1])[0]


def get_dynamics_from_intensity(intensity: str) -> Dict[str, Any]:
    """
    Get velocity range from intensity.

    Args:
        intensity: Intensity level

    Returns:
        Dictionary with velocity_min, velocity_max, and marking
    """
    intensity_lower = intensity.lower()

    velocity_ranges = {
        "low": {"min": 40, "max": 60, "marking": "p"},
        "moderate": {"min": 60, "max": 80, "marking": "mf"},
        "high": {"min": 80, "max": 100, "marking": "f"},
        "intense": {"min": 100, "max": 120, "marking": "ff"},
        "extreme": {"min": 110, "max": 127, "marking": "fff"},
        "overwhelming": {"min": 115, "max": 127, "marking": "fff"},
    }

    return velocity_ranges.get(intensity_lower, velocity_ranges["moderate"])


def map_emotion_to_music(
    base: str,
    intensity: str,
    sub: Optional[str] = None
) -> Dict[str, Any]:
    """
    Map emotional parameters to complete musical configuration.
    Supports all 216 emotion nodes from the 6×6×6 thesaurus.

    Args:
        base: Base emotion (sad, happy, fear, angry, disgust, surprise, neutral)
        intensity: Intensity level (low, moderate, high, intense, extreme, overwhelming)
        sub: Specific emotion variant (grief, joy, rage, etc.)

    Returns:
        Dictionary with:
        - key: Musical key (e.g., "F", "C")
        - mode: Mode name (e.g., "aeolian", "ionian") - lowercase
        - tempo: Tempo in BPM (exact value)
        - progression: Chord progression (string format: "i-VI-III-VII")
        - progression_list: Chord progression (list format: ["i", "VI", "III", "VII"])
        - dynamics: Dynamic level marking (p, mf, f, ff, fff)
        - velocity_min: Minimum MIDI velocity
        - velocity_max: Maximum MIDI velocity
        - articulation: Articulation style (legato, staccato)
    """
    # Get key from emotion
    key = get_key_from_emotion(base, sub)

    # Map emotion to mode (lowercase)
    mode = emotion_to_mode(base)

    # Map intensity to tempo (exact BPM)
    tempo = emotion_to_tempo(intensity)

    # Map emotion to progression (as list)
    progression_list = emotion_to_progression(base, intensity, sub)
    progression = "-".join(progression_list)  # Convert to string for compatibility

    # Map intensity to dynamics
    dynamics_dict = get_dynamics_from_intensity(intensity)
    dynamics = dynamics_dict["marking"]

    # Determine articulation based on emotion
    articulation = "legato"  # Default
    if "angry" in base.lower() or "rage" in (sub or "").lower():
        articulation = "staccato"
    elif "fear" in base.lower() or "anxiety" in (sub or "").lower():
        articulation = "staccato"
    elif "sad" in base.lower() or "grief" in (sub or "").lower():
        articulation = "legato"
    elif "happy" in base.lower() or "joy" in (sub or "").lower():
        articulation = "legato"

    return {
        "key": key,
        "mode": mode,
        "tempo": tempo,
        "progression": progression,
        "progression_list": progression_list,
        "dynamics": dynamics,
        "velocity_min": dynamics_dict["min"],
        "velocity_max": dynamics_dict["max"],
        "articulation": articulation,
    }
