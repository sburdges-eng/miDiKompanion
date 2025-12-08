"""
Emotion Mapper - Maps emotional states to musical parameters.

Converts base emotions, intensity levels, and specific emotions into
musical parameters like key, mode, tempo, and chord progressions.
"""

from typing import Dict, Optional, List
import random


def emotion_to_mode(base: str, intensity: str, sub: Optional[str] = None) -> str:
    """
    Map emotion to musical mode.

    Args:
        base: Base emotion (sad, happy, fear, angry, etc.)
        intensity: Intensity level (low, moderate, high, intense)
        sub: Specific emotion (grief, joy, etc.)

    Returns:
        Mode name (Aeolian, Ionian, Phrygian, Locrian, etc.)
    """
    base_lower = base.lower()
    intensity_lower = intensity.lower()
    sub_lower = sub.lower() if sub else ""

    # Specific emotion overrides
    if "grief" in sub_lower:
        return "Aeolian"  # Natural minor - most sorrowful
    if "joy" in sub_lower or "euphoria" in sub_lower:
        return "Ionian"  # Major - brightest
    if "rage" in sub_lower or "fury" in sub_lower:
        return "Locrian"  # Most unstable, tense
    if "anxiety" in sub_lower or "panic" in sub_lower:
        return "Phrygian"  # Dark, Spanish, tense

    # Base emotion mapping
    if base_lower in ["sad", "sadness", "melancholy", "depression"]:
        return "Aeolian"  # Natural minor
    elif base_lower in ["happy", "happiness", "joy", "elation"]:
        return "Ionian"  # Major
    elif base_lower in ["fear", "anxiety", "worry", "dread"]:
        return "Phrygian"  # Dark, tense
    elif base_lower in ["angry", "anger", "rage", "fury"]:
        return "Locrian"  # Most unstable
    elif base_lower in ["calm", "peaceful", "serene"]:
        return "Ionian"  # Major, but could be Lydian for dreamy
    elif base_lower in ["excited", "energetic", "intense"]:
        return "Mixolydian"  # Bluesy, energetic
    else:
        # Default based on intensity
        if intensity_lower in ["low", "moderate"]:
            return "Aeolian"  # Default to minor
        else:
            return "Ionian"  # Default to major


def emotion_to_tempo(intensity: str, base: Optional[str] = None) -> int:
    """
    Map intensity to tempo range and return a value.

    Args:
        intensity: Intensity level (low, moderate, high, intense)
        base: Optional base emotion for fine-tuning

    Returns:
        Tempo in BPM
    """
    intensity_lower = intensity.lower()

    # Intensity-based tempo mapping
    if intensity_lower == "low":
        tempo_range = (60, 80)
    elif intensity_lower == "moderate":
        tempo_range = (80, 100)
    elif intensity_lower == "high":
        tempo_range = (100, 120)
    elif intensity_lower == "intense":
        tempo_range = (120, 140)
    else:
        # Default moderate
        tempo_range = (80, 100)

    # Fine-tune based on base emotion
    if base:
        base_lower = base.lower()
        if base_lower in ["sad", "grief", "melancholy"]:
            # Slow down sad emotions
            tempo_range = (max(50, tempo_range[0] - 10), max(70, tempo_range[1] - 10))
        elif base_lower in ["angry", "rage", "fury"]:
            # Speed up angry emotions
            tempo_range = (min(120, tempo_range[0] + 20), min(180, tempo_range[1] + 20))

    # Return random value in range
    return random.randint(tempo_range[0], tempo_range[1])


def emotion_to_progression(base: str, intensity: str, sub: Optional[str] = None) -> List[str]:
    """
    Map emotion to chord progression (Roman numerals).

    Args:
        base: Base emotion
        intensity: Intensity level
        sub: Specific emotion

    Returns:
        List of Roman numeral chord symbols (e.g., ["i", "VI", "III", "VII"])
    """
    base_lower = base.lower()
    intensity_lower = intensity.lower()
    sub_lower = sub.lower() if sub else ""

    # Specific emotion progressions
    if "grief" in sub_lower:
        return ["i", "VI", "III", "VII"]  # i-VI-III-VII (sad but with borrowed chords)
    if "joy" in sub_lower or "euphoria" in sub_lower:
        return ["I", "V", "vi", "IV"]  # I-V-vi-IV (pop progression)
    if "longing" in sub_lower:
        return ["vi", "IV", "I", "V"]  # vi-IV-I-V (melancholy but hopeful)
    if "rage" in sub_lower:
        return ["i", "bII", "bVII", "i"]  # i-bII-bVII-i (tense, unresolved)

    # Base emotion progressions
    if base_lower in ["sad", "sadness", "melancholy"]:
        if intensity_lower == "intense":
            return ["i", "VI", "III", "VII"]  # i-VI-III-VII
        else:
            return ["i", "iv", "VI", "V"]  # i-iv-VI-V (classic minor)

    elif base_lower in ["happy", "happiness", "joy"]:
        return ["I", "V", "vi", "IV"]  # I-V-vi-IV (pop)

    elif base_lower in ["fear", "anxiety"]:
        return ["i", "bII", "bVII", "i"]  # i-bII-bVII-i (Phrygian, tense)

    elif base_lower in ["angry", "anger"]:
        return ["i", "bII", "bVII", "i"]  # i-bII-bVII-i (Locrian/Phrygian)

    elif base_lower in ["calm", "peaceful"]:
        return ["I", "IV", "I", "V"]  # I-IV-I-V (simple, resolved)

    else:
        # Default progression
        return ["I", "V", "vi", "IV"]


def emotion_to_key(base: str, intensity: str, sub: Optional[str] = None) -> str:
    """
    Suggest a key based on emotion.

    Args:
        base: Base emotion
        intensity: Intensity level
        sub: Specific emotion

    Returns:
        Key name (C, F, D, etc.)
    """
    # For now, return a sensible default
    # Could be enhanced with more sophisticated mapping
    keys = ["C", "D", "E", "F", "G", "A", "Bb"]

    # Prefer certain keys for certain emotions
    if base.lower() in ["sad", "grief", "melancholy"]:
        # F minor, D minor are common for sad songs
        return random.choice(["F", "D", "A", "C"])
    elif base.lower() in ["happy", "joy"]:
        return random.choice(["C", "G", "F", "D"])
    else:
        return random.choice(keys)


def map_emotion_to_music(
    base: str,
    intensity: str,
    sub: Optional[str] = None
) -> Dict[str, any]:
    """
    Map emotion to complete musical parameters.

    Args:
        base: Base emotion (sad, happy, fear, angry)
        intensity: Intensity level (low, moderate, high, intense)
        sub: Specific emotion (grief, joy, etc.)

    Returns:
        Dict with:
        - key: Musical key (e.g., "F")
        - mode: Mode name (e.g., "Aeolian")
        - tempo: Tempo in BPM (e.g., 82)
        - progression: List of Roman numeral chords (e.g., ["i", "VI", "III", "VII"])
        - dynamics: Suggested dynamic level (pp, p, mp, mf, f, ff)
    """
    mode = emotion_to_mode(base, intensity, sub)
    tempo = emotion_to_tempo(intensity, base)
    progression = emotion_to_progression(base, intensity, sub)
    key = emotion_to_key(base, intensity, sub)

    # Map intensity to dynamics
    intensity_lower = intensity.lower()
    if intensity_lower == "low":
        dynamics = "p"  # piano (soft)
    elif intensity_lower == "moderate":
        dynamics = "mf"  # mezzo-forte (medium)
    elif intensity_lower == "high":
        dynamics = "f"  # forte (loud)
    elif intensity_lower == "intense":
        dynamics = "ff"  # fortissimo (very loud)
    else:
        dynamics = "mf"

    return {
        "key": key,
        "mode": mode,
        "tempo": tempo,
        "progression": progression,
        "dynamics": dynamics,
    }
