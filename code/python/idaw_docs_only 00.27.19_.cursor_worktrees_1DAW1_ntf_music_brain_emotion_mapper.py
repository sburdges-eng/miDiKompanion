"""
Emotion to Music Mapper
Maps emotional states to musical parameters (mode, tempo, progression, dynamics)
Supports all 216 emotion nodes from the 6x6x6 emotion thesaurus.
"""

from typing import Dict, Any, Optional, List
import random

# EMOTION_TO_MODE mapping - maps base emotions to musical modes
EMOTION_TO_MODE = {
    "sad": "aeolian",           # Natural minor - melancholic, sorrowful
    "sadness": "aeolian",
    "happy": "ionian",          # Major - bright, uplifting
    "happiness": "ionian",
    "fear": "phrygian",         # Dark, tense, mysterious
    "angry": "locrian",         # Most dissonant - unstable, aggressive
    "anger": "locrian",
    "disgust": "dorian",        # Minor with raised 6th - conflicted
    "surprise": "lydian",       # Major with raised 4th - dreamy, floating
    "neutral": "ionian",        # Default to major
}

# INTENSITY_TO_TEMPO mapping - maps intensity levels to tempo (BPM)
INTENSITY_TO_TEMPO = {
    "low": 65,
    "subtle": 65,
    "mild": 75,
    "moderate": 90,
    "medium": 90,
    "high": 115,
    "strong": 115,
    "intense": 130,
    "extreme": 145,
    "overwhelming": 160,
}

# SUB_EMOTION_PROGRESSIONS - specific progressions for emotional nuances
SUB_EMOTION_PROGRESSIONS = {
    # Grief and loss
    "grief": ["i", "VI", "III", "VII"],          # F minor: F-Db-Ab-Eb (i-VI-III-VII)
    "bereaved": ["i", "VI", "III", "VII"],
    "mournful": ["i", "bVI", "bIII", "bVII"],    # Minor with borrowed chords
    "heartbroken": ["i", "VI", "iv", "VII"],     # Adds the iv for heartbreak

    # Joy and happiness
    "joy": ["I", "V", "vi", "IV"],               # Classic pop progression
    "elation": ["I", "V", "I", "V"],             # Simple, triumphant
    "euphoria": ["I", "V", "vi", "IV", "I", "V"], # Extended joy

    # Anger and rage
    "rage": ["i", "bII", "bVII", "i"],           # Phrygian/Locrian - tense
    "fury": ["i", "bII", "bVII", "i"],
    "wrath": ["i", "bII", "bVII", "bIII", "i"],  # Extended tension

    # Fear and anxiety
    "terror": ["i", "bII", "bVII", "i"],         # Phrygian - unsettling
    "anxiety": ["i", "bVI", "bVII", "i"],        # Minor with borrowed chords
    "dread": ["i", "bII", "bVII", "bVI", "i"],   # Extended fear

    # Disgust
    "revulsion": ["i", "bII", "bVII", "i"],      # Tense, uncomfortable
    "contempt": ["i", "bVI", "bIII", "bVII"],    # Dark minor

    # Surprise
    "astonishment": ["I", "bVII", "IV", "I"],    # Unexpected chord changes
    "wonder": ["I", "IV", "I", "V"],             # Open, spacious
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
        Mode name (aeolian, ionian, phrygian, locrian, dorian, lydian)
    """
    emotion_lower = base_emotion.lower()

    # Direct match first
    if emotion_lower in EMOTION_TO_MODE:
        return EMOTION_TO_MODE[emotion_lower]

    # Partial matches
    for key, mode in EMOTION_TO_MODE.items():
        if key in emotion_lower or emotion_lower in key:
            return mode

    # Default to Ionian (major)
    return "ionian"


def emotion_to_tempo(intensity: str, base_emotion: Optional[str] = None, specific_emotion: Optional[str] = None) -> int:
    """
    Map intensity level to tempo. Special handling for grief/intense → 82 BPM (Kelly song).

    Args:
        intensity: Intensity level (low, moderate, high, intense, etc.)
        base_emotion: Base emotion (for special cases)
        specific_emotion: Specific emotion (for special cases)

    Returns:
        Tempo in BPM
    """
    intensity_lower = intensity.lower()

    # Special case: grief with intense intensity → 82 BPM (Kelly song spec)
    if specific_emotion and "grief" in specific_emotion.lower():
        if intensity_lower in ["intense", "strong", "high"]:
            return 82

    # Use mapping or find closest match
    if intensity_lower in INTENSITY_TO_TEMPO:
        return INTENSITY_TO_TEMPO[intensity_lower]

    # Find closest intensity level
    intensity_order = ["low", "subtle", "mild", "moderate", "medium", "high", "strong", "intense", "extreme", "overwhelming"]
    for level in intensity_order:
        if level in intensity_lower:
            return INTENSITY_TO_TEMPO[level]

    # Default to moderate
    return 90


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

    # Check sub-emotion progressions first (most specific)
    for sub_emotion, progression in SUB_EMOTION_PROGRESSIONS.items():
        if sub_emotion in specific or sub_emotion in emotion_lower:
            return progression

    # Base emotion progressions
    if "grief" in emotion_lower or "grief" in specific:
        return ["i", "VI", "III", "VII"]  # Kelly song progression

    if "sad" in emotion_lower or "melancholy" in emotion_lower:
        progressions = [
            ["i", "bVI", "bIII", "bVII"],  # Minor with borrowed chords
            ["i", "iv", "bVI", "V"],        # Minor with iv
            ["vi", "IV", "I", "V"],         # Relative minor progression
        ]
        return random.choice(progressions)

    if "joy" in emotion_lower or "joy" in specific or "happy" in emotion_lower:
        return ["I", "V", "vi", "IV"]  # Classic pop progression

    if "fear" in emotion_lower or "anxiety" in emotion_lower:
        return ["i", "bII", "bVII", "i"]  # Phrygian progression

    if "angry" in emotion_lower or "rage" in emotion_lower:
        return ["i", "bII", "bVII", "i"]  # Locrian/Phrygian, tense

    if "disgust" in emotion_lower:
        return ["i", "bII", "bVII", "i"]  # Tense, uncomfortable

    if "surprise" in emotion_lower:
        return ["I", "bVII", "IV", "I"]  # Unexpected changes

    # Default progression
    return ["I", "V", "vi", "IV"]


def get_key_from_emotion(base_emotion: str, specific_emotion: Optional[str] = None) -> str:
    """
    Get appropriate musical key from emotion. Grief tends toward F minor.

    Args:
        base_emotion: Base emotion
        specific_emotion: Specific emotion variant

    Returns:
        Key name (e.g., "F", "C", "G")
    """
    emotion_lower = base_emotion.lower()
    specific = specific_emotion.lower() if specific_emotion else ""

    # Grief → F minor (Kelly song reference)
    if "grief" in emotion_lower or "grief" in specific:
        return "F"

    # Sad/melancholy → prefer F, C, A minor
    if "sad" in emotion_lower or "melancholy" in emotion_lower:
        return random.choice(["F", "C", "A", "D"])

    # Joy/happy → prefer C, G, F major
    if "happy" in emotion_lower or "joy" in emotion_lower:
        return random.choice(["C", "G", "F", "D"])

    # Fear/anxiety → prefer darker keys
    if "fear" in emotion_lower or "anxiety" in emotion_lower:
        return random.choice(["Dm", "Em", "Am"])

    # Anger → prefer sharp keys or dissonant
    if "angry" in emotion_lower:
        return random.choice(["E", "B", "F#"])

    # Default: C
    return "C"


def get_dynamics_from_intensity(intensity: str) -> str:
    """
    Map intensity to dynamic marking.

    Args:
        intensity: Intensity level

    Returns:
        Dynamic marking (pp, p, mp, mf, f, ff)
    """
    intensity_lower = intensity.lower()

    dynamics_map = {
        "low": "p",              # Piano (soft)
        "subtle": "pp",          # Pianissimo (very soft)
        "mild": "p",             # Piano
        "moderate": "mf",        # Mezzo forte (moderate)
        "medium": "mf",
        "high": "f",             # Forte (loud)
        "strong": "f",
        "intense": "ff",         # Fortissimo (very loud)
        "extreme": "ff",
        "overwhelming": "fff",   # Fortississimo (extremely loud)
    }

    if intensity_lower in dynamics_map:
        return dynamics_map[intensity_lower]

    # Find closest match
    for level in ["overwhelming", "extreme", "intense", "strong", "high", "moderate", "medium", "mild", "subtle", "low"]:
        if level in intensity_lower:
            return dynamics_map.get(level, "mf")

    return "mf"  # Default


def get_articulation_from_emotion(base_emotion: str, intensity: str) -> str:
    """
    Map emotion to articulation style.

    Args:
        base_emotion: Base emotion
        intensity: Intensity level

    Returns:
        Articulation marking (legato, staccato, tenuto, etc.)
    """
    emotion_lower = base_emotion.lower()
    intensity_lower = intensity.lower()

    # High intensity → more detached, staccato
    if intensity_lower in ["intense", "extreme", "overwhelming"]:
        if "angry" in emotion_lower or "fear" in emotion_lower:
            return "staccato"

    # Low intensity → legato, smooth
    if intensity_lower in ["low", "subtle", "mild"]:
        if "sad" in emotion_lower or "melancholy" in emotion_lower:
            return "legato"

    # Grief → expressive, legato with rubato
    if "grief" in emotion_lower:
        return "legato"

    # Joy → bouncy, staccato
    if "happy" in emotion_lower or "joy" in emotion_lower:
        if intensity_lower in ["high", "strong", "intense"]:
            return "staccato"

    # Default
    return "legato"


def map_emotion_to_music(
    base: str,
    intensity: str,
    sub: Optional[str] = None
) -> Dict[str, Any]:
    """
    Map emotional parameters to complete musical configuration.
    Supports all 216 emotion nodes from the 6x6x6 emotion thesaurus.

    Args:
        base: Base emotion (sad, happy, fear, angry, disgust, surprise, neutral)
        intensity: Intensity level (low, moderate, high, intense, extreme, overwhelming)
        sub: Specific emotion variant (grief, joy, rage, etc.)

    Returns:
        Dictionary with:
        - key: Musical key (e.g., "F", "C")
        - mode: Mode name (e.g., "aeolian", "ionian")
        - tempo: Tempo in BPM
        - progression: Chord progression as string (e.g., "i-VI-III-VII")
        - progression_list: Chord progression as list (e.g., ["i", "VI", "III", "VII"])
        - dynamics: Dynamic level (pp, p, mf, f, ff)
        - articulation: Articulation style (legato, staccato)
    """
    # Get key from emotion (grief → F)
    key = get_key_from_emotion(base, sub)

    # Map emotion to mode
    mode = emotion_to_mode(base)

    # Map intensity to tempo (special handling for grief/intense → 82 BPM)
    tempo = emotion_to_tempo(intensity, base, sub)

    # Map emotion to progression
    progression_list = emotion_to_progression(base, intensity, sub)
    # Convert list to string format for backward compatibility
    progression_str = "-".join(progression_list)

    # Map intensity to dynamics
    dynamics = get_dynamics_from_intensity(intensity)

    # Get articulation style
    articulation = get_articulation_from_emotion(base, intensity)

    return {
        "key": key,
        "mode": mode,
        "tempo": tempo,
        "progression": progression_str,  # String format for compatibility
        "progression_list": progression_list,  # Also provide as list
        "dynamics": dynamics,
        "articulation": articulation,
    }
