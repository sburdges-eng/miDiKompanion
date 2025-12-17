"""
Chord Progression Parser
========================
Parses chord symbols like "Cm", "F#maj7", "Bbm7" into structured data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Note name to semitone offset from C
NOTE_MAP = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 0,
}

# Chord quality patterns and their interval structures
CHORD_QUALITIES = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "m": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "m7": (0, 3, 7, 10),
    "7": (0, 4, 7, 10),
    "dim7": (0, 3, 6, 9),
    "hdim7": (0, 3, 6, 10),  # half-diminished
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "add9": (0, 4, 7, 14),
    "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9),
    "9": (0, 4, 7, 10, 14),
    "maj9": (0, 4, 7, 11, 14),
    "m9": (0, 3, 7, 10, 14),
}


@dataclass
class ParsedChord:
    """Represents a parsed chord symbol."""
    original: str
    root: str
    root_num: int  # semitone offset from C
    quality: str
    intervals: Tuple[int, ...]
    bass_note: Optional[str] = None
    bass_num: Optional[int] = None


def _parse_note(note_str: str) -> Optional[Tuple[str, int]]:
    """Parse a note name and return (name, semitone)."""
    note_str = note_str.strip()
    if not note_str:
        return None
    
    # Try two-character notes first (with accidental)
    if len(note_str) >= 2 and note_str[:2] in NOTE_MAP:
        return note_str[:2], NOTE_MAP[note_str[:2]]
    
    # Single character note
    if note_str[0].upper() in NOTE_MAP:
        return note_str[0].upper(), NOTE_MAP[note_str[0].upper()]
    
    return None


def parse_chord(chord_str: str) -> Optional[ParsedChord]:
    """
    Parse a single chord symbol.
    
    Examples:
        "C" -> C major
        "Am" -> A minor
        "F#m7" -> F# minor 7
        "Bb/D" -> Bb major with D bass
    """
    if not chord_str:
        return None
    
    chord_str = chord_str.strip()
    original = chord_str
    
    # Handle slash chords (bass note)
    bass_note = None
    bass_num = None
    if "/" in chord_str:
        parts = chord_str.split("/")
        chord_str = parts[0]
        if len(parts) > 1:
            bass_result = _parse_note(parts[1])
            if bass_result:
                bass_note, bass_num = bass_result
    
    # Parse root note
    root_result = _parse_note(chord_str)
    if not root_result:
        return None
    
    root, root_num = root_result
    
    # Get remaining string (quality)
    remaining = chord_str[len(root):]
    
    # Determine quality
    quality = "maj"
    intervals = CHORD_QUALITIES["maj"]
    
    # Check for explicit qualities
    remaining_lower = remaining.lower()
    
    for q in sorted(CHORD_QUALITIES.keys(), key=len, reverse=True):
        if remaining_lower.startswith(q.lower()):
            quality = q
            intervals = CHORD_QUALITIES[q]
            break
    else:
        # Check if lowercase letter implies minor
        if remaining and remaining[0] == "m" and not remaining.startswith("maj"):
            quality = "m"
            intervals = CHORD_QUALITIES["m"]
    
    return ParsedChord(
        original=original,
        root=root,
        root_num=root_num,
        quality=quality,
        intervals=intervals,
        bass_note=bass_note,
        bass_num=bass_num
    )


def parse_progression_string(progression: str) -> List[ParsedChord]:
    """
    Parse a progression string like "F - C - Am - Dm" into chord objects.
    """
    # Split on common delimiters
    chords_raw = re.split(r'[\s\-,|]+', progression)
    
    parsed = []
    for chord_str in chords_raw:
        chord_str = chord_str.strip()
        if chord_str:
            result = parse_chord(chord_str)
            if result:
                parsed.append(result)
    
    return parsed


def chord_to_midi_notes(chord: ParsedChord, octave: int = 4) -> List[int]:
    """
    Convert a parsed chord to MIDI note numbers.
    
    Args:
        chord: Parsed chord object
        octave: Base octave (4 = middle C)
    
    Returns:
        List of MIDI note numbers
    """
    base_midi = 12 * (octave + 1) + chord.root_num  # C4 = 60
    return [base_midi + interval for interval in chord.intervals]


def analyze_progression(progression: str) -> dict:
    """
    Analyze a chord progression and return key information.
    """
    chords = parse_progression_string(progression)
    
    if not chords:
        return {"error": "No valid chords found"}
    
    # Count root notes
    root_counts = {}
    for chord in chords:
        root_counts[chord.root] = root_counts.get(chord.root, 0) + 1
    
    # Estimate key (most common root, or first chord)
    estimated_key = max(root_counts, key=root_counts.get)
    
    # Convert to Roman numerals
    roman_numerals = []
    key_root_num = NOTE_MAP.get(estimated_key, 0)
    
    numeral_map = ["I", "bII", "II", "bIII", "III", "IV", "#IV/bV", "V", "bVI", "VI", "bVII", "VII"]
    
    for chord in chords:
        interval = (chord.root_num - key_root_num) % 12
        numeral = numeral_map[interval]
        
        # Adjust for minor
        if chord.quality in ["m", "min", "min7", "m7"]:
            numeral = numeral.lower()
        elif chord.quality in ["dim", "dim7"]:
            numeral = numeral.lower() + "°"
        
        roman_numerals.append(numeral)
    
    return {
        "chords": [c.original for c in chords],
        "estimated_key": estimated_key,
        "roman_numerals": roman_numerals,
        "progression_string": " - ".join(roman_numerals),
        "num_chords": len(chords),
    }
