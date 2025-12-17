"""
DAiW Chord Generator
====================

Generates chord progressions based on emotional intent, technical constraints,
and rule-breaking directives.

Uses the rule_breaking_masterpieces.md database for intentional rule violations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    from midiutil import MIDIFile
    HAS_MIDIUTIL = True
except ImportError:
    HAS_MIDIUTIL = False


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chord:
    """Represents a single chord."""
    root: str
    quality: str
    midi_notes: List[int]
    duration_beats: float = 4.0
    velocity: int = 90
    roman_numeral: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

NOTE_TO_MIDI = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11
}

CHORD_INTERVALS = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dom7": [0, 4, 7, 10],
    "7": [0, 4, 7, 10],
    "dim7": [0, 3, 6, 9],
    "5": [0, 7],  # Power chord
}

# Scale degrees in major
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]

# Roman numeral to scale degree
ROMAN_TO_DEGREE = {
    "I": 0, "II": 1, "III": 2, "IV": 3, "V": 4, "VI": 5, "VII": 6,
    "i": 0, "ii": 1, "iii": 2, "iv": 3, "v": 4, "vi": 5, "vii": 6,
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def note_to_midi(note: str, octave: int = 4) -> int:
    """Convert note name to MIDI number."""
    base = NOTE_TO_MIDI.get(note, 0)
    return base + (octave + 1) * 12


def chord_to_midi_notes(root: str, quality: str, octave: int = 4) -> List[int]:
    """Build MIDI notes for a chord."""
    intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS["maj"])
    root_midi = note_to_midi(root, octave)
    return [root_midi + interval for interval in intervals]


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE ACCESS
# ═══════════════════════════════════════════════════════════════════════════════

def load_chord_database() -> Dict[str, Any]:
    """Load the chord progression database."""
    db_path = Path(__file__).parent.parent / "data" / "chord_progressions.json"
    
    if db_path.exists():
        with open(db_path, 'r') as f:
            return json.load(f)
    
    # Return default database if file not found
    return get_default_database()


def get_default_database() -> Dict[str, Any]:
    """Return the default chord progression database."""
    return {
        "narrative_arc_progressions": {
            "Climb-to-Climax": {
                "description": "Gradual emotional build",
                "patterns": [
                    {"name": "Standard Build", "roman": ["I", "IV", "vi", "V"]},
                    {"name": "Minor Build", "roman": ["i", "VI", "III", "VII"]},
                ]
            },
            "Slow Reveal": {
                "description": "Subtle progression that unfolds",
                "patterns": [
                    {"name": "Gentle Reveal", "roman": ["I", "iii", "IV", "I"]},
                ]
            },
            "Repetitive Despair": {
                "description": "Circular, non-resolving",
                "patterns": [
                    {"name": "Loop", "roman": ["i", "VI", "i", "VI"]},
                ]
            },
            "Sudden Shift": {
                "description": "Dramatic tonal shift",
                "patterns": [
                    {
                        "name": "Two Worlds",
                        "blocks": {
                            "A": {"roman": ["I", "V", "vi", "IV"]},
                            "B": {"roman": ["i", "iv", "VI", "V"]}
                        },
                        "structure": ["A", "A", "B", "A-modified"]
                    }
                ]
            },
            "Static Reflection": {
                "description": "Minimal harmonic movement",
                "patterns": [
                    {"name": "Drone", "roman": ["I", "I", "I", "IV"]},
                ]
            },
        },
        "mood_chord_mapping": {
            "grief": {"preferred_roots": ["i", "VI", "iv"], "avoid": ["V"]},
            "anger": {"preferred_roots": ["i", "bVII", "bVI"], "avoid": ["IV"]},
            "nostalgia": {"preferred_roots": ["I", "IV", "vi"], "avoid": []},
        },
        "rule_breaking_implementations": {
            "HARMONY_ModalInterchange": {
                "description": "Borrow chords from parallel mode",
                "technique": "Replace IV with iv, or add bVII, bVI",
                "example": "In F major: Replace Bb with Bbm",
            },
            "HARMONY_ParallelMotion": {
                "description": "Move chords in parallel fifths",
                "technique": "Use power chords or parallel triads",
            },
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_progression(payload: Dict[str, Any]) -> List[Chord]:
    """
    Generate a chord progression based on the payload.
    
    Args:
        payload: Dictionary containing all generation parameters
    
    Returns:
        List of Chord objects
    """
    db = load_chord_database()
    
    # Extract parameters
    mood_primary = payload.get("mood_primary", "")
    narrative_arc = payload.get("narrative_arc", "Climb-to-Climax")
    harmonic_complexity = payload.get("harmonic_complexity", "Moderate")
    tension = payload.get("mood_secondary_tension", 0.5)
    key = payload.get("technical_key", "C")
    mode = payload.get("technical_mode", "major")
    rule_to_break = payload.get("technical_rule_to_break", "NONE")
    song_length_bars = payload.get("song_length_bars", 64)
    
    # Select base progression
    pattern = _select_progression_pattern(db, narrative_arc, mood_primary)
    
    # Build chord sequence
    chords = _build_chord_sequence(pattern, key, mode, harmonic_complexity)
    
    # Apply rule-breaking
    if rule_to_break != "NONE":
        chords = _apply_rule_break(chords, rule_to_break, db, tension)
    
    # Extend to song length
    chords = _extend_to_length(chords, song_length_bars)
    
    # Apply tension curve
    chords = _apply_tension_curve(chords, narrative_arc, tension)
    
    return chords


def _select_progression_pattern(
    db: Dict[str, Any],
    narrative_arc: str,
    mood: str
) -> Dict[str, Any]:
    """Select a progression pattern based on narrative arc."""
    arc_progressions = db.get("narrative_arc_progressions", {})
    
    if narrative_arc in arc_progressions:
        arc_data = arc_progressions[narrative_arc]
        patterns = arc_data.get("patterns", [])
        if patterns:
            return patterns[0]
    
    # Default fallback
    return {"name": "Default", "roman": ["I", "V", "vi", "IV"]}


def _build_chord_sequence(
    pattern: Dict[str, Any],
    key: str,
    mode: str,
    complexity: str
) -> List[Chord]:
    """Build chord objects from a pattern."""
    chords = []
    
    # Handle block-based patterns
    if "blocks" in pattern:
        blocks = pattern["blocks"]
        structure = pattern.get("structure", ["A", "B"])
        
        for block_name in structure:
            block_key = block_name.replace("-modified", "")
            if block_key in blocks:
                block = blocks[block_key]
                for numeral in block.get("roman", []):
                    chord = _roman_to_chord(numeral, key, mode, complexity)
                    chords.append(chord)
    else:
        # Simple pattern
        for numeral in pattern.get("roman", ["I", "IV", "V", "I"]):
            chord = _roman_to_chord(numeral, key, mode, complexity)
            chords.append(chord)
    
    return chords


def _roman_to_chord(numeral: str, key: str, mode: str, complexity: str) -> Chord:
    """Convert a roman numeral to a Chord object."""
    # Parse the numeral
    clean_numeral = numeral.replace("dim", "").replace("7", "").replace("5", "").replace("b", "").replace("#", "")
    
    # Determine root offset
    major_scale_roots = {
        "I": 0, "II": 2, "III": 4, "IV": 5, "V": 7, "VI": 9, "VII": 11,
    }
    
    # Handle flats and sharps
    root_offset = major_scale_roots.get(clean_numeral.upper(), 0)
    if numeral.startswith("b"):
        root_offset -= 1
    elif numeral.startswith("#"):
        root_offset += 1
    
    # Get key root
    root_midi_base = NOTE_TO_MIDI.get(key, 0)
    root_midi = (root_midi_base + root_offset) % 12
    
    # Find note name
    midi_to_note = {v: k for k, v in NOTE_TO_MIDI.items() if '#' not in k or k == "F#"}
    root_name = midi_to_note.get(root_midi, "C")
    
    # Determine quality
    is_minor = clean_numeral.islower()
    is_dim = "dim" in numeral
    is_seventh = "7" in numeral
    
    if is_dim:
        quality = "dim7" if is_seventh else "dim"
    elif is_seventh:
        quality = "min7" if is_minor else "dom7"
    else:
        quality = "min" if is_minor else "maj"
    
    # Build MIDI notes
    midi_notes = chord_to_midi_notes(root_name, quality, octave=4)
    
    return Chord(
        root=root_name,
        quality=quality,
        midi_notes=midi_notes,
        roman_numeral=numeral
    )


def _apply_rule_break(
    chords: List[Chord],
    rule: str,
    db: Dict[str, Any],
    tension: float
) -> List[Chord]:
    """Apply rule-breaking modifications."""
    if rule == "HARMONY_ModalInterchange":
        # Replace major IV with minor iv at certain points
        for i, chord in enumerate(chords):
            if chord.roman_numeral == "IV" and tension > 0.5:
                chords[i] = Chord(
                    root=chord.root,
                    quality="min",
                    midi_notes=chord_to_midi_notes(chord.root, "min"),
                    roman_numeral="iv"
                )
    
    elif rule == "HARMONY_ParallelMotion":
        # Convert to power chords
        for i, chord in enumerate(chords):
            chords[i] = Chord(
                root=chord.root,
                quality="5",
                midi_notes=chord_to_midi_notes(chord.root, "5"),
                roman_numeral=chord.roman_numeral + "5"
            )
    
    return chords


def _extend_to_length(chords: List[Chord], target_bars: int) -> List[Chord]:
    """Extend chord sequence to target length."""
    if not chords:
        return chords
    
    bars_per_chord = 2  # Assume 2 bars per chord
    needed = target_bars // bars_per_chord
    
    extended = []
    while len(extended) < needed:
        extended.extend(chords)
    
    return extended[:needed]


def _apply_tension_curve(
    chords: List[Chord],
    narrative_arc: str,
    base_tension: float
) -> List[Chord]:
    """Apply velocity/tension adjustments based on arc."""
    if not chords:
        return chords
    
    for i, chord in enumerate(chords):
        position = i / len(chords)
        
        if narrative_arc == "Climb-to-Climax":
            # Gradual increase
            tension_mult = 0.7 + (position * 0.5)
        elif narrative_arc == "Sudden Shift":
            # Steady then jump
            tension_mult = 0.8 if position < 0.7 else 1.2
        else:
            tension_mult = 1.0
        
        chord.velocity = int(min(127, 70 + (base_tension * tension_mult * 50)))
    
    return chords


# ═══════════════════════════════════════════════════════════════════════════════
# MIDI EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_to_midi(
    chords: List[Chord],
    output_path: str,
    tempo: int = 120
) -> str:
    """Export chord progression to MIDI file."""
    if not HAS_MIDIUTIL:
        raise ImportError("midiutil required: pip install midiutil")
    
    midi = MIDIFile(1)
    midi.addTrackName(0, 0, "Chord Progression")
    midi.addTempo(0, 0, tempo)
    
    time = 0
    for chord in chords:
        for note in chord.midi_notes:
            midi.addNote(
                track=0,
                channel=0,
                pitch=note,
                time=time,
                duration=chord.duration_beats,
                volume=chord.velocity
            )
        time += chord.duration_beats
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, "wb") as f:
        midi.writeFile(f)
    
    return str(output)
