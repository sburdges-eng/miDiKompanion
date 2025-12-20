"""
Kelly MIDI Companion - Python Interface

High-level Python wrapper around the C++ kelly_bridge module.
Provides a clean, Pythonic API for emotion-driven MIDI generation.
"""

try:
    from ._bridge import (
        KellyBrain,
        Wound,
        EmotionNode,
        IntentResult,
        MidiNote,
        Chord,
        GeneratedMidi,
        MusicalParameters,
        EmotionCategory,
        RuleBreakType,
        IntentPipeline,
        EmotionThesaurus,
        SideA,
        SideB,
        midi_note_to_name,
        note_name_to_midi,
        ticks_to_ms,
        ms_to_ticks,
        category_to_string,
    )
except ImportError:
    # Try importing from parent directory if _bridge is not in package
    try:
        import sys
        import os
        # Add parent directory to path
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from kelly_bridge import (
            KellyBrain,
            Wound,
            EmotionNode,
            IntentResult,
            MidiNote,
            Chord,
            GeneratedMidi,
            EmotionCategory,
            RuleBreakType,
            IntentPipeline,
            EmotionThesaurus,
            midi_note_to_name,
            note_name_to_midi,
            ticks_to_ms,
            ms_to_ticks,
            category_to_string,
        )
    except ImportError as e:
        raise ImportError(
            "Could not import kelly_bridge. Make sure the C++ bridge is built.\n"
            "Build with: cmake -B build -DBUILD_PYTHON_BRIDGE=ON && cmake --build build\n"
            f"Original error: {e}"
        )

__version__ = "2.0.0"
__all__ = [
    "KellyBrain",
    "Wound",
    "EmotionNode",
    "IntentResult",
    "MidiNote",
    "Chord",
    "GeneratedMidi",
    "MusicalParameters",
    "EmotionCategory",
    "RuleBreakType",
    "IntentPipeline",
    "EmotionThesaurus",
    "SideA",
    "SideB",
    "midi_note_to_name",
    "note_name_to_midi",
    "ticks_to_ms",
    "ms_to_ticks",
    "category_to_string",
]
