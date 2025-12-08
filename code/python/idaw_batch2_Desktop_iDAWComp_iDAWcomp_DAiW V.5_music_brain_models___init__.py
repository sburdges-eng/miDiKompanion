"""
Music Brain Models

Emotional → Musical parameter mapping for therapeutic composition.

Philosophy: "Interrogate Before Generate"
"""

from music_brain.models.emotional_mapping import (
    # Enums
    TimingFeel,
    Register,
    HarmonicRhythm,
    Density,
    # Dataclasses
    EmotionalState,
    MusicalParameters,
    # Presets
    EMOTIONAL_PRESETS,
    EMOTION_MODIFIERS,
    INTERVAL_EMOTIONS,
    CHORD_PROGRESSION_EMOTIONS,
    EMOTIONAL_STATE_PRESETS,
    # Functions
    get_parameters_for_state,
    get_interrogation_prompts,
    get_misdirection_technique,
)

__all__ = [
    # Enums
    "TimingFeel",
    "Register",
    "HarmonicRhythm",
    "Density",
    # Dataclasses
    "EmotionalState",
    "MusicalParameters",
    # Presets
    "EMOTIONAL_PRESETS",
    "EMOTION_MODIFIERS",
    "INTERVAL_EMOTIONS",
    "CHORD_PROGRESSION_EMOTIONS",
    "EMOTIONAL_STATE_PRESETS",
    # Functions
    "get_parameters_for_state",
    "get_interrogation_prompts",
    "get_misdirection_technique",
]
