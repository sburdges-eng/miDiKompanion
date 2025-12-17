"""
miDEE - Music Generation and Processing Engine

A comprehensive music generation and analysis toolkit:
- Groove extraction and application
- Chord progression analysis and generation
- Section detection
- Feel/timing analysis
- DAW integration
- Emotion-to-music pipeline
- Lyrical fragment generation
- Reference track DNA analysis
- AI-powered instrument education

Part of miDiKompanion therapeutic iDAW system.
"""

__version__ = "1.0.0"
__author__ = "miDiKompanion Development Team"

from midee.groove import extract_groove, apply_groove, GrooveTemplate
from midee.structure import analyze_chords, detect_sections, ChordProgression
from midee.audio import analyze_feel, AudioFeatures

# Harmony generation
from midee.harmony import HarmonyGenerator, HarmonyResult, generate_midi_from_harmony

# New comprehensive engine exports
from midee.structure.comprehensive_engine import (
    AffectAnalyzer,
    TherapySession,
    HarmonyPlan,
    render_plan_to_midi,
)
from midee.groove_engine import apply_groove as apply_groove_events
from midee.text.lyrical_mirror import generate_lyrical_fragments

# Emotion API (Emotion -> Music -> Mixer)
from midee.emotion_api import (
    MusicBrain,
    GeneratedMusic,
    FluentChain,
    quick_generate,
    quick_export,
    INTENT_EXAMPLES,
)

# Emotional mapping
from midee.data.emotional_mapping import (
    EmotionalState,
    MusicalParameters,
    Valence,
    Arousal,
    TimingFeel,
    Mode,
    EMOTIONAL_PRESETS,
    get_parameters_for_state,
)

# Mixer parameters
from midee.daw.mixer_params import (
    MixerParameters,
    EmotionMapper,
    export_to_logic_automation,
    MIXER_PRESETS,
)

# Learning Module - AI-powered instrument education
from midee.learning import (
    DifficultyLevel,
    InstrumentFamily,
    Instrument,
    INSTRUMENTS,
    get_instrument,
    get_instruments_by_family,
    get_beginner_instruments,
    CurriculumBuilder,
    LearningPath,
    PedagogyEngine,
    AdaptiveTeacher,
    generate_learning_plan,
)

__all__ = [
    # Groove (file-based)
    "extract_groove",
    "apply_groove",
    "GrooveTemplate",
    # Groove (event-based)
    "apply_groove_events",
    # Structure
    "analyze_chords",
    "detect_sections",
    "ChordProgression",
    # Audio
    "analyze_feel",
    "AudioFeatures",
    # Harmony
    "HarmonyGenerator",
    "HarmonyResult",
    "generate_midi_from_harmony",
    # Comprehensive Engine
    "AffectAnalyzer",
    "TherapySession",
    "HarmonyPlan",
    "render_plan_to_midi",
    # Text/Lyrical
    "generate_lyrical_fragments",
    # Emotion API
    "MusicBrain",
    "GeneratedMusic",
    "FluentChain",
    "quick_generate",
    "quick_export",
    "INTENT_EXAMPLES",
    # Emotional mapping
    "EmotionalState",
    "MusicalParameters",
    "Valence",
    "Arousal",
    "TimingFeel",
    "Mode",
    "EMOTIONAL_PRESETS",
    "get_parameters_for_state",
    # Mixer parameters
    "MixerParameters",
    "EmotionMapper",
    "export_to_logic_automation",
    "MIXER_PRESETS",
    # Learning Module
    "DifficultyLevel",
    "InstrumentFamily",
    "Instrument",
    "INSTRUMENTS",
    "get_instrument",
    "get_instruments_by_family",
    "get_beginner_instruments",
    "CurriculumBuilder",
    "LearningPath",
    "PedagogyEngine",
    "AdaptiveTeacher",
    "generate_learning_plan",
]
