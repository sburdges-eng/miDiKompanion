"""
Music Brain - Intelligent Music Analysis Toolkit

A Python package for music production analysis:
- Groove extraction and application
- Chord progression analysis
- Section detection
- Feel/timing analysis
- Audio analysis (BPM, key, chord detection)
- DAW integration
- Therapy-to-music pipeline (Comprehensive Engine)
- Lyrical fragment generation
- Reference track DNA analysis
- Real-time MIDI processing
"""

__version__ = "0.4.0"
__author__ = "Sean Burdges"

from music_brain.groove import extract_groove, apply_groove, GrooveTemplate
from music_brain.structure import analyze_chords, detect_sections, ChordProgression
from music_brain.audio import (
    analyze_feel,
    AudioFeatures,
    AudioAnalyzer,
    AudioAnalysis,
    ChordDetector,
    FrequencyAnalyzer,
    TheoryAnalyzer,
    detect_key,
    detect_bpm,
    detect_chords_from_audio,
    detect_scale,
    detect_mode,
    analyze_harmony,
    SCALES,
    MODE_CHARACTERISTICS,
)
from music_brain.harmony import HarmonyGenerator, HarmonyResult, generate_midi_from_harmony

# Comprehensive engine exports
from music_brain.structure.comprehensive_engine import (
    AffectAnalyzer,
    TherapySession,
    HarmonyPlan,
    render_plan_to_midi,
)
from music_brain.groove_engine import apply_groove as apply_groove_events
from music_brain.text.lyrical_mirror import generate_lyrical_fragments

# Real-time MIDI processing
from music_brain.realtime import (
    RealtimeMidiProcessor,
    MidiProcessorConfig,
    RealtimeEngine,
    RealtimeClock,
    EventScheduler,
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
    # Audio (feel)
    "analyze_feel",
    "AudioFeatures",
    # Audio (analysis)
    "AudioAnalyzer",
    "AudioAnalysis",
    "ChordDetector",
    "FrequencyAnalyzer",
    "TheoryAnalyzer",
    "detect_key",
    "detect_bpm",
    "detect_chords_from_audio",
    # Theory detection
    "detect_scale",
    "detect_mode",
    "analyze_harmony",
    "SCALES",
    "MODE_CHARACTERISTICS",
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
    # Real-time MIDI
    "RealtimeMidiProcessor",
    "MidiProcessorConfig",
    "RealtimeEngine",
    "RealtimeClock",
    "EventScheduler",
]
