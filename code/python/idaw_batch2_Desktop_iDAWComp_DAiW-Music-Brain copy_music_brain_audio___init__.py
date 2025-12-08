"""
Audio Analysis - Analyze feel and characteristics of audio files.

Features:
- Feel/groove analysis
- Energy extraction
- Tempo detection
- Key detection
- Chord detection from audio
- Spectral analysis
- Frequency/pitch analysis
- Reference track DNA analysis
"""

from music_brain.audio.feel import analyze_feel, AudioFeatures, compare_feel
from music_brain.audio.reference_dna import (
    analyze_reference,
    apply_reference_to_plan,
    ReferenceProfile,
)
from music_brain.audio.analyzer import (
    AudioAnalyzer,
    AudioAnalysis,
    KeyDetectionResult,
    BPMDetectionResult,
    AudioSegment,
    detect_key,
    detect_bpm,
    extract_features,
)
from music_brain.audio.chord_detection import (
    ChordDetector,
    ChordDetection,
    ChordProgressionDetection,
    detect_chords_from_audio,
)
from music_brain.audio.frequency import (
    FrequencyAnalyzer,
    FFTAnalysis,
    PitchDetection,
    HarmonicContent,
    freq_to_midi,
    midi_to_freq,
    midi_to_note_name,
    analyze_frequency_spectrum,
    detect_pitch_from_audio,
)
from music_brain.audio.theory_analyzer import (
    TheoryAnalyzer,
    TheoryAnalysis,
    ScaleDetection,
    TriadDetection,
    ArpeggioDetection,
    IntervalAnalysis,
    detect_scale,
    detect_mode,
    analyze_harmony,
    SCALES,
    TRIADS,
    SEVENTH_CHORDS,
    MODE_CHARACTERISTICS,
)

__all__ = [
    # Feel analysis
    "analyze_feel",
    "AudioFeatures",
    "compare_feel",
    # Reference DNA
    "analyze_reference",
    "apply_reference_to_plan",
    "ReferenceProfile",
    # Main analyzer
    "AudioAnalyzer",
    "AudioAnalysis",
    "KeyDetectionResult",
    "BPMDetectionResult",
    "AudioSegment",
    "detect_key",
    "detect_bpm",
    "extract_features",
    # Chord detection
    "ChordDetector",
    "ChordDetection",
    "ChordProgressionDetection",
    "detect_chords_from_audio",
    # Frequency analysis
    "FrequencyAnalyzer",
    "FFTAnalysis",
    "PitchDetection",
    "HarmonicContent",
    "freq_to_midi",
    "midi_to_freq",
    "midi_to_note_name",
    "analyze_frequency_spectrum",
    "detect_pitch_from_audio",
    # Theory analyzer
    "TheoryAnalyzer",
    "TheoryAnalysis",
    "ScaleDetection",
    "TriadDetection",
    "ArpeggioDetection",
    "IntervalAnalysis",
    "detect_scale",
    "detect_mode",
    "analyze_harmony",
    "SCALES",
    "TRIADS",
    "SEVENTH_CHORDS",
    "MODE_CHARACTERISTICS",
]
