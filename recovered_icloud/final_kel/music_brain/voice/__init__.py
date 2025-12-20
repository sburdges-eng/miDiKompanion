"""
Voice Processing - Text-to-speech synthesis, voice modulation, and Parrot singing synthesis.

This module provides voice processing capabilities including:
- Parrot: Unified singing voice synthesizer with voice mimicking and instrument conversion
- AutoTuneProcessor: Pitch correction for vocals
- VoiceModulator: Voice character modification
- VoiceSynthesizer: Text-to-speech and guide vocal generation
"""

# Parrot - Main singing voice synthesizer
from music_brain.voice.parrot import (
    Parrot,
    create_parrot,
)

# Phoneme processing
from music_brain.voice.phoneme_processor import (
    PhonemeProcessor,
    Phoneme,
    PhonemeSequence,
    process_lyrics,
)

# Pitch control
from music_brain.voice.pitch_controller import (
    PitchController,
    PitchCurve,
    ExpressionParams,
)

# Synthesis backends
from music_brain.voice.singing_synthesizer import (
    SingingSynthesizer,
    FormantConfig,
)

from music_brain.voice.neural_backend import (
    NeuralBackend,
    create_neural_backend,
)

# Unified singing voice API (formant + neural switcher)
from music_brain.voice.singing_voice import (
    SingingVoice,
    create_singing_voice,
)

# Dev-focused copy with embedded improvement prompt
from music_brain.voice.singing_voice_dev import (
    SingingVoiceDev,
    create_singing_voice_dev,
    DEV_PROMPT as SINGING_VOICE_DEV_PROMPT,
)

# Voice input and mimicking
from music_brain.voice.voice_input import (
    VoiceRecorder,
    VoiceMimic,
)

# Instrument synthesis
from music_brain.voice.instrument_synth import (
    InstrumentSynthesizer,
    InstrumentConfig,
    get_instrument_preset,
    INSTRUMENT_PRESETS,
)

# Voice learning
from music_brain.voice.voice_learning import (
    VoiceLearningManager,
    VoiceSampleStore,
    VoiceLearner,
    VoiceSample,
    LearnedVoiceProfile,
)

# Legacy exports (for backward compatibility)
from music_brain.voice.auto_tune import (
    AutoTuneProcessor,
    AutoTuneSettings,
    get_auto_tune_preset,
)

from music_brain.voice.modulator import (
    VoiceModulator,
    ModulationSettings,
    get_modulation_preset,
)

from music_brain.voice.synthesizer import (
    VoiceSynthesizer,
    SynthConfig,
    get_voice_profile,
)

__all__ = [
    # Parrot - Main API
    "Parrot",
    "create_parrot",

    # Phoneme processing
    "PhonemeProcessor",
    "Phoneme",
    "PhonemeSequence",
    "process_lyrics",

    # Pitch control
    "PitchController",
    "PitchCurve",
    "ExpressionParams",

    # Synthesis
    "SingingSynthesizer",
    "FormantConfig",
    "NeuralBackend",
    "create_neural_backend",
    "SingingVoice",
    "create_singing_voice",
    "SingingVoiceDev",
    "create_singing_voice_dev",
    "SINGING_VOICE_DEV_PROMPT",

    # Voice input/mimicking
    "VoiceRecorder",
    "VoiceMimic",

    # Instrument synthesis
    "InstrumentSynthesizer",
    "InstrumentConfig",
    "get_instrument_preset",
    "INSTRUMENT_PRESETS",

    # Voice learning
    "VoiceLearningManager",
    "VoiceSampleStore",
    "VoiceLearner",
    "VoiceSample",
    "LearnedVoiceProfile",

    # Legacy
    "AutoTuneProcessor",
    "AutoTuneSettings",
    "get_auto_tune_preset",
    "VoiceModulator",
    "ModulationSettings",
    "get_modulation_preset",
    "VoiceSynthesizer",
    "SynthConfig",
    "get_voice_profile",
]
