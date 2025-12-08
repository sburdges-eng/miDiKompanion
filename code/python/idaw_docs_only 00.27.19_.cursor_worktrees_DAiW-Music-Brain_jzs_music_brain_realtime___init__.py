"""
Real-time MIDI Processing - Live MIDI input/output and analysis.

This module provides real-time MIDI processing capabilities:
- MIDI input capture from hardware/software ports
- Real-time chord detection
- Real-time groove analysis
- MIDI transformation and routing
- MIDI output to hardware/software ports
"""

from music_brain.realtime.midi_processor import (
    RealtimeMidiProcessor,
    MidiProcessorConfig,
    ChordDetectionCallback,
    GrooveAnalysisCallback,
    MidiTransformCallback,
)

from music_brain.realtime.transformers import (
    create_transpose_transformer,
    create_velocity_scale_transformer,
    create_chord_generator_transformer,
    create_arpeggiator_transformer,
    create_humanize_transformer,
    create_channel_router_transformer,
    create_filter_transformer,
)

__all__ = [
    # Core processor
    "RealtimeMidiProcessor",
    "MidiProcessorConfig",
    "ChordDetectionCallback",
    "GrooveAnalysisCallback",
    "MidiTransformCallback",
    # Transformers
    "create_transpose_transformer",
    "create_velocity_scale_transformer",
    "create_chord_generator_transformer",
    "create_arpeggiator_transformer",
    "create_humanize_transformer",
    "create_channel_router_transformer",
    "create_filter_transformer",
]

