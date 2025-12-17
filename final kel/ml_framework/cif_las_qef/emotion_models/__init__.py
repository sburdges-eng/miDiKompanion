"""
Emotional Models - Base Layer

Classical and quantum emotional models for the framework:
- VAD (Valence-Arousal-Dominance) Model
- Plutchik's Wheel
- Quantum Emotional Field (superposition, interference, entanglement)
- Hybrid Emotional Field Equation

Status: Research Phase
"""

from .classical import VADModel, VADState, PlutchikWheel, EmotionBasis
from .quantum import QuantumEmotionalField, EmotionSuperposition, EmotionalInterference, EmotionalEntanglement
from .hybrid import HybridEmotionalField, EmotionalHamiltonian
from .simulation import EmotionalFieldSimulator
from .music_generation import (
    EmotionToMusicMapper, QuantumMusicalField, EmotionScaleMapper,
    EmotionTimbreMapper, TemporalEmotionFlow, EmotionalMusicField, MusicalParameters
)
from .voice_synthesis import (
    EmotionToVoiceMapper, QuantumVoiceField, VoiceMorphing,
    QuantumEmotionalVoiceField, VoiceParameters
)
from .field_dynamics import (
    EmotionalPotential, QuantumEmotionalHamiltonian,
    EmotionalNetworkDynamics, PhysiologicalResonance,
    TemporalEmotionalDynamics, GeometricEmotionalProperties,
    QuantumEmotionalEntropy, ResonanceFormulas,
    UnifiedEmotionalField, EmotionalFieldController
)
from .color_mapping import EmotionColorMapper, ColorMapping

__all__ = [
    # Classical
    "VADModel",
    "VADState",
    "PlutchikWheel",
    "EmotionBasis",
    # Quantum
    "QuantumEmotionalField",
    "EmotionSuperposition",
    "EmotionalInterference",
    "EmotionalEntanglement",
    # Hybrid
    "HybridEmotionalField",
    "EmotionalHamiltonian",
    # Simulation
    "EmotionalFieldSimulator",
    # Music Generation
    "EmotionToMusicMapper",
    "QuantumMusicalField",
    "EmotionScaleMapper",
    "EmotionTimbreMapper",
    "TemporalEmotionFlow",
    "EmotionalMusicField",
    "MusicalParameters",
    # Voice Synthesis
    "EmotionToVoiceMapper",
    "QuantumVoiceField",
    "VoiceMorphing",
    "QuantumEmotionalVoiceField",
    "VoiceParameters",
    # Field Dynamics
    "EmotionalPotential",
    "QuantumEmotionalHamiltonian",
    "EmotionalNetworkDynamics",
    "PhysiologicalResonance",
    "TemporalEmotionalDynamics",
    "GeometricEmotionalProperties",
    "QuantumEmotionalEntropy",
    "ResonanceFormulas",
    "UnifiedEmotionalField",
    "EmotionalFieldController",
    # Color Mapping
    "EmotionColorMapper",
    "ColorMapping",
]
