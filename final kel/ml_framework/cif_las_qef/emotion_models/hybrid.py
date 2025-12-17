"""
Hybrid Emotional Field Equation

Combines classical (VAD) and quantum (superposition) parts.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from .classical import VADModel, VADState
from .quantum import QuantumEmotionalField, EmotionSuperposition


@dataclass
class EmotionalHamiltonian:
    """
    Emotional Hamiltonian Operator

    Describes emotional influences (memory, stimuli, empathy links).
    """
    memory_influence: np.ndarray      # Memory effects
    stimulus_influence: np.ndarray     # External stimuli
    empathy_links: np.ndarray          # Empathy connections
    self_regulation: np.ndarray        # Self-regulation

    def __init__(self, n_emotions: int):
        """
        Initialize Hamiltonian.

        Args:
            n_emotions: Number of basis emotions
        """
        self.n_emotions = n_emotions
        self.memory_influence = np.eye(n_emotions, dtype=complex)
        self.stimulus_influence = np.zeros((n_emotions, n_emotions), dtype=complex)
        self.empathy_links = np.zeros((n_emotions, n_emotions), dtype=complex)
        self.self_regulation = np.eye(n_emotions, dtype=complex) * 0.1

    def get_hamiltonian(self) -> np.ndarray:
        """
        Get full Hamiltonian: Ĥ = H_memory + H_stimulus + H_empathy + H_self

        Returns:
            Hamiltonian matrix
        """
        return (
            self.memory_influence +
            self.stimulus_influence +
            self.empathy_links +
            self.self_regulation
        )

    def set_memory_influence(self, memory_strength: float = 0.5):
        """
        Set memory influence (tends to preserve past states).

        Args:
            memory_strength: Strength of memory (0-1)
        """
        self.memory_influence = np.eye(self.n_emotions, dtype=complex) * memory_strength

    def add_stimulus(
        self,
        emotion_idx: int,
        strength: float,
        coupling: Optional[np.ndarray] = None
    ):
        """
        Add external stimulus influence.

        Args:
            emotion_idx: Index of stimulated emotion
            strength: Stimulus strength
            coupling: Optional coupling matrix (default: diagonal)
        """
        if coupling is None:
            coupling = np.eye(self.n_emotions, dtype=complex)

        # Stimulus creates transitions
        self.stimulus_influence += strength * coupling

    def add_empathy_link(
        self,
        emotion_i: int,
        emotion_j: int,
        strength: float
    ):
        """
        Add empathy link between emotions.

        Args:
            emotion_i: First emotion index
            emotion_j: Second emotion index
            strength: Link strength
        """
        self.empathy_links[emotion_i, emotion_j] += strength
        self.empathy_links[emotion_j, emotion_i] += strength  # Symmetric


class HybridEmotionalField:
    """
    Hybrid Emotional Field

    Combines classical (VAD) and quantum (superposition) parts:
    F_E(t) = VAD(t) + Re[Σ α_i(t) e^(iφ_i(t)) |e_i⟩]
    """

    def __init__(
        self,
        quantum_field: Optional[QuantumEmotionalField] = None,
        vad_model: Optional[VADModel] = None
    ):
        """
        Initialize Hybrid Emotional Field.

        Args:
            quantum_field: Quantum emotional field (default: creates new)
            vad_model: VAD model (default: creates new)
        """
        from .quantum import QuantumEmotionalField
        from .classical import VADModel

        self.quantum_field = quantum_field or QuantumEmotionalField()
        self.vad_model = vad_model or VADModel()

        self.current_vad: Optional[VADState] = None
        self.current_superposition: Optional[EmotionSuperposition] = None
        self.hamiltonian: Optional[EmotionalHamiltonian] = None

    def initialize(
        self,
        vad: Optional[VADState] = None,
        superposition: Optional[EmotionSuperposition] = None
    ):
        """
        Initialize field with VAD and/or superposition.

        Args:
            vad: Classical VAD state
            superposition: Quantum superposition
        """
        if vad is None:
            vad = VADState()
        self.current_vad = vad

        if superposition is None:
            superposition = self.quantum_field.create_superposition()
        self.current_superposition = superposition

        # Initialize Hamiltonian
        n = len(self.current_superposition.basis_emotions)
        self.hamiltonian = EmotionalHamiltonian(n)

    def compute_field(self, t: float = 0.0) -> Dict:
        """
        Compute hybrid field: F_E(t) = VAD(t) + Re[quantum part]

        Args:
            t: Time

        Returns:
            Field state
        """
        if self.current_vad is None or self.current_superposition is None:
            raise ValueError("Field not initialized. Call initialize() first.")

        # Classical part
        classical = self.current_vad.to_array()  # Shape: (3,)

        # Quantum part: Re[Σ α_i e^(iφ_i) |e_i⟩]
        amplitudes = self.current_superposition.amplitudes
        phases = self.current_superposition.get_phases()

        # Complex exponential: α_i e^(iφ_i)
        quantum_complex = amplitudes * np.exp(1j * phases)

        # Real part
        quantum_real = np.real(quantum_complex)  # Shape: (8,) for 8 basis emotions

        # Project quantum part to VAD space (3D) using emotion-to-VAD mapping
        # Map 8 quantum emotions to 3 VAD dimensions
        from .classical import PlutchikWheel
        plutchik = PlutchikWheel()

        # Convert quantum amplitudes to VAD contribution
        quantum_vad = np.zeros(3)  # [valence, arousal, dominance]
        for i, emotion_enum in enumerate(self.current_superposition.basis_emotions):
            # Get VAD for this emotion (emotion_enum is already EmotionBasis)
            # Use absolute value of quantum_real as intensity
            intensity = abs(quantum_real[i])
            emotion_vad = plutchik.emotion_to_vad(emotion_enum, intensity=intensity)
            # Weight by quantum amplitude magnitude
            weight = abs(quantum_real[i])
            quantum_vad[0] += emotion_vad.valence * weight
            quantum_vad[1] += emotion_vad.arousal * weight
            quantum_vad[2] += emotion_vad.dominance * weight

        # Normalize quantum VAD contribution
        total_weight = np.sum(np.abs(quantum_real))
        if total_weight > 0:
            quantum_vad = quantum_vad / total_weight

        # Combined field: blend classical and quantum VAD
        # In practice, quantum part modulates classical
        combined = classical + 0.3 * quantum_vad  # Weight quantum part

        return {
            "time": t,
            "classical_vad": {
                "valence": float(classical[0]),
                "arousal": float(classical[1]),
                "dominance": float(classical[2])
            },
            "quantum_amplitudes": quantum_real.tolist(),
            "quantum_probabilities": self.current_superposition.get_probabilities().tolist(),
            "combined_field": combined.tolist(),
            "coherence": self.current_superposition.compute_coherence(),
            "entropy": self.current_superposition.compute_entropy()
        }

    def evolve(self, dt: float = 0.01) -> Dict:
        """
        Evolve field over time: dΨ/dt = -i Ĥ Ψ

        Args:
            dt: Time step

        Returns:
            Evolution result
        """
        if self.current_superposition is None or self.hamiltonian is None:
            raise ValueError("Field not initialized. Call initialize() first.")

        # Evolve quantum part
        h = self.hamiltonian.get_hamiltonian()
        self.current_superposition = self.quantum_field.evolve_superposition(
            self.current_superposition,
            h,
            dt
        )

        # Evolve classical part (simplified: decay toward equilibrium)
        if self.current_vad:
            # Decay toward neutral
            decay_rate = 0.01
            self.current_vad.valence *= (1.0 - decay_rate)
            self.current_vad.arousal = 0.5 + (self.current_vad.arousal - 0.5) * (1.0 - decay_rate)
            self.current_vad.dominance *= (1.0 - decay_rate)
            self.current_vad.clip()

        return {
            "evolved": True,
            "new_coherence": self.current_superposition.compute_coherence(),
            "new_entropy": self.current_superposition.compute_entropy()
        }

    def observe(self, random_state: Optional[np.random.RandomState] = None) -> Dict:
        """
        Observe field (collapse quantum part).

        Args:
            random_state: Optional random state

        Returns:
            Observation result
        """
        if self.current_superposition is None:
            raise ValueError("Field not initialized.")

        # Collapse quantum part
        collapsed_emotion, probability = self.current_superposition.collapse(random_state)

        # Update classical VAD based on collapsed emotion
        from .classical import PlutchikWheel
        plutchik = PlutchikWheel()
        vad_from_emotion = plutchik.emotion_to_vad(collapsed_emotion, intensity=probability)

        # Blend with current VAD
        if self.current_vad:
            blend_factor = 0.3
            self.current_vad.valence = (
                self.current_vad.valence * (1.0 - blend_factor) +
                vad_from_emotion.valence * blend_factor
            )
            self.current_vad.arousal = (
                self.current_vad.arousal * (1.0 - blend_factor) +
                vad_from_emotion.arousal * blend_factor
            )
            self.current_vad.dominance = (
                self.current_vad.dominance * (1.0 - blend_factor) +
                vad_from_emotion.dominance * blend_factor
            )
            self.current_vad.clip()

        return {
            "collapsed_emotion": collapsed_emotion.value,
            "probability": probability,
            "new_vad": {
                "valence": self.current_vad.valence if self.current_vad else 0.0,
                "arousal": self.current_vad.arousal if self.current_vad else 0.5,
                "dominance": self.current_vad.dominance if self.current_vad else 0.0
            }
        }

    def add_stimulus(
        self,
        emotion: str,
        strength: float = 0.5
    ):
        """
        Add external stimulus.

        Args:
            emotion: Emotion name
            strength: Stimulus strength
        """
        if self.hamiltonian is None or self.current_superposition is None:
            raise ValueError("Field not initialized.")

        # Find emotion index
        try:
            from .quantum import EmotionBasis
            emotion_enum = EmotionBasis(emotion.lower())
            idx = self.current_superposition.basis_emotions.index(emotion_enum)
        except (ValueError, AttributeError):
            return  # Emotion not in basis

        # Add to Hamiltonian
        self.hamiltonian.add_stimulus(idx, strength)

    def get_status(self) -> Dict:
        """Get current field status."""
        if self.current_vad is None or self.current_superposition is None:
            return {"initialized": False}

        return {
            "initialized": True,
            "vad": {
                "valence": self.current_vad.valence,
                "arousal": self.current_vad.arousal,
                "dominance": self.current_vad.dominance
            },
            "quantum_coherence": self.current_superposition.compute_coherence(),
            "quantum_entropy": self.current_superposition.compute_entropy(),
            "probabilities": self.current_superposition.get_probabilities().tolist()
        }
