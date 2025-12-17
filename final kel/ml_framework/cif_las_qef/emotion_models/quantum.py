"""
Quantum Emotional Field Models

Wave-based emotional representation with superposition, interference,
entanglement, and collapse functions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from enum import Enum


class EmotionBasis(Enum):
    """Basic emotions for quantum superposition."""
    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"


@dataclass
class EmotionSuperposition:
    """
    Emotion Superposition State
    
    |Ψ_E⟩ = Σ α_i |e_i⟩
    
    where α_i ∈ C are complex amplitudes.
    """
    basis_emotions: List[EmotionBasis]
    amplitudes: np.ndarray  # Complex amplitudes α_i
    phases: np.ndarray      # Phases φ_i
    
    def __post_init__(self):
        """Normalize amplitudes after initialization."""
        self.normalize()
    
    def normalize(self):
        """
        Normalize: Σ |α_i|² = 1
        """
        probabilities = np.abs(self.amplitudes)**2
        total = np.sum(probabilities)
        
        if total > 0:
            self.amplitudes = self.amplitudes / np.sqrt(total)
    
    def get_probabilities(self) -> np.ndarray:
        """
        Get probability distribution: |α_i|²
        
        Returns:
            Probability array
        """
        return np.abs(self.amplitudes)**2
    
    def get_phases(self) -> np.ndarray:
        """Get phases."""
        return np.angle(self.amplitudes)
    
    def collapse(self, random_state: Optional[np.random.RandomState] = None) -> Tuple[EmotionBasis, float]:
        """
        Collapse function: |Ψ_E⟩ → |e_j⟩ with probability |α_j|²
        
        Args:
            random_state: Optional random state for reproducibility
        
        Returns:
            (collapsed_emotion, probability)
        """
        if random_state is None:
            random_state = np.random
        
        probabilities = self.get_probabilities()
        
        # Sample from distribution
        idx = random_state.choice(len(self.basis_emotions), p=probabilities)
        
        return (self.basis_emotions[idx], float(probabilities[idx]))
    
    def compute_coherence(self) -> float:
        """
        Compute emotional coherence: C = |Σ α_i|
        
        Returns:
            Coherence (0-1)
        """
        return float(np.abs(np.sum(self.amplitudes)))
    
    def compute_entropy(self) -> float:
        """
        Compute emotional entropy: S = -Σ |α_i|² log(|α_i|²)
        
        Returns:
            Entropy (0-∞, higher = more uncertainty)
        """
        probabilities = self.get_probabilities()
        # Avoid log(0)
        probabilities = probabilities + 1e-10
        entropy = -np.sum(probabilities * np.log(probabilities))
        return float(entropy)


class EmotionalInterference:
    """
    Emotional Interference
    
    Models interference between overlapping emotional fields.
    """
    
    def __init__(self):
        """Initialize interference calculator."""
        pass
    
    def compute_interference(
        self,
        psi1: EmotionSuperposition,
        psi2: EmotionSuperposition
    ) -> Dict[str, float]:
        """
        Compute interference: I = |Ψ₁ + Ψ₂|²
        
        Args:
            psi1: First emotional field
            psi2: Second emotional field
        
        Returns:
            Interference metrics
        """
        # Ensure same basis
        if psi1.basis_emotions != psi2.basis_emotions:
            # Align bases (simplified - would need proper basis transformation)
            raise ValueError("Emotional fields must have same basis")
        
        # Combined amplitude
        combined = psi1.amplitudes + psi2.amplitudes
        
        # Interference intensity
        intensity = np.abs(combined)**2
        
        # Individual intensities
        intensity1 = np.abs(psi1.amplitudes)**2
        intensity2 = np.abs(psi2.amplitudes)**2
        
        # Interference term: 2 Re(Ψ₁* Ψ₂)
        interference_term = 2.0 * np.real(np.conj(psi1.amplitudes) * psi2.amplitudes)
        
        # Total interference
        total_interference = np.sum(interference_term)
        
        # Determine if constructive or destructive
        is_constructive = total_interference > 0
        
        return {
            "total_intensity": float(np.sum(intensity)),
            "individual_intensity_1": float(np.sum(intensity1)),
            "individual_intensity_2": float(np.sum(intensity2)),
            "interference_term": float(total_interference),
            "is_constructive": is_constructive,
            "interference_type": "constructive" if is_constructive else "destructive"
        }


@dataclass
class EmotionalEntanglement:
    """
    Emotional Entanglement
    
    Two agents (A and B) share a coupled emotional state.
    """
    agent_a_emotions: List[EmotionBasis]
    agent_b_emotions: List[EmotionBasis]
    entangled_state: np.ndarray  # Combined state vector
    
    def __init__(
        self,
        agent_a_emotions: List[EmotionBasis],
        agent_b_emotions: List[EmotionBasis],
        entanglement_strength: float = 0.5
    ):
        """
        Initialize entangled state.
        
        Args:
            agent_a_emotions: Emotions for agent A
            agent_b_emotions: Emotions for agent B
            entanglement_strength: Strength of entanglement (0-1)
        """
        self.agent_a_emotions = agent_a_emotions
        self.agent_b_emotions = agent_b_emotions
        
        # Create entangled state: |Ψ_AB⟩ = (1/√2)(|Joy_A, Joy_B⟩ + |Fear_A, Fear_B⟩)
        # Simplified: equal superposition of matching states
        n_a = len(agent_a_emotions)
        n_b = len(agent_b_emotions)
        
        # Initialize as product state
        self.entangled_state = np.ones((n_a, n_b), dtype=complex) / np.sqrt(n_a * n_b)
        
        # Add entanglement (correlation)
        if n_a == n_b:
            # Create correlation matrix
            for i in range(n_a):
                self.entangled_state[i, i] *= (1.0 + entanglement_strength)
        
        # Normalize
        norm = np.linalg.norm(self.entangled_state)
        if norm > 0:
            self.entangled_state = self.entangled_state / norm
    
    def observe_agent_a(self, emotion: EmotionBasis, random_state: Optional[np.random.RandomState] = None) -> Dict:
        """
        Observe agent A's emotion (collapses both A and B).
        
        Args:
            emotion: Observed emotion for A
            random_state: Optional random state
        
        Returns:
            Observation result with B's state
        """
        if random_state is None:
            random_state = np.random
        
        # Find index of emotion in A's list
        try:
            idx_a = self.agent_a_emotions.index(emotion)
        except ValueError:
            return {"error": "Emotion not in agent A's basis"}
        
        # Get probability distribution for B given A's state
        b_probabilities = np.abs(self.entangled_state[idx_a, :])**2
        b_probabilities = b_probabilities / np.sum(b_probabilities)  # Normalize
        
        # Sample B's emotion
        idx_b = random_state.choice(len(self.agent_b_emotions), p=b_probabilities)
        
        return {
            "agent_a_emotion": emotion,
            "agent_b_emotion": self.agent_b_emotions[idx_b],
            "agent_b_probability": float(b_probabilities[idx_b]),
            "entangled": True
        }


class QuantumEmotionalField:
    """
    Quantum Emotional Field
    
    Main class for quantum emotional modeling.
    """
    
    def __init__(
        self,
        basis_emotions: Optional[List[EmotionBasis]] = None,
        hbar: float = 1.0  # Emotional sensitivity constant
    ):
        """
        Initialize Quantum Emotional Field.
        
        Args:
            basis_emotions: Basis emotions (default: all 8 Plutchik emotions)
            hbar: Emotional sensitivity constant (scaling factor)
        """
        if basis_emotions is None:
            basis_emotions = list(EmotionBasis)
        
        self.basis_emotions = basis_emotions
        self.hbar = hbar
        self.current_superposition: Optional[EmotionSuperposition] = None
    
    def create_superposition(
        self,
        amplitudes: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None
    ) -> EmotionSuperposition:
        """
        Create emotion superposition state.
        
        Args:
            amplitudes: Complex amplitudes (default: random)
            phases: Phases (default: random)
        
        Returns:
            EmotionSuperposition
        """
        n = len(self.basis_emotions)
        
        if amplitudes is None:
            # Random amplitudes
            real = np.random.rand(n)
            imag = np.random.rand(n)
            amplitudes = real + 1j * imag
        
        if phases is None:
            phases = np.random.rand(n) * 2 * np.pi
        
        return EmotionSuperposition(
            basis_emotions=self.basis_emotions,
            amplitudes=amplitudes,
            phases=phases
        )
    
    def compute_quantum_energy(
        self,
        superposition: EmotionSuperposition,
        omega: float = 1.0,
        n: int = 0
    ) -> float:
        """
        Compute quantum emotional energy: E = ℏω(n + 1/2)
        
        Args:
            superposition: Emotion superposition
            omega: Frequency of emotional fluctuation
            n: Emotional excitation level (0 = calm, 1 = agitated, ...)
        
        Returns:
            Quantum emotional energy
        """
        return self.hbar * omega * (n + 0.5)
    
    def compute_emotional_temperature(
        self,
        energy: float,
        k_b: float = 1.0
    ) -> float:
        """
        Compute emotional temperature: T_E = k_B^(-1) E_emotion
        
        Args:
            energy: Quantum emotional energy
            k_b: Boltzmann-like constant
        
        Returns:
            Emotional temperature
        """
        return energy / k_b if k_b > 0 else energy
    
    def compute_resonance(
        self,
        superposition: EmotionSuperposition,
        external_stimulus: np.ndarray
    ) -> float:
        """
        Compute resonance with external stimulus: R = Re(Ψ* · Φ_stim)
        
        Args:
            superposition: Emotion superposition
            external_stimulus: External stimulus vector (complex)
        
        Returns:
            Resonance strength
        """
        # Ensure same dimension
        if len(external_stimulus) != len(superposition.amplitudes):
            # Pad or truncate
            min_len = min(len(external_stimulus), len(superposition.amplitudes))
            external_stimulus = external_stimulus[:min_len]
            amplitudes = superposition.amplitudes[:min_len]
        else:
            amplitudes = superposition.amplitudes
        
        # Compute resonance
        resonance = np.real(np.dot(np.conj(amplitudes), external_stimulus))
        
        return float(resonance)
    
    def evolve_superposition(
        self,
        superposition: EmotionSuperposition,
        hamiltonian: np.ndarray,
        dt: float = 0.01
    ) -> EmotionSuperposition:
        """
        Evolve superposition: dΨ/dt = -i Ĥ Ψ
        
        Args:
            superposition: Current superposition
            hamiltonian: Emotional Hamiltonian operator
            dt: Time step
        
        Returns:
            Evolved superposition
        """
        # Ensure dimensions match
        n = len(superposition.amplitudes)
        if hamiltonian.shape != (n, n):
            raise ValueError(f"Hamiltonian must be {n}x{n} matrix")
        
        # Time evolution: Ψ(t+dt) = exp(-i H dt) Ψ(t)
        # Simplified: Ψ(t+dt) ≈ (1 - i H dt) Ψ(t)
        evolution_op = np.eye(n, dtype=complex) - 1j * hamiltonian * dt
        new_amplitudes = evolution_op @ superposition.amplitudes
        
        # Create new superposition
        new_superposition = EmotionSuperposition(
            basis_emotions=superposition.basis_emotions,
            amplitudes=new_amplitudes,
            phases=superposition.phases  # Phases evolve with amplitudes
        )
        
        return new_superposition
