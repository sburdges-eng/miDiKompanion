"""
Quantum Emotional Field (QEF) - Complete Mathematical Framework

Implements the unified quantum field theory of emotion, music, voice, and consciousness.

Based on the mathematical formulas for:
- Emotional State Vectors (VAD)
- Quantum Wavefunctions
- Music Synthesis
- Voice Modulation
- Network Dynamics
- Resonance & Coherence
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import math
from scipy.integrate import odeint
from scipy.fft import fft, ifft


# ============================================================================
# I. EMOTIONAL STATE VECTOR
# ============================================================================

@dataclass
class EmotionalStateVector:
    """Emotional State Vector E(t) = [V(t), A(t), D(t)]"""
    valence: float  # V ∈ [-1, +1] (pleasantness)
    arousal: float  # A ∈ [0, 1] (energy level)
    dominance: float  # D ∈ [-1, +1] (control/submission)
    
    def __post_init__(self):
        """Clamp values to valid ranges"""
        self.valence = np.clip(self.valence, -1.0, 1.0)
        self.arousal = np.clip(self.arousal, 0.0, 1.0)
        self.dominance = np.clip(self.dominance, -1.0, 1.0)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.valence, self.arousal, self.dominance])
    
    def distance(self, other: 'EmotionalStateVector') -> float:
        """Calculate emotional distance metric"""
        return np.sqrt(
            (self.valence - other.valence)**2 +
            (self.arousal - other.arousal)**2 +
            (self.dominance - other.dominance)**2
        )
    
    def emotional_potential_energy(self, k_v: float = 1.0, k_a: float = 1.0, k_d: float = 1.0) -> float:
        """
        Emotional Potential Energy:
        U_E = (1/2) * k_V * V^2 + (1/2) * k_A * A^2 + (1/2) * k_D * D^2
        """
        return 0.5 * (k_v * self.valence**2 + k_a * self.arousal**2 + k_d * self.dominance**2)
    
    def emotional_force(self, k_v: float = 1.0, k_a: float = 1.0, k_d: float = 1.0) -> np.ndarray:
        """
        Emotional Force (Gradient):
        F_E = -∇U_E = [-k_V*V, -k_A*A, -k_D*D]
        """
        return np.array([-k_v * self.valence, -k_a * self.arousal, -k_d * self.dominance])
    
    def emotional_stability(self) -> float:
        """
        Emotional Stability:
        S_E = 1 - sqrt((V^2 + A^2 + D^2) / 3)
        """
        return 1.0 - np.sqrt((self.valence**2 + self.arousal**2 + self.dominance**2) / 3.0)


# ============================================================================
# II. QUANTUM EMOTIONAL FIELD
# ============================================================================

class EmotionBasis(Enum):
    """Fundamental emotion basis states"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"


@dataclass
class QuantumEmotionalWavefunction:
    """
    Emotional Wavefunction: |Ψ_E(t)⟩ = Σ α_i(t) |e_i⟩
    """
    amplitudes: Dict[EmotionBasis, complex] = field(default_factory=dict)
    time: float = 0.0
    
    def __post_init__(self):
        """Initialize with equal amplitudes if empty"""
        if not self.amplitudes:
            for emotion in EmotionBasis:
                self.amplitudes[emotion] = 0.0 + 0.0j
    
    def normalize(self):
        """Normalize: Σ |α_i|^2 = 1"""
        total = sum(abs(amp)**2 for amp in self.amplitudes.values())
        if total > 0:
            norm = 1.0 / np.sqrt(total)
            for emotion in self.amplitudes:
                self.amplitudes[emotion] *= norm
    
    def probability_density(self, emotion: EmotionBasis) -> float:
        """
        Emotional Probability Density:
        P_i = |α_i|^2
        """
        return abs(self.amplitudes.get(emotion, 0.0))**2
    
    def all_probabilities(self) -> Dict[EmotionBasis, float]:
        """Get probability for all emotions"""
        return {emotion: self.probability_density(emotion) 
                for emotion in EmotionBasis}
    
    def emotional_entropy(self) -> float:
        """
        Emotional Entropy:
        S_E = -Σ P_i * ln(P_i)
        """
        probs = self.all_probabilities()
        entropy = 0.0
        for prob in probs.values():
            if prob > 0:
                entropy -= prob * np.log(prob)
        return entropy
    
    def expectation_value(self, observable: Dict[EmotionBasis, float]) -> float:
        """Calculate expectation value: <Ψ|O|Ψ>"""
        result = 0.0
        for emotion, amp in self.amplitudes.items():
            result += abs(amp)**2 * observable.get(emotion, 0.0)
        return result


class EmotionalHamiltonian:
    """
    Emotional Hamiltonian (Evolution Operator):
    Ĥ_E = Σ ℏω_i |e_i⟩⟨e_i|
    """
    
    def __init__(self, frequencies: Optional[Dict[EmotionBasis, float]] = None):
        """
        Initialize with emotion frequencies (ω_i)
        Default: frequencies based on typical emotional energy levels
        """
        if frequencies is None:
            # Default frequencies (in Hz, converted to angular frequency)
            self.frequencies = {
                EmotionBasis.JOY: 2.0 * np.pi * 2.0,      # 2 Hz
                EmotionBasis.SADNESS: 2.0 * np.pi * 0.5,  # 0.5 Hz
                EmotionBasis.ANGER: 2.0 * np.pi * 3.0,    # 3 Hz
                EmotionBasis.FEAR: 2.0 * np.pi * 4.0,     # 4 Hz
                EmotionBasis.SURPRISE: 2.0 * np.pi * 2.5,  # 2.5 Hz
                EmotionBasis.DISGUST: 2.0 * np.pi * 1.5,  # 1.5 Hz
                EmotionBasis.TRUST: 2.0 * np.pi * 1.0,    # 1 Hz
                EmotionBasis.ANTICIPATION: 2.0 * np.pi * 2.2,  # 2.2 Hz
            }
        else:
            self.frequudes = frequencies
        
        self.hbar = 1.0  # Reduced Planck constant (normalized)
    
    def evolve(self, wavefunction: QuantumEmotionalWavefunction, dt: float) -> QuantumEmotionalWavefunction:
        """
        Time Evolution:
        iℏ d|Ψ_E(t)⟩/dt = Ĥ_E |Ψ_E(t)⟩
        
        Solution: |Ψ(t+dt)⟩ = exp(-i Ĥ_E dt / ℏ) |Ψ(t)⟩
        """
        new_amplitudes = {}
        for emotion, amp in wavefunction.amplitudes.items():
            omega = self.frequencies.get(emotion, 0.0)
            # Time evolution: exp(-i * omega * dt)
            phase = -1j * omega * dt
            new_amplitudes[emotion] = amp * np.exp(phase)
        
        new_wf = QuantumEmotionalWavefunction(new_amplitudes, wavefunction.time + dt)
        new_wf.normalize()
        return new_wf


def emotional_interference(wf1: QuantumEmotionalWavefunction, 
                          wf2: QuantumEmotionalWavefunction) -> float:
    """
    Emotional Interference:
    I(t) = |Ψ_1 + Ψ_2|^2 = |Ψ_1|^2 + |Ψ_2|^2 + 2*Re(Ψ_1* * Ψ_2)
    
    Returns interference strength (positive = constructive, negative = destructive)
    """
    interference = 0.0
    for emotion in EmotionBasis:
        amp1 = wf1.amplitudes.get(emotion, 0.0)
        amp2 = wf2.amplitudes.get(emotion, 0.0)
        interference += 2 * np.real(np.conj(amp1) * amp2)
    return interference


def emotional_entanglement(wf1: QuantumEmotionalWavefunction,
                          wf2: QuantumEmotionalWavefunction) -> QuantumEmotionalWavefunction:
    """
    Emotional Entanglement (Two Agents):
    |Ψ_AB⟩ = (1/√2) (|Joy_A, Joy_B⟩ + |Fear_A, Fear_B⟩)
    
    Simplified: creates correlated state
    """
    # Create entangled state (simplified version)
    entangled_amplitudes = {}
    for emotion in EmotionBasis:
        amp1 = wf1.amplitudes.get(emotion, 0.0)
        amp2 = wf2.amplitudes.get(emotion, 0.0)
        # Entangled amplitude (correlated)
        entangled_amplitudes[emotion] = (amp1 + amp2) / np.sqrt(2.0)
    
    entangled = QuantumEmotionalWavefunction(entangled_amplitudes)
    entangled.normalize()
    return entangled


# ============================================================================
# III. EMOTION → MUSIC FORMULAS
# ============================================================================

class MusicGenerator:
    """Converts emotional state to musical parameters"""
    
    def __init__(self, base_frequency: float = 440.0):
        """Initialize with base frequency (A4 = 440 Hz)"""
        self.f0 = base_frequency
    
    def base_frequency_mapping(self, state: EmotionalStateVector) -> float:
        """
        Base Frequency Mapping:
        f_E = f_0 * (1 + 0.4*A + 0.2*V)
        """
        return self.f0 * (1.0 + 0.4 * state.arousal + 0.2 * state.valence)
    
    def harmonic_structure(self, state: EmotionalStateVector, t: np.ndarray, 
                          n_harmonics: int = 5) -> np.ndarray:
        """
        Harmonic Structure:
        H(t) = Σ a_n * sin(2π * n * f_E * t + φ_n)
        """
        f_e = self.base_frequency_mapping(state)
        harmonics = np.zeros_like(t)
        
        for n in range(1, n_harmonics + 1):
            # Amplitude decreases with harmonic number
            a_n = 1.0 / n
            # Phase based on emotion
            phi_n = state.valence * np.pi / 4.0
            harmonics += a_n * np.sin(2.0 * np.pi * n * f_e * t + phi_n)
        
        return harmonics
    
    def chordal_shift(self, state: EmotionalStateVector) -> float:
        """
        Chordal Shift:
        Δf = (V + D) * 30 Hz
        """
        return (state.valence + state.dominance) * 30.0
    
    def emotional_resonance_energy(self, harmonics: np.ndarray, 
                                   frequencies: np.ndarray) -> float:
        """
        Emotional Resonance Energy:
        E_res = Σ a_i^2 * f_i
        """
        return np.sum(harmonics**2 * frequencies)


# ============================================================================
# IV. VOICE MODULATION FORMULAS
# ============================================================================

class VoiceModulator:
    """Modulates voice parameters based on emotion"""
    
    def __init__(self, base_freq: float = 200.0, base_amplitude: float = 1.0):
        """Initialize with base voice parameters"""
        self.f_base = base_freq
        self.A_base = base_amplitude
    
    def pitch(self, state: EmotionalStateVector) -> float:
        """
        Pitch (Fundamental Frequency):
        f_0 = f_base * (1 + 0.5*A + 0.3*V)
        """
        return self.f_base * (1.0 + 0.5 * state.arousal + 0.3 * state.valence)
    
    def amplitude(self, state: EmotionalStateVector) -> float:
        """
        Amplitude (Volume):
        A_voice = A_base * (1 + 0.4*D + 0.2*A)
        """
        return self.A_base * (1.0 + 0.4 * state.dominance + 0.2 * state.arousal)
    
    def formant_shift(self, state: EmotionalStateVector, 
                     base_formant: float = 1000.0) -> float:
        """
        Formant Shifts (Vowel Color):
        F_i' = F_i * (1 + 0.2*V - 0.1*D)
        """
        return base_formant * (1.0 + 0.2 * state.valence - 0.1 * state.dominance)
    
    def vibrato(self, state: EmotionalStateVector, t: np.ndarray) -> np.ndarray:
        """
        Vibrato & Tremor:
        f_0(t) = f_0 * (1 + v_d * sin(2π * v_r * t))
        where v_d = 0.01*A, v_r = 5 + 3*A
        """
        v_d = 0.01 * state.arousal
        v_r = 5.0 + 3.0 * state.arousal
        f0 = self.pitch(state)
        return f0 * (1.0 + v_d * np.sin(2.0 * np.pi * v_r * t))
    
    def speech_rate(self, state: EmotionalStateVector, base_rate: float = 150.0) -> float:
        """
        Speech Rate:
        R = R_0 * (1 + 0.7*A - 0.3*V)
        """
        return base_rate * (1.0 + 0.7 * state.arousal - 0.3 * state.valence)
    
    def voice_entropy(self, probabilities: Dict[EmotionBasis, float]) -> float:
        """
        Emotional Voice Entropy:
        S_V = -Σ P_i * log(P_i)
        """
        entropy = 0.0
        for prob in probabilities.values():
            if prob > 0:
                entropy -= prob * np.log(prob)
        return entropy


# ============================================================================
# V. SOUND SYNTHESIS
# ============================================================================

class EmotionalTimbre:
    """Generates emotional timbre spectrum"""
    
    def timbre_spectrum(self, state: EmotionalStateVector, 
                       frequencies: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Emotional Timbre Spectrum:
        S(f, t) = A(t) * exp(-β(t) * f)
        where β(t) = β_0 - 0.1*V + 0.2*A
        """
        A_t = 1.0 + 0.3 * state.arousal
        beta_0 = 0.001
        beta_t = beta_0 - 0.1 * state.valence + 0.2 * state.arousal
        beta_t = max(0.0001, beta_t)  # Prevent negative
        
        return A_t * np.exp(-beta_t * frequencies)
    
    def resonance_filter(self, frequencies: np.ndarray, 
                        formants: List[Tuple[float, float]]) -> np.ndarray:
        """
        Emotional Resonance Filter:
        H(f) = Π 1 / (1 + j*Q_i*(f/F_i' - F_i'/f))
        Simplified version (magnitude only)
        """
        H = np.ones_like(frequencies)
        for F_i, Q_i in formants:
            # Simplified magnitude response
            response = 1.0 / (1.0 + Q_i * np.abs(frequencies / F_i - F_i / frequencies))
            H *= response
        return H
    
    def intermodulation(self, amplitudes: np.ndarray, 
                      frequencies: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Emotional Intermodulation:
        s(t) = Σ a_i * a_j * cos(2π * (f_i ± f_j) * t)
        """
        signal = np.zeros_like(t)
        n = len(amplitudes)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Sum and difference frequencies
                    f_sum = frequencies[i] + frequencies[j]
                    f_diff = abs(frequencies[i] - frequencies[j])
                    signal += amplitudes[i] * amplitudes[j] * (
                        np.cos(2.0 * np.pi * f_sum * t) +
                        np.cos(2.0 * np.pi * f_diff * t)
                    )
        
        return signal


# ============================================================================
# VI. NETWORK DYNAMICS
# ============================================================================

class EmotionalNetwork:
    """Multi-agent emotional field network"""
    
    def __init__(self, n_agents: int, coupling_strength: float = 0.1):
        """Initialize network with n agents"""
        self.n_agents = n_agents
        self.k = coupling_strength
        self.states = [EmotionalStateVector(0.0, 0.5, 0.0) for _ in range(n_agents)]
        self.coupling_matrix = np.ones((n_agents, n_agents)) * self.k
        np.fill_diagonal(self.coupling_matrix, 0.0)  # No self-coupling
    
    def emotional_coupling(self, dt: float = 0.01):
        """
        Emotional Coupling:
        dE_i/dt = Σ k_ij * (E_j - E_i)
        """
        new_states = []
        for i in range(self.n_agents):
            state_i = self.states[i]
            dE = np.zeros(3)
            
            for j in range(self.n_agents):
                if i != j:
                    state_j = self.states[j]
                    k_ij = self.coupling_matrix[i, j]
                    dE += k_ij * (state_j.to_array() - state_i.to_array())
            
            # Update state
            new_array = state_i.to_array() + dE * dt
            new_states.append(EmotionalStateVector(new_array[0], new_array[1], new_array[2]))
        
        self.states = new_states
    
    def coherence(self) -> float:
        """
        Coherence:
        C = (1/N^2) * Σ cos(θ_i - θ_j)
        Simplified: using valence as phase
        """
        phases = [state.valence * np.pi for state in self.states]
        coherence = 0.0
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                coherence += np.cos(phases[i] - phases[j])
        return coherence / (self.n_agents ** 2)
    
    def phase_locking_value(self, i: int, j: int, n_samples: int = 100) -> float:
        """
        Phase Locking Value:
        PLV = |(1/N) * Σ exp(i*(φ_1(n) - φ_2(n)))|
        """
        phases_i = [self.states[i].valence * np.pi for _ in range(n_samples)]
        phases_j = [self.states[j].valence * np.pi for _ in range(n_samples)]
        
        complex_sum = sum(np.exp(1j * (phi_i - phi_j)) 
                         for phi_i, phi_j in zip(phases_i, phases_j))
        return abs(complex_sum / n_samples)


# ============================================================================
# VII. TEMPORAL & MEMORY
# ============================================================================

class EmotionalMemory:
    """Handles emotional hysteresis and temporal decay"""
    
    def __init__(self, tau: float = 1.0):
        """Initialize with emotional half-life τ"""
        self.tau = tau
        self.memory_kernel = None
    
    def temporal_decay(self, E0: float, dt: float) -> float:
        """
        Temporal Decay:
        E(t+Δt) = E(t) * exp(-Δt / τ_E)
        """
        return E0 * np.exp(-dt / self.tau)
    
    def emotional_hysteresis(self, stimulus: Callable, t: np.ndarray, 
                            kernel: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Emotional Hysteresis (Memory):
        E(t) = E_0 + ∫ K(τ) * S(t-τ) dτ
        
        Simplified: convolution with exponential kernel
        """
        if kernel is None:
            # Exponential memory kernel
            kernel = np.exp(-t / self.tau)
            kernel = kernel / np.sum(kernel)  # Normalize
        
        stimulus_values = np.array([stimulus(ti) for ti in t])
        # Convolution
        memory = np.convolve(stimulus_values, kernel, mode='same')
        return memory
    
    def emotional_momentum(self, state: EmotionalStateVector, 
                          previous_state: EmotionalStateVector, 
                          m_e: float = 1.0) -> np.ndarray:
        """
        Emotional Momentum:
        p_E = m_E * dE/dt
        """
        dE_dt = (state.to_array() - previous_state.to_array())
        return m_e * dE_dt


# ============================================================================
# VIII. RESONANCE & COHERENCE
# ============================================================================

class EmotionalResonance:
    """Calculates resonance frequencies and coherence"""
    
    def resonance_frequency(self, k_e: float, m_e: float) -> float:
        """
        Emotional Frequency Resonance:
        f_res = (1/(2π)) * sqrt(k_E / m_E)
        """
        return (1.0 / (2.0 * np.pi)) * np.sqrt(k_e / m_e)
    
    def beat_frequency(self, f1: float, f2: float) -> float:
        """
        Beat Frequency (Interference):
        f_beat = |f1 - f2|
        """
        return abs(f1 - f2)
    
    def quality_factor(self, f_res: float, delta_f: float) -> float:
        """
        Emotional Quality Factor:
        Q_E = f_res / Δf
        """
        return f_res / delta_f if delta_f > 0 else float('inf')
    
    def resonant_coherence_energy(self, wf1: QuantumEmotionalWavefunction,
                                  wf2: QuantumEmotionalWavefunction) -> float:
        """
        Resonant Coherence Energy:
        E_coh = ∫ |Ψ_1* * Ψ_2|^2 dx
        Simplified: sum over basis states
        """
        coherence = 0.0
        for emotion in EmotionBasis:
            amp1 = wf1.amplitudes.get(emotion, 0.0)
            amp2 = wf2.amplitudes.get(emotion, 0.0)
            coherence += abs(np.conj(amp1) * amp2)**2
        return coherence


# ============================================================================
# IX. UNIFIED FIELD EQUATION
# ============================================================================

class QuantumEmotionalField:
    """
    Complete Quantum Emotional Field System
    
    Implements the unified field equation:
    iℏ dΨ_E/dt = Ĥ_E Ψ_E + g_M S(f,t) + g_V H(f) + g_N Σ k_ij (Ψ_j - Ψ_E)
    """
    
    def __init__(self, 
                 music_coupling: float = 0.1,
                 voice_coupling: float = 0.1,
                 network_coupling: float = 0.05):
        """Initialize with coupling constants"""
        self.g_M = music_coupling  # Music coupling
        self.g_V = voice_coupling  # Voice coupling
        self.g_N = network_coupling  # Network coupling
        
        self.hamiltonian = EmotionalHamiltonian()
        self.music_gen = MusicGenerator()
        self.voice_mod = VoiceModulator()
        self.timbre = EmotionalTimbre()
        self.network = None
    
    def evolve(self, wavefunction: QuantumEmotionalWavefunction,
               state: EmotionalStateVector,
               dt: float,
               music_signal: Optional[np.ndarray] = None,
               voice_signal: Optional[np.ndarray] = None) -> QuantumEmotionalWavefunction:
        """
        Evolve the quantum emotional field
        
        Simplified implementation of the full equation
        """
        # Hamiltonian evolution
        new_wf = self.hamiltonian.evolve(wavefunction, dt)
        
        # Music coupling (if provided)
        if music_signal is not None:
            # Couple music to emotional field (simplified)
            music_energy = np.mean(np.abs(music_signal)**2)
            for emotion in EmotionBasis:
                new_wf.amplitudes[emotion] += self.g_M * music_energy * 0.01
        
        # Voice coupling (if provided)
        if voice_signal is not None:
            voice_energy = np.mean(np.abs(voice_signal)**2)
            for emotion in EmotionBasis:
                new_wf.amplitudes[emotion] += self.g_V * voice_energy * 0.01
        
        # Network coupling (if network exists)
        if self.network is not None:
            # Average state of network
            avg_state = np.mean([s.to_array() for s in self.network.states], axis=0)
            state_diff = state.to_array() - avg_state
            network_influence = np.linalg.norm(state_diff) * self.g_N
            
            for emotion in EmotionBasis:
                new_wf.amplitudes[emotion] += network_influence * 0.01
        
        new_wf.normalize()
        return new_wf
    
    def total_energy(self, wavefunction: QuantumEmotionalWavefunction,
                    state: EmotionalStateVector) -> float:
        """
        Total Field Energy:
        E_total = E_emotion + E_music + E_voice + E_network + E_resonance
        """
        # Emotional energy
        E_emotion = state.emotional_potential_energy()
        
        # Wavefunction energy
        E_wf = sum(abs(amp)**2 for amp in wavefunction.amplitudes.values())
        
        return E_emotion + E_wf


# ============================================================================
# X. UTILITY FUNCTIONS
# ============================================================================

def vad_to_wavefunction(state: EmotionalStateVector) -> QuantumEmotionalWavefunction:
    """
    Convert VAD state to quantum wavefunction
    
    Maps valence/arousal/dominance to emotion probabilities
    """
    # Simple mapping (can be improved)
    amplitudes = {}
    
    # Positive valence -> Joy, Trust
    if state.valence > 0:
        amplitudes[EmotionBasis.JOY] = state.valence * (1.0 + 0.5j)
        amplitudes[EmotionBasis.TRUST] = state.valence * 0.7 * (1.0 + 0.3j)
    else:
        amplitudes[EmotionBasis.JOY] = 0.0 + 0.0j
        amplitudes[EmotionBasis.TRUST] = 0.0 + 0.0j
    
    # Negative valence -> Sadness, Disgust
    if state.valence < 0:
        amplitudes[EmotionBasis.SADNESS] = abs(state.valence) * (1.0 + 0.5j)
        amplitudes[EmotionBasis.DISGUST] = abs(state.valence) * 0.7 * (1.0 + 0.3j)
    else:
        amplitudes[EmotionBasis.SADNESS] = 0.0 + 0.0j
        amplitudes[EmotionBasis.DISGUST] = 0.0 + 0.0j
    
    # High arousal -> Anger, Fear, Surprise
    if state.arousal > 0.5:
        amplitudes[EmotionBasis.ANGER] = state.arousal * (1.0 + 0.4j)
        amplitudes[EmotionBasis.FEAR] = state.arousal * 0.8 * (1.0 + 0.4j)
        amplitudes[EmotionBasis.SURPRISE] = state.arousal * 0.6 * (1.0 + 0.3j)
    else:
        amplitudes[EmotionBasis.ANGER] = 0.0 + 0.0j
        amplitudes[EmotionBasis.FEAR] = 0.0 + 0.0j
        amplitudes[EmotionBasis.SURPRISE] = 0.0 + 0.0j
    
    # Anticipation based on dominance
    amplitudes[EmotionBasis.ANTICIPATION] = abs(state.dominance) * (1.0 + 0.2j)
    
    wf = QuantumEmotionalWavefunction(amplitudes)
    wf.normalize()
    return wf


def color_frequency_mapping(valence: float) -> float:
    """
    Color/Frequency Mapping:
    f_color = f_min + (V+1) * (f_max - f_min) / 2
    """
    f_min = 400.0  # Violet (THz)
    f_max = 750.0  # Red (THz)
    return f_min + (valence + 1.0) * (f_max - f_min) / 2.0


# ============================================================================
# XI. SIMULATION RUNNER
# ============================================================================

def simulate_emotional_field(initial_state: EmotionalStateVector,
                            duration: float = 10.0,
                            dt: float = 0.01,
                            qef: Optional[QuantumEmotionalField] = None) -> Dict:
    """
    Run a complete simulation of the quantum emotional field
    
    Returns time series of all relevant quantities
    """
    if qef is None:
        qef = QuantumEmotionalField()
    
    # Initialize
    wavefunction = vad_to_wavefunction(initial_state)
    state = initial_state
    
    # Time arrays
    t = np.arange(0, duration, dt)
    n_steps = len(t)
    
    # Storage
    results = {
        'time': t,
        'valence': np.zeros(n_steps),
        'arousal': np.zeros(n_steps),
        'dominance': np.zeros(n_steps),
        'probabilities': {emotion: np.zeros(n_steps) for emotion in EmotionBasis},
        'entropy': np.zeros(n_steps),
        'energy': np.zeros(n_steps),
        'music_freq': np.zeros(n_steps),
        'voice_pitch': np.zeros(n_steps),
    }
    
    # Simulation loop
    for i in range(n_steps):
        # Store current state
        results['valence'][i] = state.valence
        results['arousal'][i] = state.arousal
        results['dominance'][i] = state.dominance
        
        probs = wavefunction.all_probabilities()
        for emotion, prob in probs.items():
            results['probabilities'][emotion][i] = prob
        
        results['entropy'][i] = wavefunction.emotional_entropy()
        results['energy'][i] = qef.total_energy(wavefunction, state)
        results['music_freq'][i] = qef.music_gen.base_frequency_mapping(state)
        results['voice_pitch'][i] = qef.voice_mod.pitch(state)
        
        # Evolve
        wavefunction = qef.evolve(wavefunction, state, dt)
        
        # Simple state evolution (can be made more sophisticated)
        force = state.emotional_force()
        new_array = state.to_array() + force * dt * 0.1
        state = EmotionalStateVector(new_array[0], new_array[1], new_array[2])
    
    return results
