"""
Kelly Quantum Emotional Voice Field (QEVF)
Integrates VAD system with quantum emotional modeling for voice synthesis
Based on VAD_SYSTEM_IMPLEMENTATION.md and quantum emotional field formulas
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import math
import random
import numpy as np
from enum import Enum


# =============================================================================
# VAD CORE SYSTEM
# =============================================================================

@dataclass
class VADVector:
    """Valence-Arousal-Dominance emotional state."""
    valence: float = 0.0      # -1 (unpleasant) to +1 (pleasant)
    arousal: float = 0.5      # 0 (calm) to 1 (excited)
    dominance: float = 0.5    # -1 (submissive) to +1 (dominant)
    
    def energy(self) -> float:
        """Emotional energy level."""
        return self.arousal * (1 + abs(self.valence))
    
    def tension(self) -> float:
        """Emotional tension."""
        return abs(self.valence) * (1 - self.dominance)
    
    def stability(self) -> float:
        """Stability index (1 = stable, 0 = unstable)."""
        magnitude = math.sqrt(self.valence**2 + self.arousal**2 + self.dominance**2)
        return 1 - magnitude / math.sqrt(3)
    
    def distance(self, other: 'VADVector') -> float:
        """Euclidean distance to another VAD state."""
        return math.sqrt(
            (self.valence - other.valence)**2 +
            (self.arousal - other.arousal)**2 +
            (self.dominance - other.dominance)**2
        )
    
    def blend(self, other: 'VADVector', weight: float = 0.5) -> 'VADVector':
        """Blend with another VAD state."""
        return VADVector(
            valence=self.valence * (1-weight) + other.valence * weight,
            arousal=self.arousal * (1-weight) + other.arousal * weight,
            dominance=self.dominance * (1-weight) + other.dominance * weight
        )


# =============================================================================
# BIOMETRIC → VAD MAPPING
# =============================================================================

@dataclass
class BiometricData:
    """Biometric input signals."""
    heart_rate: float = 75.0           # BPM
    heart_rate_variability: float = 50.0  # ms (RMSSD)
    skin_conductance: float = 5.0      # microsiemens (EDA)
    temperature: float = 36.5          # Celsius
    timestamp: float = 0.0


def biometric_to_vad(bio: BiometricData) -> VADVector:
    """Convert biometric signals to VAD vector."""
    # Heart rate → Arousal
    hr_norm = (bio.heart_rate - 60) / 60  # Normalize around 60-120 BPM
    arousal = max(0, min(1, 0.5 + hr_norm * 0.5))
    
    # HRV → Dominance (high HRV = more control)
    hrv_norm = (bio.heart_rate_variability - 30) / 40  # Normalize 30-70ms
    dominance = max(-1, min(1, hrv_norm))
    
    # EDA → Negative valence modifier (high = stress)
    eda_stress = (bio.skin_conductance - 5) / 10
    
    # Temperature deviation → Arousal modifier
    temp_dev = abs(bio.temperature - 36.5) / 2
    
    # Combined valence
    valence = max(-1, min(1, bio.heart_rate_variability / 50 - eda_stress * 0.5))
    
    return VADVector(
        valence=valence,
        arousal=arousal + temp_dev * 0.2,
        dominance=dominance
    )


# =============================================================================
# CIRCADIAN ADJUSTMENTS
# =============================================================================

def circadian_adjustment(hour: int, day_of_week: int = 0) -> VADVector:
    """Get circadian rhythm adjustments for VAD."""
    # Hour adjustments (0-23)
    arousal_adj = 0.0
    valence_adj = 0.0
    dominance_adj = 0.0
    
    if 4 <= hour < 6:      # Early morning
        arousal_adj = -0.3
        valence_adj = -0.1
        dominance_adj = -0.2
    elif 6 <= hour < 10:   # Morning
        arousal_adj = -0.1 + (hour - 6) * 0.05
        valence_adj = 0.1
        dominance_adj = -0.1 + (hour - 6) * 0.05
    elif 10 <= hour < 14:  # Late morning/early afternoon
        arousal_adj = 0.1
        valence_adj = 0.1
        dominance_adj = 0.1
    elif 14 <= hour < 16:  # Afternoon peak
        arousal_adj = 0.2
        valence_adj = 0.1
        dominance_adj = 0.15
    elif 16 <= hour < 18:  # Late afternoon
        arousal_adj = 0.0
        valence_adj = 0.05
        dominance_adj = 0.0
    elif 18 <= hour < 22:  # Evening
        arousal_adj = -0.1
        valence_adj = 0.05
        dominance_adj = -0.05
    else:                  # Night
        arousal_adj = -0.2
        valence_adj = -0.05
        dominance_adj = -0.1
    
    # Day of week adjustments
    if day_of_week == 0:   # Monday
        valence_adj -= 0.1
    elif day_of_week == 4:  # Friday
        valence_adj += 0.1
    elif day_of_week in [5, 6]:  # Weekend
        valence_adj += 0.05
    
    return VADVector(valence=valence_adj, arousal=arousal_adj, dominance=dominance_adj)


# =============================================================================
# QUANTUM EMOTIONAL FIELD
# =============================================================================

class EmotionBasis(Enum):
    """Basic emotion basis vectors (Plutchik's wheel)."""
    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"


# VAD coordinates for each basis emotion
EMOTION_VAD = {
    EmotionBasis.JOY: VADVector(1.0, 0.6, 0.3),
    EmotionBasis.TRUST: VADVector(0.5, 0.4, 0.2),
    EmotionBasis.FEAR: VADVector(-0.6, 0.8, -0.4),
    EmotionBasis.SURPRISE: VADVector(0.0, 0.9, -0.1),
    EmotionBasis.SADNESS: VADVector(-0.8, 0.2, -0.5),
    EmotionBasis.DISGUST: VADVector(-0.6, 0.4, 0.3),
    EmotionBasis.ANGER: VADVector(-0.5, 0.8, 0.4),
    EmotionBasis.ANTICIPATION: VADVector(0.3, 0.6, 0.2),
}


@dataclass
class QuantumEmotionState:
    """Quantum superposition of emotional states."""
    # Complex amplitudes for each basis emotion
    amplitudes: Dict[EmotionBasis, complex] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.amplitudes:
            # Default: equal superposition
            n = len(EmotionBasis)
            amp = 1.0 / math.sqrt(n)
            self.amplitudes = {e: complex(amp, 0) for e in EmotionBasis}
    
    def normalize(self):
        """Ensure probabilities sum to 1."""
        total = sum(abs(a)**2 for a in self.amplitudes.values())
        if total > 0:
            factor = 1.0 / math.sqrt(total)
            self.amplitudes = {k: v * factor for k, v in self.amplitudes.items()}
    
    def probabilities(self) -> Dict[EmotionBasis, float]:
        """Get probability of each emotion."""
        return {e: abs(a)**2 for e, a in self.amplitudes.items()}
    
    def expected_vad(self) -> VADVector:
        """Calculate expected VAD from superposition."""
        probs = self.probabilities()
        v = sum(probs[e] * EMOTION_VAD[e].valence for e in EmotionBasis)
        a = sum(probs[e] * EMOTION_VAD[e].arousal for e in EmotionBasis)
        d = sum(probs[e] * EMOTION_VAD[e].dominance for e in EmotionBasis)
        return VADVector(v, a, d)
    
    def coherence(self) -> float:
        """Calculate quantum coherence (phase alignment)."""
        total = sum(self.amplitudes.values())
        return abs(total) / math.sqrt(len(self.amplitudes))
    
    def entropy(self) -> float:
        """Calculate emotional entropy."""
        probs = self.probabilities()
        return -sum(p * math.log(p + 1e-10) for p in probs.values())
    
    def collapse(self) -> EmotionBasis:
        """Collapse superposition to single emotion (measurement)."""
        probs = self.probabilities()
        r = random.random()
        cumulative = 0.0
        for emotion, prob in probs.items():
            cumulative += prob
            if r <= cumulative:
                return emotion
        return EmotionBasis.JOY  # Fallback
    
    def evolve(self, dt: float, frequencies: Dict[EmotionBasis, float] = None):
        """Evolve state over time (phase rotation)."""
        if frequencies is None:
            frequencies = {e: 0.5 + i * 0.1 for i, e in enumerate(EmotionBasis)}
        
        for emotion, freq in frequencies.items():
            if emotion in self.amplitudes:
                phase = 2 * math.pi * freq * dt
                self.amplitudes[emotion] *= complex(math.cos(phase), math.sin(phase))


def emotional_interference(state1: QuantumEmotionState, state2: QuantumEmotionState) -> float:
    """Calculate interference between two emotional fields."""
    interference = 0.0
    for emotion in EmotionBasis:
        a1 = state1.amplitudes.get(emotion, 0)
        a2 = state2.amplitudes.get(emotion, 0)
        interference += 2 * (a1.conjugate() * a2).real
    return interference


# =============================================================================
# VAD → VOICE PARAMETERS
# =============================================================================

@dataclass
class VoiceParameters:
    """Complete voice synthesis parameters."""
    # Pitch
    f0_base: float = 200.0       # Base frequency (Hz)
    f0_modulated: float = 200.0  # After emotional modulation
    
    # Vibrato
    vibrato_rate: float = 5.0    # Hz
    vibrato_depth: float = 2.0   # Semitones
    
    # Formants (F1, F2, F3)
    formants: Tuple[float, float, float] = (500.0, 1500.0, 2500.0)
    formant_bandwidths: Tuple[float, float, float] = (100.0, 150.0, 200.0)
    
    # Timbre
    spectral_tilt: float = -6.0  # dB/octave
    brightness: float = 0.5      # 0-1
    
    # Dynamics
    amplitude: float = 0.7       # 0-1
    jitter: float = 0.01         # Pitch perturbation
    shimmer: float = 0.02        # Amplitude perturbation
    
    # Temporal
    speech_rate: float = 1.0     # Relative to normal


def vad_to_voice(vad: VADVector, f_base: float = 200.0) -> VoiceParameters:
    """Convert VAD vector to voice synthesis parameters."""
    V, A, D = vad.valence, vad.arousal, vad.dominance
    
    # Pitch modulation
    f0 = f_base * (1 + 0.5 * A + 0.3 * V)
    
    # Amplitude from dominance and arousal
    amplitude = 0.5 + 0.3 * D + 0.2 * A
    amplitude = max(0.1, min(1.0, amplitude))
    
    # Vibrato increases with arousal and emotional openness
    vibrato_rate = 4.5 + 2.5 * A
    vibrato_depth = 1.5 + V + 0.5 * A
    vibrato_depth = max(0.5, min(4.0, vibrato_depth))
    
    # Formant shifts
    f1 = 500 * (1 + 0.2 * V - 0.1 * D)
    f2 = 1500 * (1 + 0.15 * V + 0.1 * A)
    f3 = 2500 * (1 + 0.1 * V)
    
    # Spectral tilt: joy brightens, sadness dulls
    spectral_tilt = -6 + 4 * V - 3 * A
    brightness = 0.5 + 0.3 * V + 0.2 * A
    
    # Speech rate from arousal
    speech_rate = 1.0 + 0.5 * A - 0.3 * (1 - V) if V < 0 else 1.0 + 0.3 * A
    
    # Jitter/shimmer increase with negative emotions
    jitter = 0.01 + 0.02 * max(0, -V) + 0.01 * A
    shimmer = 0.02 + 0.03 * max(0, -V) + 0.02 * A
    
    return VoiceParameters(
        f0_base=f_base,
        f0_modulated=f0,
        vibrato_rate=vibrato_rate,
        vibrato_depth=vibrato_depth,
        formants=(f1, f2, f3),
        spectral_tilt=spectral_tilt,
        brightness=brightness,
        amplitude=amplitude,
        jitter=jitter,
        shimmer=shimmer,
        speech_rate=speech_rate
    )


# =============================================================================
# EMOTION → FREQUENCY MAPPING
# =============================================================================

F0_REFERENCE = 440.0  # A4

def emotion_to_frequency(emotion: EmotionBasis, vad: VADVector = None) -> float:
    """Map emotion to base frequency."""
    if vad is None:
        vad = EMOTION_VAD[emotion]
    
    V, A = vad.valence, vad.arousal
    
    # Emotion-specific formulas
    freq_formulas = {
        EmotionBasis.JOY: F0_REFERENCE * (1 + V + 0.5 * A),
        EmotionBasis.SADNESS: F0_REFERENCE * (1 - abs(V)) * (1 - 0.3 * A),
        EmotionBasis.FEAR: F0_REFERENCE * (1 + 0.3 * A - 0.6 * V),
        EmotionBasis.ANGER: F0_REFERENCE * (1 + 0.8 * A) * abs(math.sin(math.pi * V)),
        EmotionBasis.TRUST: F0_REFERENCE * (1 + 0.2 * V + 0.2 * A),
        EmotionBasis.SURPRISE: F0_REFERENCE * (1 + 0.5 * A),
        EmotionBasis.DISGUST: F0_REFERENCE * (1 - 0.2 * V),
        EmotionBasis.ANTICIPATION: F0_REFERENCE * (1 + 0.3 * V + 0.4 * A),
    }
    
    return freq_formulas.get(emotion, F0_REFERENCE)


# =============================================================================
# QUANTUM VOICE FIELD
# =============================================================================

def quantum_voice_field(
    quantum_state: QuantumEmotionState,
    t: float,
    f_base: float = 200.0
) -> Tuple[float, float]:
    """
    Generate quantum emotional voice field value at time t.
    Returns (pitch_hz, amplitude).
    """
    pitch_sum = complex(0, 0)
    amp_sum = complex(0, 0)
    
    for emotion, alpha in quantum_state.amplitudes.items():
        freq = emotion_to_frequency(emotion)
        vad = EMOTION_VAD[emotion]
        voice_params = vad_to_voice(vad, f_base)
        
        # Phase evolution
        phase = 2 * math.pi * freq * t / F0_REFERENCE
        wave = alpha * complex(math.cos(phase), math.sin(phase))
        
        pitch_sum += wave * voice_params.f0_modulated
        amp_sum += wave * voice_params.amplitude
    
    # Take real part (observation)
    pitch = abs(pitch_sum.real)
    amplitude = abs(amp_sum.real)
    
    return (pitch if pitch > 50 else f_base, min(1.0, amplitude))


def generate_emotional_waveform(
    quantum_state: QuantumEmotionState,
    duration: float,
    sample_rate: int = 44100,
    f_base: float = 200.0
) -> np.ndarray:
    """Generate audio waveform from quantum emotional state."""
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    
    # Get voice parameters from expected VAD
    vad = quantum_state.expected_vad()
    voice = vad_to_voice(vad, f_base)
    
    # Base frequency with vibrato
    f0 = voice.f0_modulated
    vibrato = voice.vibrato_depth * np.sin(2 * np.pi * voice.vibrato_rate * t)
    freq = f0 * (1 + vibrato / 100)
    
    # Add jitter
    jitter = 1 + voice.jitter * np.random.randn(samples)
    freq = freq * jitter
    
    # Generate carrier
    phase = np.cumsum(2 * np.pi * freq / sample_rate)
    carrier = np.sin(phase)
    
    # Apply shimmer
    amplitude = voice.amplitude * (1 + voice.shimmer * np.random.randn(samples))
    
    # Apply emotional interference pattern
    interference = 1 + 0.1 * np.sin(2 * np.pi * quantum_state.coherence() * t)
    
    return carrier * amplitude * interference


# =============================================================================
# RESONANCE CALCULATIONS
# =============================================================================

@dataclass
class ResonanceMetrics:
    """Coherence/resonance measurements."""
    coherence: float = 0.0           # Overall coherence
    emotion_biometric_match: float = 0.0
    temporal_stability: float = 0.0
    quantum_coherence: float = 0.0


def calculate_resonance(
    emotion_vad: VADVector,
    biometric_vad: VADVector,
    quantum_state: QuantumEmotionState = None,
    history: List[VADVector] = None
) -> ResonanceMetrics:
    """Calculate resonance between emotion and biometric states."""
    # Emotion-biometric match
    distance = emotion_vad.distance(biometric_vad)
    match = 1.0 / (1.0 + distance)
    
    # Temporal stability
    stability = 1.0
    if history and len(history) >= 2:
        variations = [history[i].distance(history[i-1]) for i in range(1, len(history))]
        avg_variation = sum(variations) / len(variations)
        stability = 1.0 / (1.0 + avg_variation * 5)
    
    # Quantum coherence
    q_coherence = quantum_state.coherence() if quantum_state else 0.5
    
    # Overall coherence
    coherence = 0.4 * match + 0.3 * stability + 0.3 * q_coherence
    
    return ResonanceMetrics(
        coherence=coherence,
        emotion_biometric_match=match,
        temporal_stability=stability,
        quantum_coherence=q_coherence
    )


# =============================================================================
# TREND ANALYSIS
# =============================================================================

@dataclass
class TrendMetrics:
    """VAD trend analysis."""
    valence_trend: float = 0.0      # -1 to 1 (direction * strength)
    arousal_trend: float = 0.0
    dominance_trend: float = 0.0
    confidence: float = 0.0
    prediction: Optional[VADVector] = None


def analyze_trends(history: List[VADVector], horizon: float = 1.0) -> TrendMetrics:
    """Analyze VAD trends and predict future state."""
    if len(history) < 3:
        return TrendMetrics()
    
    n = len(history)
    
    # Linear regression for each dimension
    x = np.arange(n)
    v_vals = np.array([h.valence for h in history])
    a_vals = np.array([h.arousal for h in history])
    d_vals = np.array([h.dominance for h in history])
    
    def linear_trend(y):
        slope = np.polyfit(x, y, 1)[0]
        return slope * n  # Normalized slope
    
    v_trend = linear_trend(v_vals)
    a_trend = linear_trend(a_vals)
    d_trend = linear_trend(d_vals)
    
    # Confidence from R²
    def r_squared(y):
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - ss_res / (ss_tot + 1e-10)
    
    confidence = (r_squared(v_vals) + r_squared(a_vals) + r_squared(d_vals)) / 3
    
    # Prediction
    prediction = VADVector(
        valence=max(-1, min(1, history[-1].valence + v_trend * horizon)),
        arousal=max(0, min(1, history[-1].arousal + a_trend * horizon)),
        dominance=max(-1, min(1, history[-1].dominance + d_trend * horizon))
    )
    
    return TrendMetrics(
        valence_trend=max(-1, min(1, v_trend)),
        arousal_trend=max(-1, min(1, a_trend)),
        dominance_trend=max(-1, min(1, d_trend)),
        confidence=confidence,
        prediction=prediction
    )


# =============================================================================
# MAIN API
# =============================================================================

class QuantumEmotionalVoiceField:
    """Main interface for quantum emotional voice synthesis."""
    
    def __init__(self, f_base: float = 200.0):
        self.f_base = f_base
        self.quantum_state = QuantumEmotionState()
        self.vad_history: List[VADVector] = []
        self.max_history = 50
    
    def set_emotion(self, emotion: EmotionBasis, intensity: float = 1.0):
        """Set dominant emotion."""
        # Reset amplitudes
        self.quantum_state.amplitudes = {e: complex(0.1, 0) for e in EmotionBasis}
        self.quantum_state.amplitudes[emotion] = complex(intensity, 0)
        self.quantum_state.normalize()
    
    def set_vad(self, vad: VADVector):
        """Set VAD state directly."""
        # Find closest emotion and set quantum state
        min_dist = float('inf')
        closest = EmotionBasis.JOY
        for emotion, e_vad in EMOTION_VAD.items():
            dist = vad.distance(e_vad)
            if dist < min_dist:
                min_dist = dist
                closest = emotion
        
        # Set primary emotion with others as background
        for emotion in EmotionBasis:
            dist = vad.distance(EMOTION_VAD[emotion])
            amp = 1.0 / (1.0 + dist * 2)
            self.quantum_state.amplitudes[emotion] = complex(amp, 0)
        
        self.quantum_state.normalize()
        self._record_history(vad)
    
    def set_from_biometrics(self, bio: BiometricData):
        """Set emotional state from biometrics."""
        vad = biometric_to_vad(bio)
        self.set_vad(vad)
    
    def apply_circadian(self, hour: int, day_of_week: int = 0):
        """Apply circadian rhythm adjustments."""
        adj = circadian_adjustment(hour, day_of_week)
        current = self.quantum_state.expected_vad()
        adjusted = VADVector(
            valence=max(-1, min(1, current.valence + adj.valence)),
            arousal=max(0, min(1, current.arousal + adj.arousal)),
            dominance=max(-1, min(1, current.dominance + adj.dominance))
        )
        self.set_vad(adjusted)
    
    def evolve(self, dt: float):
        """Evolve quantum state over time."""
        self.quantum_state.evolve(dt)
    
    def get_voice_params(self) -> VoiceParameters:
        """Get current voice synthesis parameters."""
        vad = self.quantum_state.expected_vad()
        return vad_to_voice(vad, self.f_base)
    
    def get_resonance(self, biometric_vad: VADVector = None) -> ResonanceMetrics:
        """Get resonance metrics."""
        emotion_vad = self.quantum_state.expected_vad()
        if biometric_vad is None:
            biometric_vad = emotion_vad
        return calculate_resonance(
            emotion_vad, biometric_vad,
            self.quantum_state, self.vad_history
        )
    
    def get_trends(self) -> TrendMetrics:
        """Get VAD trend analysis."""
        return analyze_trends(self.vad_history)
    
    def generate_audio(self, duration: float, sample_rate: int = 44100) -> np.ndarray:
        """Generate audio waveform."""
        return generate_emotional_waveform(
            self.quantum_state, duration, sample_rate, self.f_base
        )
    
    def _record_history(self, vad: VADVector):
        """Record VAD state to history."""
        self.vad_history.append(vad)
        if len(self.vad_history) > self.max_history:
            self.vad_history = self.vad_history[-self.max_history:]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=== Quantum Emotional Voice Field Test ===\n")
    
    qevf = QuantumEmotionalVoiceField(f_base=200.0)
    
    # Test 1: Set grief emotion (for "When I Found You Sleeping")
    print("Setting grief emotion...")
    qevf.set_emotion(EmotionBasis.SADNESS, intensity=0.8)
    vad = qevf.quantum_state.expected_vad()
    print(f"VAD: V={vad.valence:.2f}, A={vad.arousal:.2f}, D={vad.dominance:.2f}")
    print(f"Energy: {vad.energy():.2f}, Tension: {vad.tension():.2f}")
    
    voice = qevf.get_voice_params()
    print(f"\nVoice Parameters:")
    print(f"  F0: {voice.f0_modulated:.1f} Hz")
    print(f"  Vibrato: {voice.vibrato_rate:.1f} Hz, depth={voice.vibrato_depth:.1f} st")
    print(f"  Brightness: {voice.brightness:.2f}")
    print(f"  Jitter: {voice.jitter:.3f}")
    
    # Test 2: Quantum coherence
    print(f"\nQuantum coherence: {qevf.quantum_state.coherence():.3f}")
    print(f"Entropy: {qevf.quantum_state.entropy():.3f}")
    
    # Test 3: Biometric input
    print("\n--- Biometric Test ---")
    bio = BiometricData(
        heart_rate=85,
        heart_rate_variability=35,  # Low HRV = stress
        skin_conductance=8,         # High = arousal
        temperature=37.0
    )
    qevf.set_from_biometrics(bio)
    vad2 = qevf.quantum_state.expected_vad()
    print(f"From biometrics: V={vad2.valence:.2f}, A={vad2.arousal:.2f}, D={vad2.dominance:.2f}")
    
    # Test 4: Resonance
    resonance = qevf.get_resonance()
    print(f"\nResonance: {resonance.coherence:.3f}")
    print(f"Quantum coherence: {resonance.quantum_coherence:.3f}")
    
    # Test 5: Generate short audio clip info
    print("\n--- Audio Generation Info ---")
    audio = qevf.generate_audio(1.0)
    print(f"Generated {len(audio)} samples")
    print(f"Max amplitude: {np.max(np.abs(audio)):.3f}")
