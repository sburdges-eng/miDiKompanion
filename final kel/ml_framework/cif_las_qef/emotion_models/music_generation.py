"""
Music Generation from Emotional Models

Converts emotional states (VAD, quantum superposition) into musical parameters:
- Emotion → Frequency mapping
- Emotional chord generation
- Quantum emotional harmonic field
- Resonance and interference
- Temporal emotion flow (rhythm)
- Emotion-scale mapping
- Emotion-sound texture (timbre)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from .classical import VADState, EmotionBasis
from .quantum import EmotionSuperposition


@dataclass
class MusicalParameters:
    """Musical parameters derived from emotions."""
    frequencies: List[float]  # Hz
    amplitudes: List[float]  # 0-1
    phases: List[float]  # radians
    tempo: float  # BPM
    time_signature: Tuple[int, int]  # (numerator, denominator)
    key: str  # e.g., "C", "A"
    mode: str  # e.g., "major", "minor", "lydian"
    chord: List[int]  # MIDI note numbers
    timbre_type: str  # synthesis type
    volume: float  # 0-1


class EmotionToMusicMapper:
    """
    Maps emotions to musical frequencies and parameters.
    """
    
    def __init__(self, f0: float = 440.0):
        """
        Initialize mapper.
        
        Args:
            f0: Base frequency in Hz (default: A4 = 440 Hz)
        """
        self.f0 = f0
        
    def emotion_to_frequency(
        self,
        emotion: EmotionBasis,
        vad: Optional[VADState] = None
    ) -> float:
        """
        Map emotion to frequency using formulas.
        
        Args:
            emotion: Basic emotion
            vad: Optional VAD state for modulation
        
        Returns:
            Frequency in Hz
        """
        if vad is None:
            vad = VADState()
        
        V, A = vad.valence, vad.arousal
        
        # Emotion-specific formulas
        if emotion == EmotionBasis.JOY:
            # f_J = f_0 (1 + V + 0.5A)
            freq = self.f0 * (1.0 + V + 0.5 * A)
        elif emotion == EmotionBasis.SADNESS:
            # f_S = f_0 (1 - |V|)
            freq = self.f0 * (1.0 - abs(V))
        elif emotion == EmotionBasis.FEAR:
            # f_F = f_0 (1 + 0.3A - 0.6V)
            freq = self.f0 * (1.0 + 0.3 * A - 0.6 * V)
        elif emotion == EmotionBasis.ANGER:
            # f_A = f_0 (1 + 0.8A) sin(πV)
            freq = self.f0 * (1.0 + 0.8 * A) * np.sin(np.pi * (V + 1.0) / 2.0)
        elif emotion == EmotionBasis.TRUST:
            # f_T = f_0 (1 + 0.2V + 0.2A)
            freq = self.f0 * (1.0 + 0.2 * V + 0.2 * A)
        else:
            # Default: linear combination
            freq = self.f0 * (1.0 + 0.3 * V + 0.4 * A)
        
        return float(np.clip(freq, 80.0, 2000.0))
    
    def vad_to_chord(self, vad: VADState, base_note: int = 60) -> List[int]:
        """
        Generate chord from VAD: Chord(V,A,D) = BaseChord + Δ_valence + Δ_arousal + Δ_dominance
        
        Args:
            vad: VAD state
            base_note: Base MIDI note (default: 60 = C4)
        
        Returns:
            List of MIDI note numbers
        """
        # Base chord (major triad)
        base_chord = [base_note, base_note + 4, base_note + 7]  # I, III, V
        
        # Valence shift: +4V semitones (move to major when positive)
        delta_valence = int(4 * vad.valence)
        
        # Arousal: affects rhythm/tempo (not chord structure directly)
        # But can add extensions for high arousal
        if vad.arousal > 0.7:
            # Add 9th for high arousal
            base_chord.append(base_note + 14)
        
        # Dominance: affects dynamics (not chord structure)
        # But can add bass note for high dominance
        if vad.dominance > 0.5:
            # Add octave below
            base_chord.insert(0, base_note - 12)
        
        # Apply valence shift
        chord = [note + delta_valence for note in base_chord]
        
        # Ensure in valid MIDI range
        chord = [max(0, min(127, note)) for note in chord]
        
        return chord
    
    def emotion_to_chord(self, emotion: EmotionBasis, base_note: int = 60) -> List[int]:
        """
        Get chord for specific emotion.
        
        Args:
            emotion: Basic emotion
            base_note: Base MIDI note
        
        Returns:
            List of MIDI note numbers
        """
        # Emotion-chord mappings
        if emotion == EmotionBasis.JOY:
            # Major triad (I-III-V)
            return [base_note, base_note + 4, base_note + 7]
        elif emotion == EmotionBasis.SADNESS:
            # Minor triad (I-♭III-V)
            return [base_note, base_note + 3, base_note + 7]
        elif emotion == EmotionBasis.ANGER:
            # Diminished (I-♭III-♭V)
            return [base_note, base_note + 3, base_note + 6]
        elif emotion == EmotionBasis.FEAR:
            # Suspended (I-IV-V)
            return [base_note, base_note + 5, base_note + 7]
        elif emotion == EmotionBasis.TRUST:
            # Add9 (I-III-V-IX)
            return [base_note, base_note + 4, base_note + 7, base_note + 14]
        elif emotion == EmotionBasis.SURPRISE:
            # Lydian (#IV) - add #4
            return [base_note, base_note + 4, base_note + 6, base_note + 7]
        else:
            # Default: major triad
            return [base_note, base_note + 4, base_note + 7]
    
    def vad_to_tempo(self, vad: VADState, base_tempo: float = 120.0) -> float:
        """
        Compute tempo from VAD: T = T_0 (1 - A) (faster for high arousal)
        
        Actually, should be: T = T_0 (1 + A) for faster with high arousal
        
        Args:
            vad: VAD state
            base_tempo: Base tempo in BPM
        
        Returns:
            Tempo in BPM
        """
        # Higher arousal = faster tempo
        tempo = base_tempo * (1.0 + vad.arousal)
        return float(np.clip(tempo, 60.0, 200.0))
    
    def vad_to_volume(self, vad: VADState, base_volume: float = 0.7) -> float:
        """
        Compute volume from VAD: Volume = 70 + 30D (MIDI velocity scale)
        
        Args:
            vad: VAD state
            base_volume: Base volume (0-1)
        
        Returns:
            Volume (0-1)
        """
        # Volume = 70 + 30D (converted to 0-1 scale)
        midi_velocity = 70 + 30 * vad.dominance
        volume = midi_velocity / 127.0
        return float(np.clip(volume, 0.0, 1.0))


class QuantumMusicalField:
    """
    Quantum Emotional Harmonic Field
    
    Ψ_music(t) = Σ α_i e^(i2πf_i t + φ_i)
    """
    
    def __init__(self, f0: float = 440.0):
        """
        Initialize quantum musical field.
        
        Args:
            f0: Base frequency
        """
        self.f0 = f0
        self.mapper = EmotionToMusicMapper(f0)
        
    def superposition_to_music(
        self,
        superposition: EmotionSuperposition,
        duration: float = 1.0,
        sample_rate: int = 44100
    ) -> Tuple[np.ndarray, Dict]:
        """
        Convert emotion superposition to musical waveform.
        
        Args:
            superposition: Emotion superposition
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
        
        Returns:
            (waveform, parameters_dict)
        """
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Get frequencies for each emotion
        frequencies = []
        amplitudes = []
        phases = []
        
        for i, emotion in enumerate(superposition.basis_emotions):
            # Get frequency for this emotion
            freq = self.mapper.emotion_to_frequency(emotion)
            frequencies.append(freq)
            
            # Amplitude from superposition
            amp = np.abs(superposition.amplitudes[i])
            amplitudes.append(amp)
            
            # Phase from superposition
            phase = np.angle(superposition.amplitudes[i])
            phases.append(phase)
        
        # Generate waveform: Σ α_i e^(i2πf_i t + φ_i)
        waveform = np.zeros_like(t, dtype=complex)
        for i in range(len(frequencies)):
            waveform += amplitudes[i] * np.exp(1j * (2 * np.pi * frequencies[i] * t + phases[i]))
        
        # Take real part
        waveform_real = np.real(waveform)
        
        # Normalize
        max_amp = np.max(np.abs(waveform_real))
        if max_amp > 0:
            waveform_real = waveform_real / max_amp
        
        parameters = {
            "frequencies": frequencies,
            "amplitudes": amplitudes,
            "phases": phases,
            "duration": duration,
            "sample_rate": sample_rate
        }
        
        return waveform_real, parameters
    
    def compute_resonance_energy(
        self,
        superposition: EmotionSuperposition
    ) -> float:
        """
        Compute resonance energy: E_res = |Σ a_i e^(iφ_i)|²
        
        Args:
            superposition: Emotion superposition
        
        Returns:
            Resonance energy
        """
        amplitudes = np.abs(superposition.amplitudes)
        phases = np.angle(superposition.amplitudes)
        
        # E_res = |Σ a_i e^(iφ_i)|²
        complex_sum = np.sum(amplitudes * np.exp(1j * phases))
        energy = np.abs(complex_sum)**2
        
        return float(energy)
    
    def compute_beat_frequency(
        self,
        freq1: float,
        freq2: float
    ) -> float:
        """
        Compute beat frequency: R = cos(2π(f1 - f2)t)
        
        Args:
            freq1: First frequency
            freq2: Second frequency
        
        Returns:
            Beat frequency (f1 - f2)
        """
        return abs(freq1 - freq2)


class EmotionScaleMapper:
    """
    Maps emotions to musical scales.
    """
    
    SCALE_MAPPINGS = {
        EmotionBasis.JOY: "lydian",  # or "ionian"
        EmotionBasis.SADNESS: "aeolian",  # or "dorian"
        EmotionBasis.FEAR: "phrygian",
        EmotionBasis.ANGER: "locrian",
        EmotionBasis.TRUST: "mixolydian",
        EmotionBasis.TRUST: "major_pentatonic",  # Love
    }
    
    SCALE_INTERVALS = {
        "ionian": [0, 2, 4, 5, 7, 9, 11],  # Major
        "lydian": [0, 2, 4, 6, 7, 9, 11],
        "aeolian": [0, 2, 3, 5, 7, 8, 10],  # Natural minor
        "dorian": [0, 2, 3, 5, 7, 9, 10],
        "phrygian": [0, 1, 3, 5, 7, 8, 10],
        "locrian": [0, 1, 3, 5, 6, 8, 10],
        "mixolydian": [0, 2, 4, 5, 7, 9, 10],
        "major_pentatonic": [0, 2, 4, 7, 9],
    }
    
    def __init__(self, f0: float = 440.0):
        """
        Initialize scale mapper.
        
        Args:
            f0: Base frequency
        """
        self.f0 = f0
    
    def emotion_to_scale_frequencies(
        self,
        emotion: EmotionBasis,
        vad: Optional[VADState] = None,
        octaves: int = 2
    ) -> List[float]:
        """
        Get scale frequencies for emotion.
        
        Args:
            emotion: Basic emotion
            vad: Optional VAD state
            octaves: Number of octaves
        
        Returns:
            List of frequencies in Hz
        """
        if vad is None:
            vad = VADState()
        
        scale_name = self.SCALE_MAPPINGS.get(emotion, "ionian")
        intervals = self.SCALE_INTERVALS.get(scale_name, self.SCALE_INTERVALS["ionian"])
        
        frequencies = []
        V = vad.valence
        
        for octave in range(octaves):
            for interval in intervals:
                # Formula: f_n = f_0 × 2^((n + offset) / 12)
                # Offset depends on emotion
                if emotion == EmotionBasis.JOY:
                    offset = 7 * V  # Lydian/Ionian
                elif emotion == EmotionBasis.SADNESS:
                    offset = -3  # Aeolian/Dorian
                elif emotion == EmotionBasis.FEAR:
                    offset = -1 * vad.arousal  # Phrygian
                elif emotion == EmotionBasis.ANGER:
                    offset = -5 * vad.arousal  # Locrian
                elif emotion == EmotionBasis.TRUST:
                    offset = 2 * V  # Mixolydian
                else:
                    offset = 0
                
                semitone = interval + offset + (octave * 12)
                freq = self.f0 * (2 ** (semitone / 12.0))
                frequencies.append(freq)
        
        return frequencies


class EmotionTimbreMapper:
    """
    Maps emotions to sound texture (timbre) synthesis types.
    """
    
    TIMBRE_MAPPINGS = {
        EmotionBasis.JOY: "additive",
        EmotionBasis.FEAR: "fm",
        EmotionBasis.ANGER: "distortion",
        EmotionBasis.SADNESS: "lowpass",
        EmotionBasis.TRUST: "chorus_reverb",
    }
    
    def synthesize_additive(
        self,
        frequencies: List[float],
        amplitudes: List[float],
        t: np.ndarray,
        harmonics: int = 5
    ) -> np.ndarray:
        """
        Additive synthesis: s(t) = Σ a_i sin(2πf_i t)
        
        Args:
            frequencies: Fundamental frequencies
            amplitudes: Amplitudes
            t: Time array
            harmonics: Number of harmonics per frequency
        
        Returns:
            Waveform
        """
        waveform = np.zeros_like(t)
        
        for f, a in zip(frequencies, amplitudes):
            for h in range(1, harmonics + 1):
                waveform += (a / h) * np.sin(2 * np.pi * h * f * t)
        
        return waveform
    
    def synthesize_fm(
        self,
        fc: float,
        fm: float,
        I: float,
        t: np.ndarray
    ) -> np.ndarray:
        """
        Frequency modulation: s(t) = sin(2πf_c t + I sin(2πf_m t))
        
        Args:
            fc: Carrier frequency
            fm: Modulator frequency
            I: Modulation index
            t: Time array
        
        Returns:
            Waveform
        """
        return np.sin(2 * np.pi * fc * t + I * np.sin(2 * np.pi * fm * t))
    
    def apply_distortion(
        self,
        signal: np.ndarray,
        gain: float = 2.0
    ) -> np.ndarray:
        """
        Distortion/clipping: s'(t) = tanh(g s(t))
        
        Args:
            signal: Input signal
            gain: Distortion gain
        
        Returns:
            Distorted signal
        """
        return np.tanh(gain * signal)
    
    def apply_lowpass(
        self,
        signal: np.ndarray,
        cutoff: float,
        sample_rate: int = 44100
    ) -> np.ndarray:
        """
        Low-pass filter: s'(t) = s(t) * e^(-t/τ)
        
        Simplified: uses exponential decay
        
        Args:
            signal: Input signal
            cutoff: Cutoff frequency
            sample_rate: Sample rate
        
        Returns:
            Filtered signal
        """
        # Simplified: exponential moving average
        alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff / sample_rate)
        filtered = np.zeros_like(signal)
        filtered[0] = signal[0]
        
        for i in range(1, len(signal)):
            filtered[i] = alpha * signal[i] + (1 - alpha) * filtered[i-1]
        
        return filtered


class TemporalEmotionFlow:
    """
    Temporal emotion flow for rhythm generation.
    """
    
    def compute_rhythm_density(
        self,
        vad: VADState,
        vad_history: Optional[List[VADState]] = None
    ) -> float:
        """
        Compute rhythm density: dA/dt + |V|
        
        Args:
            vad: Current VAD state
            vad_history: History of VAD states for dA/dt
        
        Returns:
            Rhythm density (0-2)
        """
        base_density = abs(vad.valence)
        
        if vad_history and len(vad_history) > 1:
            # Compute dA/dt
            dA_dt = (vad.arousal - vad_history[-1].arousal) / (len(vad_history) * 0.01)  # Assuming 0.01s steps
            base_density += abs(dA_dt)
        
        return float(np.clip(base_density, 0.0, 2.0))
    
    def generate_rhythmic_pulse(
        self,
        tempo: float,
        duration: float,
        density: float = 1.0
    ) -> np.ndarray:
        """
        Generate rhythmic pulse pattern.
        
        Args:
            tempo: Tempo in BPM
            duration: Duration in seconds
            density: Rhythm density (0-2)
        
        Returns:
            Pulse pattern (0s and 1s)
        """
        interval = 60.0 / tempo
        n_beats = int(duration / interval)
        
        # Density affects subdivision
        subdivisions = max(1, int(density))
        
        pulse = np.zeros(int(n_beats * subdivisions))
        pulse[::subdivisions] = 1.0  # Main beats
        
        return pulse


class EmotionalMusicField:
    """
    Complete emotional music field generator.
    
    S(t) = Σ A_i(V,A,D,t) sin(2πf_i(V,A)t + φ_i)
    """
    
    def __init__(self, f0: float = 440.0):
        """
        Initialize music field generator.
        
        Args:
            f0: Base frequency
        """
        self.f0 = f0
        self.mapper = EmotionToMusicMapper(f0)
        self.quantum_field = QuantumMusicalField(f0)
        self.scale_mapper = EmotionScaleMapper(f0)
        self.timbre_mapper = EmotionTimbreMapper()
        self.temporal_flow = TemporalEmotionFlow()
        
    def generate_from_vad(
        self,
        vad: VADState,
        duration: float = 2.0,
        sample_rate: int = 44100
    ) -> Tuple[np.ndarray, MusicalParameters]:
        """
        Generate music from VAD state.
        
        Args:
            vad: VAD state
            duration: Duration in seconds
            sample_rate: Sample rate
        
        Returns:
            (waveform, musical_parameters)
        """
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Get chord
        chord = self.mapper.vad_to_chord(vad)
        frequencies = [440.0 * (2 ** ((note - 69) / 12.0)) for note in chord]
        
        # Get amplitudes (equal for now, could vary by emotion)
        amplitudes = [0.3] * len(frequencies)
        
        # Get tempo and volume
        tempo = self.mapper.vad_to_tempo(vad)
        volume = self.mapper.vad_to_volume(vad)
        
        # Generate waveform
        waveform = np.zeros_like(t)
        for f, a in zip(frequencies, amplitudes):
            waveform += a * np.sin(2 * np.pi * f * t)
        
        # Normalize
        max_amp = np.max(np.abs(waveform))
        if max_amp > 0:
            waveform = waveform / max_amp * volume
        
        # Create parameters
        params = MusicalParameters(
            frequencies=frequencies,
            amplitudes=amplitudes,
            phases=[0.0] * len(frequencies),
            tempo=tempo,
            time_signature=(4, 4),
            key="C",
            mode="major" if vad.valence > 0 else "minor",
            chord=chord,
            timbre_type="additive",
            volume=volume
        )
        
        return waveform, params
    
    def generate_from_superposition(
        self,
        superposition: EmotionSuperposition,
        duration: float = 2.0,
        sample_rate: int = 44100
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate music from quantum superposition.
        
        Args:
            superposition: Emotion superposition
            duration: Duration in seconds
            sample_rate: Sample rate
        
        Returns:
            (waveform, parameters_dict)
        """
        return self.quantum_field.superposition_to_music(
            superposition,
            duration,
            sample_rate
        )
