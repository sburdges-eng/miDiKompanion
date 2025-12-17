"""
Voice Synthesis from Emotional Models

Converts emotional states into voice parameters and synthesis:
- Emotion → Voice parameter mapping
- Quantum emotional voice field
- Voice morphing between emotions
- Advanced voice synthesis formulas
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from .classical import VADState, EmotionBasis
from .quantum import EmotionSuperposition


@dataclass
class VoiceParameters:
    """Voice synthesis parameters."""
    f0: float  # Fundamental frequency (pitch) in Hz
    amplitude: float  # Volume (0-1)
    formants: List[float]  # F1, F2, F3 in Hz
    spectral_tilt: float  # dB/octave
    vibrato_rate: float  # Hz
    vibrato_depth: float  # semitones
    speech_rate: float  # relative to base
    jitter: float  # percentage
    shimmer: float  # percentage


class EmotionToVoiceMapper:
    """
    Maps emotions to voice parameters.
    """
    
    def __init__(
        self,
        f0_base: float = 200.0,
        amplitude_base: float = 0.7
    ):
        """
        Initialize voice mapper.
        
        Args:
            f0_base: Base fundamental frequency in Hz
            amplitude_base: Base amplitude (0-1)
        """
        self.f0_base = f0_base
        self.amplitude_base = amplitude_base
        
        # Default formants (vowel "ah")
        self.formants_base = [730.0, 1090.0, 2440.0]  # F1, F2, F3
    
    def vad_to_pitch(
        self,
        vad: VADState
    ) -> float:
        """
        Compute pitch: f_0 = f_base (1 + 0.5A + 0.3V)
        
        Args:
            vad: VAD state
        
        Returns:
            Fundamental frequency in Hz
        """
        f0 = self.f0_base * (1.0 + 0.5 * vad.arousal + 0.3 * vad.valence)
        return float(np.clip(f0, 80.0, 400.0))
    
    def vad_to_volume(
        self,
        vad: VADState
    ) -> float:
        """
        Compute volume: A = A_base (1 + 0.4D + 0.3A)
        
        Args:
            vad: VAD state
        
        Returns:
            Amplitude (0-1)
        """
        amplitude = self.amplitude_base * (1.0 + 0.4 * vad.dominance + 0.3 * vad.arousal)
        return float(np.clip(amplitude, 0.0, 1.0))
    
    def vad_to_formants(
        self,
        vad: VADState
    ) -> List[float]:
        """
        Compute formant shift: F_i' = F_i (1 + 0.2V - 0.1D)
        
        Args:
            vad: VAD state
        
        Returns:
            List of formant frequencies [F1, F2, F3]
        """
        formants = []
        for F_base in self.formants_base:
            F_shifted = F_base * (1.0 + 0.2 * vad.valence - 0.1 * vad.dominance)
            formants.append(float(np.clip(F_shifted, 200.0, 4000.0)))
        return formants
    
    def vad_to_spectral_tilt(
        self,
        vad: VADState,
        base_tilt: float = -6.0
    ) -> float:
        """
        Compute spectral tilt: T_s' = T_s + (6V - 4A)
        
        Args:
            vad: VAD state
            base_tilt: Base spectral tilt in dB/octave
        
        Returns:
            Spectral tilt in dB/octave
        """
        tilt = base_tilt + (6.0 * vad.valence - 4.0 * vad.arousal)
        return float(np.clip(tilt, -12.0, 6.0))
    
    def vad_to_vibrato_rate(
        self,
        vad: VADState
    ) -> float:
        """
        Compute vibrato rate: v_r' = 5 + 3A
        
        Args:
            vad: VAD state
        
        Returns:
            Vibrato rate in Hz
        """
        rate = 5.0 + 3.0 * vad.arousal
        return float(np.clip(rate, 4.0, 8.0))
    
    def vad_to_vibrato_depth(
        self,
        vad: VADState
    ) -> float:
        """
        Compute vibrato depth: v_d' = 2 + V + 0.5A
        
        Args:
            vad: VAD state
        
        Returns:
            Vibrato depth in semitones
        """
        depth = 2.0 + vad.valence + 0.5 * vad.arousal
        return float(np.clip(depth, 1.0, 3.0))
    
    def vad_to_speech_rate(
        self,
        vad: VADState,
        base_rate: float = 1.0
    ) -> float:
        """
        Compute speech rate: R = R_0 (1 + 0.7A - 0.4V)
        
        Args:
            vad: VAD state
            base_rate: Base speech rate
        
        Returns:
            Speech rate multiplier
        """
        rate = base_rate * (1.0 + 0.7 * vad.arousal - 0.4 * vad.valence)
        return float(np.clip(rate, 0.5, 2.0))
    
    def vad_to_jitter(
        self,
        vad: VADState,
        base_jitter: float = 0.01
    ) -> float:
        """
        Compute jitter (pitch instability).
        
        Args:
            vad: VAD state
            base_jitter: Base jitter percentage
        
        Returns:
            Jitter percentage
        """
        # Higher arousal and negative valence increase jitter
        jitter = base_jitter * (1.0 + 2.0 * vad.arousal - 0.5 * vad.valence)
        return float(np.clip(jitter, 0.0, 0.05))
    
    def vad_to_voice_parameters(
        self,
        vad: VADState
    ) -> VoiceParameters:
        """
        Compute all voice parameters from VAD.
        
        Args:
            vad: VAD state
        
        Returns:
            VoiceParameters object
        """
        return VoiceParameters(
            f0=self.vad_to_pitch(vad),
            amplitude=self.vad_to_volume(vad),
            formants=self.vad_to_formants(vad),
            spectral_tilt=self.vad_to_spectral_tilt(vad),
            vibrato_rate=self.vad_to_vibrato_rate(vad),
            vibrato_depth=self.vad_to_vibrato_depth(vad),
            speech_rate=self.vad_to_speech_rate(vad),
            jitter=self.vad_to_jitter(vad),
            shimmer=self.vad_to_jitter(vad) * 0.5  # Shimmer related to jitter
        )
    
    def emotion_to_voice_pattern(
        self,
        emotion: EmotionBasis
    ) -> Dict[str, str]:
        """
        Get voice pattern for specific emotion.
        
        Args:
            emotion: Basic emotion
        
        Returns:
            Pattern description
        """
        patterns = {
            EmotionBasis.JOY: {
                "f0": "↑",
                "amplitude": "↑",
                "F1": "↑",
                "vibrato_rate": "↑",
                "spectral_tilt": "↑",
                "description": "bright, resonant"
            },
            EmotionBasis.SADNESS: {
                "f0": "↓",
                "amplitude": "↓",
                "F1": "↓",
                "spectral_tilt": "↓",
                "description": "low, mellow"
            },
            EmotionBasis.ANGER: {
                "f0": "↑",
                "amplitude": "↑↑",
                "F1": "↑",
                "jitter": "↑",
                "description": "sharp, intense"
            },
            EmotionBasis.FEAR: {
                "f0": "↑↑",
                "amplitude": "↓",
                "vibrato_depth": "↑",
                "jitter": "↑↑",
                "description": "trembling"
            },
            EmotionBasis.TRUST: {
                "f0": "≈",
                "amplitude": "↑",
                "F2": "↑",
                "description": "stable, open"
            },
        }
        
        return patterns.get(emotion, {"description": "neutral"})


class QuantumVoiceField:
    """
    Quantum Emotional Voice Field
    
    Voice as quantum superposition of emotional states.
    """
    
    def __init__(self, f0_base: float = 200.0):
        """
        Initialize quantum voice field.
        
        Args:
            f0_base: Base fundamental frequency
        """
        self.f0_base = f0_base
        self.mapper = EmotionToVoiceMapper(f0_base)
    
    def superposition_to_voice_parameters(
        self,
        superposition: EmotionSuperposition
    ) -> VoiceParameters:
        """
        Convert superposition to voice parameters using expectation values.
        
        ⟨f_0(t)⟩ = Σ |α_i|² f_0,i
        ⟨A(t)⟩ = Σ |α_i|² A_i
        
        Args:
            superposition: Emotion superposition
        
        Returns:
            VoiceParameters
        """
        probabilities = superposition.get_probabilities()
        
        # Compute expectation values
        f0_expected = 0.0
        amplitude_expected = 0.0
        formants_expected = [0.0, 0.0, 0.0]
        vibrato_rate_expected = 0.0
        vibrato_depth_expected = 0.0
        
        for i, emotion in enumerate(superposition.basis_emotions):
            prob = probabilities[i]
            
            # Create VAD from emotion (simplified)
            from .classical import PlutchikWheel
            plutchik = PlutchikWheel()
            vad = plutchik.emotion_to_vad(emotion, intensity=1.0)
            
            # Get parameters for this emotion
            params = self.mapper.vad_to_voice_parameters(vad)
            
            # Weight by probability
            f0_expected += prob * params.f0
            amplitude_expected += prob * params.amplitude
            for j, F in enumerate(params.formants):
                formants_expected[j] += prob * F
            vibrato_rate_expected += prob * params.vibrato_rate
            vibrato_depth_expected += prob * params.vibrato_depth
        
        return VoiceParameters(
            f0=f0_expected,
            amplitude=amplitude_expected,
            formants=formants_expected,
            spectral_tilt=-6.0,  # Average
            vibrato_rate=vibrato_rate_expected,
            vibrato_depth=vibrato_depth_expected,
            speech_rate=1.0,
            jitter=0.01,
            shimmer=0.005
        )
    
    def generate_voice_waveform(
        self,
        voice_params: VoiceParameters,
        duration: float = 1.0,
        sample_rate: int = 44100
    ) -> np.ndarray:
        """
        Generate voice waveform from parameters.
        
        s(t) = A(t) sin(2πf_0(t)t + φ(t))
        
        with f_0(t) = f_base (1 + v_d sin(2πv_r t))
        
        Args:
            voice_params: Voice parameters
            duration: Duration in seconds
            sample_rate: Sample rate
        
        Returns:
            Waveform
        """
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Modulated pitch with vibrato
        f0_modulated = voice_params.f0 * (
            1.0 + (voice_params.vibrato_depth / 12.0) * 
            np.sin(2 * np.pi * voice_params.vibrato_rate * t)
        )
        
        # Add jitter (random micro-perturbations)
        if voice_params.jitter > 0:
            jitter_noise = np.random.normal(0, voice_params.jitter, len(t))
            f0_modulated = f0_modulated * (1.0 + jitter_noise)
        
        # Generate fundamental
        phase = np.cumsum(2 * np.pi * f0_modulated / sample_rate)
        waveform = voice_params.amplitude * np.sin(phase)
        
        # Add formants (simplified: add harmonics at formant frequencies)
        for F in voice_params.formants:
            # Add harmonic at formant frequency
            harmonic_amp = 0.3 * voice_params.amplitude
            waveform += harmonic_amp * np.sin(2 * np.pi * F * t)
        
        # Apply spectral tilt (simplified: high-pass filter effect)
        # In practice, would use proper filter
        
        # Normalize
        max_amp = np.max(np.abs(waveform))
        if max_amp > 0:
            waveform = waveform / max_amp
        
        return waveform
    
    def generate_from_superposition(
        self,
        superposition: EmotionSuperposition,
        duration: float = 1.0,
        sample_rate: int = 44100
    ) -> Tuple[np.ndarray, VoiceParameters]:
        """
        Generate voice from quantum superposition.
        
        Args:
            superposition: Emotion superposition
            duration: Duration in seconds
            sample_rate: Sample rate
        
        Returns:
            (waveform, voice_parameters)
        """
        voice_params = self.superposition_to_voice_parameters(superposition)
        waveform = self.generate_voice_waveform(voice_params, duration, sample_rate)
        
        return waveform, voice_params


class VoiceMorphing:
    """
    Emotional voice morphing between states.
    """
    
    def __init__(self, f0_base: float = 200.0):
        """
        Initialize voice morphing.
        
        Args:
            f0_base: Base fundamental frequency
        """
        self.f0_base = f0_base
        self.mapper = EmotionToVoiceMapper(f0_base)
    
    def morph_voice_parameters(
        self,
        params1: VoiceParameters,
        params2: VoiceParameters,
        lambda_t: float
    ) -> VoiceParameters:
        """
        Morph between two voice parameter sets.
        
        P_blend(t) = (1 - λ(t)) P_1 + λ(t) P_2
        
        Args:
            params1: First voice parameters
            params2: Second voice parameters
            lambda_t: Blend factor (0-1)
        
        Returns:
            Blended voice parameters
        """
        lambda_t = np.clip(lambda_t, 0.0, 1.0)
        
        # Blend formants
        formants_blend = [
            (1 - lambda_t) * f1 + lambda_t * f2
            for f1, f2 in zip(params1.formants, params2.formants)
        ]
        
        return VoiceParameters(
            f0=(1 - lambda_t) * params1.f0 + lambda_t * params2.f0,
            amplitude=(1 - lambda_t) * params1.amplitude + lambda_t * params2.amplitude,
            formants=formants_blend,
            spectral_tilt=(1 - lambda_t) * params1.spectral_tilt + lambda_t * params2.spectral_tilt,
            vibrato_rate=(1 - lambda_t) * params1.vibrato_rate + lambda_t * params2.vibrato_rate,
            vibrato_depth=(1 - lambda_t) * params1.vibrato_depth + lambda_t * params2.vibrato_depth,
            speech_rate=(1 - lambda_t) * params1.speech_rate + lambda_t * params2.speech_rate,
            jitter=(1 - lambda_t) * params1.jitter + lambda_t * params2.jitter,
            shimmer=(1 - lambda_t) * params1.shimmer + lambda_t * params2.shimmer
        )
    
    def generate_morphing_sequence(
        self,
        vad1: VADState,
        vad2: VADState,
        duration: float = 2.0,
        morph_frequency: float = 0.5,
        sample_rate: int = 44100
    ) -> Tuple[np.ndarray, List[VoiceParameters]]:
        """
        Generate smooth morphing sequence.
        
        λ(t) = 0.5 (1 + sin(2πf_morph t))
        
        Args:
            vad1: Starting VAD state
            vad2: Ending VAD state
            duration: Duration in seconds
            morph_frequency: Morphing frequency
            sample_rate: Sample rate
        
        Returns:
            (waveform, parameter_sequence)
        """
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Morphing function
        lambda_t_array = 0.5 * (1.0 + np.sin(2 * np.pi * morph_frequency * t))
        
        # Get parameters
        params1 = self.mapper.vad_to_voice_parameters(vad1)
        params2 = self.mapper.vad_to_voice_parameters(vad2)
        
        # Generate waveform
        waveform = np.zeros_like(t)
        param_sequence = []
        
        quantum_field = QuantumVoiceField(self.f0_base)
        
        for i, lambda_t in enumerate(lambda_t_array):
            # Morph parameters
            params_blend = self.morph_voice_parameters(params1, params2, lambda_t)
            param_sequence.append(params_blend)
            
            # Generate sample (simplified: use current params)
            if i < len(t) - 1:
                dt = t[i+1] - t[i]
                sample = quantum_field.generate_voice_waveform(params_blend, dt, sample_rate)
                if len(sample) > 0:
                    waveform[i] = sample[0] if len(sample) == 1 else sample[0]
        
        return waveform, param_sequence


class QuantumEmotionalVoiceField:
    """
    Complete Quantum Emotional Voice Field (QEVF)
    
    Voice(t,f) = Σ α_i(t) · G(f; F_i'(t), Q_i(t)) · e^(i2πf_0(t)t)
    """
    
    def __init__(self, f0_base: float = 200.0):
        """
        Initialize QEVF.
        
        Args:
            f0_base: Base fundamental frequency
        """
        self.f0_base = f0_base
        self.mapper = EmotionToVoiceMapper(f0_base)
        self.quantum_field = QuantumVoiceField(f0_base)
    
    def generate_voice_field(
        self,
        superposition: EmotionSuperposition,
        duration: float = 1.0,
        sample_rate: int = 44100
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate complete voice field from superposition.
        
        Args:
            superposition: Emotion superposition
            duration: Duration in seconds
            sample_rate: Sample rate
        
        Returns:
            (waveform, field_parameters)
        """
        # Get voice parameters
        voice_params = self.quantum_field.superposition_to_voice_parameters(superposition)
        
        # Generate waveform
        waveform = self.quantum_field.generate_voice_waveform(
            voice_params,
            duration,
            sample_rate
        )
        
        # Field parameters
        field_params = {
            "f0": voice_params.f0,
            "formants": voice_params.formants,
            "amplitude": voice_params.amplitude,
            "vibrato_rate": voice_params.vibrato_rate,
            "vibrato_depth": voice_params.vibrato_depth,
            "probabilities": superposition.get_probabilities().tolist()
        }
        
        return waveform, field_params
