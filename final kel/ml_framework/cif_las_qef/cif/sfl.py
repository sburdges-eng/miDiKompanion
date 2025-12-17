"""
Sensory Fusion Layer (SFL)

Translates human biological/affective data into machine-readable
Emotional State Vectors (ESVs) and vice versa.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class EmotionalStateVector:
    """
    Emotional State Vector (ESV)
    
    Represents emotional state as a multi-dimensional vector.
    Typically includes valence, arousal, and other affective dimensions.
    """
    valence: float = 0.0      # -1.0 (negative) to +1.0 (positive)
    arousal: float = 0.0       # 0.0 (calm) to 1.0 (intense)
    dominance: float = 0.0     # 0.0 (submissive) to 1.0 (dominant)
    tension: float = 0.0       # 0.0 (relaxed) to 1.0 (tense)
    
    def to_array(self) -> np.ndarray:
        """Convert ESV to numpy array."""
        return np.array([self.valence, self.arousal, self.dominance, self.tension])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'EmotionalStateVector':
        """Create ESV from numpy array."""
        return cls(
            valence=float(arr[0]) if len(arr) > 0 else 0.0,
            arousal=float(arr[1]) if len(arr) > 1 else 0.0,
            dominance=float(arr[2]) if len(arr) > 2 else 0.0,
            tension=float(arr[3]) if len(arr) > 3 else 0.0
        )


class SensoryFusionLayer:
    """
    Sensory Fusion Layer (SFL)
    
    Maps bio-affective data (heart rate, EEG, microexpressions, voice tone)
    to Emotional State Vectors (ESVs) and vice versa.
    """
    
    def __init__(self):
        """Initialize SFL with default mappings."""
        # Default mapping parameters (research phase - simplified)
        self.hr_to_arousal_scale = 0.01  # Heart rate impact on arousal
        self.eeg_alpha_to_valence = 0.5  # EEG alpha band to valence
        self.voice_tone_to_valence = 0.3  # Voice tone to valence
        
    def map_bio_to_esv(self, bio_data: Dict) -> np.ndarray:
        """
        Map biological/affective data to Emotional State Vector.
        
        Args:
            bio_data: Dictionary containing:
                - heart_rate: Heart rate (BPM)
                - eeg_alpha: EEG alpha band power (0-1)
                - voice_tone: Voice tone analysis (-1 to 1)
                - facial_tension: Facial microexpression tension (0-1)
                - gsr: Galvanic skin response (0-1)
        
        Returns:
            ESV as numpy array [valence, arousal, dominance, tension]
        """
        # Extract bio signals (with defaults)
        heart_rate = bio_data.get("heart_rate", 70.0)
        eeg_alpha = bio_data.get("eeg_alpha", 0.5)
        voice_tone = bio_data.get("voice_tone", 0.0)
        facial_tension = bio_data.get("facial_tension", 0.5)
        gsr = bio_data.get("gsr", 0.5)
        
        # Map to ESV dimensions (simplified research-phase mapping)
        # Valence: positive/negative emotion
        valence = (
            eeg_alpha * self.eeg_alpha_to_valence +
            voice_tone * self.voice_tone_to_valence
        )
        valence = np.clip(valence, -1.0, 1.0)
        
        # Arousal: intensity of emotion
        arousal = (
            (heart_rate - 60) / 100.0 * self.hr_to_arousal_scale * 100 +
            gsr * 0.5
        )
        arousal = np.clip(arousal, 0.0, 1.0)
        
        # Dominance: sense of control (simplified)
        dominance = 1.0 - facial_tension  # Less tension = more dominance
        dominance = np.clip(dominance, 0.0, 1.0)
        
        # Tension: stress/anxiety level
        tension = facial_tension * 0.7 + (1.0 - eeg_alpha) * 0.3
        tension = np.clip(tension, 0.0, 1.0)
        
        return np.array([valence, arousal, dominance, tension])
    
    def map_esv_to_music_params(self, esv: np.ndarray) -> Dict:
        """
        Map Emotional State Vector to musical parameters.
        
        Args:
            esv: Emotional State Vector [valence, arousal, dominance, tension]
        
        Returns:
            Dictionary of musical parameters
        """
        valence, arousal, dominance, tension = esv
        
        # Map to musical parameters (research phase - simplified)
        return {
            "tempo": int(60 + arousal * 120),  # 60-180 BPM
            "key_mode": "major" if valence > 0 else "minor",
            "harmonic_tension": float(tension),
            "dynamic_range": float(arousal * 0.8 + 0.2),
            "timbre_brightness": float((valence + 1.0) / 2.0),
            "rhythmic_density": float(arousal)
        }
    
    def map_music_to_esv(self, music_params: Dict) -> np.ndarray:
        """
        Map musical parameters back to Emotional State Vector.
        
        Inverse mapping for feedback loops.
        
        Args:
            music_params: Dictionary of musical parameters
        
        Returns:
            ESV as numpy array
        """
        tempo = music_params.get("tempo", 120)
        key_mode = music_params.get("key_mode", "major")
        harmonic_tension = music_params.get("harmonic_tension", 0.5)
        dynamic_range = music_params.get("dynamic_range", 0.5)
        
        # Reverse mapping (simplified)
        arousal = (tempo - 60) / 120.0
        arousal = np.clip(arousal, 0.0, 1.0)
        
        valence = 1.0 if key_mode == "major" else -1.0
        valence *= (1.0 - harmonic_tension * 0.5)  # Adjust by tension
        
        tension = harmonic_tension
        dominance = dynamic_range
        
        return np.array([valence, arousal, dominance, tension])
