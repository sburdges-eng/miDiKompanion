"""
Emotion Interface (EI)

Captures, translates, and transmits affective input into
Emotional State Vectors (ESVs) that guide composition.
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class EmotionalStateVector:
    """
    Emotional State Vector for LAS.
    
    Similar to CIF ESV but optimized for creative generation.
    """
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    tension: float = 0.0
    creativity: float = 0.0  # Additional dimension for creative potential
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.valence, self.arousal, self.dominance, self.tension, self.creativity])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "tension": self.tension,
            "creativity": self.creativity
        }


class EmotionInterface:
    """
    Emotion Interface (EI)
    
    Converts multimodal emotional data into Emotional State Vectors (ESVs).
    """
    
    def __init__(self):
        """Initialize Emotion Interface."""
        self.input_modalities = ["biofeedback", "voice", "text", "facial", "gesture"]
        
    def process_emotional_input(self, input_data: Dict) -> EmotionalStateVector:
        """
        Process multimodal emotional input into ESV.
        
        Args:
            input_data: Dictionary with keys:
                - biofeedback: Heart rate, EEG, GSR, etc.
                - voice: Voice tone analysis
                - text: Text sentiment
                - facial: Facial expression data
                - gesture: Gesture/posture data
        
        Returns:
            EmotionalStateVector
        """
        # Extract from different modalities
        bio = input_data.get("biofeedback", {})
        voice = input_data.get("voice", {})
        text = input_data.get("text", {})
        facial = input_data.get("facial", {})
        
        # Compute ESV dimensions from various inputs
        valence = self._compute_valence(bio, voice, text, facial)
        arousal = self._compute_arousal(bio, voice, text)
        dominance = self._compute_dominance(bio, gesture=input_data.get("gesture", {}))
        tension = self._compute_tension(bio, facial)
        creativity = self._compute_creativity(text, bio)
        
        return EmotionalStateVector(
            valence=float(np.clip(valence, -1.0, 1.0)),
            arousal=float(np.clip(arousal, 0.0, 1.0)),
            dominance=float(np.clip(dominance, 0.0, 1.0)),
            tension=float(np.clip(tension, 0.0, 1.0)),
            creativity=float(np.clip(creativity, 0.0, 1.0))
        )
    
    def _compute_valence(self, bio: Dict, voice: Dict, text: Dict, facial: Dict) -> float:
        """Compute valence (positive/negative emotion)."""
        # Voice tone (most direct)
        voice_valence = voice.get("tone", 0.0)
        
        # Text sentiment
        text_sentiment = text.get("sentiment", 0.0)
        
        # Facial expression
        facial_valence = facial.get("valence", 0.0)
        
        # EEG alpha (higher = more positive)
        eeg_alpha = bio.get("eeg_alpha", 0.5)
        eeg_valence = (eeg_alpha - 0.5) * 2.0
        
        # Weighted average
        return (
            voice_valence * 0.3 +
            text_sentiment * 0.3 +
            facial_valence * 0.2 +
            eeg_valence * 0.2
        )
    
    def _compute_arousal(self, bio: Dict, voice: Dict, text: Dict) -> float:
        """Compute arousal (intensity of emotion)."""
        # Heart rate
        hr = bio.get("heart_rate", 70.0)
        hr_arousal = (hr - 60) / 100.0
        hr_arousal = np.clip(hr_arousal, 0.0, 1.0)
        
        # Voice intensity
        voice_intensity = voice.get("intensity", 0.5)
        
        # Text energy
        text_energy = text.get("energy", 0.5)
        
        # GSR
        gsr = bio.get("gsr", 0.5)
        
        return (
            hr_arousal * 0.3 +
            voice_intensity * 0.3 +
            text_energy * 0.2 +
            gsr * 0.2
        )
    
    def _compute_dominance(self, bio: Dict, gesture: Dict) -> float:
        """Compute dominance (sense of control)."""
        # Posture
        posture = gesture.get("posture", 0.5)
        
        # Heart rate variability (lower HRV = more stress = less dominance)
        hrv = bio.get("hrv", 50.0)
        hrv_dominance = hrv / 100.0
        hrv_dominance = np.clip(hrv_dominance, 0.0, 1.0)
        
        return (posture * 0.6 + hrv_dominance * 0.4)
    
    def _compute_tension(self, bio: Dict, facial: Dict) -> float:
        """Compute tension (stress/anxiety)."""
        # Facial tension
        facial_tension = facial.get("tension", 0.5)
        
        # Muscle tension (if available)
        muscle_tension = bio.get("muscle_tension", 0.5)
        
        # EEG beta (higher = more alert/stressed)
        eeg_beta = bio.get("eeg_beta", 0.5)
        
        return (
            facial_tension * 0.4 +
            muscle_tension * 0.3 +
            eeg_beta * 0.3
        )
    
    def _compute_creativity(self, text: Dict, bio: Dict) -> float:
        """Compute creativity potential."""
        # Text novelty
        text_novelty = text.get("novelty", 0.5)
        
        # EEG gamma (associated with creative insight)
        eeg_gamma = bio.get("eeg_gamma", 0.5)
        
        # Theta/alpha ratio (creative flow state)
        theta_alpha = bio.get("theta_alpha_ratio", 0.5)
        
        return (
            text_novelty * 0.4 +
            eeg_gamma * 0.3 +
            theta_alpha * 0.3
        )
