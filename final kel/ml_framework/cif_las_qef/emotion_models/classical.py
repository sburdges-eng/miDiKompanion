"""
Classical Emotional Models

Base layer for classical emotional representation:
- VAD (Valence-Arousal-Dominance) Model
- Plutchik's Wheel with derived formulas
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from enum import Enum


class EmotionBasis(Enum):
    """Basic emotions from Plutchik's Wheel."""
    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"


@dataclass
class VADState:
    """
    Valence-Arousal-Dominance State
    
    E = (V, A, D)
    """
    valence: float = 0.0      # V: [-1, 1] (pleasant-unpleasant)
    arousal: float = 0.5     # A: [0, 1] (calm-excited)
    dominance: float = 0.0   # D: [-1, 1] (control-submission)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.valence, self.arousal, self.dominance])
    
    def clip(self):
        """Clip values to valid ranges."""
        self.valence = np.clip(self.valence, -1.0, 1.0)
        self.arousal = np.clip(self.arousal, 0.0, 1.0)
        self.dominance = np.clip(self.dominance, -1.0, 1.0)


class VADModel:
    """
    Valence-Arousal-Dominance Model
    
    Classical 3D coordinate system for emotional states.
    """
    
    def __init__(self):
        """Initialize VAD Model."""
        pass
    
    def compute_energy_level(self, vad: VADState) -> float:
        """
        Compute energy level: E_n = A × (1 + |V|)
        
        Args:
            vad: VAD state
        
        Returns:
            Energy level (0-2)
        """
        return vad.arousal * (1.0 + abs(vad.valence))
    
    def compute_emotional_tension(self, vad: VADState) -> float:
        """
        Compute emotional tension: T = |V| × (1 - D)
        
        Args:
            vad: VAD state
        
        Returns:
            Tension level (0-2)
        """
        return abs(vad.valence) * (1.0 - vad.dominance)
    
    def compute_stability_index(self, vad: VADState) -> float:
        """
        Compute stability index: S = 1 - sqrt((V² + A² + D²) / 3)
        
        Args:
            vad: VAD state
        
        Returns:
            Stability index (0-1, higher = more stable)
        """
        magnitude = np.sqrt(
            (vad.valence**2 + vad.arousal**2 + vad.dominance**2) / 3.0
        )
        return 1.0 - magnitude
    
    def compute_all_metrics(self, vad: VADState) -> Dict[str, float]:
        """
        Compute all VAD-derived metrics.
        
        Args:
            vad: VAD state
        
        Returns:
            Dictionary of all metrics
        """
        return {
            "energy_level": self.compute_energy_level(vad),
            "emotional_tension": self.compute_emotional_tension(vad),
            "stability_index": self.compute_stability_index(vad),
            "valence": vad.valence,
            "arousal": vad.arousal,
            "dominance": vad.dominance
        }


class PlutchikWheel:
    """
    Plutchik's Wheel of Emotions
    
    Each basic emotion has intensity and relationships.
    """
    
    # Emotion definitions: (V, A, D, intensity_coefficient)
    EMOTION_DEFINITIONS = {
        EmotionBasis.JOY: (1.0, 0.6, 0.3, 1.0),
        EmotionBasis.TRUST: (0.5, 0.4, 0.2, 0.8),
        EmotionBasis.FEAR: (-0.6, 0.8, -0.4, 1.0),
        EmotionBasis.SURPRISE: (0.0, 0.9, -0.1, 0.9),
        EmotionBasis.SADNESS: (-0.8, 0.3, -0.2, 1.0),
        EmotionBasis.DISGUST: (-0.7, 0.5, -0.3, 0.9),
        EmotionBasis.ANGER: (-0.5, 0.9, 0.4, 1.0),
        EmotionBasis.ANTICIPATION: (0.3, 0.7, 0.1, 0.8),
    }
    
    # Opposite pairs
    OPPOSITES = {
        EmotionBasis.JOY: EmotionBasis.SADNESS,
        EmotionBasis.TRUST: EmotionBasis.DISGUST,
        EmotionBasis.FEAR: EmotionBasis.ANGER,
        EmotionBasis.SURPRISE: EmotionBasis.ANTICIPATION,
    }
    
    def __init__(self):
        """Initialize Plutchik's Wheel."""
        pass
    
    def get_emotion_vad(self, emotion: EmotionBasis) -> Tuple[float, float, float]:
        """
        Get VAD coordinates for a basic emotion.
        
        Args:
            emotion: Basic emotion
        
        Returns:
            (V, A, D) tuple
        """
        definition = self.EMOTION_DEFINITIONS.get(emotion)
        if definition:
            return definition[:3]
        return (0.0, 0.5, 0.0)
    
    def compute_emotion_intensity(
        self,
        emotion: EmotionBasis,
        arousal: float,
        valence: float
    ) -> float:
        """
        Compute emotion intensity: k × A × (1 + V)
        
        Args:
            emotion: Basic emotion
            arousal: Arousal level
            valence: Valence level
        
        Returns:
            Emotion intensity (0-2)
        """
        definition = self.EMOTION_DEFINITIONS.get(emotion)
        if not definition:
            return 0.0
        
        k = definition[3]  # intensity coefficient
        return k * arousal * (1.0 + valence)
    
    def get_opposite(self, emotion: EmotionBasis) -> Optional[EmotionBasis]:
        """
        Get opposite emotion.
        
        Args:
            emotion: Basic emotion
        
        Returns:
            Opposite emotion or None
        """
        return self.OPPOSITES.get(emotion)
    
    def combine_emotions(
        self,
        emotion1: EmotionBasis,
        emotion2: EmotionBasis,
        weight1: float = 0.5,
        weight2: float = 0.5
    ) -> VADState:
        """
        Combine two emotions (e.g., Joy + Trust → Love).
        
        Args:
            emotion1: First emotion
            emotion2: Second emotion
            weight1: Weight for first emotion
            weight2: Weight for second emotion
        
        Returns:
            Combined VAD state
        """
        v1, a1, d1 = self.get_emotion_vad(emotion1)
        v2, a2, d2 = self.get_emotion_vad(emotion2)
        
        # Weighted combination
        total_weight = weight1 + weight2
        if total_weight > 0:
            v = (v1 * weight1 + v2 * weight2) / total_weight
            a = (a1 * weight1 + a2 * weight2) / total_weight
            d = (d1 * weight1 + d2 * weight2) / total_weight
        else:
            v, a, d = 0.0, 0.5, 0.0
        
        return VADState(valence=v, arousal=a, dominance=d)
    
    def emotion_to_vad(self, emotion: EmotionBasis, intensity: float = 1.0) -> VADState:
        """
        Convert emotion to VAD state with intensity scaling.
        
        Args:
            emotion: Basic emotion
            intensity: Intensity multiplier (0-1)
        
        Returns:
            VAD state
        """
        v, a, d = self.get_emotion_vad(emotion)
        
        # Scale by intensity
        return VADState(
            valence=v * intensity,
            arousal=a * intensity,
            dominance=d * intensity
        )
    
    def vad_to_emotion(self, vad: VADState, threshold: float = 0.3) -> List[Tuple[EmotionBasis, float]]:
        """
        Map VAD state to closest emotions with similarity scores.
        
        Args:
            vad: VAD state
            threshold: Minimum similarity threshold
        
        Returns:
            List of (emotion, similarity) tuples
        """
        results = []
        vad_array = vad.to_array()
        
        for emotion in EmotionBasis:
            v, a, d = self.get_emotion_vad(emotion)
            emotion_array = np.array([v, a, d])
            
            # Cosine similarity
            dot_product = np.dot(vad_array, emotion_array)
            norm_vad = np.linalg.norm(vad_array)
            norm_emotion = np.linalg.norm(emotion_array)
            
            if norm_vad > 0 and norm_emotion > 0:
                similarity = dot_product / (norm_vad * norm_emotion)
                # Normalize to 0-1
                similarity = (similarity + 1.0) / 2.0
                
                if similarity >= threshold:
                    results.append((emotion, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
