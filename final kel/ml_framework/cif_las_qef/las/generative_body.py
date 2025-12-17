"""
Generative Body (GB)

Produces audio/visual/textual output based on creative intent
and aesthetic DNA.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime


@dataclass
class CreativeOutput:
    """
    Creative Output
    
    Represents generated creative content.
    """
    content_type: str  # "audio", "visual", "text", "hybrid"
    content_data: Dict
    parameters: Dict
    aesthetic_score: float = 0.0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "content_type": self.content_type,
            "content_data": self.content_data,
            "parameters": self.parameters,
            "aesthetic_score": self.aesthetic_score,
            "timestamp": self.timestamp
        }


class GenerativeBody:
    """
    Generative Body (GB)
    
    Produces creative output based on intent and aesthetic DNA.
    """
    
    def __init__(self):
        """Initialize Generative Body."""
        self.generation_modes = ["audio", "visual", "text", "hybrid"]
        self.current_mode = "audio"
        
    def generate(
        self,
        intent: 'CreativeIntent',
        aesthetic_dna: 'AestheticDNA'
    ) -> CreativeOutput:
        """
        Generate creative output from intent and aesthetic DNA.
        
        Args:
            intent: Creative Intent
            aesthetic_dna: Aesthetic DNA
        
        Returns:
            CreativeOutput
        """
        # Apply aesthetic DNA to style preferences
        style = self._apply_dna_to_style(intent.style_preferences, aesthetic_dna)
        
        # Generate based on content type
        if self.current_mode == "audio":
            content_data = self._generate_audio(intent, style)
        elif self.current_mode == "visual":
            content_data = self._generate_visual(intent, style)
        elif self.current_mode == "text":
            content_data = self._generate_text(intent, style)
        else:  # hybrid
            content_data = self._generate_hybrid(intent, style)
        
        # Compute aesthetic score
        aesthetic_score = self._compute_aesthetic_score(intent, content_data)
        
        # Parameters used
        parameters = {
            "style": style,
            "emotional_target": intent.emotional_target,
            "novelty_weight": intent.novelty_weight,
            "coherence_weight": intent.coherence_weight
        }
        
        return CreativeOutput(
            content_type=self.current_mode,
            content_data=content_data,
            parameters=parameters,
            aesthetic_score=aesthetic_score
        )
    
    def _apply_dna_to_style(
        self,
        style_prefs: Dict,
        aesthetic_dna: 'AestheticDNA'
    ) -> Dict:
        """
        Apply aesthetic DNA mutations to style preferences.
        
        Args:
            style_prefs: Style preferences from intent
            aesthetic_dna: Aesthetic DNA
        
        Returns:
            Modified style dictionary
        """
        style = style_prefs.copy()
        
        # Apply style genes
        for key, value in aesthetic_dna.style_genes.items():
            if key in style:
                style[key] = style[key] * 0.7 + value * 0.3
            else:
                style[key] = value
        
        return style
    
    def _generate_audio(
        self,
        intent: 'CreativeIntent',
        style: Dict
    ) -> Dict:
        """
        Generate audio content (simplified for research phase).
        
        Args:
            intent: Creative Intent
            style: Style parameters
        
        Returns:
            Audio content data
        """
        # Simplified audio generation (research phase)
        return {
            "tempo": int(60 + style.get("tempo_preference", 0.5) * 120),
            "key": "C",
            "mode": style.get("mode_preference", "major"),
            "time_signature": "4/4",
            "harmonic_progression": self._generate_progression(intent),
            "rhythmic_pattern": self._generate_rhythm(style),
            "timbre": {
                "brightness": style.get("timbre_brightness", 0.5),
                "warmth": 1.0 - style.get("timbre_brightness", 0.5)
            },
            "dynamics": {
                "range": style.get("dynamic_range", 0.5),
                "curve": "exponential"
            }
        }
    
    def _generate_visual(
        self,
        intent: 'CreativeIntent',
        style: Dict
    ) -> Dict:
        """
        Generate visual content (simplified for research phase).
        
        Args:
            intent: Creative Intent
            style: Style parameters
        
        Returns:
            Visual content data
        """
        # Map emotional target to visual parameters
        valence = intent.emotional_target.get("valence", 0.0)
        arousal = intent.emotional_target.get("arousal", 0.5)
        
        return {
            "color_palette": {
                "hue": (valence + 1.0) / 2.0,  # 0-1
                "saturation": arousal,
                "brightness": (valence + 1.0) / 2.0
            },
            "composition": {
                "balance": intent.emotional_target.get("dominance", 0.5),
                "complexity": intent.emotional_target.get("tension", 0.5)
            },
            "texture": {
                "smoothness": 1.0 - intent.emotional_target.get("tension", 0.5)
            }
        }
    
    def _generate_text(
        self,
        intent: 'CreativeIntent',
        style: Dict
    ) -> Dict:
        """
        Generate text content (simplified for research phase).
        
        Args:
            intent: Creative Intent
            style: Style parameters
        
        Returns:
            Text content data
        """
        # Map to text parameters
        return {
            "sentiment": intent.emotional_target.get("valence", 0.0),
            "energy": intent.emotional_target.get("arousal", 0.5),
            "complexity": intent.emotional_target.get("tension", 0.5),
            "length": intent.structural_constraints.get("min_length", 8),
            "style": "poetic" if intent.emotional_target.get("creativity", 0.5) > 0.7 else "prose"
        }
    
    def _generate_hybrid(
        self,
        intent: 'CreativeIntent',
        style: Dict
    ) -> Dict:
        """
        Generate hybrid (multi-modal) content.
        
        Args:
            intent: Creative Intent
            style: Style parameters
        
        Returns:
            Hybrid content data
        """
        return {
            "audio": self._generate_audio(intent, style),
            "visual": self._generate_visual(intent, style),
            "text": self._generate_text(intent, style),
            "synchronization": {
                "temporal_alignment": 1.0,
                "emotional_coherence": 0.8
            }
        }
    
    def _generate_progression(self, intent: 'CreativeIntent') -> List[str]:
        """Generate harmonic progression (simplified)."""
        mode = intent.style_preferences.get("mode_preference", "major")
        complexity = intent.emotional_target.get("tension", 0.5)
        
        if mode == "major":
            if complexity < 0.3:
                return ["I", "V", "vi", "IV"]
            elif complexity < 0.7:
                return ["I", "vi", "IV", "V"]
            else:
                return ["I", "iii", "vi", "ii", "V"]
        else:  # minor
            if complexity < 0.3:
                return ["i", "v", "VI", "iv"]
            else:
                return ["i", "VI", "iv", "V"]
    
    def _generate_rhythm(self, style: Dict) -> Dict:
        """Generate rhythmic pattern (simplified)."""
        density = style.get("rhythmic_density", 0.5)
        
        return {
            "pattern": "standard" if density < 0.5 else "complex",
            "subdivision": "16th" if density > 0.7 else "8th",
            "syncopation": density * 0.5
        }
    
    def _compute_aesthetic_score(
        self,
        intent: 'CreativeIntent',
        content: Dict
    ) -> float:
        """
        Compute aesthetic score for generated content.
        
        Args:
            intent: Creative Intent
            content: Generated content
        
        Returns:
            Aesthetic score (0-1)
        """
        # Simplified scoring (research phase)
        # Higher intentionality = higher base score
        base_score = intent.intentionality_score
        
        # Novelty/coherence balance contributes
        balance_score = 1.0 - abs(intent.novelty_weight - intent.coherence_weight)
        
        # Final score
        score = base_score * 0.6 + balance_score * 0.4
        
        return float(np.clip(score, 0.0, 1.0))
    
    def update_from_feedback(self, feedback: Dict):
        """
        Update generation parameters from feedback.
        
        Args:
            feedback: Feedback data
        """
        # Adjust generation mode if feedback suggests
        if feedback.get("preferred_mode"):
            self.current_mode = feedback["preferred_mode"]
