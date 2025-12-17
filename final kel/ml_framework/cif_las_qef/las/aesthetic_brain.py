"""
Aesthetic Brain Core (ABC)

Interprets goals, forms creative intent, and directs
the generative process toward aesthetic objectives.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .emotion_interface import EmotionalStateVector


@dataclass
class CreativeIntent:
    """
    Creative Intent

    Represents the aesthetic goal and direction for generation.
    """
    emotional_target: Dict[str, float] = field(default_factory=dict)
    style_preferences: Dict[str, float] = field(default_factory=dict)
    structural_constraints: Dict[str, any] = field(default_factory=dict)
    novelty_weight: float = 0.5
    coherence_weight: float = 0.5
    intentionality_score: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "emotional_target": self.emotional_target,
            "style_preferences": self.style_preferences,
            "structural_constraints": self.structural_constraints,
            "novelty_weight": self.novelty_weight,
            "coherence_weight": self.coherence_weight,
            "intentionality_score": self.intentionality_score
        }


class AestheticBrainCore:
    """
    Aesthetic Brain Core (ABC)

    Interprets emotional input and forms creative intent for generation.
    """

    def __init__(self):
        """Initialize Aesthetic Brain Core."""
        self.style_library: Dict[str, Dict] = {}
        self.emotional_profiles: Dict[str, Dict] = {}
        self.learned_preferences: Dict[str, float] = {}

    def form_creative_intent(
        self,
        esv: 'EmotionalStateVector',
        creative_goal: Optional[Dict] = None
    ) -> CreativeIntent:
        """
        Form creative intent from emotional state and optional goal.

        Args:
            esv: Emotional State Vector
            creative_goal: Optional creative goal/intent

        Returns:
            CreativeIntent
        """
        # Map ESV to emotional target
        emotional_target = {
            "valence": esv.valence,
            "arousal": esv.arousal,
            "dominance": esv.dominance,
            "tension": esv.tension,
            "creativity": esv.creativity
        }

        # Determine style preferences from ESV
        style_preferences = self._infer_style_from_esv(esv)

        # Apply creative goal if provided
        if creative_goal:
            style_value = creative_goal.get("style", {})
            if isinstance(style_value, dict):
                style_preferences.update(style_value)
            elif isinstance(style_value, str):
                # If style is a string, use it as the style name
                style_preferences["style_name"] = style_value

            emotion_value = creative_goal.get("emotion", {})
            if isinstance(emotion_value, dict):
                emotional_target.update(emotion_value)
            elif isinstance(emotion_value, str):
                # If emotion is a string, use it as emotion name
                emotional_target["emotion_name"] = emotion_value

        # Compute novelty/coherence balance
        novelty_weight, coherence_weight = self._compute_balance(esv)

        # Compute intentionality score
        intentionality = self._compute_intentionality(esv, creative_goal)

        # Structural constraints (simplified)
        structural_constraints = {
            "min_length": creative_goal.get("min_length", 8) if creative_goal else 8,
            "max_length": creative_goal.get("max_length", 64) if creative_goal else 64,
            "complexity": esv.creativity
        }

        return CreativeIntent(
            emotional_target=emotional_target,
            style_preferences=style_preferences,
            structural_constraints=structural_constraints,
            novelty_weight=novelty_weight,
            coherence_weight=coherence_weight,
            intentionality_score=intentionality
        )

    def _infer_style_from_esv(self, esv: 'EmotionalStateVector') -> Dict[str, float]:
        """
        Infer style preferences from Emotional State Vector.

        Args:
            esv: Emotional State Vector

        Returns:
            Style preferences dictionary
        """
        # Map ESV dimensions to musical/artistic style parameters
        style = {}

        # Tempo based on arousal
        style["tempo_preference"] = float(esv.arousal)

        # Mode based on valence
        style["mode_preference"] = "major" if esv.valence > 0 else "minor"

        # Harmonic complexity based on tension
        style["harmonic_complexity"] = float(esv.tension)

        # Rhythmic density based on arousal
        style["rhythmic_density"] = float(esv.arousal)

        # Timbre brightness based on valence
        style["timbre_brightness"] = float((esv.valence + 1.0) / 2.0)

        # Dynamic range based on dominance
        style["dynamic_range"] = float(esv.dominance)

        return style

    def _compute_balance(self, esv: 'EmotionalStateVector') -> tuple:
        """
        Compute novelty/coherence balance.

        Args:
            esv: Emotional State Vector

        Returns:
            (novelty_weight, coherence_weight) tuple
        """
        # Higher creativity = more novelty
        novelty_weight = esv.creativity * 0.7 + 0.3
        coherence_weight = 1.0 - novelty_weight

        return (float(novelty_weight), float(coherence_weight))

    def _compute_intentionality(
        self,
        esv: 'EmotionalStateVector',
        creative_goal: Optional[Dict]
    ) -> float:
        """
        Compute intentionality score (how goal-directed the creation is).

        Args:
            esv: Emotional State Vector
            creative_goal: Optional creative goal

        Returns:
            Intentionality score (0-1)
        """
        if not creative_goal:
            # Lower intentionality without explicit goal
            return 0.3

        # Higher intentionality with clear goal
        goal_clarity = creative_goal.get("clarity", 0.5)
        goal_specificity = len(creative_goal.get("constraints", {}))

        intentionality = (
            goal_clarity * 0.6 +
            min(goal_specificity / 5.0, 1.0) * 0.4
        )

        return float(np.clip(intentionality, 0.0, 1.0))

    def update_from_feedback(self, feedback: Dict):
        """
        Update learned preferences from feedback.

        Args:
            feedback: Feedback data
        """
        # Extract style preferences from positive feedback
        if feedback.get("aesthetic_rating", 0.0) > 0.7:
            style_feedback = feedback.get("style_preferences", {})
            for key, value in style_feedback.items():
                if key in self.learned_preferences:
                    # Update with exponential moving average
                    self.learned_preferences[key] = (
                        self.learned_preferences[key] * 0.7 + value * 0.3
                    )
                else:
                    self.learned_preferences[key] = value
