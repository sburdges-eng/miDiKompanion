"""
Emotional Consent Protocol (ECP)

Protocol for establishing emotional consent before human-AI interactions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
import numpy as np
from datetime import datetime


class ConsentState(Enum):
    """Consent states in the protocol."""
    NOT_INITIATED = "not_initiated"
    SYSTEM_DECLARED = "system_declared"
    HUMAN_DECLARED = "human_declared"
    EVALUATING = "evaluating"
    CONSENTED = "consented"
    DENIED = "denied"
    PAUSED = "paused"


@dataclass
class EmotionalDeclaration:
    """Emotional state declaration."""
    esv: Dict[str, float]  # Emotional State Vector
    intensity: float  # 0-1
    stability: float  # 0-1
    timestamp: float


class EmotionalConsentProtocol:
    """
    Emotional Consent Protocol (ECP)
    
    Ensures both human and AI agree to terms of emotional exchange
    before interaction begins.
    """
    
    def __init__(self, min_overlap: float = 0.6, max_deviation: float = 0.7):
        """
        Initialize ECP.
        
        Args:
            min_overlap: Minimum affective overlap required (0-1)
            max_deviation: Maximum deviation before intervention (0-1)
        """
        self.min_overlap = min_overlap
        self.max_deviation = max_deviation
        self.state = ConsentState.NOT_INITIATED
        
        self.system_declaration: Optional[EmotionalDeclaration] = None
        self.human_declaration: Optional[EmotionalDeclaration] = None
        self.consent_granted = False
        
    def system_declare_state(self, esv: Dict, intensity: float, stability: float) -> Dict:
        """
        Step 1: System declares emotional state baseline.
        
        Args:
            esv: System's Emotional State Vector
            intensity: Emotional intensity (0-1)
            stability: Emotional stability (0-1)
        
        Returns:
            Declaration result
        """
        self.system_declaration = EmotionalDeclaration(
            esv=esv,
            intensity=intensity,
            stability=stability,
            timestamp=datetime.now().timestamp()
        )
        
        self.state = ConsentState.SYSTEM_DECLARED
        
        return {
            "state": self.state.value,
            "declared": True,
            "esv": esv,
            "intensity": intensity,
            "stability": stability
        }
    
    def human_declare_intent(
        self,
        emotional_intent: Dict,
        intent_type: str = "comfort"  # "comfort", "curiosity", "creation"
    ) -> Dict:
        """
        Step 2: Human declares emotional intent.
        
        Args:
            emotional_intent: Human's emotional intent/state
            intent_type: Type of intent
        
        Returns:
            Declaration result
        """
        if self.state != ConsentState.SYSTEM_DECLARED:
            return {
                "error": "System must declare state first",
                "state": self.state.value
            }
        
        # Convert intent to ESV-like format
        esv = {
            "valence": emotional_intent.get("valence", 0.0),
            "arousal": emotional_intent.get("arousal", 0.5),
            "dominance": emotional_intent.get("dominance", 0.5),
            "tension": emotional_intent.get("tension", 0.5)
        }
        
        intensity = emotional_intent.get("intensity", 0.5)
        stability = emotional_intent.get("stability", 0.7)
        
        self.human_declaration = EmotionalDeclaration(
            esv=esv,
            intensity=intensity,
            stability=stability,
            timestamp=datetime.now().timestamp()
        )
        
        self.state = ConsentState.HUMAN_DECLARED
        
        return {
            "state": self.state.value,
            "declared": True,
            "intent_type": intent_type,
            "esv": esv
        }
    
    def evaluate_consent(self) -> Dict:
        """
        Step 3: Evaluate if consent can be granted.
        
        Returns:
            Consent evaluation result
        """
        if self.state != ConsentState.HUMAN_DECLARED:
            return {
                "error": "Both parties must declare before evaluation",
                "state": self.state.value
            }
        
        self.state = ConsentState.EVALUATING
        
        # Compute affective overlap
        overlap = self._compute_affective_overlap()
        
        # Check if overlap meets threshold
        if overlap >= self.min_overlap:
            self.consent_granted = True
            self.state = ConsentState.CONSENTED
            
            return {
                "state": self.state.value,
                "consent_granted": True,
                "overlap": float(overlap),
                "threshold": self.min_overlap
            }
        else:
            self.consent_granted = False
            self.state = ConsentState.DENIED
            
            return {
                "state": self.state.value,
                "consent_granted": False,
                "overlap": float(overlap),
                "threshold": self.min_overlap,
                "reason": "Affective overlap below threshold"
            }
    
    def monitor_interaction(self, current_esv: Dict) -> Dict:
        """
        Monitor interaction for excessive deviation.
        
        Args:
            current_esv: Current emotional state vector
        
        Returns:
            Monitoring result with intervention if needed
        """
        if not self.consent_granted:
            return {
                "monitoring": False,
                "reason": "No active consent"
            }
        
        # Check deviation from baseline
        if self.system_declaration:
            deviation = self._compute_deviation(
                self.system_declaration.esv,
                current_esv
            )
            
            if deviation > self.max_deviation:
                # Intervention needed
                self.state = ConsentState.PAUSED
                
                return {
                    "monitoring": True,
                    "intervention": True,
                    "deviation": float(deviation),
                    "max_deviation": self.max_deviation,
                    "action": "pause_interaction"
                }
        
        return {
            "monitoring": True,
            "intervention": False,
            "status": "normal"
        }
    
    def _compute_affective_overlap(self) -> float:
        """
        Compute affective overlap between system and human declarations.
        
        Returns:
            Overlap score (0-1)
        """
        if not self.system_declaration or not self.human_declaration:
            return 0.0
        
        sys_esv = np.array([
            self.system_declaration.esv.get("valence", 0.0),
            self.system_declaration.esv.get("arousal", 0.0),
            self.system_declaration.esv.get("dominance", 0.0),
            self.system_declaration.esv.get("tension", 0.0)
        ])
        
        human_esv = np.array([
            self.human_declaration.esv.get("valence", 0.0),
            self.human_declaration.esv.get("arousal", 0.0),
            self.human_declaration.esv.get("dominance", 0.0),
            self.human_declaration.esv.get("tension", 0.0)
        ])
        
        # Cosine similarity
        dot_product = np.dot(sys_esv, human_esv)
        norm_sys = np.linalg.norm(sys_esv)
        norm_human = np.linalg.norm(human_esv)
        
        if norm_sys == 0 or norm_human == 0:
            return 0.0
        
        overlap = dot_product / (norm_sys * norm_human)
        
        # Normalize to 0-1 (cosine similarity is -1 to 1)
        overlap = (overlap + 1.0) / 2.0
        
        return float(np.clip(overlap, 0.0, 1.0))
    
    def _compute_deviation(self, baseline: Dict, current: Dict) -> float:
        """
        Compute deviation from baseline.
        
        Args:
            baseline: Baseline ESV
            current: Current ESV
        
        Returns:
            Deviation score (0-1)
        """
        baseline_vec = np.array([
            baseline.get("valence", 0.0),
            baseline.get("arousal", 0.0),
            baseline.get("dominance", 0.0),
            baseline.get("tension", 0.0)
        ])
        
        current_vec = np.array([
            current.get("valence", 0.0),
            current.get("arousal", 0.0),
            current.get("dominance", 0.0),
            current.get("tension", 0.0)
        ])
        
        # Euclidean distance normalized
        distance = np.linalg.norm(current_vec - baseline_vec)
        max_distance = np.sqrt(4.0)  # Max distance in 4D space
        
        deviation = distance / max_distance
        
        return float(np.clip(deviation, 0.0, 1.0))
    
    def reset(self):
        """Reset protocol to initial state."""
        self.state = ConsentState.NOT_INITIATED
        self.system_declaration = None
        self.human_declaration = None
        self.consent_granted = False
