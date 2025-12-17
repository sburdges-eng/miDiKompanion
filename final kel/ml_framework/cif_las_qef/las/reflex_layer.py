"""
Reflex Layer (RL)

Adjusts models dynamically for creative balance and homeostasis.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class HomeostaticBalance:
    """
    Homeostatic Balance
    
    Represents the balance between novelty and coherence,
    structure and chaos, etc.
    """
    novelty_coherence: float = 0.5  # 0.0 = pure coherence, 1.0 = pure novelty
    structure_chaos: float = 0.5     # 0.0 = pure structure, 1.0 = pure chaos
    stability: float = 0.5           # Overall system stability
    entropy: float = 0.5             # System entropy level
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "novelty_coherence": self.novelty_coherence,
            "structure_chaos": self.structure_chaos,
            "stability": self.stability,
            "entropy": self.entropy
        }


class ReflexLayer:
    """
    Reflex Layer (RL)
    
    Adjusts system parameters dynamically to maintain creative homeostasis.
    """
    
    def __init__(self):
        """Initialize Reflex Layer."""
        self.balance = HomeostaticBalance()
        self.target_balance = HomeostaticBalance(
            novelty_coherence=0.5,
            structure_chaos=0.5,
            stability=0.7,
            entropy=0.4
        )
        self.adjustment_history: list = []
        
    def adjust_for_balance(
        self,
        output: 'CreativeOutput',
        memory: Optional['AestheticMemory'] = None
    ) -> HomeostaticBalance:
        """
        Adjust system for creative balance.
        
        Args:
            output: Generated creative output
            memory: Optional memory context
        
        Returns:
            Updated HomeostaticBalance
        """
        # Measure current state
        novelty = output.parameters.get("novelty_weight", 0.5)
        coherence = output.parameters.get("coherence_weight", 0.5)
        
        # Update novelty/coherence balance
        self.balance.novelty_coherence = novelty
        
        # Compute stability from memory if available
        if memory and memory.feedback:
            # Stability based on feedback consistency
            feedback_scores = [f["data"].get("aesthetic_rating", 0.5) for f in memory.feedback]
            if feedback_scores:
                stability = 1.0 - np.std(feedback_scores)
                self.balance.stability = float(np.clip(stability, 0.0, 1.0))
        
        # Compute entropy (diversity of outputs)
        # Simplified: based on novelty weight
        self.balance.entropy = novelty
        
        # Adjust toward target if too far
        self._adjust_toward_target()
        
        return self.balance
    
    def _adjust_toward_target(self):
        """Gradually adjust balance toward target."""
        # Novelty/coherence
        diff = self.target_balance.novelty_coherence - self.balance.novelty_coherence
        self.balance.novelty_coherence += diff * 0.1  # Slow adjustment
        
        # Stability
        diff = self.target_balance.stability - self.balance.stability
        self.balance.stability += diff * 0.1
        
        # Entropy
        diff = self.target_balance.entropy - self.balance.entropy
        self.balance.entropy += diff * 0.1
        
        # Clip values
        self.balance.novelty_coherence = np.clip(self.balance.novelty_coherence, 0.0, 1.0)
        self.balance.stability = np.clip(self.balance.stability, 0.0, 1.0)
        self.balance.entropy = np.clip(self.balance.entropy, 0.0, 1.0)
    
    def get_current_balance(self) -> HomeostaticBalance:
        """Get current homeostatic balance."""
        return self.balance
    
    def set_target_balance(self, target: HomeostaticBalance):
        """Set target balance for homeostasis."""
        self.target_balance = target
