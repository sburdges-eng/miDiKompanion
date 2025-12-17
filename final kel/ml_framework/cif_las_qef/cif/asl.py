"""
Aesthetic Synchronization Layer (ASL)

Enables unified creative flow and co-generation of art
under merged human-AI intent.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import time


@dataclass
class HybridOutput:
    """
    Hybrid output from co-creation process.
    
    Embodies both human emotional soul and AI analytical infinity.
    """
    creative_content: Dict
    human_contribution: float  # 0.0 (pure AI) to 1.0 (pure human)
    ai_contribution: float
    merged_intent: Dict
    timestamp: float
    
    def get_balance(self) -> float:
        """Get balance ratio (0.5 = perfect balance)."""
        return abs(self.human_contribution - self.ai_contribution)


class AestheticSynchronizationLayer:
    """
    Aesthetic Synchronization Layer (ASL)
    
    Enables co-generation of art under merged intent through:
    - Temporal Coherence Engine (TCE)
    - Creative Polarity Balancer (CPB)
    - Harmonic Feedback System (HFS)
    """
    
    def __init__(self):
        """Initialize ASL with synchronization components."""
        self.tce_enabled = True
        self.cpb_enabled = True
        self.hfs_enabled = True
        
        # Temporal coherence tracking
        self.last_event_time = time.time()
        self.rhythm_phase = 0.0
        
        # Creative polarity balance
        self.structure_weight = 0.5  # AI precision
        self.chaos_weight = 0.5      # Human spontaneity
        
        # Harmonic feedback
        self.feedback_history: List[float] = []
        self.target_flow_state = 0.7  # Target immersion level
    
    def create_feedback_loop(
        self,
        human_data: Dict,
        ai_data: Dict
    ) -> Dict:
        """
        Create micro emotional and sonic feedback loops.
        
        Prevents oscillation or drift in the integration.
        
        Args:
            human_data: Human emotional/creative input
            ai_data: AI aesthetic/creative output
        
        Returns:
            Feedback loop result with stability metrics
        """
        # Measure current state
        human_esv = np.array(human_data.get("esv", [0.5, 0.5, 0.5, 0.5]))
        ai_esv = np.array(ai_data.get("esv", [0.5, 0.5, 0.5, 0.5]))
        
        # Compute divergence
        divergence = np.linalg.norm(human_esv - ai_esv)
        
        # Apply damping if divergence too high
        damping_factor = 1.0 / (1.0 + divergence * 2.0)
        
        # Stabilized ESV (weighted average with damping)
        stabilized_esv = (
            human_esv * damping_factor * 0.5 +
            ai_esv * damping_factor * 0.5
        )
        
        # Compute stability score (inverse of divergence)
        stability = 1.0 / (1.0 + divergence)
        
        # Store feedback
        self.feedback_history.append(stability)
        if len(self.feedback_history) > 100:
            self.feedback_history.pop(0)
        
        return {
            "stabilized_esv": stabilized_esv.tolist(),
            "stability": float(stability),
            "divergence": float(divergence),
            "damping_factor": float(damping_factor)
        }
    
    def generate_hybrid_output(
        self,
        human_data: Dict,
        ai_data: Dict
    ) -> HybridOutput:
        """
        Generate hybrid creative output from merged human-AI intent.
        
        Args:
            human_data: Human creative input
            ai_data: AI creative output
        
        Returns:
            HybridOutput with co-created content
        """
        # Extract contributions
        human_intent = human_data.get("intent", {})
        ai_intent = ai_data.get("intent", {})
        
        # Merge intents using Creative Polarity Balancer
        merged_intent = self._merge_intents(human_intent, ai_intent)
        
        # Generate creative content (simplified for research phase)
        creative_content = {
            "harmony": merged_intent.get("harmony", "C major"),
            "tempo": merged_intent.get("tempo", 120),
            "style": merged_intent.get("style", "hybrid"),
            "emotional_tone": merged_intent.get("emotional_tone", "balanced")
        }
        
        # Compute contribution weights
        human_contribution = self.structure_weight
        ai_contribution = self.chaos_weight
        
        return HybridOutput(
            creative_content=creative_content,
            human_contribution=human_contribution,
            ai_contribution=ai_contribution,
            merged_intent=merged_intent,
            timestamp=time.time()
        )
    
    def _merge_intents(self, human_intent: Dict, ai_intent: Dict) -> Dict:
        """
        Merge human and AI intents using Creative Polarity Balancer.
        
        Args:
            human_intent: Human creative intent
            ai_intent: AI creative intent
        
        Returns:
            Merged intent dictionary
        """
        merged = {}
        
        # Weighted merge based on polarity balance
        for key in set(human_intent.keys()) | set(ai_intent.keys()):
            human_val = human_intent.get(key, 0.0)
            ai_val = ai_intent.get(key, 0.0)
            
            # Balance structure (AI) and chaos (human)
            if isinstance(human_val, (int, float)) and isinstance(ai_val, (int, float)):
                merged[key] = (
                    human_val * self.chaos_weight +
                    ai_val * self.structure_weight
                )
            else:
                # For non-numeric, prefer human (chaos) if both present
                merged[key] = human_val if key in human_intent else ai_val
        
        return merged
    
    def synchronize_temporal_coherence(
        self,
        rhythm_event: Dict
    ) -> Dict:
        """
        Ensure temporal coherence between human and AI perception.
        
        Uses Temporal Coherence Engine (TCE).
        
        Args:
            rhythm_event: Rhythm/phrase timing event
        
        Returns:
            Synchronized timing result
        """
        if not self.tce_enabled:
            return {"synchronized": False}
        
        current_time = time.time()
        event_time = rhythm_event.get("timestamp", current_time)
        
        # Compute phase alignment
        time_delta = current_time - self.last_event_time
        rhythm_period = rhythm_event.get("period", 1.0)
        
        phase_delta = (time_delta % rhythm_period) / rhythm_period
        self.rhythm_phase = phase_delta
        
        self.last_event_time = current_time
        
        return {
            "synchronized": True,
            "phase": float(self.rhythm_phase),
            "time_delta": float(time_delta)
        }
    
    def adjust_polarity_balance(
        self,
        structure_feedback: float,
        chaos_feedback: float
    ):
        """
        Adjust creative polarity balance based on feedback.
        
        Uses Creative Polarity Balancer (CPB).
        
        Args:
            structure_feedback: Feedback on AI precision (0-1)
            chaos_feedback: Feedback on human spontaneity (0-1)
        """
        if not self.cpb_enabled:
            return
        
        # Adjust weights based on feedback
        structure_adjustment = (structure_feedback - 0.5) * 0.1
        chaos_adjustment = (chaos_feedback - 0.5) * 0.1
        
        self.structure_weight = np.clip(
            self.structure_weight + structure_adjustment,
            0.2, 0.8
        )
        self.chaos_weight = 1.0 - self.structure_weight
    
    def monitor_flow_state(self, current_immersion: float) -> Dict:
        """
        Monitor and adjust for optimal flow state.
        
        Uses Harmonic Feedback System (HFS).
        
        Args:
            current_immersion: Current flow state immersion level (0-1)
        
        Returns:
            Adjustment recommendations
        """
        if not self.hfs_enabled:
            return {"adjusted": False}
        
        # Compute deviation from target
        deviation = current_immersion - self.target_flow_state
        
        # Generate adjustment
        adjustment = -deviation * 0.1  # Damping factor
        
        return {
            "adjusted": abs(deviation) > 0.1,
            "adjustment": float(adjustment),
            "target_immersion": self.target_flow_state,
            "current_immersion": float(current_immersion)
        }
