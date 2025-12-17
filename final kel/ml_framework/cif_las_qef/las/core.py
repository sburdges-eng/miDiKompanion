"""
LAS Core: Living Art System

Main orchestrator for the multi-agent creative ecosystem.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
import json


@dataclass
class AestheticDNA:
    """
    Aesthetic DNA (aDNA)
    
    Digital genome describing the creative identity of a Living Art System.
    """
    style_genes: Dict[str, float] = field(default_factory=dict)
    emotion_genes: Dict[str, float] = field(default_factory=dict)
    structure_genes: Dict[str, float] = field(default_factory=dict)
    evolution_generation: int = 0
    
    def mutate(self, mutation_rate: float = 0.1) -> 'AestheticDNA':
        """
        Create mutated version of aDNA.
        
        Args:
            mutation_rate: Probability of mutation per gene
        
        Returns:
            New AestheticDNA with mutations
        """
        new_dna = AestheticDNA(
            style_genes=self.style_genes.copy(),
            emotion_genes=self.emotion_genes.copy(),
            structure_genes=self.structure_genes.copy(),
            evolution_generation=self.evolution_generation + 1
        )
        
        # Mutate style genes
        for key in new_dna.style_genes:
            if np.random.random() < mutation_rate:
                new_dna.style_genes[key] += np.random.normal(0, 0.1)
                new_dna.style_genes[key] = np.clip(new_dna.style_genes[key], -1.0, 1.0)
        
        # Mutate emotion genes
        for key in new_dna.emotion_genes:
            if np.random.random() < mutation_rate:
                new_dna.emotion_genes[key] += np.random.normal(0, 0.1)
                new_dna.emotion_genes[key] = np.clip(new_dna.emotion_genes[key], 0.0, 1.0)
        
        return new_dna
    
    def to_dict(self) -> Dict:
        """Serialize aDNA to dictionary."""
        return {
            "style_genes": self.style_genes,
            "emotion_genes": self.emotion_genes,
            "structure_genes": self.structure_genes,
            "evolution_generation": self.evolution_generation
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AestheticDNA':
        """Deserialize aDNA from dictionary."""
        return cls(
            style_genes=data.get("style_genes", {}),
            emotion_genes=data.get("emotion_genes", {}),
            structure_genes=data.get("structure_genes", {}),
            evolution_generation=data.get("evolution_generation", 0)
        )


class LAS:
    """
    Living Art System
    
    Multi-agent ecosystem for autonomous creative generation and evolution.
    """
    
    def __init__(
        self,
        emotion_interface: Optional['EmotionInterface'] = None,
        aesthetic_brain: Optional['AestheticBrainCore'] = None,
        generative_body: Optional['GenerativeBody'] = None,
        recursive_memory: Optional['RecursiveMemory'] = None,
        reflex_layer: Optional['ReflexLayer'] = None
    ):
        """
        Initialize LAS with core components.
        
        Args:
            emotion_interface: Emotion Interface (default: creates new)
            aesthetic_brain: Aesthetic Brain Core (default: creates new)
            generative_body: Generative Body (default: creates new)
            recursive_memory: Recursive Memory (default: creates new)
            reflex_layer: Reflex Layer (default: creates new)
        """
        from .emotion_interface import EmotionInterface
        from .aesthetic_brain import AestheticBrainCore
        from .generative_body import GenerativeBody
        from .recursive_memory import RecursiveMemory
        from .reflex_layer import ReflexLayer
        
        self.ei = emotion_interface or EmotionInterface()
        self.abc = aesthetic_brain or AestheticBrainCore()
        self.gb = generative_body or GenerativeBody()
        self.rm = recursive_memory or RecursiveMemory()
        self.rl = reflex_layer or ReflexLayer()
        
        # Aesthetic DNA
        self.aesthetic_dna = AestheticDNA()
        
        # System state
        self.creative_history: List[Dict] = []
        self.feedback_history: List[Dict] = []
        self.evolution_count = 0
        
    def generate(
        self,
        emotional_input: Dict,
        creative_goal: Optional[Dict] = None
    ) -> Dict:
        """
        Generate creative output from emotional input.
        
        Args:
            emotional_input: Emotional data (biofeedback, text, etc.)
            creative_goal: Optional creative goal/intent
        
        Returns:
            Generated creative output
        """
        # 1. Emotion Interface: Convert input to ESV
        esv = self.ei.process_emotional_input(emotional_input)
        
        # 2. Aesthetic Brain: Form creative intent
        intent = self.abc.form_creative_intent(esv, creative_goal)
        
        # 3. Generative Body: Produce output
        output = self.gb.generate(intent, self.aesthetic_dna)
        
        # 4. Store in memory
        memory = self.rm.store_creation(output, esv, intent)
        
        # 5. Reflex Layer: Adjust for homeostasis
        balance = self.rl.adjust_for_balance(output, memory)
        
        # Store in history
        self.creative_history.append({
            "timestamp": datetime.now().isoformat(),
            "esv": esv.to_dict() if hasattr(esv, 'to_dict') else esv,
            "intent": intent.to_dict() if hasattr(intent, 'to_dict') else intent,
            "output": output.to_dict() if hasattr(output, 'to_dict') else output,
            "balance": balance.to_dict() if hasattr(balance, 'to_dict') else balance
        })
        
        return {
            "output": output.to_dict() if hasattr(output, 'to_dict') else output,
            "esv": esv.to_dict() if hasattr(esv, 'to_dict') else esv,
            "intent": intent.to_dict() if hasattr(intent, 'to_dict') else intent,
            "memory_id": memory.id if hasattr(memory, 'id') else None,
            "homeostatic_balance": balance.to_dict() if hasattr(balance, 'to_dict') else balance
        }
    
    def evolve(self, feedback: Dict) -> Dict:
        """
        Evolve system based on feedback.
        
        Args:
            feedback: Feedback data (emotional response, aesthetic rating, etc.)
        
        Returns:
            Evolution result
        """
        # Store feedback
        self.feedback_history.append({
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback
        })
        
        # Update memory with feedback
        self.rm.integrate_feedback(feedback)
        
        # Compute reward for evolution
        reward = self._compute_aesthetic_reward(feedback)
        
        # Mutate aesthetic DNA based on reward
        if reward > 0.6:  # High reward - preserve and slightly mutate
            self.aesthetic_dna = self.aesthetic_dna.mutate(mutation_rate=0.05)
        elif reward < 0.4:  # Low reward - more aggressive mutation
            self.aesthetic_dna = self.aesthetic_dna.mutate(mutation_rate=0.2)
        else:  # Medium reward - moderate mutation
            self.aesthetic_dna = self.aesthetic_dna.mutate(mutation_rate=0.1)
        
        # Update components based on feedback
        self.abc.update_from_feedback(feedback)
        self.gb.update_from_feedback(feedback)
        
        self.evolution_count += 1
        
        return {
            "evolution_count": self.evolution_count,
            "reward": reward,
            "new_dna_generation": self.aesthetic_dna.evolution_generation,
            "mutated": True
        }
    
    def _compute_aesthetic_reward(self, feedback: Dict) -> float:
        """
        Compute aesthetic reward from feedback.
        
        Args:
            feedback: Feedback data
        
        Returns:
            Reward score (0-1)
        """
        # Extract reward signals (simplified for research phase)
        emotional_resonance = feedback.get("emotional_resonance", 0.5)
        aesthetic_rating = feedback.get("aesthetic_rating", 0.5)
        engagement_level = feedback.get("engagement", 0.5)
        
        # Weighted average
        reward = (
            emotional_resonance * 0.4 +
            aesthetic_rating * 0.4 +
            engagement_level * 0.2
        )
        
        return float(np.clip(reward, 0.0, 1.0))
    
    def get_status(self) -> Dict:
        """Get current LAS status."""
        return {
            "evolution_count": self.evolution_count,
            "dna_generation": self.aesthetic_dna.evolution_generation,
            "creative_history_count": len(self.creative_history),
            "feedback_history_count": len(self.feedback_history),
            "homeostatic_balance": self.rl.get_current_balance().to_dict() if hasattr(self.rl, 'get_current_balance') else {}
        }
    
    def save_state(self, filepath: str):
        """Save LAS state to file."""
        state = {
            "aesthetic_dna": self.aesthetic_dna.to_dict(),
            "evolution_count": self.evolution_count,
            "creative_history": self.creative_history[-100:],  # Last 100
            "feedback_history": self.feedback_history[-100:]   # Last 100
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load LAS state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.aesthetic_dna = AestheticDNA.from_dict(state["aesthetic_dna"])
        self.evolution_count = state.get("evolution_count", 0)
        self.creative_history = state.get("creative_history", [])
        self.feedback_history = state.get("feedback_history", [])
