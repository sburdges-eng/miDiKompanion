"""
Cognitive Resonance Layer (CRL)

Aligns human thought patterns with AI aesthetic logic,
creating a shared interpretive space for co-creation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
import numpy as np


@dataclass
class SharedInterpretiveSpace:
    """
    Shared Interpretive Space
    
    Represents the aligned semantic and conceptual space
    between human and AI consciousness.
    """
    shared_symbols: Dict[str, float] = field(default_factory=dict)
    semantic_alignment: float = 0.0
    conceptual_overlap: Set[str] = field(default_factory=set)
    translation_matrix: np.ndarray = field(default_factory=lambda: np.eye(4))
    
    def compute_alignment_score(self) -> float:
        """Compute overall alignment score (0-1)."""
        if not self.shared_symbols:
            return 0.0
        return float(np.mean(list(self.shared_symbols.values())))


class CognitiveResonanceLayer:
    """
    Cognitive Resonance Layer (CRL)
    
    Aligns human thought patterns with AI aesthetic logic,
    building a shared symbolic lexicon for co-creation.
    """
    
    def __init__(self):
        """Initialize CRL with default semantic mappings."""
        self.human_concepts: Dict[str, float] = {}
        self.ai_concepts: Dict[str, float] = {}
        self.shared_space: Optional[SharedInterpretiveSpace] = None
        
    def build_shared_space(
        self,
        human_data: Dict,
        ai_data: Dict
    ) -> SharedInterpretiveSpace:
        """
        Build shared interpretive space from human and AI data.
        
        Args:
            human_data: Human cognitive/linguistic data
            ai_data: AI aesthetic/conceptual data
        
        Returns:
            SharedInterpretiveSpace with aligned concepts
        """
        # Extract concepts from human data
        human_concepts = human_data.get("concepts", {})
        human_intent = human_data.get("intent", "")
        human_emotions = human_data.get("emotions", [])
        
        # Extract concepts from AI data
        ai_concepts = ai_data.get("aesthetic_concepts", {})
        ai_style = ai_data.get("style", "")
        ai_emotions = ai_data.get("emotional_vectors", [])
        
        # Find overlapping concepts
        shared_symbols = {}
        conceptual_overlap = set()
        
        for concept, value in human_concepts.items():
            if concept in ai_concepts:
                # Average alignment
                shared_symbols[concept] = (value + ai_concepts[concept]) / 2.0
                conceptual_overlap.add(concept)
        
        # Compute semantic alignment
        if human_concepts and ai_concepts:
            all_concepts = set(human_concepts.keys()) | set(ai_concepts.keys())
            overlap_ratio = len(conceptual_overlap) / len(all_concepts) if all_concepts else 0.0
        else:
            overlap_ratio = 0.0
        
        # Create translation matrix (simplified - identity for research phase)
        translation_matrix = np.eye(4)
        
        self.shared_space = SharedInterpretiveSpace(
            shared_symbols=shared_symbols,
            semantic_alignment=overlap_ratio,
            conceptual_overlap=conceptual_overlap,
            translation_matrix=translation_matrix
        )
        
        return self.shared_space
    
    def translate_human_to_ai(self, human_input: Dict) -> Dict:
        """
        Translate human cognitive input to AI aesthetic space.
        
        Args:
            human_input: Human cognitive/emotional input
        
        Returns:
            Translated AI aesthetic representation
        """
        if not self.shared_space:
            # Return default translation if no shared space built
            return {"aesthetic_vector": np.zeros(4).tolist()}
        
        # Use translation matrix to map human concepts to AI space
        human_vector = np.array([
            human_input.get("valence", 0.0),
            human_input.get("arousal", 0.0),
            human_input.get("dominance", 0.0),
            human_input.get("tension", 0.0)
        ])
        
        ai_vector = self.shared_space.translation_matrix @ human_vector
        
        return {
            "aesthetic_vector": ai_vector.tolist(),
            "translation_confidence": self.shared_space.semantic_alignment
        }
    
    def translate_ai_to_human(self, ai_output: Dict) -> Dict:
        """
        Translate AI aesthetic output to human cognitive space.
        
        Args:
            ai_output: AI aesthetic/emotional output
        
        Returns:
            Translated human cognitive representation
        """
        if not self.shared_space:
            return {"cognitive_vector": np.zeros(4).tolist()}
        
        # Inverse translation
        ai_vector = np.array(ai_output.get("aesthetic_vector", [0.0, 0.0, 0.0, 0.0]))
        translation_inv = np.linalg.pinv(self.shared_space.translation_matrix)
        human_vector = translation_inv @ ai_vector
        
        return {
            "cognitive_vector": human_vector.tolist(),
            "translation_confidence": self.shared_space.semantic_alignment
        }
    
    def get_alignment_quality(self) -> float:
        """
        Get current alignment quality score.
        
        Returns:
            Alignment score (0-1), 1.0 = perfect alignment
        """
        if not self.shared_space:
            return 0.0
        return self.shared_space.compute_alignment_score()
