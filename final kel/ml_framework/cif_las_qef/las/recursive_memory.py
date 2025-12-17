"""
Recursive Memory (RM)

Stores feedback, context, and self-reflections for
continuous learning and evolution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
import uuid


@dataclass
class AestheticMemory:
    """
    Aesthetic Memory
    
    Stores a single creative output with its context and feedback.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    esv: Dict = field(default_factory=dict)
    intent: Dict = field(default_factory=dict)
    output: Dict = field(default_factory=dict)
    feedback: List[Dict] = field(default_factory=list)
    aesthetic_score: float = 0.0
    emotional_resonance: float = 0.0
    
    def add_feedback(self, feedback_data: Dict):
        """Add feedback to this memory."""
        self.feedback.append({
            "timestamp": datetime.now().timestamp(),
            "data": feedback_data
        })
        
        # Update scores if provided
        if "aesthetic_rating" in feedback_data:
            self.aesthetic_score = feedback_data["aesthetic_rating"]
        if "emotional_resonance" in feedback_data:
            self.emotional_resonance = feedback_data["emotional_resonance"]


class RecursiveMemory:
    """
    Recursive Memory (RM)
    
    Stores and retrieves aesthetic memories for learning and evolution.
    """
    
    def __init__(self, max_memories: int = 1000):
        """
        Initialize Recursive Memory.
        
        Args:
            max_memories: Maximum number of memories to store
        """
        self.memories: List[AestheticMemory] = []
        self.max_memories = max_memories
        self.memory_index: Dict[str, AestheticMemory] = {}
        
    def store_creation(
        self,
        output: 'CreativeOutput',
        esv: 'EmotionalStateVector',
        intent: 'CreativeIntent'
    ) -> AestheticMemory:
        """
        Store a creation in memory.
        
        Args:
            output: Creative Output
            esv: Emotional State Vector
            intent: Creative Intent
        
        Returns:
            Stored AestheticMemory
        """
        memory = AestheticMemory(
            esv=esv.to_dict() if hasattr(esv, 'to_dict') else esv,
            intent=intent.to_dict() if hasattr(intent, 'to_dict') else intent,
            output=output.to_dict() if hasattr(output, 'to_dict') else output,
            aesthetic_score=output.aesthetic_score if hasattr(output, 'aesthetic_score') else 0.0
        )
        
        self.memories.append(memory)
        self.memory_index[memory.id] = memory
        
        # Trim if over limit
        if len(self.memories) > self.max_memories:
            removed = self.memories.pop(0)
            del self.memory_index[removed.id]
        
        return memory
    
    def integrate_feedback(self, feedback: Dict):
        """
        Integrate feedback into recent memories.
        
        Args:
            feedback: Feedback data with optional memory_id
        """
        memory_id = feedback.get("memory_id")
        
        if memory_id and memory_id in self.memory_index:
            # Add to specific memory
            self.memory_index[memory_id].add_feedback(feedback)
        else:
            # Add to most recent memory
            if self.memories:
                self.memories[-1].add_feedback(feedback)
    
    def retrieve_similar(
        self,
        esv: Dict,
        intent: Dict,
        top_k: int = 5
    ) -> List[AestheticMemory]:
        """
        Retrieve similar memories based on ESV and intent.
        
        Args:
            esv: Emotional State Vector to match
            intent: Creative Intent to match
            top_k: Number of similar memories to return
        
        Returns:
            List of similar AestheticMemory objects
        """
        if not self.memories:
            return []
        
        # Compute similarity scores
        similarities = []
        for memory in self.memories:
            # ESV similarity
            esv_sim = self._compute_esv_similarity(esv, memory.esv)
            
            # Intent similarity (simplified)
            intent_sim = self._compute_intent_similarity(intent, memory.intent)
            
            # Combined similarity
            combined = (esv_sim * 0.6 + intent_sim * 0.4)
            similarities.append((combined, memory))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k
        return [mem for _, mem in similarities[:top_k]]
    
    def _compute_esv_similarity(self, esv1: Dict, esv2: Dict) -> float:
        """Compute similarity between two ESVs."""
        # Extract vectors
        vec1 = np.array([
            esv1.get("valence", 0.0),
            esv1.get("arousal", 0.0),
            esv1.get("dominance", 0.0),
            esv1.get("tension", 0.0)
        ])
        vec2 = np.array([
            esv2.get("valence", 0.0),
            esv2.get("arousal", 0.0),
            esv2.get("dominance", 0.0),
            esv2.get("tension", 0.0)
        ])
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(np.clip(similarity, 0.0, 1.0))
    
    def _compute_intent_similarity(self, intent1: Dict, intent2: Dict) -> float:
        """Compute similarity between two intents (simplified)."""
        # Compare style preferences
        style1 = intent1.get("style_preferences", {})
        style2 = intent2.get("style_preferences", {})
        
        if not style1 or not style2:
            return 0.5
        
        # Average similarity of common keys
        common_keys = set(style1.keys()) & set(style2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            diff = abs(style1[key] - style2[key])
            similarities.append(1.0 - diff)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def get_high_score_memories(self, threshold: float = 0.7, top_k: int = 10) -> List[AestheticMemory]:
        """
        Get memories with high aesthetic scores.
        
        Args:
            threshold: Minimum aesthetic score
            top_k: Maximum number to return
        
        Returns:
            List of high-scoring memories
        """
        high_score = [m for m in self.memories if m.aesthetic_score >= threshold]
        high_score.sort(key=lambda m: m.aesthetic_score, reverse=True)
        return high_score[:top_k]
