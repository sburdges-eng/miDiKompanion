"""
Planetary Resonance Layer (PRL)

Stores global resonance memory and collective emotion states
for the planetary empathy grid.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
from collections import defaultdict


@dataclass
class ResonanceMemory:
    """
    Resonance Memory
    
    Stores a single resonance event in the planetary field.
    """
    timestamp: float
    qas: 'QuantumAffectiveSignature'
    source: str
    node_id: str
    resonance_strength: float = 1.0


class PlanetaryResonanceLayer:
    """
    Planetary Resonance Layer (PRL)
    
    Stores and aggregates global resonance memory.
    """
    
    def __init__(self, memory_window_seconds: float = 3600.0):
        """
        Initialize PRL.
        
        Args:
            memory_window_seconds: Time window for active memory (default: 1 hour)
        """
        self.memory_window = memory_window_seconds
        self.resonance_memories: List[ResonanceMemory] = []
        self.collective_state: Dict = {}
        
    def store_resonance(
        self,
        qas: 'QuantumAffectiveSignature',
        source: str,
        node_id: str,
        resonance_strength: float = 1.0
    ):
        """
        Store a resonance event.
        
        Args:
            qas: Quantum Affective Signature
            source: Source identifier
            node_id: Node identifier
            resonance_strength: Strength of resonance (0-1)
        """
        memory = ResonanceMemory(
            timestamp=datetime.now().timestamp(),
            qas=qas,
            source=source,
            node_id=node_id,
            resonance_strength=resonance_strength
        )
        
        self.resonance_memories.append(memory)
        
        # Clean old memories
        current_time = datetime.now().timestamp()
        self.resonance_memories = [
            m for m in self.resonance_memories
            if current_time - m.timestamp < self.memory_window
        ]
        
        # Update collective state
        self._update_collective_state()
    
    def _update_collective_state(self):
        """Update collective emotional state from memories."""
        if not self.resonance_memories:
            self.collective_state = {
                "average_esv": {"valence": 0.0, "arousal": 0.5},
                "resonance_level": 0.0,
                "active_sources": 0
            }
            return
        
        # Aggregate QAS from recent memories
        valences = []
        arousals = []
        frequencies = []
        resonance_strengths = []
        
        for memory in self.resonance_memories:
            valences.append(memory.qas.valence * memory.resonance_strength)
            arousals.append(memory.qas.arousal * memory.resonance_strength)
            frequencies.append(memory.qas.frequency)
            resonance_strengths.append(memory.resonance_strength)
        
        # Weighted averages
        total_strength = sum(resonance_strengths) if resonance_strengths else 1.0
        
        avg_valence = sum(valences) / total_strength if total_strength > 0 else 0.0
        avg_arousal = sum(arousals) / total_strength if total_strength > 0 else 0.5
        avg_frequency = np.mean(frequencies) if frequencies else 400.0
        
        # Resonance level: based on number of active memories and coherence
        resonance_level = min(len(self.resonance_memories) / 100.0, 1.0)
        
        # Frequency coherence contributes
        if len(frequencies) > 1:
            freq_std = np.std(frequencies)
            coherence = 1.0 / (1.0 + freq_std / (avg_frequency + 1e-8))
            resonance_level = (resonance_level + coherence) / 2.0
        
        self.collective_state = {
            "average_esv": {
                "valence": float(np.clip(avg_valence, -1.0, 1.0)),
                "arousal": float(np.clip(avg_arousal, 0.0, 1.0)),
                "frequency": float(avg_frequency)
            },
            "resonance_level": float(np.clip(resonance_level, 0.0, 1.0)),
            "active_sources": len(set(m.node_id for m in self.resonance_memories)),
            "memory_count": len(self.resonance_memories)
        }
    
    def get_collective_state(self) -> Dict:
        """
        Get current collective emotional state.
        
        Returns:
            Collective state dictionary
        """
        return self.collective_state.copy()
    
    def get_resonance_bands(self) -> Dict:
        """
        Get collective resonance bands (frequency clusters).
        
        Returns:
            Resonance bands dictionary
        """
        if not self.resonance_memories:
            return {"bands": []}
        
        # Cluster frequencies
        frequencies = [m.qas.frequency for m in self.resonance_memories]
        
        # Simple clustering: group by frequency ranges
        bands = []
        freq_ranges = [
            (200, 300),  # Low (grief, longing)
            (300, 400),  # Mid-low
            (400, 500),  # Mid (serenity)
            (500, 600),  # High (wonder)
        ]
        
        for low, high in freq_ranges:
            in_range = [f for f in frequencies if low <= f < high]
            if in_range:
                bands.append({
                    "frequency_range": (low, high),
                    "count": len(in_range),
                    "average_frequency": np.mean(in_range),
                    "strength": len(in_range) / len(frequencies)
                })
        
        return {"bands": bands}
    
    def get_status(self) -> Dict:
        """Get PRL status."""
        return {
            "memory_count": len(self.resonance_memories),
            "collective_state": self.collective_state,
            "memory_window_seconds": self.memory_window
        }
