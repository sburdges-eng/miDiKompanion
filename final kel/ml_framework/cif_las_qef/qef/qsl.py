"""
Quantum Synchronization Layer (QSL)

Uses quantum or ultra-low-latency connections for phase locking
and real-time entanglement between nodes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
import numpy as np
from datetime import datetime


@dataclass
class PhaseCoupling:
    """
    Phase Coupling State
    
    Represents the synchronization state between multiple nodes.
    """
    coherence: float = 0.0  # 0-1, higher = more synchronized
    phase_alignment: float = 0.0  # 0-1
    node_count: int = 0
    coupling_strength: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "coherence": self.coherence,
            "phase_alignment": self.phase_alignment,
            "node_count": self.node_count,
            "coupling_strength": self.coupling_strength
        }


class QuantumSynchronizationLayer:
    """
    Quantum Synchronization Layer (QSL)
    
    Handles real-time synchronization and phase coupling between nodes.
    """
    
    def __init__(self):
        """Initialize QSL."""
        self.connected_nodes: Set[str] = set()
        self.node_qas_history: Dict[str, List] = {}  # node_id -> [QAS]
        self.phase_coupling: Optional[PhaseCoupling] = None
        self.sync_threshold = 0.7  # Minimum coherence for phase coupling
        
    def synchronize(
        self,
        qas: 'QuantumAffectiveSignature',
        target_nodes: Set[str]
    ):
        """
        Synchronize QAS with target nodes.
        
        Args:
            qas: Quantum Affective Signature to synchronize
            target_nodes: Set of node IDs to synchronize with
        """
        # Store QAS for this node
        node_id = "local"  # In real implementation, would have actual node ID
        if node_id not in self.node_qas_history:
            self.node_qas_history[node_id] = []
        
        self.node_qas_history[node_id].append({
            "timestamp": datetime.now().timestamp(),
            "qas": qas
        })
        
        # Keep only recent history
        if len(self.node_qas_history[node_id]) > 100:
            self.node_qas_history[node_id].pop(0)
        
        # Update connected nodes
        self.connected_nodes.update(target_nodes)
        
        # Compute phase coupling if enough nodes
        if len(self.connected_nodes) >= 2:
            self.phase_coupling = self.compute_phase_coupling()
    
    def compute_phase_coupling(self) -> PhaseCoupling:
        """
        Compute phase coupling across all connected nodes.
        
        Returns:
            PhaseCoupling state
        """
        if len(self.node_qas_history) < 2:
            return PhaseCoupling()
        
        # Get most recent QAS from each node
        recent_qas = []
        for node_id, history in self.node_qas_history.items():
            if history:
                recent_qas.append(history[-1]["qas"])
        
        if len(recent_qas) < 2:
            return PhaseCoupling()
        
        # Compute coherence (frequency alignment)
        frequencies = [qas.frequency for qas in recent_qas]
        freq_std = np.std(frequencies)
        freq_mean = np.mean(frequencies)
        
        # Coherence: inverse of normalized standard deviation
        coherence = 1.0 / (1.0 + freq_std / (freq_mean + 1e-8))
        coherence = np.clip(coherence, 0.0, 1.0)
        
        # Phase alignment (valence/arousal alignment)
        valences = [qas.valence for qas in recent_qas]
        arousals = [qas.arousal for qas in recent_qas]
        
        # Cosine similarity of valence vectors
        valence_vec = np.array(valences)
        arousal_vec = np.array(arousals)
        
        # Normalize
        if np.linalg.norm(valence_vec) > 0:
            valence_vec = valence_vec / np.linalg.norm(valence_vec)
        if np.linalg.norm(arousal_vec) > 0:
            arousal_vec = arousal_vec / np.linalg.norm(arousal_vec)
        
        # Average alignment
        phase_alignment = (
            np.mean(np.abs(valence_vec)) +
            np.mean(np.abs(arousal_vec))
        ) / 2.0
        
        # Coupling strength
        coupling_strength = (coherence + phase_alignment) / 2.0
        
        return PhaseCoupling(
            coherence=float(coherence),
            phase_alignment=float(phase_alignment),
            node_count=len(recent_qas),
            coupling_strength=float(coupling_strength)
        )
    
    def is_phase_coupled(self) -> bool:
        """
        Check if nodes are in phase coupling state.
        
        Returns:
            True if phase coupling achieved
        """
        if not self.phase_coupling:
            return False
        
        return self.phase_coupling.coherence >= self.sync_threshold
    
    def get_status(self) -> Dict:
        """Get QSL status."""
        return {
            "connected_nodes": list(self.connected_nodes),
            "node_count": len(self.connected_nodes),
            "phase_coupled": self.is_phase_coupled(),
            "phase_coupling": self.phase_coupling.to_dict() if self.phase_coupling else None
        }
