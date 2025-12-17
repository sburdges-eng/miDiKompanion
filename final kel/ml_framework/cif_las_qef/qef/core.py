"""
QEF Core: Quantum Emotional Field

Main orchestrator for the three-layer planetary empathy grid.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import numpy as np
from datetime import datetime
import asyncio
import json


@dataclass
class QuantumAffectiveSignature:
    """
    Quantum Affective Signature (QAS)
    
    Encodes emotional state as waveform cluster:
    QAS = [Valence Vector] + [Arousal Vector] + [Resonant Frequency]
    """
    valence: float = 0.0
    arousal: float = 0.0
    frequency: float = 0.0  # Resonant frequency in Hz
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.valence, self.arousal, self.frequency])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "frequency": self.frequency
        }
    
    @classmethod
    def from_emotion(cls, emotion: str) -> 'QuantumAffectiveSignature':
        """
        Create QAS from emotion name (simplified mapping).
        
        Args:
            emotion: Emotion name (e.g., "serenity", "longing", "wonder", "grief")
        
        Returns:
            QuantumAffectiveSignature
        """
        # Emotion to QAS mappings (from documentation)
        mappings = {
            "serenity": cls(valence=0.7, arousal=-0.3, frequency=512.0),
            "longing": cls(valence=-0.2, arousal=0.4, frequency=288.0),
            "wonder": cls(valence=0.9, arousal=0.6, frequency=432.0),
            "grief": cls(valence=-0.8, arousal=0.2, frequency=204.0),
        }
        
        return mappings.get(emotion.lower(), cls())


class QEF:
    """
    Quantum Emotional Field
    
    Distributed consciousness mesh for planetary empathy grid.
    """
    
    def __init__(
        self,
        len_layer: Optional['LocalEmpathicNode'] = None,
        qsl_layer: Optional['QuantumSynchronizationLayer'] = None,
        prl_layer: Optional['PlanetaryResonanceLayer'] = None,
        node_id: Optional[str] = None
    ):
        """
        Initialize QEF with three layers.
        
        Args:
            len_layer: Local Empathic Node layer
            qsl_layer: Quantum Synchronization Layer
            prl_layer: Planetary Resonance Layer
            node_id: Unique identifier for this node
        """
        from .len import LocalEmpathicNode
        from .qsl import QuantumSynchronizationLayer
        from .prl import PlanetaryResonanceLayer
        
        self.node_id = node_id or f"qef_node_{datetime.now().timestamp()}"
        self.len = len_layer or LocalEmpathicNode(node_id=self.node_id)
        self.qsl = qsl_layer or QuantumSynchronizationLayer()
        self.prl = prl_layer or PlanetaryResonanceLayer()
        
        # Network state
        self.connected_nodes: Set[str] = set()
        self.resonance_history: List[Dict] = []
        self.is_active = False
        
    def connect_node(self, node_id: str, node_address: Optional[str] = None) -> bool:
        """
        Connect to another QEF node.
        
        Args:
            node_id: ID of node to connect to
            node_address: Optional network address
        
        Returns:
            True if connection successful
        """
        # In research phase: simulate connection
        self.connected_nodes.add(node_id)
        return True
    
    def disconnect_node(self, node_id: str):
        """Disconnect from a node."""
        self.connected_nodes.discard(node_id)
    
    def emit_emotional_state(
        self,
        esv: Dict,
        source: str = "local"
    ) -> QuantumAffectiveSignature:
        """
        Emit emotional state to the QEF.
        
        Args:
            esv: Emotional State Vector
            source: Source identifier
        
        Returns:
            QuantumAffectiveSignature
        """
        # Convert ESV to QAS
        qas = QuantumAffectiveSignature(
            valence=esv.get("valence", 0.0),
            arousal=esv.get("arousal", 0.5),
            frequency=self._compute_resonant_frequency(esv)
        )
        
        # Emit through LEN
        self.len.emit_qas(qas, source)
        
        # Synchronize through QSL
        if self.connected_nodes:
            self.qsl.synchronize(qas, self.connected_nodes)
        
        # Store in PRL
        self.prl.store_resonance(qas, source, self.node_id)
        
        return qas
    
    def receive_collective_resonance(self) -> Dict:
        """
        Receive collective resonance from the field.
        
        Returns:
            Collective resonance data
        """
        # Get from PRL
        collective = self.prl.get_collective_state()
        
        # Compute phase coupling if multiple nodes
        if len(self.connected_nodes) > 0:
            phase_coupling = self.qsl.compute_phase_coupling()
        else:
            phase_coupling = None
        
        return {
            "collective_esv": collective.get("average_esv", {}),
            "resonance_level": collective.get("resonance_level", 0.0),
            "active_nodes": len(self.connected_nodes) + 1,  # +1 for self
            "phase_coupling": phase_coupling.to_dict() if phase_coupling else None
        }
    
    def _compute_resonant_frequency(self, esv: Dict) -> float:
        """
        Compute resonant frequency from ESV.
        
        Args:
            esv: Emotional State Vector
        
        Returns:
            Frequency in Hz
        """
        # Simplified mapping: valence and arousal determine frequency
        valence = esv.get("valence", 0.0)
        arousal = esv.get("arousal", 0.5)
        
        # Base frequency around 400 Hz
        base_freq = 400.0
        
        # Adjust by valence (positive = higher)
        freq = base_freq + valence * 100.0
        
        # Adjust by arousal (higher arousal = higher freq)
        freq += arousal * 50.0
        
        # Clip to reasonable range
        freq = np.clip(freq, 200.0, 600.0)
        
        return float(freq)
    
    def activate(self):
        """Activate QEF node."""
        self.is_active = True
        self.len.activate()
    
    def deactivate(self):
        """Deactivate QEF node."""
        self.is_active = False
        self.len.deactivate()
    
    def get_status(self) -> Dict:
        """Get QEF node status."""
        return {
            "node_id": self.node_id,
            "is_active": self.is_active,
            "connected_nodes": list(self.connected_nodes),
            "connection_count": len(self.connected_nodes),
            "len_status": self.len.get_status(),
            "qsl_status": self.qsl.get_status(),
            "prl_status": self.prl.get_status()
        }
