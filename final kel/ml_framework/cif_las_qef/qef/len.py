"""
Local Empathic Node (LEN)

Captures emotional data via sensory and affective channels
and emits Quantum Affective Signatures (QAS).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime


@dataclass
class NodeState:
    """State of a Local Empathic Node."""
    node_id: str
    is_active: bool = False
    last_qas: Optional['QuantumAffectiveSignature'] = None
    emission_count: int = 0
    last_emission_time: float = 0.0


class LocalEmpathicNode:
    """
    Local Empathic Node (LEN)
    
    Captures emotional data and converts to QAS for emission.
    """
    
    def __init__(self, node_id: str):
        """
        Initialize LEN.
        
        Args:
            node_id: Unique identifier for this node
        """
        self.node_id = node_id
        self.state = NodeState(node_id=node_id)
        self.emission_history: List[Dict] = []
        
        # Input channels
        self.biofeedback_enabled = True
        self.voice_enabled = True
        self.text_enabled = True
        self.facial_enabled = True
        
    def capture_emotional_data(
        self,
        biofeedback: Optional[Dict] = None,
        voice: Optional[Dict] = None,
        text: Optional[Dict] = None,
        facial: Optional[Dict] = None
    ) -> Dict:
        """
        Capture emotional data from various channels.
        
        Args:
            biofeedback: Biometric data (HR, EEG, GSR, etc.)
            voice: Voice tone analysis
            text: Text sentiment
            facial: Facial expression data
        
        Returns:
            Aggregated emotional data
        """
        # Aggregate from available channels
        esv = {
            "valence": 0.0,
            "arousal": 0.5,
            "dominance": 0.5,
            "tension": 0.5
        }
        
        weights = []
        
        # Biofeedback
        if biofeedback and self.biofeedback_enabled:
            hr = biofeedback.get("heart_rate", 70.0)
            eeg_alpha = biofeedback.get("eeg_alpha", 0.5)
            
            esv["valence"] += (eeg_alpha - 0.5) * 2.0
            esv["arousal"] += (hr - 60) / 100.0
            weights.append(0.4)
        
        # Voice
        if voice and self.voice_enabled:
            tone = voice.get("tone", 0.0)
            intensity = voice.get("intensity", 0.5)
            
            esv["valence"] += tone
            esv["arousal"] += intensity
            weights.append(0.3)
        
        # Text
        if text and self.text_enabled:
            sentiment = text.get("sentiment", 0.0)
            energy = text.get("energy", 0.5)
            
            esv["valence"] += sentiment
            esv["arousal"] += energy
            weights.append(0.2)
        
        # Facial
        if facial and self.facial_enabled:
            facial_valence = facial.get("valence", 0.0)
            facial_tension = facial.get("tension", 0.5)
            
            esv["valence"] += facial_valence
            esv["tension"] += facial_tension
            weights.append(0.1)
        
        # Normalize by weights
        total_weight = sum(weights) if weights else 1.0
        for key in esv:
            esv[key] = esv[key] / total_weight if total_weight > 0 else esv[key]
            esv[key] = np.clip(esv[key], -1.0 if key == "valence" else 0.0, 1.0)
        
        return esv
    
    def emit_qas(
        self,
        qas: 'QuantumAffectiveSignature',
        source: str = "local"
    ):
        """
        Emit Quantum Affective Signature.
        
        Args:
            qas: Quantum Affective Signature
            source: Source identifier
        """
        if not self.state.is_active:
            return
        
        self.state.last_qas = qas
        self.state.emission_count += 1
        self.state.last_emission_time = datetime.now().timestamp()
        
        # Store in history
        self.emission_history.append({
            "timestamp": self.state.last_emission_time,
            "qas": qas.to_dict(),
            "source": source,
            "node_id": self.node_id
        })
        
        # Keep only last 1000 emissions
        if len(self.emission_history) > 1000:
            self.emission_history.pop(0)
    
    def activate(self):
        """Activate LEN."""
        self.state.is_active = True
    
    def deactivate(self):
        """Deactivate LEN."""
        self.state.is_active = False
    
    def get_status(self) -> Dict:
        """Get LEN status."""
        return {
            "node_id": self.node_id,
            "is_active": self.state.is_active,
            "emission_count": self.state.emission_count,
            "last_emission_time": self.state.last_emission_time,
            "has_last_qas": self.state.last_qas is not None
        }
