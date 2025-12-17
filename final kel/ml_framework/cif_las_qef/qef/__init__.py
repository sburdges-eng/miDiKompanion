"""
Quantum Emotional Field (QEF)

Network-based collective emotion synchronization system with three layers:
1. Local Empathic Nodes (LENs)
2. Quantum Synchronization Layer (QSL)
3. Planetary Resonance Layer (PRL)

Status: Research Phase (Requires Network Infrastructure)
"""

from .core import QEF, QuantumAffectiveSignature
from .len import LocalEmpathicNode, NodeState
from .qsl import QuantumSynchronizationLayer, PhaseCoupling
from .prl import PlanetaryResonanceLayer, ResonanceMemory

__all__ = [
    "QEF",
    "QuantumAffectiveSignature",
    "LocalEmpathicNode",
    "NodeState",
    "QuantumSynchronizationLayer",
    "PhaseCoupling",
    "PlanetaryResonanceLayer",
    "ResonanceMemory",
]
