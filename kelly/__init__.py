"""
KELLY - Emotion Understanding and Mapping System

KELLY is the emotion intelligence engine for miDiKompanion.
It provides comprehensive emotion understanding, mapping, and translation
to musical parameters.

Core Components:
- Thesaurus: 216-node emotion mapping system
- Emotional Mapping: Emotion → musical parameter translation
- Emotion Sampler: Real-time emotion processing

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "miDiKompanion Development Team"

# Import submodules
from kelly import thesaurus
from kelly import emotional_mapping

__all__ = [
    "thesaurus",
    "emotional_mapping",
]
