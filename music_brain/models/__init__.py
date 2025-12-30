"""
Neural network model definitions for music_brain.

Provides lightweight PyTorch modules used by Tier 1 inference and Tier 2 LoRA
fine-tuning workflows.
"""

from .melody_transformer import MelodyTransformer
from .harmony_predictor import HarmonyPredictor
from .groove_predictor import GroovePredictor

__all__ = [
    "MelodyTransformer",
    "HarmonyPredictor",
    "GroovePredictor",
]
