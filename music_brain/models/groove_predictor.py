"""
GroovePredictor model definition.

Emotion embedding → groove parameters.
"""

import torch
import torch.nn as nn


class GroovePredictor(nn.Module):
    """Emotion → 32-dim groove parameters (~25K params)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 64) emotion embedding
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x  # (batch, 32) groove parameters
