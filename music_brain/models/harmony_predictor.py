"""
HarmonyPredictor model definition.

Context → chord probability distribution.
"""

import torch
import torch.nn as nn


class HarmonyPredictor(nn.Module):
    """Context → 64-dim chord probabilities (~100K params)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 128) context (emotion + state)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x  # (batch, 64) chord probabilities
