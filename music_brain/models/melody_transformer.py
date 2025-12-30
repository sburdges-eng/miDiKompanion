"""
MelodyTransformer model definition.

Emotion embedding → MIDI note probability distribution.
"""

import torch
import torch.nn as nn


class MelodyTransformer(nn.Module):
    """Emotion → 128-dim MIDI note probabilities (~400K params)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 256)
        self.lstm = nn.LSTM(256, 256, batch_first=True)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 64) emotion embedding
        x = self.relu(self.fc1(x))
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x.squeeze(1)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x  # (batch, 128) note probabilities
