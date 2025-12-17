#!/usr/bin/env python3
"""
Kelly MIDI Companion - Multi-Model Training Pipeline
=====================================================
Trains all 5 neural network models for the Kelly plugin:
1. EmotionRecognizer: Audio → Emotion (128→512→256→128→64)
2. MelodyTransformer: Emotion → MIDI (64→256→256→256→128)
3. HarmonyPredictor: Context → Chords (128→256→128→64)
4. DynamicsEngine: Context → Expression (32→128→64→16)
5. GroovePredictor: Emotion → Groove (64→128→64→32)

Total: ~1M parameters, ~4MB memory, <10ms inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

# Import real dataset loaders
import sys
from pathlib import Path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from data_loaders import (
        EmotionDataset,
        MelodyDataset,
        DynamicsDataset,
        GrooveDataset
    )
    REAL_DATASETS_AVAILABLE = True
except ImportError as e:
    REAL_DATASETS_AVAILABLE = False
    print(f"Warning: data_loaders not found ({e}). Using synthetic data.")

# Import training utilities
try:
    from training_utils import (
        EarlyStopping,
        CheckpointManager,
        MetricsTracker,
        train_epoch,
        validate_epoch,
        split_dataset
    )
    TRAINING_UTILS_AVAILABLE = True
except ImportError as e:
    TRAINING_UTILS_AVAILABLE = False
    print(f"Warning: training_utils not found ({e}). Some features will be disabled.")


# =============================================================================
# Model Definitions
# =============================================================================

class EmotionRecognizer(nn.Module):
    """Audio features → 64-dim emotion embedding (~500K params)"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.fc3 = nn.Linear(128, 64)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (batch, 128) mel features
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = x.unsqueeze(1)  # (batch, 1, 256) for LSTM
        x, _ = self.lstm(x)
        x = x.squeeze(1)  # (batch, 128)
        x = self.tanh(self.fc3(x))
        return x  # (batch, 64) emotion embedding


class MelodyTransformer(nn.Module):
    """Emotion → 128-dim MIDI note probabilities (~400K params)"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 256)
        self.lstm = nn.LSTM(256, 256, batch_first=True)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, 64) emotion embedding
        x = self.relu(self.fc1(x))
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x.squeeze(1)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x  # (batch, 128) note probabilities


class HarmonyPredictor(nn.Module):
    """Context → 64-dim chord probabilities (~100K params)"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (batch, 128) context (emotion + state)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x  # (batch, 64) chord probabilities


class DynamicsEngine(nn.Module):
    """Compact context → 16-dim expression params (~20K params)"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, 32) compact context
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x  # (batch, 16) velocity/timing/expression


class GroovePredictor(nn.Module):
    """Emotion → 32-dim groove parameters (~25K params)"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (batch, 64) emotion embedding
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x  # (batch, 32) groove parameters


# =============================================================================
# RTNeural Export
# =============================================================================

def detect_activation(model: nn.Module, layer_name: str) -> str:
    """Detect activation function from model structure."""
    for name, module in model.named_modules():
        if name == layer_name or name.endswith('.' + layer_name):
            found = False
            for next_name, next_module in model.named_modules():
                if found:
                    if isinstance(next_module, nn.Tanh):
                        return "tanh"
                    elif isinstance(next_module, nn.ReLU):
                        return "relu"
                    elif isinstance(next_module, nn.Sigmoid):
                        return "sigmoid"
                    elif isinstance(next_module, nn.Softmax):
                        return "softmax"
                    break
                if name == next_name:
                    found = True

    if 'emotion' in model.__class__.__name__.lower():
        return "tanh"
    elif 'melody' in model.__class__.__name__.lower():
        return "sigmoid"
    elif 'harmony' in model.__class__.__name__.lower():
        return "softmax"
    elif 'dynamics' in model.__class__.__name__.lower():
        return "sigmoid"
    else:
        return "tanh"


def split_lstm_weights(weight_tensor, hidden_size):
    """Split PyTorch LSTM weight tensor into RTNeural format."""
    import numpy as np
    weight_np = weight_tensor.detach().cpu().numpy()
    gate_size = hidden_size
    weights_ih = []
    for gate_idx in range(4):
        start = gate_idx * gate_size
        end = start + gate_size
        weights_ih.append(weight_np[start:end].tolist())
    return weights_ih


def export_to_rtneural(
    model: nn.Module,
    model_name: str,
    output_dir: Path
) -> Dict:
    """
    Export PyTorch model to RTNeural JSON format.
    RTNeural expects: {"layers": [...], "metadata": {...}}
    """
    model.eval()
    state_dict = model.state_dict()
    layers = []
    layer_order = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_order.append(('linear', name, module))
        elif isinstance(module, nn.LSTM):
            layer_order.append(('lstm', name, module))

    for layer_type, layer_name, module in layer_order:
        if layer_type == 'linear':
            weight_key = layer_name + '.weight'
            bias_key = layer_name + '.bias'

            if weight_key not in state_dict:
                continue

            weight = state_dict[weight_key]
            bias = state_dict.get(bias_key)
            weights = weight.detach().cpu().numpy().tolist()
            bias_list = bias.detach().cpu().numpy().tolist() if bias is not None else []
            activation = detect_activation(model, layer_name)

            is_last = layer_order.index(('linear', layer_name, module)) == len(layer_order) - 1
            if is_last:
                if 'MelodyTransformer' in model_name:
                    activation = "sigmoid"
                elif 'HarmonyPredictor' in model_name:
                    activation = "softmax"
                elif 'DynamicsEngine' in model_name:
                    activation = "sigmoid"

            layers.append({
                "type": "dense",
                "in_size": int(weight.shape[1]),
                "out_size": int(weight.shape[0]),
                "activation": activation,
                "weights": weights,
                "bias": bias_list
            })

        elif layer_type == 'lstm':
            base_name = layer_name.replace('.', '_')
            weight_ih_key = None
            weight_hh_key = None
            bias_ih_key = None
            bias_hh_key = None

            for key in state_dict.keys():
                if layer_name in key or base_name in key:
                    if 'weight_ih' in key:
                        weight_ih_key = key
                    elif 'weight_hh' in key:
                        weight_hh_key = key
                    elif 'bias_ih' in key:
                        bias_ih_key = key
                    elif 'bias_hh' in key:
                        bias_hh_key = key

            if not weight_ih_key:
                for key in state_dict.keys():
                    if 'weight_ih' in key and ('lstm' in key.lower() or layer_name in key):
                        weight_ih_key = key
                        break

            if not weight_ih_key:
                print(f"Warning: Could not find LSTM weights for {layer_name}")
                continue

            weight_ih = state_dict[weight_ih_key]
            weight_hh = state_dict.get(weight_hh_key) if weight_hh_key else None
            bias_ih = state_dict.get(bias_ih_key) if bias_ih_key else None
            bias_hh = state_dict.get(bias_hh_key) if bias_hh_key else None

            hidden_size = weight_ih.shape[0] // 4
            input_size = weight_ih.shape[1]

            weights_ih = split_lstm_weights(weight_ih, hidden_size)
            weights_hh = split_lstm_weights(weight_hh, hidden_size) if weight_hh is not None else [[0.0] * hidden_size for _ in range(4)]

            if bias_ih is not None:
                bias_ih_np = bias_ih.detach().cpu().numpy()
                bias_ih_split = [bias_ih_np[i*hidden_size:(i+1)*hidden_size].tolist() for i in range(4)]
            else:
                bias_ih_split = [[0.0] * hidden_size for _ in range(4)]

            if bias_hh is not None:
                bias_hh_np = bias_hh.detach().cpu().numpy()
                bias_hh_split = [bias_hh_np[i*hidden_size:(i+1)*hidden_size].tolist() for i in range(4)]
            else:
                bias_hh_split = [[0.0] * hidden_size for _ in range(4)]

            layers.append({
                "type": "lstm",
                "in_size": int(input_size),
                "out_size": int(hidden_size),
                "weights_ih": weights_ih,
                "weights_hh": weights_hh,
                "bias_ih": bias_ih_split,
                "bias_hh": bias_hh_split
            })

    param_count = sum(p.numel() for p in model.parameters())
    input_size = layers[0]["in_size"] if layers else 0
    output_size = layers[-1]["out_size"] if layers else 0

    rtneural_json = {
        "layers": layers,
        "metadata": {
            "model_name": model_name,
            "framework": "PyTorch",
            "export_version": "2.0",
            "parameter_count": param_count,
            "memory_bytes": param_count * 4,
            "input_size": input_size,
            "output_size": output_size
        }
    }

    output_path = output_dir / f"{model_name.lower()}.json"
    with open(output_path, 'w') as f:
        json.dump(rtneural_json, f, indent=2)

    print(f"Exported {model_name} to {output_path}")
    print(f"  Parameters: {param_count:,}")
    print(f"  Memory: {param_count * 4 / 1024:.1f} KB")
    print(f"  Layers: {len(layers)}")

    return rtneural_json


# =============================================================================
# Synthetic Dataset (Replace with real data for production)
# =============================================================================

class SyntheticEmotionDataset(Dataset):
    """Synthetic dataset for testing training pipeline."""

    def __init__(self, num_samples: int = 10000, seed: int = 42):
        np.random.seed(seed)
        self.num_samples = num_samples

        # Generate synthetic mel features
        self.mel_features = np.random.randn(
            num_samples, 128).astype(np.float32)

        # Generate synthetic emotion labels (valence-arousal space)
        # First 32 dims: valence-related, last 32 dims: arousal-related
        self.emotion_labels = np.random.randn(
            num_samples, 64).astype(np.float32)
        self.emotion_labels = np.tanh(self.emotion_labels)  # Bound to [-1, 1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'mel_features': torch.tensor(self.mel_features[idx]),
            'emotion': torch.tensor(self.emotion_labels[idx])
        }


class SyntheticMelodyDataset(Dataset):
    """Synthetic dataset for melody generation."""

    def __init__(self, num_samples: int = 10000, seed: int = 42):
        np.random.seed(seed)
        self.num_samples = num_samples

        # Emotion embeddings as input
        self.emotions = np.random.randn(num_samples, 64).astype(np.float32)
        self.emotions = np.tanh(self.emotions)

        # MIDI note probabilities as output (128 notes)
        self.note_probs = np.random.rand(num_samples, 128).astype(np.float32)
        self.note_probs = (self.note_probs /
                           self.note_probs.sum(axis=1, keepdims=True))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'emotion': torch.tensor(self.emotions[idx]),
            'notes': torch.tensor(self.note_probs[idx])
        }


# =============================================================================
# Training Functions
# =============================================================================

def train_emotion_recognizer(
    model: EmotionRecognizer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu',
    early_stop_patience: int = 10,
    checkpoint_dir: Optional[Path] = None
) -> Dict[str, List[float]]:
    """Train the emotion recognition model with validation and early stopping."""

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            mel_features = batch['mel_features'].to(device)
            emotion_target = batch['emotion'].to(device)

            optimizer.zero_grad()
            emotion_pred = model(mel_features)
            loss = criterion(emotion_pred, emotion_target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    mel_features = batch['mel_features'].to(device)
                    emotion_target = batch['emotion'].to(device)
                    emotion_pred = model(mel_features)
                    loss = criterion(emotion_pred, emotion_target)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                if checkpoint_dir:
                    torch.save(model.state_dict(), checkpoint_dir / "emotionrecognizer_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_str = f", Val Loss: {avg_val_loss:.6f}" if val_loss is not None else ""
            print(f"EmotionRecognizer Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {avg_train_loss:.6f}{val_str}")

    return {'train': train_losses, 'val': val_losses}


def train_melody_transformer(
    model: MelodyTransformer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu',
    early_stop_patience: int = 10,
    checkpoint_dir: Optional[Path] = None
) -> Dict[str, List[float]]:
    """Train the melody transformer model with validation and early stopping."""

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            emotion = batch['emotion'].to(device)
            notes_target = batch['notes'].to(device)

            optimizer.zero_grad()
            notes_pred = model(emotion)
            loss = criterion(notes_pred, notes_target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    emotion = batch['emotion'].to(device)
                    notes_target = batch['notes'].to(device)
                    notes_pred = model(emotion)
                    loss = criterion(notes_pred, notes_target)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if checkpoint_dir:
                    torch.save(model.state_dict(), checkpoint_dir / "melodytransformer_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_str = f", Val Loss: {avg_val_loss:.6f}" if val_loss is not None else ""
            print(f"MelodyTransformer Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {avg_train_loss:.6f}{val_str}")

    return {'train': train_losses, 'val': val_losses}


def train_harmony_predictor(
    model: HarmonyPredictor,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu',
    early_stop_patience: int = 10,
    checkpoint_dir: Optional[Path] = None
) -> Dict[str, List[float]]:
    """Train the harmony predictor model with validation and early stopping."""

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            context = batch['context'].to(device)
            chords_target = batch['chords'].to(device)

            optimizer.zero_grad()
            chords_pred = model(context)
            loss = criterion(torch.log(chords_pred + 1e-8), chords_target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    context = batch['context'].to(device)
                    chords_target = batch['chords'].to(device)
                    chords_pred = model(context)
                    loss = criterion(torch.log(chords_pred + 1e-8), chords_target)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if checkpoint_dir:
                    torch.save(model.state_dict(), checkpoint_dir / "harmonypredictor_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_str = f", Val Loss: {avg_val_loss:.6f}" if val_loss is not None else ""
            print(f"HarmonyPredictor Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {avg_train_loss:.6f}{val_str}")

    return {'train': train_losses, 'val': val_losses}


def train_dynamics_engine(
    model: DynamicsEngine,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu',
    early_stop_patience: int = 10,
    checkpoint_dir: Optional[Path] = None
) -> Dict[str, List[float]]:
    """Train the dynamics engine model with validation and early stopping."""

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            context = batch['context'].to(device)
            dynamics_target = batch.get('dynamics', batch.get('expression')).to(device)

            optimizer.zero_grad()
            dynamics_pred = model(context)
            loss = criterion(dynamics_pred, dynamics_target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    context = batch['context'].to(device)
                    dynamics_target = batch.get('dynamics', batch.get('expression')).to(device)
                    dynamics_pred = model(context)
                    loss = criterion(dynamics_pred, dynamics_target)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if checkpoint_dir:
                    torch.save(model.state_dict(), checkpoint_dir / "dynamicsengine_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_str = f", Val Loss: {avg_val_loss:.6f}" if val_loss is not None else ""
            print(f"DynamicsEngine Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {avg_train_loss:.6f}{val_str}")

    return {'train': train_losses, 'val': val_losses}


def train_groove_predictor(
    model: GroovePredictor,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu',
    early_stop_patience: int = 10,
    checkpoint_dir: Optional[Path] = None
) -> Dict[str, List[float]]:
    """Train the groove predictor model with validation and early stopping."""

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            emotion = batch['emotion'].to(device)
            groove_target = batch['groove'].to(device)

            optimizer.zero_grad()
            groove_pred = model(emotion)
            loss = criterion(groove_pred, groove_target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    emotion = batch['emotion'].to(device)
                    groove_target = batch['groove'].to(device)
                    groove_pred = model(emotion)
                    loss = criterion(groove_pred, groove_target)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if checkpoint_dir:
                    torch.save(model.state_dict(), checkpoint_dir / "groovepredictor_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_str = f", Val Loss: {avg_val_loss:.6f}" if val_loss is not None else ""
            print(f"GroovePredictor Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {avg_train_loss:.6f}{val_str}")

    return {'train': train_losses, 'val': val_losses}


def train_all_models(
    output_dir: Path,
    datasets_dir: Optional[Path] = None,
    epochs: int = 50,
    batch_size: int = 64,
    device: str = 'cpu',
    use_synthetic: bool = False,
    val_split: float = 0.2,
    early_stop_patience: int = 10,
    save_metrics: bool = True
):
    """Train all 5 models and export to RTNeural format."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Kelly MIDI Companion - Multi-Model Training")
    print("=" * 60)

    # Create models
    models = {
        'EmotionRecognizer': EmotionRecognizer(),
        'MelodyTransformer': MelodyTransformer(),
        'HarmonyPredictor': HarmonyPredictor(),
        'DynamicsEngine': DynamicsEngine(),
        'GroovePredictor': GroovePredictor()
    }

    # Print model stats
    print("\nModel Architecture Summary:")
    print("-" * 40)
    total_params = 0
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        total_params += params
        print(f"{name}: {params:,} params ({params * 4 / 1024:.1f} KB)")
    print("-" * 40)
    print(f"TOTAL: {total_params:,} params "
          f"({total_params * 4 / 1024:.1f} KB)")
    print()

    # Create datasets with validation splits
    if use_synthetic or not REAL_DATASETS_AVAILABLE or datasets_dir is None:
        print("Using synthetic datasets...")
        emotion_dataset = SyntheticEmotionDataset(num_samples=10000)
        melody_dataset = SyntheticMelodyDataset(num_samples=10000)

        # Split into train/val
        emotion_train_size = int((1 - val_split) * len(emotion_dataset))
        emotion_val_size = len(emotion_dataset) - emotion_train_size
        emotion_train, emotion_val = torch.utils.data.random_split(
            emotion_dataset, [emotion_train_size, emotion_val_size])

        melody_train_size = int((1 - val_split) * len(melody_dataset))
        melody_val_size = len(melody_dataset) - melody_train_size
        melody_train, melody_val = torch.utils.data.random_split(
            melody_dataset, [melody_train_size, melody_val_size])

        emotion_loader = DataLoader(
            emotion_train, batch_size=batch_size, shuffle=True)
        emotion_val_loader = DataLoader(
            emotion_val, batch_size=batch_size, shuffle=False)
        melody_loader = DataLoader(
            melody_train, batch_size=batch_size, shuffle=True)
        melody_val_loader = DataLoader(
            melody_val, batch_size=batch_size, shuffle=False)
    else:
        print(f"Loading real datasets from {datasets_dir}...")
        datasets_dir = Path(datasets_dir)

        # Emotion dataset
        audio_dir = datasets_dir / "training" / "audio"
        labels_file = audio_dir / "labels.csv"
        if audio_dir.exists():
            emotion_dataset = EmotionDataset(audio_dir, labels_file)
            emotion_train_size = int((1 - val_split) * len(emotion_dataset))
            emotion_val_size = len(emotion_dataset) - emotion_train_size
            emotion_train, emotion_val = torch.utils.data.random_split(
                emotion_dataset, [emotion_train_size, emotion_val_size])
            emotion_loader = DataLoader(
                emotion_train, batch_size=batch_size, shuffle=True, num_workers=2)
            emotion_val_loader = DataLoader(
                emotion_val, batch_size=batch_size, shuffle=False, num_workers=2)
        else:
            print(f"Warning: {audio_dir} not found, using synthetic data")
            emotion_dataset = SyntheticEmotionDataset(num_samples=10000)
            emotion_train_size = int((1 - val_split) * len(emotion_dataset))
            emotion_val_size = len(emotion_dataset) - emotion_train_size
            emotion_train, emotion_val = torch.utils.data.random_split(
                emotion_dataset, [emotion_train_size, emotion_val_size])
            emotion_loader = DataLoader(
                emotion_train, batch_size=batch_size, shuffle=True)
            emotion_val_loader = DataLoader(
                emotion_val, batch_size=batch_size, shuffle=False)

        # Melody dataset
        midi_dir = datasets_dir / "training" / "midi"
        emotion_labels = datasets_dir / "training" / "emotion_labels.json"
        if midi_dir.exists():
            emotion_labels_path = emotion_labels if emotion_labels.exists() else None
            melody_dataset = MelodyDataset(midi_dir, emotion_labels_path)
            melody_loader = DataLoader(
                melody_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        else:
            print(f"Warning: {midi_dir} not found, using synthetic data")
            melody_dataset = SyntheticMelodyDataset(num_samples=10000)
            melody_loader = DataLoader(
                melody_dataset, batch_size=batch_size, shuffle=True)

        # Harmony dataset - using synthetic for now (requires chord extraction logic)
        print(f"Note: HarmonyPredictor training uses synthetic data")
        harmony_loader = None
        harmony_val_loader = None

        # Dynamics dataset
        dynamics_midi_dir = datasets_dir / "training" / "dynamics_midi"
        if dynamics_midi_dir.exists():
            dynamics_dataset = DynamicsDataset(dynamics_midi_dir)
            dynamics_train_size = int((1 - val_split) * len(dynamics_dataset))
            dynamics_val_size = len(dynamics_dataset) - dynamics_train_size
            dynamics_train, dynamics_val = torch.utils.data.random_split(
                dynamics_dataset, [dynamics_train_size, dynamics_val_size])
            dynamics_loader = DataLoader(
                dynamics_train, batch_size=batch_size, shuffle=True, num_workers=2)
            dynamics_val_loader = DataLoader(
                dynamics_val, batch_size=batch_size, shuffle=False, num_workers=2)
        else:
            print(f"Warning: {dynamics_midi_dir} not found, skipping DynamicsEngine training")
            dynamics_loader = None
            dynamics_val_loader = None

        # Groove dataset
        drums_dir = datasets_dir / "training" / "drums"
        drum_labels = datasets_dir / "training" / "drum_labels.json"
        if drums_dir.exists():
            groove_dataset = GrooveDataset(drums_dir, drum_labels)
            groove_train_size = int((1 - val_split) * len(groove_dataset))
            groove_val_size = len(groove_dataset) - groove_train_size
            groove_train, groove_val = torch.utils.data.random_split(
                groove_dataset, [groove_train_size, groove_val_size])
            groove_loader = DataLoader(
                groove_train, batch_size=batch_size, shuffle=True, num_workers=2)
            groove_val_loader = DataLoader(
                groove_val, batch_size=batch_size, shuffle=False, num_workers=2)
        else:
            print(f"Warning: {drums_dir} not found, skipping GroovePredictor training")
            groove_loader = None
            groove_val_loader = None

    # Train EmotionRecognizer
    print("\n[1/5] Training EmotionRecognizer...")
    emotion_metrics = train_emotion_recognizer(
        models['EmotionRecognizer'], emotion_loader, emotion_val_loader,
        epochs=epochs, device=device, early_stop_patience=early_stop_patience,
        checkpoint_dir=checkpoint_dir)
    if save_metrics and TRAINING_UTILS_AVAILABLE:
        metrics = TrainingMetrics()
        metrics.train_losses = emotion_metrics['train']
        metrics.val_losses = emotion_metrics['val']
        metrics.save(metrics_dir / "emotion_metrics.json")
        metrics.plot_metrics(metrics_dir / "emotion_metrics.png")

    # Train MelodyTransformer
    print("\n[2/5] Training MelodyTransformer...")
    melody_metrics = train_melody_transformer(
        models['MelodyTransformer'], melody_loader, melody_val_loader,
        epochs=epochs, device=device, early_stop_patience=early_stop_patience,
        checkpoint_dir=checkpoint_dir)
    if save_metrics and TRAINING_UTILS_AVAILABLE:
        metrics = TrainingMetrics()
        metrics.train_losses = melody_metrics['train']
        metrics.val_losses = melody_metrics['val']
        metrics.save(metrics_dir / "melody_metrics.json")
        metrics.plot_metrics(metrics_dir / "melody_metrics.png")

    # Train remaining models
    if harmony_loader is not None:
        print("\n[3/5] Training HarmonyPredictor...")
        harmony_metrics = train_harmony_predictor(
            models['HarmonyPredictor'], harmony_loader, harmony_val_loader,
            epochs=epochs, device=device, early_stop_patience=early_stop_patience,
            checkpoint_dir=checkpoint_dir)
        if save_metrics and TRAINING_UTILS_AVAILABLE:
            metrics = TrainingMetrics()
            metrics.train_losses = harmony_metrics['train']
            metrics.val_losses = harmony_metrics['val']
            metrics.save(metrics_dir / "harmony_metrics.json")
            metrics.plot_metrics(metrics_dir / "harmony_metrics.png")
    else:
        print("\n[3/5] Skipping HarmonyPredictor (no data)")

    if dynamics_loader is not None:
        print("\n[4/5] Training DynamicsEngine...")
        dynamics_metrics = train_dynamics_engine(
            models['DynamicsEngine'], dynamics_loader, dynamics_val_loader,
            epochs=epochs, device=device, early_stop_patience=early_stop_patience,
            checkpoint_dir=checkpoint_dir)
        if save_metrics and TRAINING_UTILS_AVAILABLE:
            metrics = TrainingMetrics()
            metrics.train_losses = dynamics_metrics['train']
            metrics.val_losses = dynamics_metrics['val']
            metrics.save(metrics_dir / "dynamics_metrics.json")
            metrics.plot_metrics(metrics_dir / "dynamics_metrics.png")
    else:
        print("\n[4/5] Skipping DynamicsEngine (no data)")

    if groove_loader is not None:
        print("\n[5/5] Training GroovePredictor...")
        groove_metrics = train_groove_predictor(
            models['GroovePredictor'], groove_loader, groove_val_loader,
            epochs=epochs, device=device, early_stop_patience=early_stop_patience,
            checkpoint_dir=checkpoint_dir)
        if save_metrics and TRAINING_UTILS_AVAILABLE:
            metrics = TrainingMetrics()
            metrics.train_losses = groove_metrics['train']
            metrics.val_losses = groove_metrics['val']
            metrics.save(metrics_dir / "groove_metrics.json")
            metrics.plot_metrics(metrics_dir / "groove_metrics.png")
    else:
        print("\n[5/5] Skipping GroovePredictor (no data)")

    # Export all models
    print("\n" + "=" * 60)
    print("Exporting models to RTNeural format...")
    print("=" * 60)

    for name, model in models.items():
        export_to_rtneural(model, name, output_dir)

    # Save final PyTorch checkpoints
    for name, model in models.items():
        torch.save(model.state_dict(),
                   checkpoint_dir / f"{name.lower()}_final.pt")

    print(f"\nTraining complete! Models saved to {output_dir}")
    print(f"PyTorch checkpoints saved to {checkpoint_dir}")
    if save_metrics:
        print(f"Training metrics saved to {metrics_dir}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Kelly MIDI Companion ML models")
    parser.add_argument("--output", "-o", type=str, default="./trained_models",
                        help="Output directory for trained models")
    parser.add_argument("--datasets-dir", "-d", type=str, default=None,
                        help="Directory containing training datasets")
    parser.add_argument("--epochs", "-e", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--device", "-d", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Training device")
    parser.add_argument("--synthetic", "-s", action="store_true",
                        help="Use synthetic data instead of real datasets")

    args = parser.parse_args()

    # Auto-detect best device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = "cpu"

    train_all_models(
        output_dir=args.output,
        datasets_dir=Path(args.datasets_dir) if args.datasets_dir else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        use_synthetic=args.synthetic,
        val_split=args.val_split,
        early_stop_patience=args.early_stop_patience
    )
