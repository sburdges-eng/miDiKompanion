#!/usr/bin/env python3
"""
Real Dataset Loaders for Kelly MIDI Companion ML Training
==========================================================
Loaders for actual datasets (DEAM, Lakh MIDI, MAESTRO, Groove MIDI, etc.)
"""

import json
import csv
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa

warnings.filterwarnings('ignore', category=UserWarning)


# =============================================================================
# Emotion Recognition Dataset (DEAM, PMEmo, etc.)
# =============================================================================

class EmotionDataset(Dataset):
    """
    Dataset loader for emotion recognition from audio.

    Supports multiple formats:
    1. DEAM format: audio files + CSV with valence/arousal labels
    2. PMEmo format: audio files + JSON with emotion labels
    3. Custom format: audio files + labels.csv with filename,valence,arousal
    """

    def __init__(
        self,
        audio_dir: Path,
        labels_file: Optional[Path] = None,
        n_mels: int = 128,
        duration: float = 2.0,
        sr: int = 22050,
        cache_features: bool = False
    ):
        """
        Initialize emotion dataset.

        Args:
            audio_dir: Directory containing audio files (.wav, .mp3, etc.)
            labels_file: Path to labels file (CSV or JSON)
            n_mels: Number of mel bands for spectrogram
            duration: Duration of audio to analyze (seconds)
            sr: Target sample rate
            cache_features: If True, cache extracted features in memory
        """
        self.audio_dir = Path(audio_dir)
        self.n_mels = n_mels
        self.duration = duration
        self.sr = sr
        self.cache_features = cache_features

        # Find all audio files
        self.audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']:
            self.audio_files.extend(list(self.audio_dir.glob(ext)))
            self.audio_files.extend(list(self.audio_dir.glob(f'**/{ext}')))

        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")

        # Load labels
        self.labels = self._load_labels(labels_file)

        # Filter to only files with labels (if labels provided)
        if self.labels:
            self.audio_files = [
                f for f in self.audio_files
                if f.name in self.labels or f.stem in self.labels
            ]

        if len(self.audio_files) == 0:
            raise ValueError("No audio files with labels found")

        # Cache for features (if enabled)
        self.feature_cache: Optional[Dict[Path, np.ndarray]] = {} if cache_features else None

        print(f"Loaded {len(self.audio_files)} audio files with emotion labels")

    def _load_labels(self, labels_file: Optional[Path]) -> Dict[str, np.ndarray]:
        """
        Load emotion labels from file.

        Returns dict mapping filename -> emotion vector (64-dim or 2-dim [valence, arousal])
        """
        if labels_file is None:
            # Try to find labels file automatically
            labels_file = self.audio_dir / "labels.csv"
            if not labels_file.exists():
                labels_file = self.audio_dir / "emotion_labels.json"
            if not labels_file.exists():
                labels_file = self.audio_dir.parent / "labels.csv"

        if labels_file is None or not labels_file.exists():
            # Generate placeholder labels (for testing)
            print(f"Warning: No labels file found at {labels_file}")
            print("  Using placeholder labels (random values)")
            return {}

        labels = {}

        if labels_file.suffix == '.csv':
            # CSV format: filename,valence,arousal
            with open(labels_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get('filename', row.get('file', ''))
                    if not filename:
                        continue

                    try:
                        valence = float(row.get('valence', row.get('val', 0.0)))
                        arousal = float(row.get('arousal', row.get('ar', 0.0)))

                        # Convert 2D valence/arousal to 64-dim embedding
                        # First 32 dims: valence-related, last 32 dims: arousal-related
                        emotion_vec = np.zeros(64, dtype=np.float32)
                        emotion_vec[:32] = valence * np.ones(32) * 0.5  # Valence
                        emotion_vec[32:] = arousal * np.ones(32) * 0.5  # Arousal

                        labels[filename] = emotion_vec
                        # Also store by stem (without extension)
                        labels[Path(filename).stem] = emotion_vec
                    except (ValueError, KeyError):
                        continue

        elif labels_file.suffix == '.json':
            # JSON format: {"filename": {"valence": 0.5, "arousal": 0.6}, ...}
            with open(labels_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for filename, label_data in data.items():
                    if isinstance(label_data, dict):
                        valence = float(label_data.get('valence', 0.0))
                        arousal = float(label_data.get('arousal', 0.0))
                    elif isinstance(label_data, (list, tuple)) and len(label_data) >= 2:
                        valence = float(label_data[0])
                        arousal = float(label_data[1])
                    else:
                        continue

                    # Convert to 64-dim embedding
                    emotion_vec = np.zeros(64, dtype=np.float32)
                    emotion_vec[:32] = valence * np.ones(32) * 0.5
                    emotion_vec[32:] = arousal * np.ones(32) * 0.5

                    labels[filename] = emotion_vec
                    labels[Path(filename).stem] = emotion_vec

        print(f"Loaded {len(labels)} emotion labels from {labels_file}")
        return labels

    def _extract_mel_features(self, audio_path: Path) -> np.ndarray:
        """
        Extract mel-spectrogram features from audio file.

        Returns 128-dim feature vector (average over time).
        """
        # Check cache first
        if self.feature_cache is not None and audio_path in self.feature_cache:
            return self.feature_cache[audio_path]

        try:
            # Load audio
            y, sr = librosa.load(
                str(audio_path),
                sr=self.sr,
                duration=self.duration,
                mono=True
            )

            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=self.n_mels,
                hop_length=512,
                n_fft=2048
            )

            # Convert to log scale (dB)
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)

            # Average over time to get single feature vector
            features = np.mean(log_mel, axis=1)

            # Normalize to [-1, 1] range
            # Typical dB range is -80 to 0, so divide by 80
            features = np.clip(features / 80.0, -1.0, 1.0)

            # Cache if enabled
            if self.feature_cache is not None:
                self.feature_cache[audio_path] = features

            return features.astype(np.float32)

        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            # Return zeros on error
            return np.zeros(self.n_mels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            (features, labels) where:
            - features: (128,) mel-spectrogram features
            - labels: (64,) emotion embedding
        """
        audio_path = self.audio_files[idx]

        # Extract features
        features = self._extract_mel_features(audio_path)

        # Get label (or generate placeholder)
        filename = audio_path.name
        label_key = filename if filename in self.labels else audio_path.stem
        label = self.labels.get(label_key)

        if label is None:
            # Generate placeholder label
            label = np.random.randn(64).astype(np.float32)
            label = np.tanh(label * 0.5)  # Bound to [-1, 1]

        return torch.tensor(features), torch.tensor(label)


# =============================================================================
# Melody Dataset (Lakh MIDI, etc.) - Placeholder for now
# =============================================================================

class MelodyDataset(Dataset):
    """Dataset loader for melody generation from MIDI files."""

    def __init__(self, midi_dir: Path, emotion_labels: Optional[Path] = None):
        self.midi_dir = Path(midi_dir)
        # TODO: Implement MIDI loading and processing
        raise NotImplementedError("MelodyDataset not yet implemented")

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError


# =============================================================================
# Other datasets - Placeholders
# =============================================================================

class HarmonyDataset(Dataset):
    """Dataset loader for harmony prediction."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("HarmonyDataset not yet implemented")


class DynamicsDataset(Dataset):
    """Dataset loader for dynamics engine."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DynamicsDataset not yet implemented")


class GrooveDataset(Dataset):
    """Dataset loader for groove prediction."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("GrooveDataset not yet implemented")


# =============================================================================
# Utility Functions
# =============================================================================

def create_train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into train and validation sets.

    Args:
        dataset: Full dataset
        val_ratio: Ratio of validation samples (0.0 to 1.0)
        random_seed: Random seed for shuffling

    Returns:
        (train_dataset, val_dataset)
    """
    from torch.utils.data import random_split

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    return train_dataset, val_dataset
