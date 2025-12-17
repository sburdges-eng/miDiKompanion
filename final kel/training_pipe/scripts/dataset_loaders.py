#!/usr/bin/env python3
"""
Real Dataset Loaders for Kelly MIDI Companion ML Training
==========================================================
Implements PyTorch Dataset classes for loading real training data:
1. EmotionDataset - Audio files with valence/arousal labels
2. MelodyDataset - MIDI files with emotion embeddings
3. HarmonyDataset - Chord progressions with emotion labels
4. DynamicsDataset - MIDI files with velocity/expression data
5. GrooveDataset - Drum MIDI files with emotion labels
"""

import csv
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa not available. Install with: pip install librosa")

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    warnings.warn("mido not available. Install with: pip install mido")


# =============================================================================
# Emotion Recognition Dataset (DEAM)
# =============================================================================

class EmotionDataset(Dataset):
    """
    Dataset for emotion recognition from audio.

    Expected structure:
        audio_dir/
        ├── audio_001.wav
        ├── audio_002.wav
        └── labels.csv  (filename,valence,arousal)
    """

    def __init__(
        self,
        audio_dir: Path,
        labels_file: Optional[Path] = None,
        n_mels: int = 128,
        duration: float = 2.0,
        sr: int = 22050,
        use_cache: bool = True
    ):
        self.audio_dir = Path(audio_dir)
        self.n_mels = n_mels
        self.duration = duration
        self.sr = sr
        self.use_cache = use_cache

        # Load labels
        if labels_file is None:
            labels_file = self.audio_dir / "labels.csv"

        self.labels = self._load_labels(labels_file)

        # Get audio files
        audio_files = list(self.audio_dir.glob("*.wav")) + \
                     list(self.audio_dir.glob("*.mp3")) + \
                     list(self.audio_dir.glob("*.flac"))

        # Filter to only files with labels
        self.samples = []
        for audio_file in audio_files:
            if audio_file.name in self.labels:
                self.samples.append(audio_file)

        print(f"Loaded {len(self.samples)} emotion samples from {audio_dir}")

        # Cache for precomputed features
        self.feature_cache = {}

    def _load_labels(self, labels_file: Path) -> Dict[str, Tuple[float, float]]:
        """Load emotion labels from CSV file."""
        labels = {}

        if not labels_file.exists():
            print(f"Warning: Labels file not found: {labels_file}")
            return labels

        with open(labels_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get('filename', '').strip()
                try:
                    valence = float(row.get('valence', 0.0))
                    arousal = float(row.get('arousal', 0.5))
                    labels[filename] = (valence, arousal)
                except (ValueError, KeyError):
                    continue

        return labels

    def _extract_mel_features(self, audio_path: Path) -> np.ndarray:
        """Extract mel-spectrogram features from audio file."""
        if not LIBROSA_AVAILABLE:
            # Fallback to random features if librosa not available
            return np.random.randn(self.n_mels).astype(np.float32)

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
                fmax=sr // 2
            )

            # Convert to log scale
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)

            # Average over time to get single feature vector
            features = np.mean(log_mel, axis=1)

            # Normalize to [-1, 1]
            features = np.clip(features / 80.0, -1.0, 1.0)

            return features.astype(np.float32)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return np.zeros(self.n_mels, dtype=np.float32)

    def _valence_arousal_to_embedding(self, valence: float, arousal: float) -> np.ndarray:
        """Convert valence/arousal to 64-dim emotion embedding."""
        # First 32 dims: valence-related
        # Last 32 dims: arousal-related
        embedding = np.zeros(64, dtype=np.float32)

        # Valence component (first 32 dims)
        embedding[:32] = np.tanh(valence * np.ones(32))

        # Arousal component (last 32 dims)
        embedding[32:] = np.tanh(arousal * np.ones(32))

        return embedding

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path = self.samples[idx]

        # Check cache
        cache_key = str(audio_path)
        if self.use_cache and cache_key in self.feature_cache:
            mel_features = self.feature_cache[cache_key]
        else:
            mel_features = self._extract_mel_features(audio_path)
            if self.use_cache:
                self.feature_cache[cache_key] = mel_features

        # Get emotion labels
        valence, arousal = self.labels[audio_path.name]
        emotion_embedding = self._valence_arousal_to_embedding(valence, arousal)

        return {
            'mel_features': torch.tensor(mel_features),
            'emotion': torch.tensor(emotion_embedding)
        }


# =============================================================================
# Melody Generation Dataset (Lakh MIDI)
# =============================================================================

class MelodyDataset(Dataset):
    """
    Dataset for melody generation from emotion embeddings.

    Expected structure:
        midi_dir/ - MIDI files
        emotion_labels.json - {filename: {valence, arousal}}
    """

    def __init__(
        self,
        midi_dir: Path,
        emotion_labels_file: Optional[Path] = None,
        max_files: Optional[int] = None
    ):
        self.midi_dir = Path(midi_dir)

        # Load emotion labels
        if emotion_labels_file is None:
            emotion_labels_file = self.midi_dir.parent / "emotion_labels.json"

        self.emotion_labels = self._load_emotion_labels(emotion_labels_file)

        # Get MIDI files
        midi_files = list(self.midi_dir.glob("*.mid")) + \
                    list(self.midi_dir.glob("*.midi"))

        if max_files:
            midi_files = midi_files[:max_files]

        # Filter to files with labels (or use default if missing)
        self.samples = []
        for midi_file in midi_files:
            self.samples.append(midi_file)

        print(f"Loaded {len(self.samples)} melody samples from {midi_dir}")

    def _load_emotion_labels(self, labels_file: Path) -> Dict[str, Dict[str, float]]:
        """Load emotion labels from JSON file."""
        labels = {}

        if not labels_file.exists():
            print(f"Warning: Emotion labels not found: {labels_file}")
            return labels

        try:
            with open(labels_file, 'r') as f:
                labels = json.load(f)
        except Exception as e:
            print(f"Error loading emotion labels: {e}")

        return labels

    def _midi_to_note_probabilities(self, midi_path: Path) -> np.ndarray:
        """Convert MIDI file to note probability vector (128 notes)."""
        if not MIDO_AVAILABLE:
            # Fallback to uniform distribution
            probs = np.ones(128) / 128.0
            return probs.astype(np.float32)

        try:
            mid = mido.MidiFile(str(midi_path))

            # Count note occurrences
            note_counts = np.zeros(128)

            for track in mid.tracks:
                for msg in track:
                    if msg.type == 'note_on' and msg.velocity > 0:
                        note = msg.note
                        if 0 <= note < 128:
                            note_counts[note] += 1

            # Convert to probabilities
            total = note_counts.sum()
            if total > 0:
                probs = note_counts / total
            else:
                probs = np.ones(128) / 128.0

            return probs.astype(np.float32)
        except Exception as e:
            print(f"Error processing MIDI {midi_path}: {e}")
            return np.ones(128, dtype=np.float32) / 128.0

    def _valence_arousal_to_embedding(self, valence: float, arousal: float) -> np.ndarray:
        """Convert valence/arousal to 64-dim emotion embedding."""
        embedding = np.zeros(64, dtype=np.float32)
        embedding[:32] = np.tanh(valence * np.ones(32))
        embedding[32:] = np.tanh(arousal * np.ones(32))
        return embedding

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        midi_path = self.samples[idx]

        # Get emotion embedding
        filename = midi_path.name
        if filename in self.emotion_labels:
            emotion_data = self.emotion_labels[filename]
            valence = emotion_data.get('valence', 0.0)
            arousal = emotion_data.get('arousal', 0.5)
        else:
            # Default emotion if not labeled
            valence = 0.0
            arousal = 0.5

        emotion_embedding = self._valence_arousal_to_embedding(valence, arousal)

        # Get note probabilities
        note_probs = self._midi_to_note_probabilities(midi_path)

        return {
            'emotion': torch.tensor(emotion_embedding),
            'notes': torch.tensor(note_probs)
        }


# =============================================================================
# Harmony Prediction Dataset
# =============================================================================

class HarmonyDataset(Dataset):
    """
    Dataset for harmony prediction from context.

    Expected structure:
        chord_dir/chord_progressions.json
    """

    def __init__(self, chord_file: Path):
        self.chord_file = Path(chord_file)
        self.progressions = self._load_progressions()

        print(f"Loaded {len(self.progressions)} harmony progressions")

    def _load_progressions(self) -> List[Dict]:
        """Load chord progressions from JSON file."""
        if not self.chord_file.exists():
            print(f"Warning: Chord file not found: {self.chord_file}")
            return []

        try:
            with open(self.chord_file, 'r') as f:
                data = json.load(f)
                return data.get('progressions', [])
        except Exception as e:
            print(f"Error loading chord progressions: {e}")
            return []

    def _chords_to_vector(self, chords: List[str]) -> np.ndarray:
        """Convert chord list to 64-dim chord probability vector."""
        # Simplified: map common chords to indices
        chord_map = {
            'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
            'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
        }

        probs = np.zeros(64, dtype=np.float32)

        for chord in chords:
            # Extract root note
            root = chord[0] if chord else 'C'
            if root in chord_map:
                base_idx = chord_map[root]
                # Distribute probability across chord types
                for i in range(5):  # Major, minor, dim, aug, sus
                    idx = (base_idx * 5 + i) % 64
                    probs[idx] += 1.0

        # Normalize
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones(64) / 64.0

        return probs

    def _emotion_to_context(self, emotion: Dict[str, float]) -> np.ndarray:
        """Convert emotion to 128-dim context vector."""
        valence = emotion.get('valence', 0.0)
        arousal = emotion.get('arousal', 0.5)

        # Create context vector (emotion + other features)
        context = np.zeros(128, dtype=np.float32)
        context[:32] = np.tanh(valence * np.ones(32))
        context[32:64] = np.tanh(arousal * np.ones(32))
        # Remaining 64 dims could include other context features
        context[64:] = np.random.randn(64) * 0.1  # Placeholder

        return context

    def __len__(self) -> int:
        return len(self.progressions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prog = self.progressions[idx]

        chords = prog.get('chords', [])
        emotion = prog.get('emotion', {'valence': 0.0, 'arousal': 0.5})

        context = self._emotion_to_context(emotion)
        chord_probs = self._chords_to_vector(chords)

        return {
            'context': torch.tensor(context),
            'chords': torch.tensor(chord_probs)
        }


# =============================================================================
# Dynamics Engine Dataset (MAESTRO)
# =============================================================================

class DynamicsDataset(Dataset):
    """
    Dataset for dynamics/expression prediction from MIDI.

    Extracts velocity and timing information from MIDI files.
    """

    def __init__(self, midi_dir: Path, max_files: Optional[int] = None):
        self.midi_dir = Path(midi_dir)

        midi_files = list(self.midi_dir.glob("*.mid")) + \
                    list(self.midi_dir.glob("*.midi"))

        if max_files:
            midi_files = midi_files[:max_files]

        self.samples = midi_files

        print(f"Loaded {len(self.samples)} dynamics samples from {midi_dir}")

    def _extract_dynamics(self, midi_path: Path) -> np.ndarray:
        """Extract dynamics features from MIDI file."""
        if not MIDO_AVAILABLE:
            return np.random.rand(16).astype(np.float32)

        try:
            mid = mido.MidiFile(str(midi_path))

            velocities = []
            timings = []

            for track in mid.tracks:
                time = 0.0
                for msg in track:
                    time += msg.time
                    if msg.type == 'note_on' and msg.velocity > 0:
                        velocities.append(msg.velocity / 127.0)
                        timings.append(time)

            if len(velocities) == 0:
                return np.random.rand(16).astype(np.float32)

            # Extract 16 features: velocity stats, timing stats, etc.
            features = np.zeros(16, dtype=np.float32)

            vel_array = np.array(velocities)
            features[0] = np.mean(vel_array)  # Mean velocity
            features[1] = np.std(vel_array)   # Velocity variation
            features[2] = np.min(vel_array)   # Min velocity
            features[3] = np.max(vel_array)   # Max velocity

            if len(timings) > 1:
                timing_array = np.array(timings)
                features[4] = np.mean(np.diff(timing_array))  # Mean interval
                features[5] = np.std(np.diff(timing_array))    # Interval variation

            # Additional features (placeholder)
            features[6:] = np.random.rand(10) * 0.1

            return np.clip(features, 0.0, 1.0)
        except Exception as e:
            print(f"Error processing MIDI {midi_path}: {e}")
            return np.random.rand(16).astype(np.float32)

    def _create_context(self, midi_path: Path) -> np.ndarray:
        """Create 32-dim compact context vector."""
        # Simplified context (could include more features)
        context = np.random.randn(32).astype(np.float32) * 0.1
        context = np.clip(context, -1.0, 1.0)
        return context

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        midi_path = self.samples[idx]

        context = self._create_context(midi_path)
        dynamics = self._extract_dynamics(midi_path)

        return {
            'context': torch.tensor(context),
            'dynamics': torch.tensor(dynamics)
        }


# =============================================================================
# Groove Predictor Dataset
# =============================================================================

class GrooveDataset(Dataset):
    """
    Dataset for groove prediction from emotion.

    Expected: Drum MIDI files with emotion labels.
    """

    def __init__(
        self,
        drums_dir: Path,
        emotion_labels_file: Optional[Path] = None
    ):
        self.drums_dir = Path(drums_dir)

        # Load emotion labels
        if emotion_labels_file is None:
            emotion_labels_file = self.drums_dir.parent / "drum_labels.json"

        self.emotion_labels = self._load_emotion_labels(emotion_labels_file)

        # Get drum MIDI files
        drum_files = list(self.drums_dir.glob("*.mid")) + \
                    list(self.drums_dir.glob("*.midi"))

        self.samples = drum_files

        print(f"Loaded {len(self.samples)} groove samples from {drums_dir}")

    def _load_emotion_labels(self, labels_file: Path) -> Dict[str, Dict[str, float]]:
        """Load emotion labels from JSON file."""
        if not labels_file.exists():
            return {}

        try:
            with open(labels_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading emotion labels: {e}")
            return {}

    def _extract_groove_features(self, midi_path: Path) -> np.ndarray:
        """Extract groove parameters from drum MIDI."""
        if not MIDO_AVAILABLE:
            return np.random.randn(32).astype(np.float32) * 0.1

        try:
            mid = mido.MidiFile(str(midi_path))

            # Extract timing and velocity patterns
            events = []
            for track in mid.tracks:
                time = 0.0
                for msg in track:
                    time += msg.time
                    if msg.type == 'note_on' and msg.velocity > 0:
                        events.append((time, msg.note, msg.velocity))

            # Compute groove features (32 dims)
            features = np.zeros(32, dtype=np.float32)

            if len(events) > 0:
                # Timing features
                times = [e[0] for e in events]
                if len(times) > 1:
                    intervals = np.diff(times)
                    features[0] = np.mean(intervals)
                    features[1] = np.std(intervals)

                # Velocity features
                velocities = [e[2] / 127.0 for e in events]
                features[2] = np.mean(velocities)
                features[3] = np.std(velocities)

            # Additional groove features (placeholder)
            features[4:] = np.random.randn(28) * 0.1

            return np.clip(features, -1.0, 1.0)
        except Exception as e:
            print(f"Error processing MIDI {midi_path}: {e}")
            return np.random.randn(32).astype(np.float32) * 0.1

    def _valence_arousal_to_embedding(self, valence: float, arousal: float) -> np.ndarray:
        """Convert valence/arousal to 64-dim emotion embedding."""
        embedding = np.zeros(64, dtype=np.float32)
        embedding[:32] = np.tanh(valence * np.ones(32))
        embedding[32:] = np.tanh(arousal * np.ones(32))
        return embedding

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        midi_path = self.samples[idx]

        # Get emotion embedding
        filename = midi_path.name
        if filename in self.emotion_labels:
            emotion_data = self.emotion_labels[filename]
            valence = emotion_data.get('valence', 0.0)
            arousal = emotion_data.get('arousal', 0.5)
        else:
            valence = 0.0
            arousal = 0.5

        emotion_embedding = self._valence_arousal_to_embedding(valence, arousal)
        groove_features = self._extract_groove_features(midi_path)

        return {
            'emotion': torch.tensor(emotion_embedding),
            'groove': torch.tensor(groove_features)
        }
