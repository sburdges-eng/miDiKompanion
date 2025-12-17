#!/usr/bin/env python3
"""
Real Dataset Loaders for Kelly MIDI Companion ML Training
==========================================================
Implements data loaders for all 5 models using real datasets:
1. DEAM - Emotion recognition
2. Lakh MIDI - Melody generation
3. MAESTRO - Dynamics engine
4. Groove MIDI - Groove predictor
5. Chord progressions - Harmony predictor
"""

import json
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Optional dependencies
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa not available. Audio feature extraction will use fallback.")

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    warnings.warn("mido not available. MIDI loading will use fallback.")


# =============================================================================
# Emotion Recognition Dataset (DEAM)
# =============================================================================

class EmotionDataset(Dataset):
    """Dataset for emotion recognition from audio (DEAM format)."""

    def __init__(
        self,
        audio_dir: Path,
        labels_file: Optional[Path] = None,
        sample_rate: int = 22050,
        n_mels: int = 128,
        hop_length: int = 512,
        n_fft: int = 2048,
        use_synthetic: bool = False,
        num_synthetic_samples: int = 10000
    ):
        """
        Initialize emotion dataset.

        Args:
            audio_dir: Directory containing audio files
            labels_file: CSV file with columns: filename,valence,arousal
            sample_rate: Audio sample rate
            n_mels: Number of mel bands
            use_synthetic: Fall back to synthetic data if real data unavailable
        """
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.use_synthetic = use_synthetic

        # Load labels
        self.samples = []
        if labels_file and Path(labels_file).exists():
            self._load_labels(labels_file)
        elif not use_synthetic:
            # Try to find labels in audio_dir
            labels_file = self.audio_dir / "labels.csv"
            if labels_file.exists():
                self._load_labels(labels_file)

        # If no real data, use synthetic
        if len(self.samples) == 0:
            if use_synthetic:
                print(f"⚠️  No real data found. Using {num_synthetic_samples} synthetic samples.")
                self._generate_synthetic(num_synthetic_samples)
            else:
                raise ValueError(
                    f"No audio files or labels found in {audio_dir}. "
                    "Set use_synthetic=True to use synthetic data."
                )

        print(f"✓ Loaded {len(self.samples)} emotion samples")

    def _load_labels(self, labels_file: Path):
        """Load emotion labels from CSV file."""
        audio_files = list(self.audio_dir.glob("*.wav")) + list(self.audio_dir.glob("*.mp3"))
        audio_dict = {f.name: f for f in audio_files}

        with open(labels_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get('filename', '').strip()
                if not filename:
                    continue

                try:
                    valence = float(row.get('valence', 0.0))
                    arousal = float(row.get('arousal', 0.0))

                    # Clamp to [-1, 1] for valence, [0, 1] for arousal
                    valence = np.clip(valence, -1.0, 1.0)
                    arousal = np.clip(arousal, 0.0, 1.0)

                    audio_path = audio_dict.get(filename)
                    if audio_path and audio_path.exists():
                        self.samples.append({
                            'audio_path': audio_path,
                            'valence': valence,
                            'arousal': arousal
                        })
                except (ValueError, KeyError):
                    continue

    def _generate_synthetic(self, num_samples: int):
        """Generate synthetic emotion data."""
        np.random.seed(42)
        for i in range(num_samples):
            valence = np.random.uniform(-1.0, 1.0)
            arousal = np.random.uniform(0.0, 1.0)
            self.samples.append({
                'audio_path': None,  # Will generate synthetic features
                'valence': valence,
                'arousal': arousal
            })

    def _extract_mel_features(self, audio_path: Optional[Path]) -> np.ndarray:
        """Extract 128-dim mel-spectrogram features."""
        if audio_path is None or not audio_path.exists():
            # Generate synthetic mel features
            return np.random.randn(self.n_mels).astype(np.float32)

        if not LIBROSA_AVAILABLE:
            # Fallback: random features
            return np.random.randn(self.n_mels).astype(np.float32)

        try:
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate, duration=30.0)

            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=self.n_mels,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )

            # Average over time to get single vector
            mel_features = np.mean(mel_spec, axis=1)

            # Normalize
            mel_features = (mel_features - mel_features.mean()) / (mel_features.std() + 1e-8)

            return mel_features.astype(np.float32)
        except Exception as e:
            warnings.warn(f"Error loading {audio_path}: {e}. Using synthetic features.")
            return np.random.randn(self.n_mels).astype(np.float32)

    def _valence_arousal_to_embedding(self, valence: float, arousal: float) -> np.ndarray:
        """Convert valence/arousal to 64-dim emotion embedding."""
        # Map valence-arousal space to 64-dim embedding
        # First 32 dims: valence-related, last 32 dims: arousal-related
        embedding = np.zeros(64, dtype=np.float32)

        # Valence component (first 32 dims)
        embedding[:32] = np.tanh(valence * np.random.randn(32))

        # Arousal component (last 32 dims)
        embedding[32:] = np.tanh(arousal * np.random.randn(32))

        return embedding

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Extract mel features
        mel_features = self._extract_mel_features(sample.get('audio_path'))

        # Convert valence/arousal to emotion embedding
        emotion = self._valence_arousal_to_embedding(
            sample['valence'],
            sample['arousal']
        )

        return {
            'mel_features': torch.tensor(mel_features),
            'emotion': torch.tensor(emotion)
        }


# =============================================================================
# Melody Generation Dataset (Lakh MIDI)
# =============================================================================

class MelodyDataset(Dataset):
    """Dataset for melody generation from emotion (Lakh MIDI format)."""

    def __init__(
        self,
        midi_dir: Path,
        emotion_labels_file: Optional[Path] = None,
        use_synthetic: bool = False,
        num_synthetic_samples: int = 10000
    ):
        """
        Initialize melody dataset.

        Args:
            midi_dir: Directory containing MIDI files
            emotion_labels_file: JSON file mapping MIDI files to emotion labels
            use_synthetic: Fall back to synthetic data
        """
        self.midi_dir = Path(midi_dir)
        self.use_synthetic = use_synthetic
        self.samples = []

        # Load emotion labels
        self.emotion_labels = {}
        if emotion_labels_file and Path(emotion_labels_file).exists():
            with open(emotion_labels_file, 'r') as f:
                self.emotion_labels = json.load(f)

        # Load MIDI files
        midi_files = list(self.midi_dir.glob("*.mid")) + list(self.midi_dir.glob("*.midi"))

        for midi_file in midi_files:
            filename = midi_file.name
            emotion = self.emotion_labels.get(filename, None)

            if emotion is None:
                # Generate random emotion if not labeled
                emotion = {
                    'valence': np.random.uniform(-1.0, 1.0),
                    'arousal': np.random.uniform(0.0, 1.0)
                }

            self.samples.append({
                'midi_path': midi_file,
                'emotion': emotion
            })

        # If no real data, use synthetic
        if len(self.samples) == 0:
            if use_synthetic:
                print(f"⚠️  No real data found. Using {num_synthetic_samples} synthetic samples.")
                self._generate_synthetic(num_synthetic_samples)
            else:
                raise ValueError(
                    f"No MIDI files found in {midi_dir}. "
                    "Set use_synthetic=True to use synthetic data."
                )

        print(f"✓ Loaded {len(self.samples)} melody samples")

    def _generate_synthetic(self, num_samples: int):
        """Generate synthetic melody data."""
        np.random.seed(42)
        for i in range(num_samples):
            emotion = {
                'valence': np.random.uniform(-1.0, 1.0),
                'arousal': np.random.uniform(0.0, 1.0)
            }
            self.samples.append({
                'midi_path': None,
                'emotion': emotion
            })

    def _midi_to_note_probs(self, midi_path: Optional[Path]) -> np.ndarray:
        """Convert MIDI file to 128-dim note probability vector."""
        if midi_path is None or not midi_path.exists():
            # Generate synthetic note probabilities
            probs = np.random.rand(128).astype(np.float32)
            return probs / probs.sum()

        if not MIDO_AVAILABLE:
            # Fallback: random probabilities
            probs = np.random.rand(128).astype(np.float32)
            return probs / probs.sum()

        try:
            mid = mido.MidiFile(str(midi_path))
            note_counts = np.zeros(128, dtype=np.float32)

            # Extract notes from all tracks
            for track in mid.tracks:
                current_notes = set()
                for msg in track:
                    if msg.type == 'note_on' and msg.velocity > 0:
                        current_notes.add(msg.note)
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        if msg.note in current_notes:
                            note_counts[msg.note] += 1
                            current_notes.remove(msg.note)

            # Convert to probabilities
            if note_counts.sum() > 0:
                note_probs = note_counts / note_counts.sum()
            else:
                # No notes found, use uniform distribution
                note_probs = np.ones(128) / 128.0

            return note_probs.astype(np.float32)
        except Exception as e:
            warnings.warn(f"Error loading {midi_path}: {e}. Using synthetic probabilities.")
            probs = np.random.rand(128).astype(np.float32)
            return probs / probs.sum()

    def _emotion_to_embedding(self, emotion: Dict) -> np.ndarray:
        """Convert emotion dict to 64-dim embedding."""
        valence = emotion.get('valence', 0.0)
        arousal = emotion.get('arousal', 0.5)

        embedding = np.zeros(64, dtype=np.float32)
        embedding[:32] = np.tanh(valence * np.random.randn(32))
        embedding[32:] = np.tanh(arousal * np.random.randn(32))

        return embedding

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Convert emotion to embedding
        emotion = self._emotion_to_embedding(sample['emotion'])

        # Convert MIDI to note probabilities
        notes = self._midi_to_note_probs(sample.get('midi_path'))

        return {
            'emotion': torch.tensor(emotion),
            'notes': torch.tensor(notes)
        }


# =============================================================================
# Harmony Prediction Dataset
# =============================================================================

class HarmonyDataset(Dataset):
    """Dataset for harmony prediction from context."""

    def __init__(
        self,
        chord_progressions_file: Path,
        use_synthetic: bool = False,
        num_synthetic_samples: int = 10000
    ):
        """
        Initialize harmony dataset.

        Args:
            chord_progressions_file: JSON file with chord progressions
            use_synthetic: Fall back to synthetic data
        """
        self.chord_progressions_file = Path(chord_progressions_file)
        self.use_synthetic = use_synthetic
        self.samples = []

        if self.chord_progressions_file.exists():
            with open(self.chord_progressions_file, 'r') as f:
                data = json.load(f)
                progressions = data.get('progressions', [])

                for prog in progressions:
                    self.samples.append({
                        'chords': prog.get('chords', []),
                        'emotion': prog.get('emotion', {'valence': 0.0, 'arousal': 0.5})
                    })
        elif not use_synthetic:
            raise ValueError(
                f"Chord progressions file not found: {chord_progressions_file}. "
                "Set use_synthetic=True to use synthetic data."
            )

        # If no real data, use synthetic
        if len(self.samples) == 0:
            if use_synthetic:
                print(f"⚠️  No real data found. Using {num_synthetic_samples} synthetic samples.")
                self._generate_synthetic(num_synthetic_samples)
            else:
                raise ValueError("No chord progressions found.")

        print(f"✓ Loaded {len(self.samples)} harmony samples")

    def _generate_synthetic(self, num_samples: int):
        """Generate synthetic harmony data."""
        np.random.seed(42)
        for i in range(num_samples):
            self.samples.append({
                'chords': [],
                'emotion': {
                    'valence': np.random.uniform(-1.0, 1.0),
                    'arousal': np.random.uniform(0.0, 1.0)
                }
            })

    def _chords_to_context(self, chords: List[str], emotion: Dict) -> np.ndarray:
        """Convert chord progression to 128-dim context vector."""
        # Create context from emotion + chord features
        emotion_embedding = np.zeros(64, dtype=np.float32)
        emotion_embedding[:32] = np.tanh(emotion.get('valence', 0.0) * np.random.randn(32))
        emotion_embedding[32:] = np.tanh(emotion.get('arousal', 0.5) * np.random.randn(32))

        # Chord features (simplified - would need proper chord encoding)
        chord_features = np.random.randn(64).astype(np.float32)

        context = np.concatenate([emotion_embedding, chord_features])
        return context

    def _chords_to_probabilities(self, chords: List[str]) -> np.ndarray:
        """Convert chords to 64-dim chord probability vector."""
        # Simplified: map to 64 chord classes
        # In production, would use proper chord vocabulary
        probs = np.random.rand(64).astype(np.float32)
        return probs / probs.sum()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        context = self._chords_to_context(sample['chords'], sample['emotion'])
        chord_probs = self._chords_to_probabilities(sample['chords'])

        return {
            'context': torch.tensor(context),
            'chords': torch.tensor(chord_probs)
        }


# =============================================================================
# Dynamics Engine Dataset (MAESTRO)
# =============================================================================

class DynamicsDataset(Dataset):
    """Dataset for dynamics engine (MAESTRO format)."""

    def __init__(
        self,
        midi_dir: Path,
        use_synthetic: bool = False,
        num_synthetic_samples: int = 10000
    ):
        """
        Initialize dynamics dataset.

        Args:
            midi_dir: Directory containing MIDI files with velocity data
            use_synthetic: Fall back to synthetic data
        """
        self.midi_dir = Path(midi_dir)
        self.use_synthetic = use_synthetic
        self.samples = []

        midi_files = list(self.midi_dir.glob("*.mid")) + list(self.midi_dir.glob("*.midi"))

        for midi_file in midi_files:
            self.samples.append({'midi_path': midi_file})

        # If no real data, use synthetic
        if len(self.samples) == 0:
            if use_synthetic:
                print(f"⚠️  No real data found. Using {num_synthetic_samples} synthetic samples.")
                self._generate_synthetic(num_synthetic_samples)
            else:
                raise ValueError(
                    f"No MIDI files found in {midi_dir}. "
                    "Set use_synthetic=True to use synthetic data."
                )

        print(f"✓ Loaded {len(self.samples)} dynamics samples")

    def _generate_synthetic(self, num_samples: int):
        """Generate synthetic dynamics data."""
        np.random.seed(42)
        for i in range(num_samples):
            self.samples.append({'midi_path': None})

    def _extract_context_and_expression(self, midi_path: Optional[Path]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 32-dim context and 16-dim expression from MIDI."""
        if midi_path is None or not midi_path.exists():
            # Generate synthetic
            context = np.random.randn(32).astype(np.float32)
            expression = np.random.rand(16).astype(np.float32)
            return context, expression

        if not MIDO_AVAILABLE:
            context = np.random.randn(32).astype(np.float32)
            expression = np.random.rand(16).astype(np.float32)
            return context, expression

        try:
            mid = mido.MidiFile(str(midi_path))

            # Extract velocity statistics
            velocities = []
            for track in mid.tracks:
                for msg in track:
                    if msg.type == 'note_on' and msg.velocity > 0:
                        velocities.append(msg.velocity)

            if len(velocities) > 0:
                # Create context from velocity stats
                context = np.array([
                    np.mean(velocities) / 127.0,
                    np.std(velocities) / 127.0,
                    np.min(velocities) / 127.0,
                    np.max(velocities) / 127.0,
                ] + [0.0] * 28, dtype=np.float32)

                # Create expression parameters
                expression = np.array([
                    np.mean(velocities) / 127.0,
                    np.std(velocities) / 127.0,
                ] + [np.random.rand() for _ in range(14)], dtype=np.float32)
            else:
                context = np.random.randn(32).astype(np.float32)
                expression = np.random.rand(16).astype(np.float32)

            return context, expression
        except Exception as e:
            warnings.warn(f"Error loading {midi_path}: {e}. Using synthetic data.")
            context = np.random.randn(32).astype(np.float32)
            expression = np.random.rand(16).astype(np.float32)
            return context, expression

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        context, expression = self._extract_context_and_expression(sample.get('midi_path'))

        return {
            'context': torch.tensor(context),
            'expression': torch.tensor(expression)
        }


# =============================================================================
# Groove Predictor Dataset (Groove MIDI)
# =============================================================================

class GrooveDataset(Dataset):
    """Dataset for groove prediction (Groove MIDI format)."""

    def __init__(
        self,
        drums_dir: Path,
        emotion_labels_file: Optional[Path] = None,
        use_synthetic: bool = False,
        num_synthetic_samples: int = 10000
    ):
        """
        Initialize groove dataset.

        Args:
            drums_dir: Directory containing drum MIDI files
            emotion_labels_file: JSON file mapping files to emotion labels
            use_synthetic: Fall back to synthetic data
        """
        self.drums_dir = Path(drums_dir)
        self.use_synthetic = use_synthetic
        self.samples = []

        # Load emotion labels
        self.emotion_labels = {}
        if emotion_labels_file and Path(emotion_labels_file).exists():
            with open(emotion_labels_file, 'r') as f:
                self.emotion_labels = json.load(f)

        # Load MIDI files
        midi_files = list(self.drums_dir.glob("*.mid")) + list(self.drums_dir.glob("*.midi"))

        for midi_file in midi_files:
            filename = midi_file.name
            emotion = self.emotion_labels.get(filename, None)

            if emotion is None:
                emotion = {
                    'valence': np.random.uniform(-1.0, 1.0),
                    'arousal': np.random.uniform(0.0, 1.0)
                }

            self.samples.append({
                'midi_path': midi_file,
                'emotion': emotion
            })

        # If no real data, use synthetic
        if len(self.samples) == 0:
            if use_synthetic:
                print(f"⚠️  No real data found. Using {num_synthetic_samples} synthetic samples.")
                self._generate_synthetic(num_synthetic_samples)
            else:
                raise ValueError(
                    f"No MIDI files found in {drums_dir}. "
                    "Set use_synthetic=True to use synthetic data."
                )

        print(f"✓ Loaded {len(self.samples)} groove samples")

    def _generate_synthetic(self, num_samples: int):
        """Generate synthetic groove data."""
        np.random.seed(42)
        for i in range(num_samples):
            self.samples.append({
                'midi_path': None,
                'emotion': {
                    'valence': np.random.uniform(-1.0, 1.0),
                    'arousal': np.random.uniform(0.0, 1.0)
                }
            })

    def _emotion_to_embedding(self, emotion: Dict) -> np.ndarray:
        """Convert emotion to 64-dim embedding."""
        valence = emotion.get('valence', 0.0)
        arousal = emotion.get('arousal', 0.5)

        embedding = np.zeros(64, dtype=np.float32)
        embedding[:32] = np.tanh(valence * np.random.randn(32))
        embedding[32:] = np.tanh(arousal * np.random.randn(32))

        return embedding

    def _extract_groove_params(self, midi_path: Optional[Path]) -> np.ndarray:
        """Extract 32-dim groove parameters from drum MIDI."""
        if midi_path is None or not midi_path.exists():
            # Generate synthetic groove parameters
            return np.tanh(np.random.randn(32)).astype(np.float32)

        if not MIDO_AVAILABLE:
            return np.tanh(np.random.randn(32)).astype(np.float32)

        try:
            mid = mido.MidiFile(str(midi_path))

            # Extract timing and velocity patterns
            # Simplified: extract basic statistics
            note_times = []
            velocities = []

            current_time = 0
            for track in mid.tracks:
                for msg in track:
                    current_time += msg.time
                    if msg.type == 'note_on' and msg.velocity > 0:
                        note_times.append(current_time)
                        velocities.append(msg.velocity)

            if len(note_times) > 0:
                # Create groove parameters from timing/velocity patterns
                groove = np.zeros(32, dtype=np.float32)

                # Timing features
                if len(note_times) > 1:
                    intervals = np.diff(note_times)
                    groove[:8] = np.tanh(np.array([
                        np.mean(intervals),
                        np.std(intervals),
                        np.min(intervals),
                        np.max(intervals),
                    ] + [0.0] * 4))

                # Velocity features
                if len(velocities) > 0:
                    groove[8:16] = np.tanh(np.array([
                        np.mean(velocities) / 127.0,
                        np.std(velocities) / 127.0,
                        np.min(velocities) / 127.0,
                        np.max(velocities) / 127.0,
                    ] + [0.0] * 4))

                # Fill rest with random (would be more sophisticated in production)
                groove[16:] = np.tanh(np.random.randn(16))
            else:
                groove = np.tanh(np.random.randn(32))

            return groove.astype(np.float32)
        except Exception as e:
            warnings.warn(f"Error loading {midi_path}: {e}. Using synthetic groove.")
            return np.tanh(np.random.randn(32)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        emotion = self._emotion_to_embedding(sample['emotion'])
        groove = self._extract_groove_params(sample.get('midi_path'))

        return {
            'emotion': torch.tensor(emotion),
            'groove': torch.tensor(groove)
        }


# =============================================================================
# Utility Functions
# =============================================================================

def create_data_loaders(
    datasets_dir: Path,
    batch_size: int = 64,
    val_split: float = 0.2,
    use_synthetic: bool = True,
    num_workers: int = 0
) -> Dict[str, Tuple[DataLoader, DataLoader]]:
    """
    Create data loaders for all models.

    Returns:
        Dictionary mapping model names to (train_loader, val_loader) tuples
    """
    datasets_dir = Path(datasets_dir)

    loaders = {}

    # EmotionRecognizer
    try:
        emotion_dataset = EmotionDataset(
            audio_dir=datasets_dir / "audio",
            labels_file=datasets_dir / "audio" / "labels.csv",
            use_synthetic=use_synthetic
        )
        train_size = int((1 - val_split) * len(emotion_dataset))
        val_size = len(emotion_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            emotion_dataset, [train_size, val_size]
        )
        loaders['EmotionRecognizer'] = (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        )
    except Exception as e:
        print(f"⚠️  Could not create EmotionRecognizer dataset: {e}")

    # MelodyTransformer
    try:
        melody_dataset = MelodyDataset(
            midi_dir=datasets_dir / "midi",
            emotion_labels_file=datasets_dir / "emotion_labels.json",
            use_synthetic=use_synthetic
        )
        train_size = int((1 - val_split) * len(melody_dataset))
        val_size = len(melody_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            melody_dataset, [train_size, val_size]
        )
        loaders['MelodyTransformer'] = (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        )
    except Exception as e:
        print(f"⚠️  Could not create MelodyTransformer dataset: {e}")

    # HarmonyPredictor
    try:
        harmony_dataset = HarmonyDataset(
            chord_progressions_file=datasets_dir / "chords" / "chord_progressions.json",
            use_synthetic=use_synthetic
        )
        train_size = int((1 - val_split) * len(harmony_dataset))
        val_size = len(harmony_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            harmony_dataset, [train_size, val_size]
        )
        loaders['HarmonyPredictor'] = (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        )
    except Exception as e:
        print(f"⚠️  Could not create HarmonyPredictor dataset: {e}")

    # DynamicsEngine
    try:
        dynamics_dataset = DynamicsDataset(
            midi_dir=datasets_dir / "dynamics_midi",
            use_synthetic=use_synthetic
        )
        train_size = int((1 - val_split) * len(dynamics_dataset))
        val_size = len(dynamics_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dynamics_dataset, [train_size, val_size]
        )
        loaders['DynamicsEngine'] = (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        )
    except Exception as e:
        print(f"⚠️  Could not create DynamicsEngine dataset: {e}")

    # GroovePredictor
    try:
        groove_dataset = GrooveDataset(
            drums_dir=datasets_dir / "drums",
            emotion_labels_file=datasets_dir / "groove_emotion_labels.json",
            use_synthetic=use_synthetic
        )
        train_size = int((1 - val_split) * len(groove_dataset))
        val_size = len(groove_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            groove_dataset, [train_size, val_size]
        )
        loaders['GroovePredictor'] = (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        )
    except Exception as e:
        print(f"⚠️  Could not create GroovePredictor dataset: {e}")

    return loaders
