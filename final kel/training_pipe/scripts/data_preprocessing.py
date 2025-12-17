#!/usr/bin/env python3
"""
Data Preprocessing and Augmentation for Kelly MIDI Companion ML Training
========================================================================
Includes:
- Audio augmentation (pitch shift, time stretch, noise)
- MIDI augmentation (transpose, tempo change)
- Feature normalization
- Data validation
"""

import numpy as np
import librosa
import pretty_midi
from pathlib import Path
from typing import Tuple, Optional
import random


# =============================================================================
# Audio Augmentation
# =============================================================================

def augment_audio(
    audio: np.ndarray,
    sr: int = 22050,
    pitch_shift_semitones: Optional[Tuple[float, float]] = (-2, 2),
    time_stretch_rate: Optional[Tuple[float, float]] = (0.9, 1.1),
    add_noise: bool = False,
    noise_level: float = 0.01
) -> np.ndarray:
    """
    Apply augmentations to audio signal.

    Args:
        audio: Audio signal (1D numpy array)
        sr: Sample rate
        pitch_shift_semitones: Range for pitch shifting (min, max)
        time_stretch_rate: Range for time stretching (min, max)
        add_noise: Whether to add noise
        noise_level: Noise level (relative to signal)

    Returns:
        Augmented audio signal
    """
    augmented = audio.copy()

    # Pitch shift
    if pitch_shift_semitones:
        n_steps = random.uniform(*pitch_shift_semitones)
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)

    # Time stretch
    if time_stretch_rate:
        rate = random.uniform(*time_stretch_rate)
        augmented = librosa.effects.time_stretch(augmented, rate=rate)

    # Add noise
    if add_noise:
        noise = np.random.randn(len(augmented)) * noise_level * np.std(augmented)
        augmented = augmented + noise

    # Normalize
    max_val = np.abs(augmented).max()
    if max_val > 0:
        augmented = augmented / max_val

    return augmented


def augment_mel_spectrogram(
    mel_spec: np.ndarray,
    pitch_shift: Optional[Tuple[int, int]] = (-2, 2),
    time_warp: Optional[Tuple[float, float]] = (0.9, 1.1),
    add_noise: bool = False
) -> np.ndarray:
    """
    Augment mel-spectrogram directly (faster than audio augmentation).

    Args:
        mel_spec: Mel-spectrogram (n_mels, time)
        pitch_shift: Range for shifting mel bins (min, max)
        time_warp: Range for time warping (min, max)
        add_noise: Whether to add noise

    Returns:
        Augmented mel-spectrogram
    """
    augmented = mel_spec.copy()

    # Pitch shift (shift mel bins)
    if pitch_shift:
        n_bins = random.randint(*pitch_shift)
        if n_bins != 0:
            augmented = np.roll(augmented, n_bins, axis=0)

    # Time warp (simple scaling along time axis)
    if time_warp and augmented.ndim > 1:
        rate = random.uniform(*time_warp)
        n_frames = int(augmented.shape[1] * rate)
        if n_frames > 0:
            # Simple resampling along time axis
            indices = np.linspace(0, augmented.shape[1] - 1, n_frames).astype(int)
            augmented = augmented[:, indices]
            # Crop or pad to original length
            if n_frames < augmented.shape[1]:
                padded = np.zeros_like(augmented[:, :augmented.shape[1]])
                padded[:, :n_frames] = augmented
                augmented = padded
            else:
                augmented = augmented[:, :augmented.shape[1]]

    # Add noise
    if add_noise:
        noise = np.random.randn(*augmented.shape) * 0.01 * np.std(augmented)
        augmented = augmented + noise

    return augmented


# =============================================================================
# MIDI Augmentation
# =============================================================================

def augment_midi(
    midi_data: pretty_midi.PrettyMIDI,
    transpose_semitones: Optional[Tuple[int, int]] = (-6, 6),
    tempo_change: Optional[Tuple[float, float]] = (0.8, 1.2)
) -> pretty_midi.PrettyMIDI:
    """
    Apply augmentations to MIDI data.

    Args:
        midi_data: PrettyMIDI object
        transpose_semitones: Range for transposition (min, max)
        tempo_change: Range for tempo scaling (min, max)

    Returns:
        New PrettyMIDI object with augmentations
    """
    # Create copy
    new_midi = pretty_midi.PrettyMIDI()

    # Transpose
    transpose = 0
    if transpose_semitones:
        transpose = random.randint(*transpose_semitones)

    # Tempo change
    tempo_scale = 1.0
    if tempo_change:
        tempo_scale = random.uniform(*tempo_change)

    # Copy instruments with augmentations
    for instrument in midi_data.instruments:
        new_instrument = pretty_midi.Instrument(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name
        )

        for note in instrument.notes:
            # Transpose pitch
            new_pitch = max(0, min(127, note.pitch + transpose))

            # Scale timing
            new_start = note.start * tempo_scale
            new_end = note.end * tempo_scale

            new_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=new_pitch,
                start=new_start,
                end=new_end
            )
            new_instrument.notes.append(new_note)

        new_midi.instruments.append(new_instrument)

    return new_midi


def augment_midi_notes(
    notes: np.ndarray,
    transpose_semitones: Optional[int] = None
) -> np.ndarray:
    """
    Augment MIDI note probability vector.

    Args:
        notes: Note probability vector (128,)
        transpose_semitones: Semitones to transpose (if None, random)

    Returns:
        Augmented note vector
    """
    augmented = notes.copy()

    if transpose_semitones is None:
        transpose_semitones = random.randint(-6, 6)

    if transpose_semitones != 0:
        # Circular shift
        augmented = np.roll(augmented, transpose_semitones)

    return augmented


# =============================================================================
# Feature Normalization
# =============================================================================

class FeatureNormalizer:
    """Normalize features using standardization or min-max scaling."""

    def __init__(self, method: str = 'standard'):
        """
        Args:
            method: 'standard' (z-score) or 'minmax'
        """
        self.method = method
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.fitted = False

    def fit(self, features: np.ndarray):
        """Fit normalizer to data."""
        if self.method == 'standard':
            self.mean = np.mean(features, axis=0)
            self.std = np.std(features, axis=0)
            # Avoid division by zero
            self.std = np.where(self.std == 0, 1.0, self.std)
        else:  # minmax
            self.min = np.min(features, axis=0)
            self.max = np.max(features, axis=0)
            # Avoid division by zero
            self.max = np.where(self.max == self.min, self.min + 1.0, self.max)

        self.fitted = True

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted parameters."""
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        if self.method == 'standard':
            return (features - self.mean) / self.std
        else:  # minmax
            return (features - self.min) / (self.max - self.min)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features)
        return self.transform(features)

    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        """Inverse transform (denormalize)."""
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        if self.method == 'standard':
            return features * self.std + self.mean
        else:  # minmax
            return features * (self.max - self.min) + self.min


# =============================================================================
# Data Validation
# =============================================================================

def validate_audio_file(audio_path: Path, min_duration: float = 1.0) -> bool:
    """Validate audio file exists and has minimum duration."""
    try:
        if not audio_path.exists():
            return False

        duration = librosa.get_duration(path=str(audio_path))
        return duration >= min_duration
    except Exception:
        return False


def validate_midi_file(midi_path: Path, min_notes: int = 10) -> bool:
    """Validate MIDI file exists and has minimum notes."""
    try:
        if not midi_path.exists():
            return False

        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        total_notes = sum(len(instrument.notes) for instrument in midi_data.instruments)
        return total_notes >= min_notes
    except Exception:
        return False


def validate_dataset(
    audio_files: list = None,
    midi_files: list = None,
    min_audio_duration: float = 1.0,
    min_midi_notes: int = 10
) -> Tuple[list, list]:
    """
    Validate dataset files.

    Returns:
        (valid_audio_files, valid_midi_files)
    """
    valid_audio = []
    valid_midi = []

    if audio_files:
        print(f"Validating {len(audio_files)} audio files...")
        for audio_file in audio_files:
            if validate_audio_file(audio_file, min_audio_duration):
                valid_audio.append(audio_file)
        print(f"  Valid: {len(valid_audio)}/{len(audio_files)}")

    if midi_files:
        print(f"Validating {len(midi_files)} MIDI files...")
        for midi_file in midi_files:
            if validate_midi_file(midi_file, min_midi_notes):
                valid_midi.append(midi_file)
        print(f"  Valid: {len(valid_midi)}/{len(midi_files)}")

    return valid_audio, valid_midi
