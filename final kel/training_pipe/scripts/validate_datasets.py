#!/usr/bin/env python3
"""
Dataset Validation Script for Kelly MIDI Companion ML Training
==============================================================
Validates dataset quality and structure before training.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: librosa/soundfile not available. Audio validation will be limited.")

try:
    import mido
    import pretty_midi
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False
    print("Warning: mido/pretty_midi not available. MIDI validation will be limited.")


def validate_audio_dataset(audio_dir: Path, labels_csv: Path = None) -> Dict:
    """Validate emotion recognition audio dataset."""
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        results['valid'] = False
        results['errors'].append(f"Audio directory does not exist: {audio_dir}")
        return results

    # Find audio files
    audio_files = list(audio_dir.glob("*.wav")) + \
                 list(audio_dir.glob("*.mp3")) + \
                 list(audio_dir.glob("*.flac"))

    results['stats']['total_files'] = len(audio_files)

    if len(audio_files) == 0:
        results['valid'] = False
        results['errors'].append(f"No audio files found in {audio_dir}")
        return results

    # Validate labels if provided
    if labels_csv and Path(labels_csv).exists():
        try:
            df = pd.read_csv(labels_csv)
            results['stats']['labeled_files'] = len(df)

            # Check required columns
            if 'valence' not in df.columns or 'arousal' not in df.columns:
                if 'valence_mean' in df.columns and 'arousal_mean' in df.columns:
                    results['warnings'].append("Using valence_mean/arousal_mean columns")
                else:
                    results['errors'].append("CSV missing valence/arousal columns")
                    results['valid'] = False

            # Check value ranges
            if 'valence' in df.columns:
                valence_range = (df['valence'].min(), df['valence'].max())
                if valence_range[0] < -1.0 or valence_range[1] > 1.0:
                    results['warnings'].append(
                        f"Valence values outside [-1, 1]: {valence_range}"
                    )

            if 'arousal' in df.columns:
                arousal_range = (df['arousal'].min(), df['arousal'].max())
                if arousal_range[0] < 0.0 or arousal_range[1] > 1.0:
                    results['warnings'].append(
                        f"Arousal values outside [0, 1]: {arousal_range}"
                    )

            # Check for missing files
            labeled_filenames = set(df['filename'].tolist())
            audio_filenames = {f.name for f in audio_files}
            missing_labels = audio_filenames - labeled_filenames
            if missing_labels:
                results['warnings'].append(
                    f"{len(missing_labels)} audio files without labels"
                )

        except Exception as e:
            results['errors'].append(f"Error reading labels CSV: {e}")
            results['valid'] = False

    # Validate audio files (sample a few)
    if AUDIO_AVAILABLE:
        valid_count = 0
        invalid_files = []

        for audio_file in audio_files[:min(10, len(audio_files))]:  # Sample first 10
            try:
                y, sr = librosa.load(str(audio_file), duration=1.0)  # Load just 1 second
                if len(y) == 0:
                    invalid_files.append(audio_file.name)
                else:
                    valid_count += 1
            except Exception as e:
                invalid_files.append(audio_file.name)
                results['warnings'].append(f"Could not load {audio_file.name}: {e}")

        results['stats']['sample_valid'] = valid_count
        results['stats']['sample_invalid'] = len(invalid_files)

    if len(audio_files) < 100:
        results['warnings'].append(
            f"Only {len(audio_files)} audio files - recommend at least 100 for training"
        )

    return results


def validate_midi_dataset(midi_dir: Path, emotion_labels: Path = None) -> Dict:
    """Validate melody/groove MIDI dataset."""
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    midi_dir = Path(midi_dir)
    if not midi_dir.exists():
        results['valid'] = False
        results['errors'].append(f"MIDI directory does not exist: {midi_dir}")
        return results

    # Find MIDI files
    midi_files = list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))
    results['stats']['total_files'] = len(midi_files)

    if len(midi_files) == 0:
        results['valid'] = False
        results['errors'].append(f"No MIDI files found in {midi_dir}")
        return results

    # Validate emotion labels if provided
    if emotion_labels and Path(emotion_labels).exists():
        try:
            with open(emotion_labels, 'r') as f:
                labels = json.load(f)
            results['stats']['labeled_files'] = len(labels)

            # Check label format
            midi_filenames = {f.name for f in midi_files}
            labeled_filenames = set(labels.keys())
            missing_labels = midi_filenames - labeled_filenames
            if missing_labels:
                results['warnings'].append(
                    f"{len(missing_labels)} MIDI files without labels"
                )

        except Exception as e:
            results['errors'].append(f"Error reading emotion labels JSON: {e}")
            results['valid'] = False

    # Validate MIDI files (sample a few)
    if MIDI_AVAILABLE:
        valid_count = 0
        invalid_files = []

        for midi_file in midi_files[:min(10, len(midi_files))]:  # Sample first 10
            try:
                midi = pretty_midi.PrettyMIDI(str(midi_file))
                if len(midi.instruments) == 0:
                    invalid_files.append(midi_file.name)
                else:
                    valid_count += 1
            except Exception as e:
                invalid_files.append(midi_file.name)
                results['warnings'].append(f"Could not load {midi_file.name}: {e}")

        results['stats']['sample_valid'] = valid_count
        results['stats']['sample_invalid'] = len(invalid_files)

    if len(midi_files) < 50:
        results['warnings'].append(
            f"Only {len(midi_files)} MIDI files - recommend at least 50 for training"
        )

    return results


def validate_harmony_dataset(chord_file: Path) -> Dict:
    """Validate harmony/chord progression dataset."""
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    chord_file = Path(chord_file)
    if not chord_file.exists():
        results['valid'] = False
        results['errors'].append(f"Chord file does not exist: {chord_file}")
        return results

    try:
        with open(chord_file, 'r') as f:
            data = json.load(f)

        if 'progressions' in data:
            progressions = data['progressions']
            results['stats']['total_progressions'] = len(progressions)

            # Check format
            for i, prog in enumerate(progressions[:5]):  # Check first 5
                if 'chords' not in prog:
                    results['errors'].append(
                        f"Progression {i} missing 'chords' field"
                    )
                    results['valid'] = False
                if 'emotion' not in prog:
                    results['warnings'].append(
                        f"Progression {i} missing 'emotion' field"
                    )

        if results['stats'].get('total_progressions', 0) < 20:
            results['warnings'].append(
                f"Only {results['stats'].get('total_progressions', 0)} progressions - "
                "recommend at least 20"
            )

    except Exception as e:
        results['errors'].append(f"Error reading chord file: {e}")
        results['valid'] = False

    return results


def print_validation_results(dataset_name: str, results: Dict):
    """Print validation results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"{dataset_name} Dataset Validation")
    print(f"{'='*60}")

    if results['valid']:
        print("✓ Dataset is valid")
    else:
        print("✗ Dataset has errors")

    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"  ✗ {error}")

    if results['warnings']:
        print("\nWarnings:")
        for warning in results['warnings']:
            print(f"  ⚠ {warning}")

    if results['stats']:
        print("\nStatistics:")
        for key, value in results['stats'].items():
            print(f"  {key}: {value}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Validate datasets for Kelly MIDI Companion ML training"
    )
    parser.add_argument(
        "--datasets-dir", "-d",
        type=str,
        default="./datasets",
        help="Root directory for datasets"
    )
    parser.add_argument(
        "--training-dir", "-t",
        type=str,
        default=None,
        help="Training datasets directory (default: datasets/training)"
    )

    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    training_dir = Path(args.training_dir) if args.training_dir else datasets_dir / "training"

    print("=" * 60)
    print("Kelly MIDI Companion - Dataset Validation")
    print("=" * 60)
    print(f"\nDatasets directory: {datasets_dir}")
    print(f"Training directory: {training_dir}")

    all_valid = True

    # Validate emotion dataset
    audio_dir = training_dir / "audio"
    labels_csv = audio_dir / "labels.csv"
    if audio_dir.exists():
        results = validate_audio_dataset(audio_dir, labels_csv)
        print_validation_results("Emotion Recognition (Audio)", results)
        if not results['valid']:
            all_valid = False
    else:
        print(f"\n⚠ Emotion dataset directory not found: {audio_dir}")

    # Validate melody dataset
    midi_dir = training_dir / "midi"
    emotion_labels = training_dir / "emotion_labels.json"
    if midi_dir.exists():
        results = validate_midi_dataset(midi_dir, emotion_labels)
        print_validation_results("Melody Generation (MIDI)", results)
        if not results['valid']:
            all_valid = False
    else:
        print(f"\n⚠ Melody dataset directory not found: {midi_dir}")

    # Validate harmony dataset
    chord_file = training_dir / "chords" / "chord_progressions.json"
    if chord_file.exists():
        results = validate_harmony_dataset(chord_file)
        print_validation_results("Harmony Prediction (Chords)", results)
        if not results['valid']:
            all_valid = False
    else:
        print(f"\n⚠ Harmony dataset file not found: {chord_file}")

    # Validate dynamics dataset
    dynamics_dir = training_dir / "dynamics_midi"
    if dynamics_dir.exists():
        results = validate_midi_dataset(dynamics_dir)
        print_validation_results("Dynamics Engine (MIDI)", results)
        if not results['valid']:
            all_valid = False
    else:
        print(f"\n⚠ Dynamics dataset directory not found: {dynamics_dir}")

    # Validate groove dataset
    drums_dir = training_dir / "drums"
    drum_labels = training_dir / "drum_labels.json"
    if drums_dir.exists():
        results = validate_midi_dataset(drums_dir, drum_labels)
        print_validation_results("Groove Prediction (Drums)", results)
        if not results['valid']:
            all_valid = False
    else:
        print(f"\n⚠ Groove dataset directory not found: {drums_dir}")

    # Summary
    print("=" * 60)
    print("Validation Summary")
    print("=" * 60)
    if all_valid:
        print("✓ All datasets are valid and ready for training")
    else:
        print("✗ Some datasets have errors. Please fix them before training.")
    print()


if __name__ == "__main__":
    main()
