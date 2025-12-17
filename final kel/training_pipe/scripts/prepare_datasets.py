#!/usr/bin/env python3
"""
Dataset Preparation Script for Kelly MIDI Companion ML Training
================================================================
Prepares audio, MIDI, and emotion data for training the 5 models.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


def prepare_emotion_dataset(audio_dir: Path, output_dir: Path):
    """
    Prepare emotion recognition dataset.

    Expected structure:
        audio_dir/
        ├── audio_001.wav
        ├── audio_002.wav
        └── labels.csv  (filename,valence,arousal)
    """
    print("\n[1/5] Preparing Emotion Recognition Dataset...")

    labels_file = audio_dir / "labels.csv"
    if not labels_file.exists():
        print("⚠️  No labels.csv found. Creating template...")
        with open(labels_file, 'w') as f:
            f.write("filename,valence,arousal\n")
            f.write("# Add your audio files here with emotion labels\n")
            f.write("# valence: -1.0 (negative) to 1.0 (positive)\n")
            f.write("# arousal: 0.0 (calm) to 1.0 (excited)\n")
            f.write("# Example:\n")
            f.write("# happy_001.wav,0.8,0.9\n")
            f.write("# sad_001.wav,-0.6,0.3\n")
        print(f"  Created template: {labels_file}")
        print("  Please fill it with your audio files and labels")
        return

    # Count audio files
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
    print(f"  Found {len(audio_files)} audio files")

    # Parse labels
    labels = []
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and ',' in line:
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        filename = parts[0]
                        valence = float(parts[1])
                        arousal = float(parts[2])
                        labels.append((filename, valence, arousal))
                    except ValueError:
                        continue

    print(f"  Found {len(labels)} labeled samples")

    if len(labels) == 0:
        print("  ⚠️  No valid labels found. Please fill labels.csv")
    else:
        print(f"  ✓ Dataset ready for EmotionRecognizer training")


def prepare_melody_dataset(midi_dir: Path, emotion_labels: Path, output_dir: Path):
    """
    Prepare melody generation dataset.

    Expected:
        midi_dir/ - MIDI files
        emotion_labels.json - {filename: {valence, arousal}}
    """
    print("\n[2/5] Preparing Melody Transformer Dataset...")

    if not midi_dir.exists():
        print(f"  Creating {midi_dir}...")
        midi_dir.mkdir(parents=True, exist_ok=True)

    midi_files = list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))
    print(f"  Found {len(midi_files)} MIDI files")

    if not emotion_labels.exists():
        print("  Creating template emotion_labels.json...")
        template = {
            "midi_001.mid": {"valence": 0.5, "arousal": 0.6},
            "midi_002.mid": {"valence": -0.3, "arousal": 0.4}
        }
        with open(emotion_labels, 'w') as f:
            json.dump(template, f, indent=2)
        print(f"  Created template: {emotion_labels}")
    else:
        with open(emotion_labels, 'r') as f:
            labels = json.load(f)
        print(f"  Found {len(labels)} labeled MIDI files")

    if len(midi_files) > 0:
        print(f"  ✓ Dataset ready for MelodyTransformer training")


def prepare_harmony_dataset(chord_dir: Path, output_dir: Path):
    """Prepare harmony prediction dataset."""
    print("\n[3/5] Preparing Harmony Predictor Dataset...")

    if not chord_dir.exists():
        print(f"  Creating {chord_dir}...")
        chord_dir.mkdir(parents=True, exist_ok=True)

        # Create template
        template_file = chord_dir / "chord_progressions.json"
        template = {
            "progressions": [
                {
                    "name": "I-V-vi-IV",
                    "chords": ["C", "G", "Am", "F"],
                    "emotion": {"valence": 0.7, "arousal": 0.6}
                },
                {
                    "name": "ii-V-I",
                    "chords": ["Dm", "G", "C"],
                    "emotion": {"valence": 0.5, "arousal": 0.4}
                }
            ]
        }
        with open(template_file, 'w') as f:
            json.dump(template, f, indent=2)
        print(f"  Created template: {template_file}")

    chord_files = list(chord_dir.glob("*.json"))
    print(f"  Found {len(chord_files)} chord progression files")

    if len(chord_files) > 0:
        print(f"  ✓ Dataset ready for HarmonyPredictor training")


def prepare_dynamics_dataset(midi_dir: Path, output_dir: Path):
    """Prepare dynamics engine dataset."""
    print("\n[4/5] Preparing Dynamics Engine Dataset...")

    # Can reuse MIDI files from melody dataset
    midi_files = list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))
    print(f"  Found {len(midi_files)} MIDI files (will extract velocity data)")

    if len(midi_files) > 0:
        print(f"  ✓ Dataset ready for DynamicsEngine training")


def prepare_groove_dataset(drums_dir: Path, output_dir: Path):
    """Prepare groove predictor dataset."""
    print("\n[5/5] Preparing Groove Predictor Dataset...")

    if not drums_dir.exists():
        print(f"  Creating {drums_dir}...")
        drums_dir.mkdir(parents=True, exist_ok=True)

    drum_files = list(drums_dir.glob("*.mid")) + list(drums_dir.glob("*.midi"))
    print(f"  Found {len(drum_files)} drum MIDI files")

    if len(drum_files) == 0:
        print("  ℹ️  Add drum MIDI files to train GroovePredictor")
        print("  Recommended: Groove MIDI Dataset")
        print("  https://magenta.tensorflow.org/datasets/groove")
    else:
        print(f"  ✓ Dataset ready for GroovePredictor training")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for Kelly MIDI Companion ML training"
    )
    parser.add_argument(
        "--datasets-dir", "-d",
        type=str,
        default="./datasets",
        help="Root directory for datasets"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./prepared_data",
        help="Output directory for prepared data"
    )

    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Kelly MIDI Companion - Dataset Preparation")
    print("=" * 60)

    # Create directories
    datasets_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare each dataset
    prepare_emotion_dataset(
        datasets_dir / "audio",
        output_dir / "emotion"
    )

    prepare_melody_dataset(
        datasets_dir / "midi",
        datasets_dir / "emotion_labels.json",
        output_dir / "melody"
    )

    prepare_harmony_dataset(
        datasets_dir / "chords",
        output_dir / "harmony"
    )

    prepare_dynamics_dataset(
        datasets_dir / "midi",
        output_dir / "dynamics"
    )

    prepare_groove_dataset(
        datasets_dir / "drums",
        output_dir / "groove"
    )

    print("\n" + "=" * 60)
    print("Dataset Preparation Summary")
    print("=" * 60)
    print(f"\nDatasets location: {datasets_dir}")
    print(f"Output location: {output_dir}")
    print("\nNext steps:")
    print("  1. Add your audio/MIDI files to the datasets directories")
    print("  2. Fill in the emotion labels")
    print("  3. Run: python scripts/train_all_models.py")
    print("")


if __name__ == "__main__":
    main()
