#!/usr/bin/env python3
"""
Dataset Download Script for Kelly MIDI Companion ML Training
============================================================
Automatically downloads and prepares datasets for training the 5 ML models:
1. DEAM - Emotion recognition (audio + valence/arousal labels)
2. Lakh MIDI - Melody generation (176K MIDI files)
3. MAESTRO - Dynamics engine (piano with velocity data)
4. Groove MIDI - Groove prediction (drum patterns)
5. Hooktheory/iRealPro - Harmony prediction (chord progressions)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Optional, Dict, List
import urllib.request
import urllib.error


# Dataset URLs and metadata
DATASETS = {
    "deam": {
        "name": "DEAM (Dataset for Emotion Analysis in Music)",
        "url": "https://cvml.unige.ch/databases/DEAM/DEAM_audio.zip",
        "description": "1,802 audio excerpts with valence/arousal annotations",
        "size_mb": 500,
        "format": "zip",
        "extract_to": "deam",
        "required_files": ["DEAM_audio"],
        "notes": "Manual download may be required. Visit: https://cvml.unige.ch/databases/DEAM/"
    },
    "lakh_midi": {
        "name": "Lakh MIDI Dataset",
        "url": "https://colinraffel.com/projects/lmd/lmd_full.tar.gz",
        "description": "176,581 MIDI files for melody generation",
        "size_mb": 1800,
        "format": "tar.gz",
        "extract_to": "lakh_midi",
        "required_files": ["lmd_full"],
        "notes": "Large download (~1.8GB). Consider using lmd_matched subset instead."
    },
    "lakh_matched": {
        "name": "Lakh MIDI Matched (subset)",
        "url": "https://colinraffel.com/projects/lmd/lmd_matched.tar.gz",
        "description": "45,129 matched MIDI files (smaller subset)",
        "size_mb": 500,
        "format": "tar.gz",
        "extract_to": "lakh_midi",
        "required_files": ["lmd_matched"],
        "notes": "Recommended for faster download and processing."
    },
    "maestro": {
        "name": "MAESTRO Dataset",
        "url": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
        "description": "200+ hours of piano performances with dynamics",
        "size_mb": 100,
        "format": "zip",
        "extract_to": "maestro",
        "required_files": ["maestro-v3.0.0"],
        "notes": "Official Google Magenta dataset."
    },
    "groove": {
        "name": "Groove MIDI Dataset",
        "url": "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip",
        "description": "1,150 drum patterns with timing/velocity",
        "size_mb": 5,
        "format": "zip",
        "extract_to": "groove",
        "required_files": ["groove"],
        "notes": "Small dataset, quick download."
    }
}


def download_file(url: str, dest_path: Path, description: str = "") -> bool:
    """Download a file with progress indication."""
    try:
        print(f"  Downloading {description or url}...")
        print(f"  Destination: {dest_path}")

        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                size_mb = total_size / (1024 * 1024)
                downloaded_mb = (block_num * block_size) / (1024 * 1024)
                print(f"\r  Progress: {percent}% ({downloaded_mb:.1f}/{size_mb:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, dest_path, show_progress)
        print()  # New line after progress
        return True
    except urllib.error.HTTPError as e:
        print(f"\n  ❌ HTTP Error {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"\n  ❌ URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        return False


def extract_archive(archive_path: Path, extract_to: Path, format: str) -> bool:
    """Extract archive (zip or tar.gz) to destination."""
    try:
        print(f"  Extracting to {extract_to}...")
        extract_to.mkdir(parents=True, exist_ok=True)

        if format == "zip":
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif format in ["tar.gz", "tgz"]:
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"  ❌ Unknown format: {format}")
            return False

        print(f"  ✓ Extracted successfully")
        return True
    except Exception as e:
        print(f"  ❌ Extraction error: {e}")
        return False


def verify_dataset(dataset_dir: Path, required_files: List[str]) -> bool:
    """Verify that dataset was extracted correctly."""
    print(f"  Verifying dataset structure...")

    found_files = []
    for item in dataset_dir.rglob("*"):
        if item.is_file() or item.is_dir():
            found_files.append(item.name)

    # Check for required files/directories
    for req in required_files:
        found = any(req.lower() in f.lower() for f in found_files)
        if not found:
            # Try to find similar names
            similar = [f for f in found_files if req[:5].lower() in f.lower()]
            if similar:
                print(f"  ⚠️  '{req}' not found, but found similar: {similar[:3]}")
            else:
                print(f"  ⚠️  '{req}' not found in extracted files")
        else:
            print(f"  ✓ Found '{req}'")

    # Count files
    file_count = sum(1 for _ in dataset_dir.rglob("*") if _.is_file())
    print(f"  Found {file_count} files total")

    return file_count > 0


def download_deam(datasets_dir: Path, force: bool = False) -> bool:
    """Download DEAM dataset for emotion recognition."""
    dataset_info = DATASETS["deam"]
    dataset_dir = datasets_dir / "deam"
    archive_path = datasets_dir / "DEAM_audio.zip"

    print(f"\n{'='*60}")
    print(f"Downloading: {dataset_info['name']}")
    print(f"{'='*60}")
    print(f"Description: {dataset_info['description']}")
    print(f"Size: ~{dataset_info['size_mb']} MB")
    print(f"Note: {dataset_info['notes']}")

    # Check if already exists
    if dataset_dir.exists() and not force:
        print(f"  Dataset already exists at {dataset_dir}")
        response = input("  Re-download? (y/N): ").strip().lower()
        if response != 'y':
            return True

    # Try to download
    if not archive_path.exists() or force:
        print(f"\n  ⚠️  Automatic download may not be available.")
        print(f"  Please visit: {dataset_info['url']}")
        print(f"  Or download manually and place at: {archive_path}")
        response = input("  Have you downloaded it manually? (y/N): ").strip().lower()
        if response == 'y':
            manual_path = input(f"  Enter path to downloaded file (or press Enter for {archive_path}): ").strip()
            if manual_path:
                archive_path = Path(manual_path)
        else:
            return False
    else:
        print(f"  Archive already exists: {archive_path}")

    # Extract if archive exists
    if archive_path.exists():
        if extract_archive(archive_path, dataset_dir, dataset_info['format']):
            return verify_dataset(dataset_dir, dataset_info['required_files'])

    return False


def download_lakh_midi(datasets_dir: Path, subset: str = "matched", force: bool = False) -> bool:
    """Download Lakh MIDI dataset for melody generation."""
    dataset_key = "lakh_matched" if subset == "matched" else "lakh_midi"
    dataset_info = DATASETS[dataset_key]
    dataset_dir = datasets_dir / "lakh_midi"
    archive_path = datasets_dir / f"lmd_{subset}.tar.gz"

    print(f"\n{'='*60}")
    print(f"Downloading: {dataset_info['name']}")
    print(f"{'='*60}")
    print(f"Description: {dataset_info['description']}")
    print(f"Size: ~{dataset_info['size_mb']} MB")

    # Check if already exists
    if dataset_dir.exists() and not force:
        print(f"  Dataset already exists at {dataset_dir}")
        response = input("  Re-download? (y/N): ").strip().lower()
        if response != 'y':
            return True

    # Download
    if not archive_path.exists() or force:
        if not download_file(dataset_info['url'], archive_path, dataset_info['name']):
            return False
    else:
        print(f"  Archive already exists: {archive_path}")

    # Extract
    if archive_path.exists():
        if extract_archive(archive_path, dataset_dir, dataset_info['format']):
            return verify_dataset(dataset_dir, dataset_info['required_files'])

    return False


def download_maestro(datasets_dir: Path, force: bool = False) -> bool:
    """Download MAESTRO dataset for dynamics engine."""
    dataset_info = DATASETS["maestro"]
    dataset_dir = datasets_dir / "maestro"
    archive_path = datasets_dir / "maestro-v3.0.0-midi.zip"

    print(f"\n{'='*60}")
    print(f"Downloading: {dataset_info['name']}")
    print(f"{'='*60}")
    print(f"Description: {dataset_info['description']}")
    print(f"Size: ~{dataset_info['size_mb']} MB")

    # Check if already exists
    if dataset_dir.exists() and not force:
        print(f"  Dataset already exists at {dataset_dir}")
        response = input("  Re-download? (y/N): ").strip().lower()
        if response != 'y':
            return True

    # Download
    if not archive_path.exists() or force:
        if not download_file(dataset_info['url'], archive_path, dataset_info['name']):
            return False
    else:
        print(f"  Archive already exists: {archive_path}")

    # Extract
    if archive_path.exists():
        if extract_archive(archive_path, dataset_dir, dataset_info['format']):
            return verify_dataset(dataset_dir, dataset_info['required_files'])

    return False


def download_groove(datasets_dir: Path, force: bool = False) -> bool:
    """Download Groove MIDI dataset for groove prediction."""
    dataset_info = DATASETS["groove"]
    dataset_dir = datasets_dir / "groove"
    archive_path = datasets_dir / "groove-v1.0.0-midionly.zip"

    print(f"\n{'='*60}")
    print(f"Downloading: {dataset_info['name']}")
    print(f"{'='*60}")
    print(f"Description: {dataset_info['description']}")
    print(f"Size: ~{dataset_info['size_mb']} MB")

    # Check if already exists
    if dataset_dir.exists() and not force:
        print(f"  Dataset already exists at {dataset_dir}")
        response = input("  Re-download? (y/N): ").strip().lower()
        if response != 'y':
            return True

    # Download
    if not archive_path.exists() or force:
        if not download_file(dataset_info['url'], archive_path, dataset_info['name']):
            return False
    else:
        print(f"  Archive already exists: {archive_path}")

    # Extract
    if archive_path.exists():
        if extract_archive(archive_path, dataset_dir, dataset_info['format']):
            return verify_dataset(dataset_dir, dataset_info['required_files'])

    return False


def create_harmony_template(datasets_dir: Path):
    """Create template for harmony dataset (manual collection required)."""
    harmony_dir = datasets_dir / "harmony"
    harmony_dir.mkdir(parents=True, exist_ok=True)

    template_file = harmony_dir / "chord_progressions.json"

    if template_file.exists():
        print(f"  Template already exists: {template_file}")
        return

    template = {
        "description": "Chord progressions with emotion labels for HarmonyPredictor training",
        "format": {
            "name": "Progression name (e.g., 'I-V-vi-IV')",
            "chords": ["List of chord names"],
            "emotion": {
                "valence": "Float from -1.0 to 1.0",
                "arousal": "Float from 0.0 to 1.0"
            }
        },
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
            },
            {
                "name": "vi-IV-I-V",
                "chords": ["Am", "F", "C", "G"],
                "emotion": {"valence": 0.6, "arousal": 0.5}
            }
        ],
        "notes": [
            "Add more progressions manually or use iRealPro/Hooktheory data",
            "iRealPro: https://www.irealpro.com/",
            "Hooktheory: https://www.hooktheory.com/theorytab"
        ]
    }

    with open(template_file, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"  ✓ Created template: {template_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for Kelly MIDI Companion ML training"
    )
    parser.add_argument(
        "--datasets-dir", "-d",
        type=str,
        default="./datasets",
        help="Directory to store datasets"
    )
    parser.add_argument(
        "--dataset", "-D",
        type=str,
        choices=["all", "deam", "lakh", "maestro", "groove", "harmony"],
        default="all",
        help="Which dataset to download (default: all)"
    )
    parser.add_argument(
        "--lakh-subset",
        type=str,
        choices=["full", "matched"],
        default="matched",
        help="Lakh MIDI subset: 'full' (1.8GB) or 'matched' (500MB, recommended)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if dataset exists"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip prompts (useful for scripts)"
    )

    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Kelly MIDI Companion - Dataset Downloader")
    print("=" * 60)
    print(f"\nDatasets directory: {datasets_dir}")
    print(f"Selected dataset: {args.dataset}")

    results = {}

    if args.dataset in ["all", "deam"]:
        results["deam"] = download_deam(datasets_dir, force=args.force)

    if args.dataset in ["all", "lakh"]:
        results["lakh"] = download_lakh_midi(
            datasets_dir,
            subset=args.lakh_subset,
            force=args.force
        )

    if args.dataset in ["all", "maestro"]:
        results["maestro"] = download_maestro(datasets_dir, force=args.force)

    if args.dataset in ["all", "groove"]:
        results["groove"] = download_groove(datasets_dir, force=args.force)

    if args.dataset in ["all", "harmony"]:
        create_harmony_template(datasets_dir)
        results["harmony"] = True

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)

    for name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {name:12} {status}")

    print(f"\nDatasets location: {datasets_dir}")
    print("\nNext steps:")
    print("  1. Verify datasets are correctly extracted")
    print("  2. Run: python scripts/prepare_datasets.py --datasets-dir", datasets_dir)
    print("  3. Run: python scripts/train_all_models.py")
    print()


if __name__ == "__main__":
    main()
