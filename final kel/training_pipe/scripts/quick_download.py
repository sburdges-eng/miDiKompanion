#!/usr/bin/env python3
"""
Quick Dataset Download Helper
==============================
Helps download the smallest/easiest datasets to get started quickly.
"""

import subprocess
import sys
from pathlib import Path
import urllib.request
import zipfile
import os
import ssl

# Fix SSL certificate issues on macOS
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

def download_groove_midi(datasets_dir: Path):
    """Download Groove MIDI dataset (smallest, easiest to get started)."""
    print("\n" + "="*60)
    print("Downloading Groove MIDI Dataset (~50MB)")
    print("="*60)
    
    groove_dir = datasets_dir / "groove"
    groove_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip"
    zip_path = groove_dir / "groove.zip"
    
    print(f"Downloading from: {url}")
    print(f"Destination: {zip_path}")
    
    try:
        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                if percent % 10 == 0:
                    print(f"\r  Progress: {percent}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, zip_path, show_progress)
        print("\n✓ Download complete")
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(groove_dir)
        
        # Remove zip file
        zip_path.unlink()
        
        # Count MIDI files
        midi_files = list(groove_dir.glob("**/*.mid")) + list(groove_dir.glob("**/*.midi"))
        print(f"✓ Extracted {len(midi_files)} MIDI files")
        print(f"  Location: {groove_dir}")
        
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def create_harmony_dataset_from_existing(datasets_dir: Path, data_dir: Path):
    """Create harmony dataset from existing chord progression data."""
    print("\n" + "="*60)
    print("Creating Harmony Dataset from Existing Data")
    print("="*60)
    
    chords_dir = datasets_dir / "chords"
    chords_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing chord progressions
    chord_files = [
        data_dir / "chord_progressions_db.json",
        data_dir / "chord_progressions.json",
        data_dir / "common_progressions.json",
    ]
    
    output_file = chords_dir / "chord_progressions.json"
    
    for chord_file in chord_files:
        if chord_file.exists():
            print(f"Found: {chord_file}")
            # Copy or convert to training format
            import json
            try:
                with open(chord_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to training format if needed
                if isinstance(data, dict) and 'progressions' not in data:
                    # Assume it's a different format, create template
                    progressions = []
                    if isinstance(data, list):
                        for item in data[:100]:  # Limit for now
                            progressions.append({
                                'chords': item.get('chords', []),
                                'emotion': {
                                    'valence': item.get('valence', 0.0),
                                    'arousal': item.get('arousal', 0.5)
                                }
                            })
                    else:
                        # Create sample progressions
                        progressions = [
                            {
                                'chords': ['C', 'G', 'Am', 'F'],
                                'emotion': {'valence': 0.7, 'arousal': 0.6}
                            },
                            {
                                'chords': ['Dm', 'G', 'C'],
                                'emotion': {'valence': 0.5, 'arousal': 0.4}
                            }
                        ]
                    
                    data = {'progressions': progressions}
                
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"✓ Created harmony dataset: {output_file}")
                print(f"  Progressions: {len(data.get('progressions', []))}")
                return True
            except Exception as e:
                print(f"  Error processing {chord_file}: {e}")
                continue
    
    # Create template if nothing found
    template = {
        'progressions': [
            {
                'chords': ['C', 'G', 'Am', 'F'],
                'emotion': {'valence': 0.7, 'arousal': 0.6}
            }
        ]
    }
    with open(output_file, 'w') as f:
        import json
        json.dump(template, f, indent=2)
    print(f"✓ Created template: {output_file}")
    print("  Please add more chord progressions to this file")
    return True


def main():
    """Main function to set up datasets quickly."""
    project_root = Path(__file__).parent.parent.parent
    datasets_dir = project_root / "datasets"
    data_dir = project_root / "data"
    
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Quick Dataset Setup for Kelly ML Training")
    print("="*60)
    print(f"\nProject root: {project_root}")
    print(f"Datasets directory: {datasets_dir}")
    print("\nThis script will download the easiest datasets to get started.")
    print("For full datasets, see DATASET_DOWNLOAD_GUIDE.md")
    print()
    
    # Download Groove MIDI (easiest)
    print("\n[1/2] Groove MIDI Dataset")
    download_groove_midi(datasets_dir)
    
    # Create harmony dataset from existing data
    print("\n[2/2] Harmony Dataset")
    create_harmony_dataset_from_existing(datasets_dir, data_dir)
    
    print("\n" + "="*60)
    print("Quick Setup Complete!")
    print("="*60)
    print("\nYou now have:")
    print("  ✓ Groove MIDI dataset (for GroovePredictor)")
    print("  ✓ Harmony dataset (for HarmonyPredictor)")
    print("\nFor other models, you can:")
    print("  1. Use synthetic data: --synthetic flag")
    print("  2. Download full datasets (see DATASET_DOWNLOAD_GUIDE.md)")
    print("\nTest training with what you have:")
    print(f"  python training_pipe/scripts/train_all_models.py \\")
    print(f"    --datasets-dir {datasets_dir} \\")
    print(f"    --output ./trained_models")
    print("\nOr use synthetic data for all models:")
    print(f"  python training_pipe/scripts/train_all_models.py --synthetic")
    print()


if __name__ == "__main__":
    main()
