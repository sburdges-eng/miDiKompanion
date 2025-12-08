#!/usr/bin/env python3
"""
Build Lo-Fi Drum Kit
Creates Logic Pro sampler instrument from organized samples
"""

import json
from pathlib import Path

SAMPLES_DIR = Path.home() / "Music" / "Samples"
CATALOG_FILE = SAMPLES_DIR / "sample_catalog.json"
LOGIC_SAMPLER_DIR = Path.home() / "Music" / "Audio Music Apps" / "Sampler Instruments"


def load_catalog():
    """Load sample catalog"""
    if not CATALOG_FILE.exists():
        print(f"❌ Catalog not found: {CATALOG_FILE}")
        print("   Run sample_cataloger.py first")
        return None

    with open(CATALOG_FILE) as f:
        return json.load(f)


def get_best_samples(catalog, sample_type, count=3):
    """Get best samples of a given type"""
    samples = [s for s in catalog["samples"] if s["type"].lower() == sample_type.lower()]

    if not samples:
        return []

    # Sort by file size (larger = likely better quality)
    samples.sort(key=lambda x: x["size_bytes"], reverse=True)

    return samples[:count]


def create_exs24_instrument(kit_name, sample_mapping):
    """Create Logic Pro EXS24 instrument file"""
    # EXS24 format is binary and complex - instead, create a text guide

    LOGIC_SAMPLER_DIR.mkdir(parents=True, exist_ok=True)

    guide_file = LOGIC_SAMPLER_DIR / f"{kit_name}_MAPPING.txt"

    with open(guide_file, 'w') as f:
        f.write(f"# {kit_name} - Sample Mapping Guide\n")
        f.write(f"# Created: {Path(__file__).name}\n\n")
        f.write("## MIDI Note Mapping\n\n")

        for midi_note, (note_name, sample_info) in sample_mapping.items():
            f.write(f"{midi_note} ({note_name}): {sample_info['filename']}\n")
            f.write(f"   Path: {sample_info['relative_path']}\n\n")

        f.write("\n## Logic Pro Setup Instructions\n\n")
        f.write("1. Open Logic Pro\n")
        f.write("2. Create new Software Instrument track\n")
        f.write("3. Load 'Sampler' or 'Quick Sampler' plugin\n")
        f.write("4. Drag samples onto MIDI notes as listed above\n")
        f.write("5. Save as EXS24 or Sampler preset\n")
        f.write(f"6. Name it: {kit_name}\n")

    return guide_file


def build_lofi_drum_kit():
    """Build lo-fi drum kit from available samples"""
    print("=" * 60)
    print("LO-FI DRUM KIT BUILDER")
    print("Creating sampler instrument for Logic Pro")
    print("=" * 60)

    catalog = load_catalog()
    if not catalog:
        return

    print(f"\n📊 Catalog loaded: {catalog['total_samples']} samples\n")

    # Standard GM drum mapping (compatible with MPK mini 3)
    drum_mapping = {
        36: ("C1", "Kick", "Kick"),
        38: ("D1", "Snare", "Snare"),
        40: ("E1", "Snare (Rim)", "Snare"),
        42: ("F#1", "Hi-Hat Closed", "HiHats"),
        44: ("F#1", "Hi-Hat Pedal", "HiHats"),
        46: ("A#1", "Hi-Hat Open", "HiHats"),
        49: ("C#2", "Crash", "Cymbals"),
        51: ("D#2", "Ride", "Cymbals"),
        41: ("F1", "Low Tom", "Toms"),
        43: ("G1", "Low-Mid Tom", "Toms"),
        45: ("A1", "Mid Tom", "Toms"),
        47: ("B1", "High-Mid Tom", "Toms"),
        48: ("C2", "High Tom", "Toms"),
    }

    kit_samples = {}

    print("🔍 Finding best samples for each drum voice:\n")

    for midi_note, (note_name, drum_name, sample_type) in drum_mapping.items():
        print(f"{note_name} ({midi_note}): {drum_name}")

        # Find best samples of this type
        best = get_best_samples(catalog, sample_type, count=1)

        if best:
            kit_samples[midi_note] = (note_name, best[0])
            print(f"  ✅ {best[0]['filename']}")
        else:
            print(f"  ⚠️  No {sample_type} samples found")

    if not kit_samples:
        print("\n❌ No samples available to build kit")
        print("   Run freesound_downloader.py and organize_samples.py first")
        return

    print(f"\n📦 Building kit with {len(kit_samples)} voices...\n")

    # Create EXS24 mapping guide
    kit_name = "LoFi_Bedroom_Kit_01"
    guide_file = create_exs24_instrument(kit_name, kit_samples)

    print("=" * 60)
    print(f"✅ KIT BUILT: {kit_name}")
    print("=" * 60)
    print(f"\nMapping guide saved: {guide_file}")
    print(f"\nSamples used: {len(kit_samples)} voices")
    print(f"\n🎹 MIDI Note Range: C1 (36) to C2 (48)")
    print(f"   Compatible with: MPK mini 3, most MIDI keyboards")
    print(f"\n📖 Next steps:")
    print(f"   1. Open {guide_file}")
    print(f"   2. Follow Logic Pro setup instructions")
    print(f"   3. Load samples into Sampler/Quick Sampler")
    print(f"   4. Play with your MIDI keyboard!")
    print()


if __name__ == "__main__":
    build_lofi_drum_kit()
