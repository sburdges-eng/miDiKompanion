#!/usr/bin/env python3
"""
Sample Organizer
Sorts downloaded samples into proper folder structure
Renames files to standard format: [BPM]_[Key]_[Type]_[Description]_[Number].wav
"""

import os
import re
import shutil
from pathlib import Path

DOWNLOAD_DIR = Path.home() / "Downloads" / "Freesound_Packs"
SAMPLES_DIR = Path.home() / "Music" / "Samples"

# Keyword-based categorization
CATEGORIES = {
    "Kicks": ["kick", "bd", "bassdrum", "bass drum"],
    "Snares": ["snare", "sd", "snr"],
    "HiHats": ["hihat", "hh", "hat", "hi-hat", "closed hat", "open hat"],
    "Cymbals": ["cymbal", "crash", "ride", "china"],
    "Toms": ["tom", "floor tom", "rack tom"],
    "Percussion": ["perc", "conga", "bongo", "shaker", "tamb", "cowbell", "clave"],
    "Loops": ["loop", "beat", "groove", "pattern"],
    "Bass": ["bass", "sub", "808", "low end"],
    "Pads": ["pad", "atmosphere", "ambient", "texture"],
    "Leads": ["lead", "melody", "synth lead"],
    "Keys": ["piano", "keys", "rhodes", "wurlitzer", "organ"],
    "Arps": ["arp", "arpegg"],
    "Acoustic": ["acoustic", "nylon", "steel string"],
    "Electric": ["electric", "distortion", "clean guitar"],
    "Phrases": ["phrase", "vocal phrase", "singing"],
    "Chops": ["chop", "vocal chop", "cut"],
    "FX": ["fx", "effect", "riser", "downlifter", "impact", "noise", "vinyl", "tape"],
}

# Reverse lookup: keyword -> category path
KEYWORD_TO_PATH = {}
for category, keywords in CATEGORIES.items():
    for keyword in keywords:
        KEYWORD_TO_PATH[keyword.lower()] = category


def detect_bpm(filename):
    """Extract BPM from filename if present"""
    # Look for patterns like "120bpm", "120_bpm", "120-bpm", "bpm120"
    match = re.search(r'(\d{2,3})[\s_-]?bpm|bpm[\s_-]?(\d{2,3})', filename.lower())
    if match:
        return match.group(1) or match.group(2)
    return "na"


def detect_key(filename):
    """Extract musical key from filename if present"""
    # Common key patterns: Cmaj, C_Major, Dmin, D_minor, etc.
    keys = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    modifiers = ['maj', 'major', 'min', 'minor', 'm']

    filename_lower = filename.lower()

    for key in keys:
        for mod in modifiers:
            pattern = f"{key.lower()}[\s_-]?{mod}"
            if re.search(pattern, filename_lower):
                return f"{key}{mod[:3]}"  # e.g., "Cmaj" or "Dmin"

    return "na"


def categorize_sample(filename):
    """Determine category based on filename keywords"""
    filename_lower = filename.lower()

    for keyword, category in KEYWORD_TO_PATH.items():
        if keyword in filename_lower:
            # Map category to folder path
            if category in ["Kicks", "Snares", "HiHats", "Cymbals", "Toms", "Percussion"]:
                return f"Drums/{category}"
            elif category == "Loops" and "drum" in filename_lower:
                return "Drums/Loops"
            elif category in ["Bass"]:
                return "Bass/Synth"  # Default to synth, can be refined
            elif category in ["Pads", "Leads", "Keys", "Arps"]:
                return f"Synths/{category}"
            elif category in ["Acoustic", "Electric"]:
                return f"Guitars/{category}"
            elif category in ["Phrases", "Chops"]:
                return f"Vocals/{category}"
            elif category == "FX":
                return "FX/Atmosphere"
            elif category == "Loops":
                return "Loops/Full"

    # Default: uncategorized
    return "_Uncategorized"


def sanitize_description(filename):
    """Extract clean description from filename"""
    # Remove extension, BPM, key, numbers, special chars
    desc = Path(filename).stem

    # Remove BPM
    desc = re.sub(r'\d{2,3}[\s_-]?bpm|bpm[\s_-]?\d{2,3}', '', desc, flags=re.IGNORECASE)

    # Remove key
    desc = re.sub(r'[A-G]b?[\s_-]?(maj|major|min|minor|m)', '', desc, flags=re.IGNORECASE)

    # Remove numbers at end
    desc = re.sub(r'[\s_-]?\d{1,3}$', '', desc)

    # Clean up separators
    desc = re.sub(r'[\s_-]+', '_', desc)
    desc = desc.strip('_')

    # Capitalize first letter of each word
    desc = '_'.join(word.capitalize() for word in desc.split('_'))

    return desc or "Sample"


def organize_sample(source_file):
    """Move and rename sample to proper location"""
    filename = source_file.name
    bpm = detect_bpm(filename)
    key = detect_key(filename)
    category_path = categorize_sample(filename)
    description = sanitize_description(filename)

    # Determine file extension
    ext = source_file.suffix.lower()
    if ext not in ['.wav', '.aiff', '.mp3', '.flac', '.ogg']:
        return False  # Skip non-audio files

    # Build destination path
    dest_dir = SAMPLES_DIR / category_path
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Find next available number for this sample type
    existing = list(dest_dir.glob(f"{bpm}_{key}_*{description}*{ext}"))
    number = len(existing) + 1

    # Determine sample type from category
    sample_type = Path(category_path).name

    # Build new filename: [BPM]_[Key]_[Type]_[Description]_[Number].wav
    new_filename = f"{bpm}_{key}_{sample_type}_{description}_{number:02d}{ext}"
    dest_file = dest_dir / new_filename

    # Copy file (don't delete original yet)
    try:
        shutil.copy2(source_file, dest_file)
        return True, dest_file
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False, None


def main():
    """Main organizer"""
    print("=" * 60)
    print("SAMPLE ORGANIZER")
    print("Sorting Freesound downloads into library structure")
    print("=" * 60)

    if not DOWNLOAD_DIR.exists():
        print(f"❌ Download directory not found: {DOWNLOAD_DIR}")
        print(f"   Run freesound_downloader.py first")
        return

    # Find all audio files
    audio_extensions = ['*.wav', '*.aiff', '*.mp3', '*.flac', '*.ogg']
    all_files = []
    for ext in audio_extensions:
        all_files.extend(DOWNLOAD_DIR.rglob(ext))

    print(f"\n📁 Found {len(all_files)} audio files in {DOWNLOAD_DIR}\n")

    if not all_files:
        print("⚠️  No audio files to organize")
        return

    # Organize each file
    success = 0
    for i, source_file in enumerate(all_files, 1):
        print(f"[{i}/{len(all_files)}] {source_file.name}")

        result = organize_sample(source_file)
        if result and result[0]:
            success += 1
            dest_file = result[1]
            print(f"  ✅ → {dest_file.relative_to(SAMPLES_DIR)}")
        else:
            print(f"  ⏭️  Skipped")

    print("\n" + "=" * 60)
    print(f"✅ ORGANIZED {success}/{len(all_files)} SAMPLES")
    print("=" * 60)
    print(f"\nSamples organized in: {SAMPLES_DIR}")
    print(f"\nNext step: Run sample_cataloger.py to build searchable database")


if __name__ == "__main__":
    main()
