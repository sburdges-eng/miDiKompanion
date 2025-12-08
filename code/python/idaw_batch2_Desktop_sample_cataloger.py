#!/usr/bin/env python3
"""
Sample Cataloger
Creates searchable JSON database of all samples
Enables queries like: "Find all kicks in A minor at 120 BPM"
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime

SAMPLES_DIR = Path.home() / "Music" / "Samples"
CATALOG_FILE = SAMPLES_DIR / "sample_catalog.json"


def parse_filename(filename):
    """Parse standardized filename: [BPM]_[Key]_[Type]_[Description]_[Number].ext"""
    stem = Path(filename).stem
    parts = stem.split('_')

    if len(parts) >= 5:
        return {
            "bpm": parts[0] if parts[0] != "na" else None,
            "key": parts[1] if parts[1] != "na" else None,
            "type": parts[2],
            "description": '_'.join(parts[3:-1]),
            "number": parts[-1],
        }
    else:
        # Fallback for non-standard naming
        return {
            "bpm": None,
            "key": None,
            "type": "Unknown",
            "description": stem,
            "number": "01",
        }


def get_file_hash(filepath):
    """Generate MD5 hash for file deduplication"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def scan_samples():
    """Scan all samples and build catalog"""
    print("=" * 60)
    print("SAMPLE CATALOGER")
    print("Building searchable database")
    print("=" * 60)

    if not SAMPLES_DIR.exists():
        print(f"❌ Samples directory not found: {SAMPLES_DIR}")
        return

    # Supported audio formats
    audio_extensions = ['.wav', '.aiff', '.mp3', '.flac', '.ogg', '.aif']

    # Scan all subdirectories
    samples = []
    categories = {}

    print(f"\n📁 Scanning: {SAMPLES_DIR}\n")

    for ext in audio_extensions:
        for filepath in SAMPLES_DIR.rglob(f'*{ext}'):
            # Skip the root level and only process samples in subdirectories
            if filepath.parent == SAMPLES_DIR:
                continue

            # Parse filename
            metadata = parse_filename(filepath.name)

            # Determine category from path
            relative_path = filepath.relative_to(SAMPLES_DIR)
            category = str(relative_path.parent).replace('/', ' > ')

            # Build sample entry
            sample = {
                "filename": filepath.name,
                "path": str(filepath),
                "relative_path": str(relative_path),
                "category": category,
                "type": metadata["type"],
                "description": metadata["description"],
                "bpm": metadata["bpm"],
                "key": metadata["key"],
                "number": metadata["number"],
                "extension": filepath.suffix.lower(),
                "size_bytes": filepath.stat().st_size,
                "size_mb": round(filepath.stat().st_size / 1024 / 1024, 2),
                "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
                "hash": get_file_hash(filepath),
            }

            samples.append(sample)

            # Track category stats
            if category not in categories:
                categories[category] = 0
            categories[category] += 1

            print(f"✅ {relative_path}")

    # Build catalog
    catalog = {
        "generated": datetime.now().isoformat(),
        "total_samples": len(samples),
        "categories": categories,
        "samples": samples,
    }

    # Save to JSON
    with open(CATALOG_FILE, 'w') as f:
        json.dump(catalog, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✅ CATALOGED {len(samples)} SAMPLES")
    print("=" * 60)
    print(f"\nDatabase saved: {CATALOG_FILE}")
    print(f"Total size: {sum(s['size_mb'] for s in samples):.2f} MB\n")

    print("Category breakdown:")
    for category, count in sorted(categories.items()):
        print(f"  • {category}: {count} samples")

    print("\n📖 Query examples:")
    print(f"  python3 {Path(__file__).parent / 'search_samples.py'} --type Kick")
    print(f"  python3 {Path(__file__).parent / 'search_samples.py'} --bpm 120")
    print(f"  python3 {Path(__file__).parent / 'search_samples.py'} --key Dmin")
    print()


if __name__ == "__main__":
    scan_samples()
