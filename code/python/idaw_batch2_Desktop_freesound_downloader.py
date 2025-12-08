#!/usr/bin/env python3
"""
Freesound Pack Downloader
Downloads curated packs from Freesound.org for lo-fi bedroom production
"""

import os
import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path

# Freesound API setup
API_KEY_FILE = Path.home() / ".freesound_api_key"
DOWNLOAD_DIR = Path.home() / "Downloads" / "Freesound_Packs"
SAMPLES_DIR = Path.home() / "Music" / "Samples"

# Curated pack IDs for lo-fi/bedroom emo production
PACK_LIST = {
    "drums_acoustic": {
        "pack_id": 14856,
        "name": "Acoustic Drum Kit Clean Samples",
        "user": "afleetingspeck",
    },
    "drums_vintage": {
        "pack_id": 6,
        "name": "Vintage Drum Machine Samples",
        "user": "suburban_grilla",
    },
    "drums_brush": {
        "pack_id": 28587,
        "name": "Brush Drum Samples",
        "user": "Mhistorically",
    },
    "room_ambience": {
        "pack_id": 10855,
        "name": "Room Tones and Ambiences",
        "user": "klankbeeld",
    },
    "guitar_notes": {
        "pack_id": 8647,
        "name": "Acoustic Guitar Single Notes",
        "user": "MTG",
    },
    "guitar_harmonics": {
        "pack_id": 3336,
        "name": "Guitar Harmonics",
        "user": "ERH",
    },
    "guitar_strums": {
        "pack_id": 18534,
        "name": "Acoustic Guitar Strums",
        "user": "HerbertBoland",
    },
    "bass_jazz": {
        "pack_id": 13115,
        "name": "Fender Jazz Bass Samples",
        "user": "FrusciMike",
    },
    "bass_upright": {
        "pack_id": 619,
        "name": "Upright Bass Pizzicato",
        "user": "No_Go",
    },
    "vocal_human": {
        "pack_id": 14808,
        "name": "Vocal Samples Human Voice",
        "user": "pushtobreak",
    },
    "vocal_whisper": {
        "pack_id": 30032,
        "name": "Whisper Samples",
        "user": "HerbertBoland",
    },
    "vinyl_fx": {
        "pack_id": 17926,
        "name": "Vinyl Crackle and Noise",
        "user": "OldFritz",
    },
    "tape_fx": {
        "pack_id": 15003,
        "name": "Cassette Tape Artifacts",
        "user": "j1987",
    },
    "forest_ambience": {
        "pack_id": 5246,
        "name": "Forest Ambience Pack",
        "user": "klankbeeld",
    },
}


def load_api_key():
    """Load Freesound API key from file"""
    if not API_KEY_FILE.exists():
        print(f"❌ API key not found at: {API_KEY_FILE}")
        print("\nTo get an API key:")
        print("1. Go to: https://freesound.org/apiv2/apply/")
        print("2. Create a free API key")
        print(f"3. Save it: echo 'YOUR_KEY' > {API_KEY_FILE}")
        print(f"4. Secure it: chmod 600 {API_KEY_FILE}")
        sys.exit(1)

    with open(API_KEY_FILE) as f:
        api_key = f.read().strip()

    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print(f"❌ Please update {API_KEY_FILE} with your real API key")
        sys.exit(1)

    return api_key


def get_pack_info(pack_id, api_key):
    """Get pack information from Freesound API"""
    url = f"https://freesound.org/apiv2/packs/{pack_id}/?token={api_key}"

    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        return data
    except Exception as e:
        print(f"❌ Error fetching pack {pack_id}: {e}")
        return None


def get_pack_sounds(pack_id, api_key):
    """Get list of sounds in a pack"""
    url = f"https://freesound.org/apiv2/packs/{pack_id}/sounds/?token={api_key}&fields=id,name,previews"

    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        return data.get("results", [])
    except Exception as e:
        print(f"❌ Error fetching sounds for pack {pack_id}: {e}")
        return []


def download_sound(sound_id, sound_name, preview_url, pack_name, api_key):
    """Download a single sound preview (HQ MP3)"""
    pack_dir = DOWNLOAD_DIR / pack_name.replace(" ", "_")
    pack_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    safe_name = "".join(c for c in sound_name if c.isalnum() or c in (' ', '-', '_')).strip()
    output_file = pack_dir / f"{safe_name}.mp3"

    if output_file.exists():
        print(f"  ⏭️  Skip: {safe_name} (already downloaded)")
        return True

    try:
        urllib.request.urlretrieve(preview_url, output_file)
        print(f"  ✅ Downloaded: {safe_name}")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {safe_name} - {e}")
        return False


def download_pack(pack_key, pack_info, api_key):
    """Download all sounds from a pack"""
    pack_id = pack_info["pack_id"]
    pack_name = pack_info["name"]

    print(f"\n📦 {pack_name}")
    print(f"   by {pack_info['user']}")

    # Get pack metadata
    metadata = get_pack_info(pack_id, api_key)
    if not metadata:
        print(f"   ⚠️  Could not fetch pack info")
        return

    print(f"   {metadata.get('num_sounds', 0)} sounds")

    # Get sounds list
    sounds = get_pack_sounds(pack_id, api_key)
    if not sounds:
        print(f"   ⚠️  No sounds found")
        return

    # Download each sound preview
    success = 0
    for sound in sounds:
        sound_id = sound["id"]
        sound_name = sound["name"]
        preview_url = sound["previews"]["preview-hq-mp3"]

        if download_sound(sound_id, sound_name, preview_url, pack_name, api_key):
            success += 1

    print(f"   ✅ Downloaded {success}/{len(sounds)} sounds\n")


def main():
    """Main download orchestrator"""
    print("=" * 60)
    print("FREESOUND PACK DOWNLOADER")
    print("Lo-Fi Bedroom Emo Starter Kit")
    print("=" * 60)

    # Load API key
    api_key = load_api_key()
    print(f"✅ API key loaded\n")

    # Create download directory
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Download location: {DOWNLOAD_DIR}\n")

    # Download each pack
    print(f"📥 Downloading {len(PACK_LIST)} packs...\n")

    for pack_key, pack_info in PACK_LIST.items():
        download_pack(pack_key, pack_info, api_key)

    print("=" * 60)
    print("✅ DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nSamples saved to: {DOWNLOAD_DIR}")
    print(f"\nNext steps:")
    print(f"1. Review samples in {DOWNLOAD_DIR}")
    print(f"2. Run organize_samples.py to sort into categories")
    print(f"3. Run sample_cataloger.py to build searchable database")
    print()


if __name__ == "__main__":
    main()
