#!/usr/bin/env python3
"""
CMU Pronouncing Dictionary Setup Script
========================================
Downloads and processes the CMU Pronouncing Dictionary for use in PhonemeConverter.
"""

import argparse
import urllib.request
import re
from pathlib import Path
import json


def download_cmu_dict(output_path: Path) -> bool:
    """Download CMU Pronouncing Dictionary."""
    url = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
    
    print(f"Downloading CMU Pronouncing Dictionary from {url}...")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def parse_cmu_dict(cmu_file: Path) -> dict:
    """Parse CMU dictionary file into word -> phonemes mapping."""
    word_dict = {}
    
    print(f"Parsing CMU dictionary: {cmu_file}")
    
    with open(cmu_file, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';;;'):
                continue
            
            # Format: WORD  PHONEME1 PHONEME2 ...
            parts = line.split()
            if len(parts) < 2:
                continue
            
            word = parts[0].lower()
            # Remove stress markers (1, 2, 0)
            phonemes = []
            for p in parts[1:]:
                # Remove stress markers
                p_clean = re.sub(r'[012]', '', p)
                if p_clean:
                    phonemes.append(f"/{p_clean.lower()}/")
            
            if phonemes:
                # Handle alternate pronunciations (marked with (1), (2), etc.)
                if '(' in word:
                    base_word = re.sub(r'\(\d+\)', '', word)
                    if base_word not in word_dict:
                        word_dict[base_word] = []
                    word_dict[base_word].append(phonemes)
                else:
                    word_dict[word] = phonemes
    
    print(f"✓ Parsed {len(word_dict)} words")
    return word_dict


def convert_arpabet_to_ipa(arpabet_phoneme: str) -> str:
    """Convert ARPABET phoneme to IPA symbol."""
    # ARPABET to IPA mapping
    arpabet_to_ipa = {
        'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ',
        'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'ɜr',
        'EY': 'eɪ', 'F': 'f', 'G': 'g', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i',
        'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ',
        'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'ʃ',
        'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w',
        'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
    }
    
    # Remove stress markers
    arpabet_clean = re.sub(r'[012]', '', arpabet_phoneme.upper())
    return arpabet_to_ipa.get(arpabet_clean, arpabet_clean.lower())


def create_phoneme_json(word_dict: dict, output_path: Path):
    """Create JSON file with word -> IPA phonemes mapping."""
    print(f"Creating phoneme JSON: {output_path}")
    
    # Convert ARPABET to IPA
    ipa_dict = {}
    for word, phonemes_arpabet in word_dict.items():
        if isinstance(phonemes_arpabet[0], list):
            # Multiple pronunciations
            ipa_dict[word] = [
                [convert_arpabet_to_ipa(p) for p in pron]
                for pron in phonemes_arpabet
            ]
        else:
            # Single pronunciation
            ipa_dict[word] = [convert_arpabet_to_ipa(p) for p in phonemes_arpabet]
    
    with open(output_path, 'w') as f:
        json.dump(ipa_dict, f, indent=2)
    
    print(f"✓ Created JSON with {len(ipa_dict)} words")


def main():
    parser = argparse.ArgumentParser(
        description="Setup CMU Pronouncing Dictionary for PhonemeConverter"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./data",
        help="Output directory for dictionary files"
    )
    parser.add_argument(
        "--download", "-d",
        action="store_true",
        help="Download CMU dictionary (if not already present)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmu_file = output_dir / "cmudict.dict"
    json_file = output_dir / "cmu_phonemes.json"
    
    # Download if requested or not present
    if args.download or not cmu_file.exists():
        if not download_cmu_dict(cmu_file):
            print("Failed to download CMU dictionary")
            return 1
    
    if not cmu_file.exists():
        print(f"CMU dictionary not found: {cmu_file}")
        print("Run with --download to download it")
        return 1
    
    # Parse and convert
    word_dict = parse_cmu_dict(cmu_file)
    create_phoneme_json(word_dict, json_file)
    
    print(f"\n✓ Setup complete!")
    print(f"  Dictionary file: {cmu_file}")
    print(f"  JSON file: {json_file}")
    print(f"\nTo use in PhonemeConverter, load the JSON file:")
    print(f"  converter.loadPhonemeDatabase(\"{json_file.absolute()}\")")
    
    return 0


if __name__ == "__main__":
    exit(main())
