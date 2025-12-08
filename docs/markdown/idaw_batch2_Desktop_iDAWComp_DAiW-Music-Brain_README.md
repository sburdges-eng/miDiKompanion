# DAiW - Digital Audio Intelligent Workstation

> A Python toolkit for music production intelligence: groove extraction, chord analysis, arrangement generation, and AI-assisted songwriting.
>
> **Philosophy: "Interrogate Before Generate"** — The tool shouldn't finish art for people. It should make them braver.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

DAiW (Digital Audio intelligent Workstation) combines:

- **Music Brain** - Python analysis engine for MIDI/audio
- **Intent Schema** - Three-phase deep interrogation for songwriting
- **Rule-Breaking Engine** - Intentional theory violations for emotional impact
- **Vault** - Knowledge base of songwriting guides and theory references
- **CLI** - Command-line tools for groove extraction, chord analysis, and AI-assisted composition

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DAiW-Music-Brain.git
cd DAiW-Music-Brain

# Install as package
pip install -e .
```

## Quick Start

### Intent-Based Song Generation (New!)

```bash
# Create a new song intent template
daiw intent new --title "My Song" --output my_intent.json

# Edit the JSON to fill in your emotional intent...

# Process intent to generate musical elements
daiw intent process my_intent.json

# Get suggestions for rules to break
daiw intent suggest grief

# List all rule-breaking options
daiw intent list
```

### Command Line Interface

```bash
# Extract groove from a MIDI file
daiw extract drums.mid

# Apply genre groove template
daiw apply --genre funk track.mid

# Analyze chord progression
daiw analyze --chords song.mid

<<<<<<< Current (Your changes)
# Analyze audio file (BPM, key, feel)
daiw analyze-audio song.wav --chords

# Compare two audio files
daiw compare-audio track1.wav track2.wav --detailed

# Batch analyze multiple files
daiw batch-analyze *.wav --format csv -o analysis.csv

# Export audio features
daiw export-features song.wav -o features.json --include-segments
=======
# Analyze audio file (tempo/key/spectrum)
daiw analyze-audio vocal_demo.wav

# Auto-tune and modulate vocals
daiw voice tune vocals.wav --preset transparent --key F --mode minor
daiw voice modulate vocals.wav --preset distant_radio
daiw voice synthesize "i don't feel safe" "60,62,64,65" -p guide_vulnerable
daiw voice speak "DAiW introduces Neon Ghost" -p guide_confident

# Offline chatbot (requires local LLM model)
daiw chatbot --model ~/Models/mistral-7b-instruct.gguf --persona "You are DAiW's companion."
>>>>>>> Incoming (Background Agent changes)

# Diagnose harmonic issues
daiw diagnose "F-C-Am-Dm"

# Generate reharmonizations
daiw reharm "F-C-Am-Dm" --style jazz

# Interactive teaching mode
daiw teach rulebreaking
```

### Python API

```python
from music_brain.groove import extract_groove, apply_groove
from music_brain.structure import analyze_chords
from music_brain.session import (
    CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints,
    suggest_rule_break
)
from music_brain.session.intent_processor import process_intent

# Create song intent
intent = CompleteSongIntent(
    song_root=SongRoot(
        core_event="Finding someone I loved after they chose to leave",
        core_resistance="Fear of making it about me",
        core_longing="To process without exploiting the loss",
    ),
    song_intent=SongIntent(
        mood_primary="Grief",
        mood_secondary_tension=0.3,
        vulnerability_scale="High",
        narrative_arc="Slow Reveal",
    ),
    technical_constraints=TechnicalConstraints(
        technical_key="F",
        technical_mode="major",
        technical_rule_to_break="HARMONY_ModalInterchange",
        rule_breaking_justification="Bbm makes hope feel earned and bittersweet",
    ),
)

# Process intent to generate elements
result = process_intent(intent)
print(result['harmony'].chords)  # ['F', 'C', 'Bbm', 'F']
```

### Audio Analysis (New!)

```python
from music_brain.audio import AudioAnalyzer

analyzer = AudioAnalyzer()
analysis = analyzer.analyze_file("reference_track.wav")

print(analysis.bpm)          # Tempo estimate
print(analysis.key, analysis.mode)
print(analysis.spectral.harmonic_content)
print("Chords:", analysis.chords)

# Requires optional deps
# pip install -e .[audio]
```

### MCP Server (Model Context Protocol)

Expose DAiW tools to copilots via the MCP FastMCP server:

```bash
# Install optional dependency group
pip install -e .[mcp]

# Start FastMCP server (stdio transport by default)
python -m daiw_mcp.server
```

Registered tool families:

- `daiw.analyze_chords`, `daiw.analyze_progression`, `daiw.diagnose_chords`,
  `daiw.generate_harmony`, `daiw.suggest_reharmonization`
- `daiw.extract_groove`, `daiw.apply_groove`, `daiw.humanize_midi`
- `daiw.intent.create_template`, `daiw.intent.process`, `daiw.intent.validate`,
  `daiw.intent.suggest_rulebreaks`
- `daiw.therapy.session`

All tools return JSON responses plus optional base64-encoded MIDI payloads, mirroring the
Python API so desktops, CLIs, and MCP clients stay consistent.

## The Intent Schema

### Three-Phase Deep Interrogation

#### Phase 0: Core Wound/Desire

- `core_event` — What happened?
- `core_resistance` — What holds you back from saying it?
- `core_longing` — What do you want to feel?
- `core_stakes` — What's at risk?
- `core_transformation` — How should you feel when done?

#### Phase 1: Emotional Intent

- `mood_primary` — Dominant emotion
- `mood_secondary_tension` — Internal conflict (0.0-1.0)
- `imagery_texture` — Visual/tactile quality
- `vulnerability_scale` — Emotional exposure level
- `narrative_arc` — Structural emotion pattern

#### Phase 2: Technical Implementation

- `technical_genre` — Genre/style
- `technical_key` — Musical key
- `technical_rule_to_break` — Intentional rule violation
- `rule_breaking_justification` — WHY break this rule

## Audio Analysis Tools

DAiW includes comprehensive audio analysis capabilities for working with audio files:

### Analyze Audio File

```bash
# Basic analysis (BPM, key, features)
daiw analyze-audio song.wav

# Include chord detection
daiw analyze-audio song.wav --chords

# Analyze with segments
daiw analyze-audio song.wav --segments --segment-count 8

# Save results to JSON
daiw analyze-audio song.wav -o analysis.json
```

### Compare Audio Files

Compare two audio files side-by-side for BPM, key, and feel:

```bash
# Basic comparison
daiw compare-audio original.wav remix.wav

# Detailed feature comparison
daiw compare-audio track1.wav track2.wav --detailed

# Save comparison to JSON
daiw compare-audio song1.wav song2.wav -o comparison.json
```

### Batch Analysis

Process multiple audio files or entire directories:

```bash
# Analyze specific files
daiw batch-analyze song1.wav song2.wav song3.wav

# Analyze directory (non-recursive)
daiw batch-analyze /path/to/audio/

# Analyze recursively
daiw batch-analyze /path/to/audio/ --recursive

# Export to CSV
daiw batch-analyze *.wav --format csv -o catalog.csv

# Export to JSON
daiw batch-analyze *.wav --format json -o catalog.json
```

### Export Features

Export comprehensive audio features for further analysis:

```bash
# Export to JSON (default)
daiw export-features song.wav -o features.json

# Export to CSV
daiw export-features song.wav -o features.csv --format csv

# Include segment analysis
daiw export-features song.wav -o features.json --include-segments

# Include chord detection
daiw export-features song.wav -o features.json --include-chords

# Full export
daiw export-features song.wav -o full.json --include-segments --include-chords
```

**Note:** Audio analysis requires `librosa`. Install with: `pip install librosa numpy scipy`

## Rule-Breaking Categories

### Harmony

| Rule | Effect | Use When |
|------|--------|----------|
| `HARMONY_AvoidTonicResolution` | Unresolved yearning | Grief, longing |
| `HARMONY_ModalInterchange` | Bittersweet color | Making hope feel earned |
| `HARMONY_ParallelMotion` | Power, defiance | Anger, punk energy |

### Rhythm

| Rule | Effect | Use When |
|------|--------|----------|
| `RHYTHM_ConstantDisplacement` | Off-kilter anxiety | Before a dramatic shift |
| `RHYTHM_TempoFluctuation` | Organic breathing | Intimacy, vulnerability |

### Production

| Rule | Effect | Use When |
|------|--------|----------|
| `PRODUCTION_BuriedVocals` | Dissociation, texture | Dreams, distance |
| `PRODUCTION_PitchImperfection` | Emotional honesty | Raw vulnerability |

## Project Structure

```text
DAiW-Music-Brain/
├── music_brain/              # Python analysis package
│   ├── groove/               # Groove extraction & application
│   ├── structure/            # Chord, section, progression analysis
│   ├── audio/                # Audio feel analysis
│   ├── session/              # Intent schema, teaching, interrogator
│   │   ├── intent_schema.py  # Three-phase intent system
│   │   ├── intent_processor.py # Rule-breaking execution
│   │   ├── teaching.py       # Interactive lessons
│   │   └── interrogator.py   # Song interrogation
│   ├── utils/                # MIDI I/O, instruments, PPQ
│   ├── daw/                  # DAW integration
│   └── data/                 # JSON datasets
│       ├── song_intent_schema.yaml
│       ├── song_intent_examples.json
│       └── genre_pocket_maps.json
│
├── vault/                    # Knowledge base (Obsidian-compatible)
│   └── Songwriting_Guides/
│       ├── song_intent_schema.md
│       ├── rule_breaking_practical.md
│       └── rule_breaking_masterpieces.md
│
└── tests/                    # Test suite
```

## Features

### Intent-Based Generation

- Deep interrogation before technical decisions
- Emotion-to-music mapping
- Intentional rule-breaking with justification
- Phase validation for completeness

### Groove Analysis

- Extract timing deviations (swing, push/pull)
- Velocity contours and accent patterns
- Genre-specific templates
- Cross-DAW PPQ normalization

### Chord & Harmony

- Roman numeral analysis
- Borrowed chord detection
- Modal interchange identification
- Reharmonization suggestions

### Teaching Module

- Interactive lessons on rule-breaking
- Emotion-specific technique suggestions
- Production philosophy guidance

## Requirements

- Python 3.9+
- mido (MIDI I/O)
- numpy (numerical analysis)

Optional:

- librosa (audio analysis)
- music21 (advanced theory)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built for musicians who think in sound, not spreadsheets
- Inspired by the lo-fi bedroom recording philosophy
- **"The wrong note played with conviction is the right note."**
