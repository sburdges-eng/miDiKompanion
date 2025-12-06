# iDAW - Intelligent Digital Audio Workstation

**Kelly Project Unified Repository**

## Overview

iDAW (Intelligent Digital Audio Workstation), also known as the Kelly Project, is an ambitious therapeutic music generation platform that combines professional DAW features with emotional intelligence and AI-powered music creation.

## Core Concept: Side A / Side B

Inspired by cassette tapes, iDAW features a unique dual-interface:

- **Side A**: Traditional DAW interface with professional production tools
- **Side B**: Therapeutic/creative interface with emotion-based generation

## Key Features

### Professional DAW (Side A)
- Full-featured mixer with channel strips
- Timeline and transport controls
- VU meters and audio visualization
- Plugin hosting capabilities
- MIDI and audio recording

### Therapeutic Engine (Side B)
- **6×6×6 Emotion Thesaurus**: 216 emotion nodes for precise emotional targeting
- **Emotion Wheel**: Visual emotion selection interface
- **GhostWriter**: AI-powered lyric and melody generation
- **Interrogator**: Conversational music creation assistant
- **Dreamstate Mode**: Experimental/therapeutic music exploration
- **Parrot Feature**: Learning and mimicking musical styles

### Music Brain
- Intelligent music theory engine
- Chord progression generation
- Scale and harmony analysis
- Genre-specific templates

### Music Vault
- Centralized sample and preset management
- Audio cataloging and tagging
- Smart search and retrieval

## Tech Stack

- **Frontend**: React 18, TypeScript, Tailwind CSS
- **Backend**: Tauri 2.0, Rust
- **Audio Engine**: Penta-core (C++ audio processing)
- **AI Integration**: Python-based music generation
- **Build System**: Vite

## Project Structure

```
iDAW/
├── src/                       # React/TypeScript frontend
├── src-tauri/                 # Rust backend
├── emotion_thesaurus/         # 6×6×6 emotion system
├── music_brain/               # Core music intelligence
├── vault/                     # Sample and preset management
├── penta_core/                # Audio processing engine
├── cpp_music_brain/           # C++ music algorithms
├── docs/                      # Technical documentation
├── Obsidian_Documentation/    # Knowledge base
├── Production_Workflows/      # Guides and templates
└── Python_Tools/              # Utilities and scripts
```

## Quick Start

### Prerequisites
- Node.js 18+
- Rust 1.70+
- Python 3.9+
- CMake (for C++ components)

### Installation

```bash
# Clone the repository
git clone https://github.com/sburdges-eng/1DAW1.git
cd 1DAW1

# Install dependencies
npm install

# Build and run
npm run tauri dev
```

## Documentation

- **[CONSOLIDATION_LOG.md](./CONSOLIDATION_LOG.md)**: How this repo was consolidated
- **[Obsidian_Documentation/](./Obsidian_Documentation/)**: Comprehensive knowledge base
- **[Production_Workflows/](./Production_Workflows/)**: Production guides and workflows
- **[docs/](./docs/)**: Technical documentation

## Development

This project was consolidated from 5 separate repositories:
- iDAW (base)
- DAiW-Music-Brain (emotion system)
- penta-core (audio engine)
- iDAWi (experimental)
- 1DAW1 (target)

See [CONSOLIDATION_LOG.md](./CONSOLIDATION_LOG.md) for details.

## License

MIT License - See LICENSE file for details

## Author

Sean Burdges ([@sburdges-eng](https://github.com/sburdges-eng))

## Acknowledgments

Built with Claude AI assistance for code generation, architecture, and consolidation.

---

**Status**: Active Development (Consolidated 2025-12-06)
