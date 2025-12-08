# iDAW - Intelligent Digital Audio Workstation

> **"Interrogate Before Generate"** - A therapeutic music creation tool that helps you explore emotions and generate music that captures your feelings.

## Overview

iDAW is a dual-interface music creation application:

- **Side A**: Professional DAW with timeline, mixer, and transport controls
- **Side B**: Therapeutic interface with emotion exploration and conversational music generation

The core philosophy is **"Interrogate Before Generate"** - helping users explore their emotions through conversation before generating musically appropriate compositions.

## Key Features

### 🎭 Emotion System (6×6×6 Thesaurus)

- 216 emotion nodes organized in a hierarchical structure
- Base emotions: Sad, Happy, Angry, Fear, Disgust, Surprise
- Intensity levels: Low → Moderate → High → Intense → Extreme → Overwhelming
- Specific emotions: Grief, Joy, Rage, Anxiety, Melancholy, etc.

### 🎵 Music Generation

- **Emotion-to-Music Mapping**: Automatically maps emotions to:
  - Musical modes (Aeolian for sad, Ionian for happy, etc.)
  - Tempo (65-160 BPM based on intensity)
  - Chord progressions (i-VI-III-VII for grief, etc.)
  - Dynamics and articulation
- **MIDI Export**: Generates playable MIDI files
- **Audio Preview**: Browser-based MIDI playback

### 💬 Interrogator System

- Multi-turn conversational interface
- Natural language emotion extraction
- Progressive questioning to build emotional profile
- Ready-to-generate detection

### 🎚️ Professional DAW Interface

- Timeline with track lanes and MIDI regions
- 8-channel mixer with VU meters
- Transport controls (play/pause/stop/record)
- Zoom and scroll controls
- Professional dark theme

## Installation

### Prerequisites

- Python 3.8+
- Node.js 18+
- Rust (for Tauri build)
- macOS (for Tauri app, or use browser mode)

### Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd iDAW
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Node.js dependencies**

   ```bash
   npm install
   ```

4. **Start the Music Brain API**

   ```bash
   python -m music_brain.api
   ```

   The API will run on `http://127.0.0.1:8000`

5. **Run the frontend**

   ```bash
   # Browser mode
   npm run dev

   # Or build Tauri app
   npm run tauri dev
   ```

## Usage

### Side B: Therapeutic Interface

1. **Load Emotions**
   - Click "Load Emotions" to load the 6×6×6 emotion thesaurus

2. **Select Emotion**
   - Choose base emotion (sad, happy, etc.)
   - Select intensity level
   - Pick specific emotion (grief, joy, etc.)

3. **Generate Music**
   - Click "Generate Music"
   - MIDI file is generated based on your emotion selection
   - Download or preview in browser

4. **Use Interrogator** (Alternative)
   - Start a conversation about how you're feeling
   - Answer questions naturally
   - System extracts emotions and generates music when ready

### Side A: Professional DAW

1. **View Timeline**
   - Generated MIDI appears as regions on tracks
   - Zoom and scroll to navigate

2. **Use Mixer**
   - Adjust volume, pan, mute/solo per channel
   - Monitor levels with VU meters

3. **Transport Controls**
   - Play, pause, stop, record
   - Adjust tempo and time signature

## Architecture

```
┌─────────────────┐
│   React/Tauri   │  Frontend (TypeScript/React)
│      UI          │
└────────┬────────┘
         │
         │ HTTP/WebSocket
         │
┌────────▼────────┐
│  Music Brain    │  Python FastAPI
│      API        │  - /generate
└────────┬────────┘  - /interrogate
         │          - /emotions
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────┐
│Emotion│ │  MIDI   │
│Mapper │ │Generator│
└───────┘ └─────────┘
```

## Project Structure

```
iDAW/
├── music_brain/          # Python backend
│   ├── api.py           # FastAPI server
│   ├── emotion_mapper.py # Emotion→Music mapping
│   ├── interrogator.py   # Conversational system
│   └── session/         # Song generation
├── src/                  # React frontend
│   ├── components/      # UI components
│   │   ├── EmotionWheel.tsx
│   │   ├── InterrogatorChat.tsx
│   │   ├── AudioPreview.tsx
│   │   ├── Timeline.tsx
│   │   └── Mixer.tsx
│   └── App.tsx
├── src-tauri/            # Tauri backend (Rust)
├── emotion_thesaurus/    # 6×6×6 emotion data
└── docs/                 # Documentation
```

## API Endpoints

### `POST /generate`

Generate music from emotional intent.

**Request:**

```json
{
  "intent": {
    "base_emotion": "sad",
    "intensity": "intense",
    "specific_emotion": "grief"
  }
}
```

**Response:**

```json
{
  "success": true,
  "midi_data": "base64_encoded_midi",
  "midi_path": "/tmp/file.mid",
  "music_config": {
    "key": "F",
    "mode": "Aeolian",
    "tempo": 130,
    "progression": ["i", "VI", "III", "VII"]
  }
}
```

### `POST /interrogate`

Conversational emotion exploration.

**Request:**

```json
{
  "message": "I feel sad about losing someone",
  "session_id": "optional-session-id"
}
```

**Response:**

```json
{
  "ready": false,
  "question": "How intense is this feeling?",
  "session_id": "session-id"
}
```

### `GET /emotions`

Get the full 6×6×6 emotion thesaurus.

## Development

### Running Tests

```bash
# Python tests
python -m pytest tests_music-brain/

# TypeScript compilation
npm run build
```

### Code Style

- Python: Follow PEP 8
- TypeScript: ESLint configuration
- React: Functional components with hooks

## Contributing

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for contributor guidelines.

## License

[Add license information]

## Acknowledgments

- Emotion thesaurus based on psychological research
- Music theory mappings inspired by emotional music analysis
- Built with FastAPI, React, Tauri, and Tone.js

## Roadmap

- [ ] Rust audio engine integration (CPAL)
- [ ] Voice synthesis for lyrics
- [ ] Advanced MIDI editing
- [ ] Audio effects and processing
- [ ] Export to audio formats (WAV, MP3)
- [ ] Cloud sync and collaboration

## Support

For issues and questions, please open a GitHub issue.

---

**"When I found you sleeping, everything felt right"** - Kelly
