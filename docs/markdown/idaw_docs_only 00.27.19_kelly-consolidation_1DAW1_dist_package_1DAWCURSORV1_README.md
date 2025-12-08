# 1DAWCURSORV1 - iDAW Desktop Application

## Quick Start

This is the frontend build of iDAW. To run it:

### Option 1: Simple HTTP Server (Recommended)

```bash
# Python 3
python3 -m http.server 8000

# Or Node.js
npx serve -s . -l 8000
```

Then open http://localhost:8000 in your browser.

### Option 2: Open Directly

Open `index.html` in a modern web browser (Chrome, Firefox, Edge).

## Features

- **Side A**: Professional DAW interface with:
  - Enhanced mixer with advanced sliders
  - Waveform visualization
  - Timeline and transport controls
  - EQ and mix console

- **Side B**: Therapeutic interface with:
  - Auto-prompt generation
  - Brushstroke animations
  - Doodle canvas
  - WebGL shader visualizations
  - Emotion wheel
  - Rule breaker
  - Vocal synth

## Backend Setup

For full functionality, start the Python backend:

```bash
cd /path/to/project
python -m music_brain.api
```

The API will run on http://localhost:8000

## System Requirements

- Modern web browser (Chrome 90+, Firefox 88+, Safari 14+, Edge 90+)
- WebGL support for shader visualizations
- JavaScript enabled

## Build Information

- Build Date: $(date)
- Version: 1.0.0
- Frontend: React + TypeScript + Vite
- Framework: Tauri (requires Rust for full desktop build)

## Notes

This is the frontend-only build. For the full desktop application with embedded Python backend, you'll need to build with Tauri (requires Rust 1.83+ for current dependencies).
