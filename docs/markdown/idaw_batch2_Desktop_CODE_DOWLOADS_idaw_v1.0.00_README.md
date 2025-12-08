# iDAW - intelligent Digital Audio Workspace

## Version: 1.0.00

**"Interrogate Before Generate"**

---

## What is iDAW?

iDAW is an AI-powered music generation system that translates emotional states into complete musical compositions. Unlike traditional DAWs that start with sounds, iDAW starts with *feelings*.

---

## Quick Start

### Run the Ableton-style UI:
```bash
pip install streamlit music21 mido numpy
streamlit run idaw_ableton_ui.py
```

### Run the CLI:
```bash
python idaw_complete_pipeline.py "slow grief song, lo-fi, laid back"
```

### With audio generation:
```bash
python idaw_complete_pipeline.py "grief song" --audio
```

### With AKAI MPK Mini:
```bash
python idaw_complete_pipeline.py --mpk
```

---

## Pipeline

```
USER PROMPT → Interrogation → EmotionalState → MusicalParameters
→ Structure Generator → Harmony Engine → Melody Engine
→ Groove Engine → MIDI Builder → Audio Synthesis
→ Post-Processing → Final Audio
```

---

## Emotional Presets

| Preset | BPM | Mode | Feel | Dissonance |
|--------|-----|------|------|------------|
| grief | 60-82 | minor/dorian | behind | 30% |
| anxiety | 100-140 | phrygian | ahead | 60% |
| nostalgia | 70-90 | mixolydian | behind | 25% |
| anger | 120-160 | phrygian | ahead | 50% |
| calm | 60-80 | major/lydian | behind | 10% |
| hope | 80-110 | major/lydian | on | 20% |
| intimacy | 55-75 | dorian | behind | 20% |
| defiance | 100-130 | minor/phrygian | on | 40% |

---

## Rule Breaks

Intentional theory violations for emotional effect:

- **STRUCTURE_NonResolution** - grief, longing
- **HARMONY_ModalInterchange** - bittersweet, nostalgia
- **HARMONY_ParallelMotion** - power, defiance
- **RHYTHM_TempoFluctuation** - intimacy, vulnerability
- **PRODUCTION_PitchImperfection** - rawness

---

## Files

| File | Purpose |
|------|---------|
| `idaw_complete_pipeline.py` | Core engine (1,800+ lines) |
| `idaw_ableton_ui.py` | Ableton-style Streamlit UI |
| `idaw_ui.py` | Simple Streamlit UI |
| `idaw_menubar.py` | macOS menubar app |
| `idaw_listener_public.py` | Terminal listener |
| `vernacular.py` | Casual term translator |
| `ChatGPT_Knowledge_File.md` | GPT knowledge base |
| `iDAW_GPT_Instructions.md` | GPT instructions |

---

## Version History

- **v1.0.00** (2025-11-27) - Initial release
  - Complete 12-module pipeline
  - Ableton-style UI
  - AKAI MPK Mini integration
  - Emotional presets
  - Rule-breaking database

---

## Versioning

Format: `v{MAJOR}.{MINOR}.{PATCH}`

- **MAJOR** - Fundamental architecture changes
- **MINOR** - New engines or major features
- **PATCH** - Bug fixes and minor updates (increment by 01)

---

*iDAW: Making musicians braver since 2025*
