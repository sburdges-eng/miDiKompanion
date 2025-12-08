# DAiW Music Brain - Claude Code Integration

## Philosophy: Interrogate Before Generate

**The tool shouldn't finish art for people — it should make them braver.**

### Core Principles

1. **Interrogate Before Generate** — Ask about mood, intent, imagery first
2. **Rule-Breaking Engine** — Teach when and why to break norms
3. **Groove Trainer** — Help users learn timing, feel, pocket
4. **Emotional Arrangement** — Match harmony to lyrical meaning
5. **User-Guided Translation** — Help express the sound in their head
6. **Preserve Imperfections** — Timing drift, velocity variation, human feel

---

## Reference Song: "When I Found You Sleeping" (Kelly)

### Musical Context
- **Key:** D minor (tonic)
- **Tempo:** 82 BPM
- **Time Signature:** 4/4
- **Progression:** F - C - Am - Dm
- **Genre:** Lo-fi bedroom emo
- **Guitar Pattern:** 1-5-6-4-3-2 fingerpicking (strings: high E, A, low E, D, G, B)

### The Misdirection Technique
**Major progression → Minor tonic gut punch**

The song uses a major-leaning progression (F-C-Am) that sounds hopeful/nostalgic, but resolves to Dm (minor tonic). This creates emotional misdirection — the harmony sets up one feeling, the resolution reveals the truth.

**Emotional Impact:**
- F-C-Am: "Things were good, remember?"
- Dm: "But they're not anymore"

This is the core of lo-fi bedroom emo: vulnerability through harmonic subversion.

---

## Production Aesthetic

### Timing & Feel
- **Pocket:** Behind the beat (laid back, reflective)
- **Timing drift:** ±30-50ms (human imperfection)
- **Velocity variation:** 22-95 range (NOT quantized)
- **Note duration:** Slightly shortened on upbeats

### Processing
- **Tape saturation:** 5-15% distortion (TanhDistortion)
- **Room tone:** Natural bedroom reverb
- **Tape hiss:** Low-level Gaussian noise (0.001-0.003)
- **Lo-pass filter:** 8-12kHz warmth

### Space & Arrangement
- **Sparse instrumentation:** Let silence breathe
- **Natural room reverb:** 15-20% on guitar
- **Minimal compression:** Preserve dynamics
- **Imperfect tuning:** Slightly detuned (±5 cents)

---

## Package Structure

```
DAiW/
├── .claude/
│   └── settings.json          # Claude Code project config
├── music_brain/
│   ├── models/                # NEW: Emotional → Musical mapping
│   │   ├── emotional_mapping.py
│   │   └── __init__.py
│   ├── session/               # Therapy & interrogation
│   │   ├── interrogator.py
│   │   ├── generator.py
│   │   └── intent_processor.py
│   ├── groove/                # Timing & feel
│   │   ├── extractor.py
│   │   ├── applicator.py
│   │   └── templates.py
│   ├── structure/             # Harmony & form
│   │   ├── chord.py
│   │   ├── progression.py
│   │   └── sections.py
│   ├── audio/                 # Audio analysis
│   │   └── feel.py
│   ├── daw/                   # Logic Pro integration
│   │   └── logic.py
│   ├── utils/                 # MIDI I/O, PPQ
│   │   ├── midi_io.py
│   │   └── ppq.py
│   └── data/                  # JSON data files
│       ├── genre_pocket_maps.json
│       └── chord_progressions.json
├── tests/
│   └── models/
│       └── test_emotional_mapping.py
└── docs/
    └── research/              # Artist/technique research
```

---

## Key Features Implemented

### 1. Therapy Session → MIDI Pipeline ✅
```python
"I am furious that he left" (motivation=9, chaos=7)
    ↓
TherapySession.generate_plan()
    ↓
HarmonyPlan (chords, tempo, complexity, vulnerability)
    ↓
render_plan_to_midi()
    ↓
apply_groove(notes, complexity, vulnerability, chaos)
    ↓
daiw_output.mid (95.8% off-grid notes)
```

### 2. Audio Vault & Sample Processing ✅
```
AudioVault/
├── raw/              # Original samples
├── refined/          # Lo-fi processed (audiomentations)
├── kits/             # Logic Pro kits (GM mapping)
└── output/           # Generated MIDI
```

### 3. Lo-Fi Processing Pipeline ✅
- Tape saturation (TanhDistortion)
- Warmth (LowPassFilter)
- Tape hiss (GaussianNoise)
- Type-specific chains (kick/snare/hihat)

---

## Emotional → Musical Mapping (NEW)

### EmotionalState
- Valence (negative ↔ positive)
- Arousal (calm ↔ energetic)
- Primary emotion
- Secondary emotions
- Intrusions (PTSD, anxiety)

### MusicalParameters
- Tempo range (min, max, suggested)
- Mode weights (major, minor, dorian, phrygian, etc.)
- Register (low, mid, high)
- Harmonic rhythm (slow, medium, fast)
- Dissonance level (0-1)
- Timing feel (behind, on, ahead)
- Density (sparse, medium, dense)
- Space probability (silence percentage)

### Emotion Presets
- **Grief:** 60-82 BPM, minor/dorian, behind beat, 30% dissonance
- **Anxiety:** 100-140 BPM, phrygian, ahead beat, 60% dissonance
- **Nostalgia:** 70-90 BPM, mixolydian, behind beat, 25% dissonance
- **Anger:** 120-160 BPM, phrygian/locrian, ahead beat, 50% dissonance

---

## Next Steps

1. **Complete emotional_mapping.py** with full interval/chord mappings
2. **Integrate with TherapySession** for real-time emotion → music
3. **Add PTSD intrusion modeling** (sudden register spikes, unresolved dissonance)
4. **Create Phoebe Bridgers/Julien Baker production research**
5. **Build misdirection technique library** (major→minor, modal mixture)

---

## Related Files
- [emotional_mapping.py](music_brain/models/emotional_mapping.py)
- [comprehensive_test.py](~/Music/AudioVault/comprehensive_test.py)
- [MVP_COMPLETE.md](~/Desktop/MVP_COMPLETE.md)
