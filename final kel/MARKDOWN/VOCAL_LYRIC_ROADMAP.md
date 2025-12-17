# Kelly Phase 2B: Vocal Synthesis & Lyric Generation Roadmap

**Status:** Ready for Implementation  
**Timeline:** 8-10 weeks  
**Dependencies:** Phase 2A core (VAD system, humanization engine)

---

## Current Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| VocoderEngine | ✅ Complete | C++ JUCE implementation |
| VoiceSynthesizer | ⚠️ Basic | Needs enhancement |
| LyricGenerator | ⚠️ Minimal | Hardcoded lines only |
| PhonemeProcessor | ❌ None | Not implemented |
| PitchPhonemeAlignment | ❌ None | Not implemented |
| ExpressionEngine | ❌ None | Not implemented |

---

## Phase 1: Lyric Generator Enhancement (Weeks 1-2)

### 1.1 Basic Structure & Semantic Expansion

- [ ] Create `LyricSchema` dataclass with verse/chorus/bridge sections
- [ ] Implement `LineMeter` class for syllable counting
- [ ] Build semantic expansion from emotion thesaurus (216 nodes → lyric vocabulary)
- [ ] Create theme-to-word mapping database
- [ ] Implement narrative arc templates (AABA, ABAB, verse-chorus-bridge)
- [ ] Add metaphor generation from emotion vectors
- [ ] Build rhyme dictionary integration (CMU Pronouncing Dictionary)
- [ ] Create synonym/antonym expansion engine
- [ ] Implement section transition logic
- [ ] Add repetition handler (hooks, callbacks)

### 1.2 Prosody, Meter & Flow

- [ ] Implement `StressPattern` analyzer (primary/secondary/unstressed)
- [ ] Build syllable-to-beat alignment engine
- [ ] Create meter templates (iambic, trochaic, anapestic, dactylic)
- [ ] Add syncopation detection for lyrics
- [ ] Implement enjambment detector
- [ ] Build natural pause placement (caesura)
- [ ] Create breath marker insertion
- [ ] Add emphasis word detection
- [ ] Implement rhythmic density calculator
- [ ] Build flow compatibility scorer with groove templates

### 1.3 Emotion Integration

- [ ] Map emotion vectors to lyric themes
- [ ] Implement vocabulary filtering by valence
- [ ] Add intensity modifiers based on arousal
- [ ] Create dominance → perspective mapping (I/you/we)
- [ ] Build emotional arc that matches music trajectory
- [ ] Add vulnerability level → explicitness control
- [ ] Implement rule-break justification → lyric authenticity
- [ ] Create wound description → lyric seed conversion
- [ ] Add misdirection support (surface vs. revealed meaning)
- [ ] Build lyric coherence scoring with emotion intent

---

## Phase 2: Phoneme & Text Processing (Weeks 2-3)

### 2.1 Text-to-Phoneme Conversion

- [ ] Integrate `g2p_en` library for grapheme-to-phoneme
- [ ] Build custom pronunciation dictionary for Kelly
- [ ] Create IPA output formatter
- [ ] Implement stress marker extraction (1, 2, 0)
- [ ] Add phoneme duration estimation
- [ ] Build heteronym disambiguation (read/read, live/live)
- [ ] Create proper noun handler
- [ ] Implement contractions expansion
- [ ] Add number/symbol verbalization
- [ ] Build phoneme sequence validator

### 2.2 Phoneme-to-Formant Mapping

- [ ] Create vowel formant table (F1, F2, F3)
- [ ] Build consonant characteristic mapping
- [ ] Implement coarticulation rules
- [ ] Add formant interpolation engine
- [ ] Create phoneme transition smoother
- [ ] Build aspiration/voice onset markers
- [ ] Implement nasalization detector
- [ ] Add diphthong handling
- [ ] Create formant modification for emotion
- [ ] Build F0 contour suggestions per phoneme

### 2.3 Syllable Segmentation

- [ ] Implement onset-nucleus-coda parser
- [ ] Build syllable boundary detection
- [ ] Create syllable weight calculator
- [ ] Add morpheme-aware segmentation
- [ ] Implement compound word handling
- [ ] Build hyphenation fallback
- [ ] Create syllable-to-note duration mapping
- [ ] Add melisma detection (multiple notes per syllable)
- [ ] Implement syllable merging for fast passages
- [ ] Build syllable stretch for held notes

---

## Phase 3: Vocal Synthesis Enhancements (Weeks 3-5)

### 3.1 Pitch-Phoneme Alignment

- [ ] Create `PitchPhonemeAligner` class
- [ ] Implement note-to-syllable mapping
- [ ] Build duration stretching/compression
- [ ] Add pitch target sequence generator
- [ ] Create portamento insertion rules
- [ ] Implement vibrato start delay per phoneme
- [ ] Build consonant timing adjustments
- [ ] Add vowel centering for pitch stability
- [ ] Create attack shaping per phoneme type
- [ ] Implement release tailoring

### 3.2 Expression Engine

- [ ] Create `ExpressionEngine` class
- [ ] Implement dynamic micro-expressions
- [ ] Build pitch drift for authenticity
- [ ] Add velocity curves per phrase
- [ ] Create breath noise insertion
- [ ] Implement vocal fry generator
- [ ] Build falsetto/chest voice transition
- [ ] Add voice cracking (emotional authenticity)
- [ ] Create subtle pitch corrections
- [ ] Implement emotional tremolo

### 3.3 Vocal Style Variations

- [ ] Build `VocalStyle` enum (breathy, belt, whisper, etc.)
- [ ] Implement breathiness control (HNR)
- [ ] Create belt/chest voice parameters
- [ ] Add whisper synthesis mode
- [ ] Build falsetto transition
- [ ] Implement spoken word mode
- [ ] Create call-and-response patterns
- [ ] Add harmonization generator
- [ ] Build double tracking simulation
- [ ] Implement genre-specific vocal processing chains

---

## Phase 4: Integration & Synchronization (Weeks 5-7)

### 4.1 LyriSync (Lyric-Vocal Synchronization)

- [ ] Create `LyriSync` master coordinator
- [ ] Build beat-aligned syllable scheduler
- [ ] Implement phrase grouping engine
- [ ] Add natural pause insertion
- [ ] Create breath timing calculator
- [ ] Build emphasis synchronization with music accents
- [ ] Implement melisma coordinator
- [ ] Add held note vowel sustain
- [ ] Create lyric-to-MIDI note timing
- [ ] Build real-time sync adjustment

### 4.2 MIDI Integration

- [ ] Create lyric-aware MIDI track
- [ ] Implement MIDI lyric events (0xFF 05)
- [ ] Build lyric-to-note velocity correlation
- [ ] Add expression CC mapping for lyrics
- [ ] Create pitch bend for vocal slides
- [ ] Implement aftertouch for vibrato intensity
- [ ] Build mod wheel mapping for breathiness
- [ ] Add lyric marker export
- [ ] Create karaoke-format export (KAR)
- [ ] Build DAW-compatible lyric track

### 4.3 Emotion Mapping Integration

- [ ] Connect VAD vectors to vocal parameters
- [ ] Implement emotion trajectory → vocal dynamics
- [ ] Build coherence scoring for vocal output
- [ ] Add rule-break vocal expressions
- [ ] Create wound-to-vocal-quality mapping
- [ ] Implement vulnerability → intimacy control
- [ ] Build section emotion transitions
- [ ] Add climax detection for vocal intensity
- [ ] Create resolution vocal settling
- [ ] Implement aesthetic reward for vocal coherence

---

## Phase 5: Data Structures & Configuration (Week 7)

### 5.1 New Data Structures

```python
# Target structures to implement
@dataclass
class LyricLine:
    text: str
    syllables: List[Syllable]
    phonemes: List[Phoneme]
    stress_pattern: List[int]
    emotion_intent: EmotionVector
    section_type: str

@dataclass
class Syllable:
    text: str
    phonemes: List[Phoneme]
    stress: int  # 0, 1, 2
    duration_beats: float
    start_beat: float

@dataclass
class Phoneme:
    ipa: str
    formants: Tuple[float, float, float]  # F1, F2, F3
    duration_ms: float
    is_voiced: bool
    is_vowel: bool

@dataclass
class VocalExpression:
    breathiness: float  # 0-1
    vibrato_rate: float  # Hz
    vibrato_depth: float  # cents
    pitch_drift: float  # cents
    dynamics: float  # 0-1
```

- [ ] Implement `LyricLine` dataclass
- [ ] Create `Syllable` dataclass
- [ ] Build `Phoneme` dataclass
- [ ] Add `VocalExpression` dataclass
- [ ] Create `VocalStyle` dataclass
- [ ] Implement `LyricSection` container
- [ ] Build `SongLyrics` full structure
- [ ] Add `PhonemeSequence` wrapper
- [ ] Create `VocalTrack` output structure
- [ ] Implement serialization (JSON/YAML)

### 5.2 Configuration Files

- [ ] Create `phoneme_formants.json` (IPA → formants)
- [ ] Build `vocal_styles.json` (style presets)
- [ ] Add `emotion_vocal_map.json` (VAD → vocal params)
- [ ] Create `meter_templates.json` (prosodic patterns)
- [ ] Build `rhyme_weights.json` (rhyme type preferences)
- [ ] Add `genre_vocal_presets.json`
- [ ] Create `expression_curves.json`
- [ ] Build `breath_patterns.json`
- [ ] Add `coarticulation_rules.json`
- [ ] Create `syllable_duration_defaults.json`

---

## Phase 6: UI Integration (Week 8)

### 6.1 Lyric Display Component

- [ ] Create `LyricDisplay` React component
- [ ] Implement synchronized highlighting
- [ ] Build syllable-level animation
- [ ] Add emotion color coding
- [ ] Create edit mode with meter display
- [ ] Implement drag-to-adjust timing
- [ ] Build phrase grouping visualization
- [ ] Add stress pattern overlay
- [ ] Create breath marker display
- [ ] Implement preview playback integration

### 6.2 Vocal Control Panel

- [ ] Create `VocalControlPanel` component
- [ ] Build style selector (dropdown)
- [ ] Add breathiness slider
- [ ] Create vibrato controls (rate/depth)
- [ ] Implement pitch drift toggle
- [ ] Build expression curve editor
- [ ] Add real-time preview button
- [ ] Create A/B comparison mode
- [ ] Implement preset save/load
- [ ] Build emotion-to-vocal auto-adjust toggle

---

## Phase 7: Testing & Validation (Weeks 8-9)

### 7.1 Unit Tests

- [ ] Test `LyricGenerator` output structure
- [ ] Test phoneme conversion accuracy (CMU benchmark)
- [ ] Test syllable segmentation edge cases
- [ ] Test meter analysis on known texts
- [ ] Test formant mapping values
- [ ] Test emotion-to-lyric coherence
- [ ] Test pitch-phoneme alignment timing
- [ ] Test expression parameter ranges
- [ ] Test MIDI lyric event format
- [ ] Test serialization round-trip

### 7.2 Integration Tests

- [ ] Test full pipeline: emotion → lyrics → phonemes → vocal
- [ ] Test LyriSync with MIDI playback
- [ ] Test emotion trajectory vocal dynamics
- [ ] Test UI synchronization accuracy
- [ ] Test DAW export compatibility
- [ ] Test real-time performance (<100ms latency)
- [ ] Test memory usage under load
- [ ] Test multi-section songs
- [ ] Test with rule-break scenarios
- [ ] Test with "When I Found You Sleeping" test case

### 7.3 Performance Benchmarks

- [ ] Measure lyric generation time (target: <500ms)
- [ ] Measure phoneme conversion time (target: <50ms/line)
- [ ] Measure vocal synthesis latency (target: <200ms)
- [ ] Measure memory footprint
- [ ] Profile CPU usage during generation
- [ ] Test streaming generation performance
- [ ] Benchmark multi-track generation
- [ ] Profile UI render performance
- [ ] Test under high syllable density
- [ ] Measure emotion transition smoothness

---

## Phase 8: Documentation (Week 9)

### 8.1 Code Documentation

- [ ] Document all new dataclasses
- [ ] Add docstrings to all public functions
- [ ] Create inline comments for algorithms
- [ ] Build type hints throughout
- [ ] Document configuration file formats
- [ ] Add usage examples in docstrings
- [ ] Create algorithm flowcharts
- [ ] Document edge cases and limitations
- [ ] Add performance considerations
- [ ] Create troubleshooting guide

### 8.2 User Guides

- [ ] Write "Getting Started with Lyrics"
- [ ] Create "Vocal Style Guide"
- [ ] Build "Emotion-to-Vocal Reference"
- [ ] Write "Advanced Customization"
- [ ] Create "Troubleshooting FAQ"
- [ ] Build video tutorial scripts
- [ ] Document keyboard shortcuts
- [ ] Create example workflows
- [ ] Write "Integration with DAW" guide
- [ ] Build API reference documentation

---

## Phase 9: Reference Implementations (Week 10)

### 9.1 Study Existing Python Code

- [ ] Analyze existing `kelly.thesaurus.py` structure
- [ ] Review `intent_processor.py` flow
- [ ] Study `midi_generator.py` output format
- [ ] Examine `groove_templates.py` patterns
- [ ] Review `dynamics_engine.py` approach
- [ ] Analyze `variation_engine.py` techniques
- [ ] Study existing test patterns
- [ ] Review configuration loading patterns
- [ ] Examine CLI integration approach
- [ ] Study multi-engine coordination

### 9.2 Research External Approaches

- [ ] Study Bark architecture (text → audio)
- [ ] Research Suno vocal generation
- [ ] Analyze VITS2 phoneme approach
- [ ] Review XTTS voice cloning
- [ ] Study RVC voice conversion
- [ ] Research festival/flite TTS
- [ ] Analyze eSpeak phoneme handling
- [ ] Review WORLD vocoder
- [ ] Study STRAIGHT vocoder
- [ ] Research HIFI-GAN neural vocoder

---

## Phase 10: Advanced Features (Future)

### 10.1 ML Integration

- [ ] Evaluate local LLM for lyric generation
- [ ] Research fine-tuning approaches
- [ ] Plan training data collection
- [ ] Design feedback loop architecture
- [ ] Prototype neural vocoder integration
- [ ] Research voice cloning ethics/consent
- [ ] Plan real-time inference optimization
- [ ] Design quality scoring model
- [ ] Research emotion detection from voice
- [ ] Plan A/B testing infrastructure

### 10.2 Multi-Language Support

- [ ] Research IPA universality
- [ ] Plan language-specific phoneme sets
- [ ] Design language detection
- [ ] Research cross-language rhyming
- [ ] Plan accent/dialect handling
- [ ] Design transliteration fallbacks
- [ ] Research meter in other languages
- [ ] Plan cultural-appropriate emotion mapping
- [ ] Design language pack system
- [ ] Research romanization handling

### 10.3 Real-Time Performance

- [ ] Design streaming generation
- [ ] Plan WebSocket protocol
- [ ] Research latency optimization
- [ ] Design caching strategy
- [ ] Plan GPU acceleration
- [ ] Research real-time vocal effects
- [ ] Design live performance mode
- [ ] Plan MIDI input for live lyrics
- [ ] Research adaptive timing
- [ ] Design failsafe mechanisms

---

## File Structure

```
kelly/
├── lyric/
│   ├── __init__.py
│   ├── generator.py          # Main lyric generation
│   ├── prosody.py            # Meter, stress, rhythm
│   ├── semantic.py           # Theme/vocabulary expansion
│   └── rhyme.py              # Rhyme dictionary
├── phoneme/
│   ├── __init__.py
│   ├── g2p.py                # Grapheme-to-phoneme
│   ├── formants.py           # Formant mapping
│   └── syllable.py           # Syllable processing
├── vocal/
│   ├── __init__.py
│   ├── synthesizer.py        # Main vocal synthesis
│   ├── expression.py         # Expression engine
│   ├── alignment.py          # Pitch-phoneme alignment
│   └── styles.py             # Vocal style presets
├── sync/
│   ├── __init__.py
│   ├── lyrisync.py           # Lyric-vocal coordinator
│   └── midi_lyrics.py        # MIDI lyric events
└── config/
    ├── phoneme_formants.json
    ├── vocal_styles.json
    ├── emotion_vocal_map.json
    └── meter_templates.json
```

---

## Priority Order

1. **High (Weeks 1-3):** Phoneme processing + basic lyric structure
2. **Medium (Weeks 3-5):** Vocal synthesis + expression engine  
3. **Medium (Weeks 5-7):** LyriSync integration
4. **Lower (Weeks 7-9):** UI + testing + documentation
5. **Future:** ML integration + multi-language

---

## Dependencies

```toml
# pyproject.toml additions
[project.dependencies]
g2p-en = "^2.1.0"          # Grapheme-to-phoneme
pronouncing = "^0.2.0"      # CMU dictionary
syllables = "^1.0.3"        # Syllable counting
epitran = "^1.24"           # IPA conversion
pydub = "^0.25.1"           # Audio processing
```

---

## Risk Factors

| Risk | Mitigation |
|------|------------|
| G2P accuracy | Custom dictionary + user override |
| Timing sync drift | Beat-locked anchor points |
| Expression authenticity | User feedback loop |
| Performance bottleneck | Streaming + caching |
| Memory usage | Lazy loading + pruning |

---

**Next Action:** Begin Phase 2.1 (Text-to-Phoneme) by implementing `g2p.py` with CMU dictionary integration.
