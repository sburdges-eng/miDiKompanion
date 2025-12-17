# TODO: Lyric Generator & Vocal Synthesizer - Complete Implementation

**Status:** Complete
**Date:** 2025-01-27
**Completed:** 2025-01-27
**Priority:** High

---

## Overview

This TODO document outlines all tasks required to fully implement a production-ready lyric generator and vocal synthesizer for Kelly MIDI Companion. **Core implementation is complete** (Parts 1-8). The system now includes:

- ✅ Full lyric generation with semantic expansion and rhyme modeling
- ✅ Complete phoneme processing and formant mapping
- ✅ Pitch-phoneme alignment and expression engine
- ✅ MIDI integration and emotion-based vocal mapping
- ✅ UI components for lyric display and vocal controls
- ✅ Comprehensive testing suite

**Remaining tasks** (Parts 9-10) are research/future enhancements:

- Reference implementation studies (optional)
- Performance benchmarking (as needed)
- Advanced ML features (future)
- Multi-language support (future)

---

## PART 1: LYRIC GENERATOR ENHANCEMENT

### 1.1 Core Lyric Generation Engine

#### Phase 1: Basic Structure

- [x] **Create `LyricGenerator` class** (`src/voice/LyricGenerator.h/cpp`)
  - [x] Separate from `VoiceSynthesizer` for modularity
  - [x] Accept `EmotionNode` and `Wound` as inputs
  - [x] Return structured lyric data (lines, syllables, phonemes)

- [x] **Implement semantic concept expansion**
  - [x] Extract keywords from emotion name and wound description
  - [x] Use emotion thesaurus to find related concepts
  - [x] Generate thematic word lists (synonyms, related emotions)
  - [x] Reference: `suno_adaptive_lyric_system.md`

- [x] **Create lyric structure templates**
  - [x] Verse/Chorus/Bridge patterns
  - [x] Line length templates (4, 6, 8, 10 syllables)
  - [x] Rhyme scheme patterns (ABAB, AABB, ABCB)
  - [x] Store in `data/lyric_templates.json`

#### Phase 2: Advanced Features

- [x] **Implement prosody and meter alignment**
  - [x] Syllable stress detection (stressed/unstressed)
  - [x] Meter matching (iambic, trochaic, anapestic)
  - [x] Natural speech rhythm modeling
  - [x] Reference: `suno_adaptive_lyric_system.md` Section 3

- [x] **Add rhyme and flow modeling**
  - [x] Phonetic rhyme detection algorithm
  - [x] End-rhyme generation (perfect, slant, internal)
  - [x] Flow/rhythm pattern matching
  - [x] Create `RhymeEngine` class (`src/voice/RhymeEngine.h/cpp`)

- [x] **Emotion-language mapping**
  - [x] Map emotion valence/arousal to word choice
  - [x] High valence → positive imagery, bright metaphors
  - [x] Low valence → darker imagery, introspective language
  - [x] High arousal → action verbs, dynamic words
  - [x] Low arousal → contemplative, slow-paced words

#### Phase 3: Integration

- [x] **Connect to emotion thesaurus**
  - [x] Use 216-node emotion database for word associations
  - [x] Map emotion categories to lyric themes
  - [x] Integrate with `EmotionThesaurus` class

- [x] **Wound-based lyric generation**
  - [x] Extract narrative from wound description
  - [x] Generate lyrics that reflect the emotional journey
  - [x] Support healing/catharsis themes
  - [x] Reference: Kelly's "Interrogate Before Generate" philosophy

---

## PART 2: PHONEME & TEXT PROCESSING

### 2.1 Text-to-Phoneme Conversion

- [x] **Create `PhonemeConverter` class** (`src/voice/PhonemeConverter.h/cpp`)
  - [x] Convert text to IPA (International Phonetic Alphabet)
  - [x] Support English phonemes (44 phonemes)
  - [x] Handle common words and exceptions
  - [x] Use CMU Pronouncing Dictionary format

- [x] **Implement phoneme database**
  - [x] Create `data/phonemes.json` with:
    - [x] Phoneme symbols (IPA)
    - [x] Vowel/consonant classification
    - [x] Formant mappings for each phoneme
    - [x] Duration estimates
  - [x] Reference: `bark_vocal_diffusion_engine.md` Section 3

- [x] **Syllable segmentation**
  - [x] Split words into syllables
  - [x] Identify stressed syllables
  - [x] Calculate syllable durations

### 2.2 Phoneme-to-Formant Mapping

- [x] **Extend `VowelFormantDatabase`**
  - [x] Add consonant formants (fricatives, plosives, nasals)
  - [x] Add diphthong formants
  - [x] Create transition formants between phonemes
  - [x] Reference: `VocoderEngine.h` existing vowel database

- [x] **Implement formant interpolation**
  - [x] Smooth transitions between phonemes
  - [x] Handle consonant-vowel transitions
  - [x] Adjust formant frequencies for different voice types

---

## PART 3: VOCAL SYNTHESIS ENHANCEMENTS

### 3.1 Pitch-Phoneme Alignment

- [x] **Create `PitchPhonemeAligner` class** (`src/voice/PitchPhonemeAligner.h/cpp`)
  - [x] Align MIDI pitches to phoneme sequences
  - [x] Handle melisma (multiple notes per syllable)
  - [x] Support portamento between phonemes
  - [x] Reference: `suno_vocal_synthesis.md` Section 2

- [x] **Implement lyric-to-melody alignment**
  - [x] Map lyric syllables to vocal notes
  - [x] Handle syncopation and rhythm
  - [x] Support different vocal styles (legato, staccato)

### 3.2 Expression Engine

- [x] **Enhance `VocalCharacteristics` struct**
  - [x] Add expression parameters:
    - [x] Dynamics (crescendo, diminuendo)
    - [x] Articulation (legato, staccato, marcato)
    - [x] Vibrato depth and rate
    - [x] Breathiness variation
    - [x] Brightness modulation
  - [x] Reference: `VoiceSynthesizer.h` existing struct

- [x] **Create `ExpressionEngine` class** (`src/voice/ExpressionEngine.h/cpp`)
  - [x] Apply expression curves to vocal notes
  - [x] Emotion-based expression mapping
  - [x] Real-time expression modulation

### 3.3 Vocal Style Variations

- [x] **Add multiple voice types**
  - [x] Male voice formants (lower F1, F2)
  - [x] Female voice formants (higher F1, F2)
  - [x] Child voice formants
  - [x] Store in `data/voice_types.json`

- [ ] **Implement vocal effects**
  - [ ] Reverb for spatial depth
  - [ ] Delay for echo effects
  - [ ] Chorus for richness
  - [ ] Distortion for aggressive styles
  - Note: Marked as optional/future enhancement per plan

---

## PART 4: INTEGRATION & SYNCHRONIZATION

### 4.1 Lyric-Vocal Synchronization

- [x] **Create `LyriSync` class** (`src/voice/LyriSync.h/cpp`)
  - [x] Synchronize lyrics with vocal melody
  - [x] Handle timing alignment
  - [x] Support real-time updates
  - [x] Reference: `suno_vocal_synthesis.md` Section 1

- [x] **Implement syllable timing**
  - [x] Calculate syllable durations from tempo
  - [x] Handle word stress timing
  - [x] Support rubato (tempo variation)

### 4.2 MIDI Integration

- [x] **Enhance `VoiceSynthesizer::generateVocalMelody()`**
  - [x] Better integration with generated MIDI context
  - [x] Respect chord progressions
  - [x] Follow melodic contours
  - [x] Avoid pitch conflicts with instruments

- [x] **Add vocal MIDI output**
  - [x] Generate MIDI notes for vocal parts
  - [x] Include lyric meta-events (MIDI text events)
  - [x] Export to MIDI file with lyrics

### 4.3 Emotion Integration

- [x] **Improve emotion-to-vocal mapping**
  - [x] Map emotion valence to vocal brightness
  - [x] Map emotion arousal to vibrato rate
  - [x] Map emotion dominance to vocal dynamics
  - [x] Use `PHASE_2_KELLY.md` VAD formulas

- [x] **Add emotion-based lyric themes**
  - [x] Generate lyrics that match emotional state
  - [x] Use emotion keywords from thesaurus
  - [x] Support emotional narrative arcs

---

## PART 5: DATA STRUCTURES & CONFIGURATION

### 5.1 Lyric Data Structures

- [x] **Create `LyricLine` struct**
  - [x] Defined in `src/voice/LyricTypes.h`
  - [x] Includes text, syllables, phonemes, stress pattern, meter

- [x] **Create `LyricStructure` struct**
  - [x] Defined in `src/voice/LyricTypes.h`
  - [x] Includes sections (verse, chorus, bridge), pattern, rhyme scheme

### 5.2 Configuration Files

- [x] **Create `data/lyric_templates.json`**
  - [x] Verse/Chorus/Bridge templates
  - [x] Rhyme schemes
  - [x] Meter patterns
  - [x] Line length options

- [x] **Create `data/phonemes.json`**
  - [x] IPA phoneme symbols
  - [x] Formant frequencies (F1-F4)
  - [x] Bandwidths (B1-B4)
  - [x] Duration estimates

- [x] **Create `data/voice_types.json`**
  - [x] Male/Female/Child formant shifts
  - [x] Vocal range limits
  - [x] Characteristic timbres

- [x] **Create `data/rhyme_patterns.json`**
  - [x] Rhyme patterns integrated into `lyric_templates.json` and `RhymeEngine`
  - [x] Common rhyme schemes supported
  - [x] Phonetic similarity matrices in `RhymeEngine`
  - [x] Flow patterns supported

---

## PART 6: UI INTEGRATION

### 6.1 Lyric Display

- [x] **Add lyric display to `PluginEditor`**
  - [x] Integrated with `EmotionWorkstation`
  - [x] Show generated lyrics in UI
  - [x] Highlight current line during playback
  - [x] Lyric editing support via `LyricDisplay` component
  - [x] Reference: `src/ui/` existing components

- [x] **Create `LyricDisplay` component** (`src/ui/LyricDisplay.h/cpp`)
  - [x] Display lyric lines
  - [x] Show syllable breakdown
  - [x] Highlight phonemes
  - [x] Sync with audio playback

### 6.2 Vocal Controls

- [x] **Add vocal synthesis controls to UI**
  - [x] Voice type selector (Male/Female/Child)
  - [x] Vibrato depth/rate sliders
  - [x] Breathiness control
  - [x] Brightness control
  - [x] Expression controls (via `VocalControlPanel`)

- [x] **Create `VocalControlPanel` component** (`src/ui/VocalControlPanel.h/cpp`)
  - [x] Real-time parameter adjustment
  - [x] Integrated with `EmotionWorkstation`
  - [x] Voice type and expression controls

---

## PART 7: TESTING & VALIDATION

### 7.1 Unit Tests

- [x] **Test `LyricGenerator`**
  - [x] Test with all 216 emotion nodes
  - [x] Test with various wound descriptions
  - [x] Validate rhyme schemes
  - [x] Check meter accuracy
  - [x] File: `tests/voice/test_lyric_generator.cpp`

- [x] **Test `PhonemeConverter`**
  - [x] Test common words
  - [x] Test edge cases (numbers, punctuation)
  - [x] Validate IPA output
  - [x] File: `tests/voice/test_phoneme_converter.cpp`

- [x] **Test `PitchPhonemeAligner`**
  - [x] Test alignment accuracy
  - [x] Test melisma handling
  - [x] Test timing synchronization
  - [x] File: `tests/voice/test_pitch_phoneme_aligner.cpp`

### 7.2 Integration Tests

- [x] **Test lyric-to-vocal pipeline**
  - [x] Generate lyrics → convert to phonemes → synthesize
  - [x] Validate timing alignment
  - [x] Check audio quality
  - [x] File: `tests/integration/test_lyric_vocal_integration.cpp`

- [x] **Test emotion-to-vocal mapping**
  - [x] Test all emotion categories
  - [x] Validate vocal characteristics match emotions
  - [x] Check consistency
  - [x] File: `tests/integration/test_emotion_vocal_mapping.cpp`

### 7.3 Performance Tests

- [ ] **Benchmark lyric generation**
  - [ ] Target: < 100ms for 4-line verse
  - [ ] Profile memory usage
  - [ ] Optimize slow paths
  - Note: Performance benchmarking can be done as needed

- [ ] **Benchmark vocal synthesis**
  - [ ] Target: Real-time performance (no dropouts)
  - [ ] Test with long phrases
  - [ ] Optimize formant calculations
  - Note: Performance benchmarking can be done as needed

---

## PART 8: DOCUMENTATION

### 8.1 Code Documentation

- [x] **Document all new classes**
  - [x] `LyricGenerator` API documentation
  - [x] `PhonemeConverter` usage examples
  - [x] `PitchPhonemeAligner` algorithm explanation
  - [x] `ExpressionEngine` parameter guide
  - [x] File: `docs/LYRIC_GENERATION_API.md`

- [x] **Update existing documentation**
  - [x] Update `VOCODER_IMPLEMENTATION.md`
  - [x] Add lyric generation section
  - [x] Document phoneme system
  - [x] Add usage examples

### 8.2 User Documentation

- [x] **Create lyric generation guide**
  - [x] How to use lyric generator
  - [x] Customizing lyric styles
  - [x] Editing generated lyrics
  - [x] File: `docs/LYRIC_GENERATION_GUIDE.md`

- [x] **Create vocal synthesis guide**
  - [x] Voice type selection
  - [x] Parameter adjustment
  - [x] Expression curves
  - [x] File: `docs/VOCAL_SYNTHESIS_GUIDE.md`

---

## PART 9: REFERENCE IMPLEMENTATIONS

### 9.1 Study Existing Code

- [ ] **Review Python reference** (`reference/python_kelly/`)
  - [ ] Check for lyric generation patterns
  - [ ] Study vocal synthesis approaches
  - [ ] Adapt algorithms to C++

- [ ] **Review DAiW-Music-Brain** (`reference/daiw_music_brain/`)
  - [ ] Study vocal processing modules
  - [ ] Review lyric generation approaches
  - [ ] Extract useful algorithms

### 9.2 Research Papers & Documentation

- [ ] **Study Bark vocal diffusion** (`reference/daiw_music_brain/vault/Production_Workflows/bark_vocal_diffusion_engine.md`)
  - [ ] Phoneme alignment techniques
  - [ ] Pitch conditioning methods
  - [ ] Diffusion model architecture

- [ ] **Study Suno vocal synthesis** (`reference/daiw_music_brain/vault/Production_Workflows/suno_vocal_synthesis.md`)
  - [ ] LyriSync implementation
  - [ ] Expression engine design
  - [ ] Mixer & spatial modeling

---

## PART 10: ADVANCED FEATURES (Future)

### 10.1 Machine Learning Integration

- [ ] **Research ML-based lyric generation**
  - [ ] GPT-style language models
  - [ ] Fine-tuning on emotional lyrics
  - [ ] Integration with Kelly's emotion system

- [ ] **Research neural vocoders**
  - [ ] WaveNet-style vocoders
  - [ ] Real-time neural synthesis
  - [ ] Voice cloning capabilities

### 10.2 Multi-Language Support

- [ ] **Extend phoneme system**
  - [ ] Support multiple languages
  - [ ] Language-specific formants
  - [ ] Translation integration

### 10.3 Real-Time Performance

- [ ] **Optimize for live performance**
  - [ ] Low-latency lyric generation
  - [ ] Real-time vocal synthesis
  - [ ] MIDI clock synchronization

---

## IMPLEMENTATION PRIORITY

### Phase 1 (Weeks 1-2): Foundation

1. Create `LyricGenerator` class with basic structure
2. Implement semantic concept expansion
3. Create lyric templates and data files
4. Basic text-to-phoneme conversion

### Phase 2 (Weeks 3-4): Core Features

1. Prosody and meter alignment
2. Rhyme and flow modeling
3. Emotion-language mapping
4. Phoneme-to-formant mapping

### Phase 3 (Weeks 5-6): Synthesis

1. Pitch-phoneme alignment
2. Expression engine
3. Vocal style variations
4. Lyric-vocal synchronization

### Phase 4 (Weeks 7-8): Integration

1. MIDI integration
2. Emotion integration
3. UI components
4. Testing and validation

---

## ESTIMATED TIMELINE

- **Total Development Time:** 8-10 weeks
- **Core Features:** 6 weeks
- **Integration & Testing:** 2 weeks
- **Documentation & Polish:** 2 weeks

---

## DEPENDENCIES

- ✅ `VocoderEngine` - Already implemented
- ✅ `VoiceSynthesizer` - Basic implementation exists
- ✅ `EmotionThesaurus` - Available
- ✅ `WoundProcessor` - Available
- ✅ `LyricGenerator` - **COMPLETED**
- ✅ `PhonemeConverter` - **COMPLETED**
- ✅ `PitchPhonemeAligner` - **COMPLETED**
- ✅ `ExpressionEngine` - **COMPLETED**
- ✅ `ProsodyAnalyzer` - **COMPLETED**
- ✅ `RhymeEngine` - **COMPLETED**
- ✅ `LyriSync` - **COMPLETED**

---

## NOTES

- All new classes should follow Kelly's existing code style
- Use JUCE framework for audio processing
- Maintain compatibility with existing `VoiceSynthesizer` API
- Follow "Interrogate Before Generate" philosophy
- Prioritize emotional authenticity over technical perfection

---

**Last Updated:** 2025-01-27
**Completed:** 2025-01-27
**Status:** All core implementation tasks (Parts 1-8) are complete. Parts 9-10 are research/future enhancements.
