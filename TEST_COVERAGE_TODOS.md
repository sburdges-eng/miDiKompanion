# Music Brain Test Coverage TODOs

**Priority Legend:** 🔴 HIGH | 🟡 MEDIUM | 🟢 LOW
**Effort:** S (1-2h) | M (3-5h) | L (6-10h) | XL (10+h)

---

## Sprint 1: Critical Test Coverage (🔴 HIGH PRIORITY)

### TODO-1: Structure Module - Chord Analysis Tests
**Priority:** 🔴 HIGH | **Effort:** M (4-6 hours) | **Risk:** High - Core functionality

**Description:**
Create comprehensive unit tests for `music_brain/structure/chord.py` covering chord detection, key analysis, Roman numeral analysis, and borrowed chord identification.

**Technical Parameters:**
- Input: MIDI note arrays [List[int]], MIDI files (Path)
- Output: Chord objects with root (int 0-11), quality (str), extensions (List[str])
- Performance Target: < 50ms for 4-bar progression analysis
- Accuracy Target: 90% chord recognition on test corpus
- Test Coverage Target: 85%

**Acceptance Criteria:**
- [ ] Chord detection from note clusters (triads, 7ths, extended)
- [ ] Key detection algorithm validated (major/minor)
- [ ] Roman numeral analysis (I, ii, iii, IV, V, vi, vii°)
- [ ] Borrowed chord identification (bIII, bVI, bVII, iv)
- [ ] Slash chord detection (C/E, F/A)
- [ ] Edge cases: empty input, single notes, dissonant clusters
- [ ] Invalid MIDI note handling (< 0, > 127)
- [ ] Performance benchmark: analyze 100 chords in < 500ms

**Test Cases:**
```python
def test_detect_major_triad()
def test_detect_minor_seventh()
def test_detect_borrowed_chord_from_parallel_minor()
def test_roman_numeral_analysis_in_c_major()
def test_key_detection_from_progression()
def test_empty_notes_returns_none()
def test_single_note_ambiguous_chord()
def test_dissonant_cluster_classification()
```

**Failure Scenarios:**
- Invalid MIDI notes → ValueError with helpful message
- Ambiguous chord → Return most likely with confidence score
- Empty progression → Return empty result, not crash

---

### TODO-2: Structure Module - Progression Analysis Tests
**Priority:** 🔴 HIGH | **Effort:** M (4-5 hours) | **Risk:** High - Affects harmony generation

**Description:**
Test chord progression analysis, quality assessment, tension curves, and progression suggestions.

**Technical Parameters:**
- Input: List[Chord] or chord symbol strings (List[str])
- Output: ProgressionAnalysis with quality score (0-1), tension curve (List[float])
- Validation: Common progressions database lookup
- Coverage Target: 80%

**Acceptance Criteria:**
- [ ] Progression quality scoring (diatonic vs. chromatic)
- [ ] Tension curve calculation
- [ ] Common progression detection (I-V-vi-IV, ii-V-I, etc.)
- [ ] Voice leading analysis
- [ ] Modulation detection
- [ ] Reharmonization suggestions
- [ ] Circle of fifths movement validation

**Test Cases:**
```python
def test_common_progression_i_v_vi_iv()
def test_tension_curve_rising_to_dominant()
def test_modulation_detection_c_to_g()
def test_voice_leading_quality_score()
def test_suggest_reharmonization_options()
```

---

### TODO-3: Audio Module - Feel Analysis Tests
**Priority:** 🔴 HIGH | **Effort:** M (5-6 hours) | **Risk:** High - Foundation for groove

**Description:**
Create tests for `music_brain/audio/feel.py` covering tempo detection, beat tracking, energy curves, spectral analysis, and groove characteristics.

**Technical Parameters:**
- Input: Audio files (wav, mp3, flac), sample rate 44100 or 48000
- Output: AudioFeatures dataclass
- Performance: < 5 seconds for 3-minute audio file
- Memory: < 500MB peak usage
- Accuracy: Tempo ±2 BPM, beat positions ±50ms
- Coverage Target: 75%

**Acceptance Criteria:**
- [ ] Tempo detection accuracy (±2 BPM on test corpus)
- [ ] Beat position extraction (±50ms tolerance)
- [ ] Energy curve calculation per beat
- [ ] Spectral centroid (brightness) measurement
- [ ] Dynamic range calculation (dB)
- [ ] Swing estimation (0.0-1.0)
- [ ] Groove regularity score
- [ ] Graceful degradation without librosa
- [ ] Support wav, mp3, flac formats
- [ ] Handle different sample rates (44100, 48000, 96000)

**Test Cases:**
```python
def test_tempo_detection_within_tolerance()
def test_beat_tracking_alignment()
def test_energy_curve_matches_perceptual_loudness()
def test_spectral_centroid_for_bright_vs_dark()
def test_swing_detection_in_jazz()
def test_missing_librosa_raises_import_error()
def test_invalid_audio_file_format()
def test_corrupted_audio_file_handling()
def test_very_short_audio_under_1_second()
```

**Mock Strategy:**
- Mock librosa when unavailable
- Use small test audio files (< 10 seconds)
- Pre-computed expected values for validation

**Failure Scenarios:**
- Missing librosa → ImportError with installation instructions
- Corrupted file → ValueError with file path
- Unsupported format → ValueError listing supported formats

---

### TODO-4: Error Handling Improvements
**Priority:** 🔴 HIGH | **Effort:** M (5 hours) | **Risk:** Medium - Production stability

**Description:**
Improve error messages, add custom exception classes, implement rollback logic for DAW exports, and validate all error paths are tested.

**Technical Parameters:**
- Custom exceptions: HarmonyError, GrooveError, AudioAnalysisError, DAWExportError
- Error context: Include file paths, parameter values, suggested fixes
- Rollback: Clean up partial files on export failure
- Validation: 100% of error paths tested

**Acceptance Criteria:**
- [ ] Create custom exception hierarchy
- [ ] Add context to all MIDI parsing errors
- [ ] Improve audio file error messages (format, corruption)
- [ ] Add intent validation with specific field errors
- [ ] Implement rollback/cleanup on DAW export failures
- [ ] Add error recovery mechanisms (retry with fallback)
- [ ] Document all exceptions in docstrings
- [ ] Test all error paths with pytest.raises

**Custom Exceptions:**
```python
class MusicBrainError(Exception): pass
class HarmonyError(MusicBrainError): pass
class GrooveError(MusicBrainError): pass
class AudioAnalysisError(MusicBrainError): pass
class DAWExportError(MusicBrainError): pass
class IntentValidationError(MusicBrainError): pass
```

**Error Message Templates:**
```python
# Bad
raise ValueError("Invalid")

# Good
raise HarmonyError(
    f"Cannot apply modal interchange to {mode} mode. "
    f"Modal interchange requires major or minor mode. "
    f"Current mode: {mode}, key: {key}. "
    f"Suggestion: Change mode to 'major' or 'minor'."
)
```

**Test Cases:**
```python
def test_harmony_error_includes_context()
def test_midi_parse_error_shows_file_path()
def test_daw_export_cleanup_on_failure()
def test_audio_error_suggests_librosa_install()
def test_intent_validation_lists_all_field_errors()
```

---

## Sprint 2: DAW Integration & Workflows (🟡 MEDIUM PRIORITY)

### TODO-5: DAW Integration - FL Studio Tests
**Priority:** 🟡 MEDIUM | **Effort:** L (6-8 hours) | **Risk:** Medium - Real-world usage

**Description:**
Comprehensive tests for FL Studio integration including project export, mixer automation, and MIDI import/export.

**Technical Parameters:**
- Input: Project data structures, MIDI files
- Output: FLP (FL Studio project) files, automation clips
- Validation: Parse exported FLP to verify structure
- Coverage Target: 80%

**Acceptance Criteria:**
- [ ] FLP export format validation
- [ ] Mixer track creation and routing
- [ ] Automation clip generation
- [ ] Plugin preset loading
- [ ] MIDI import/export roundtrip
- [ ] Tempo and time signature handling
- [ ] Marker/region creation
- [ ] Mock file I/O for CI/CD

**Test Cases:**
```python
def test_export_fl_studio_project()
def test_mixer_track_automation()
def test_plugin_preset_assignment()
def test_midi_roundtrip_preserves_data()
def test_tempo_map_with_changes()
def test_marker_creation_at_bars()
```

---

### TODO-6: DAW Integration - Pro Tools & Reaper Tests
**Priority:** 🟡 MEDIUM | **Effort:** L (8 hours) | **Risk:** Medium

**Description:**
Similar test coverage for Pro Tools (AAF/PTX) and Reaper (RPP) integrations.

**Acceptance Criteria:**
- [ ] Pro Tools AAF export validation
- [ ] Reaper RPP format compliance
- [ ] Cross-DAW compatibility tests
- [ ] Each module 80% coverage

---

### TODO-7: End-to-End Workflow Tests
**Priority:** 🟡 MEDIUM | **Effort:** L (8 hours) | **Risk:** High - User journeys

**Description:**
Integration tests covering complete user workflows from intent to final output.

**Technical Parameters:**
- End-to-end latency: < 2 seconds for typical workflow
- Memory usage: < 1GB peak
- Output validation: Valid MIDI, correct parameters
- Error recovery: Graceful degradation, no data loss

**Acceptance Criteria:**
- [ ] Intent → Harmony → Groove → MIDI export pipeline
- [ ] Audio → Feel analysis → Groove extraction → Application
- [ ] Reference track → DNA → Style transfer
- [ ] Kelly song complete recreation workflow
- [ ] Therapy session → Musical output
- [ ] Error recovery in pipelines (retry, fallback)
- [ ] Performance benchmarks (latency, memory)
- [ ] Multi-step rollback on failure

**Test Cases:**
```python
def test_complete_intent_to_midi_pipeline()
def test_audio_analysis_to_groove_application()
def test_kelly_song_recreation_workflow()
def test_therapy_session_generates_valid_output()
def test_pipeline_recovers_from_harmony_error()
def test_end_to_end_performance_under_2_seconds()
def test_memory_usage_under_1gb()
```

**Kelly Song Workflow Test:**
```python
def test_kelly_song_when_i_found_you_sleeping():
    """Recreate Kelly song: When I Found You Sleeping"""
    # 1. Create intent
    intent = create_intent(
        title="When I Found You Sleeping",
        core_event="Finding peace in letting go",
        mood_primary="grief",
        technical_key="F",
        technical_mode="major",
        tempo_range=(78, 86),
        rule_to_break="HARMONY_ModalInterchange",
        rule_justification="Bbm makes hope feel earned"
    )

    # 2. Generate harmony (F-C-Bbm-F)
    harmony = generate_harmony(intent)
    assert "Bbm" in harmony.chords  # Modal interchange

    # 3. Apply groove feel
    groove = apply_feel(harmony, feel="laid_back")

    # 4. Export to MIDI
    midi_path = export_to_midi(groove, "kelly_song.mid")

    # 5. Validate output
    assert Path(midi_path).exists()
    validate_midi_tempo(midi_path, expected_range=(78, 86))
    validate_chord_progression(midi_path, expected=["F", "C", "Bbm", "F"])
```

---

## Sprint 3: Compliance & Documentation (🟡 MEDIUM PRIORITY)

### TODO-8: Complete Type Hints for All Modules
**Priority:** 🟡 MEDIUM | **Effort:** M (4 hours) | **Risk:** Low - Maintainability

**Description:**
Add missing type hints to utility functions and orchestrator modules. Achieve 100% type hint coverage on public APIs.

**Files to Update:**
- `music_brain/utils/instruments.py`
- `music_brain/utils/ppq.py`
- `music_brain/utils/midi_io.py`
- `music_brain/orchestrator/processors/*.py`

**Acceptance Criteria:**
- [ ] `mypy music_brain/` passes with no errors
- [ ] All public functions have complete type hints
- [ ] Return types specified for all functions
- [ ] Complex types use TypeAlias
- [ ] Generic types properly parameterized (List[int], Dict[str, Any])

**Example:**
```python
# Before
def process_notes(notes):
    return [n + 12 for n in notes]

# After
from typing import List

def process_notes(notes: List[int]) -> List[int]:
    """Transpose notes up one octave."""
    return [n + 12 for n in notes]
```

**Validation:**
```bash
mypy music_brain/ --strict
# Should pass with 0 errors
```

---

### TODO-9: Complete Documentation for Public APIs
**Priority:** 🟡 MEDIUM | **Effort:** M (6 hours) | **Risk:** Low - User experience

**Description:**
Add missing parameter descriptions, document exceptions, add usage examples, and generate API reference.

**Acceptance Criteria:**
- [ ] All public functions have complete Google-style docstrings
- [ ] All parameters described with types
- [ ] All return values documented
- [ ] All exceptions documented in Raises section
- [ ] Complex functions have Examples section
- [ ] Module-level docstrings present
- [ ] Sphinx API reference generates successfully

**Docstring Template:**
```python
def apply_groove(
    midi_path: str,
    groove: Optional[GrooveTemplate] = None,
    genre: Optional[str] = None,
    intensity: float = 0.5,
) -> str:
    """
    Apply groove template to a MIDI file.

    Takes a groove template (extracted or preset) and applies its
    timing and velocity characteristics to the input MIDI file.

    Args:
        midi_path: Path to input MIDI file. Must exist and be readable.
        groove: GrooveTemplate to apply. Mutually exclusive with genre.
        genre: Genre template name ('funk', 'jazz', 'rock').
            Mutually exclusive with groove.
        intensity: How strongly to apply groove (0.0-1.0).
            0.0 = no change, 1.0 = full groove effect.

    Returns:
        Path to output MIDI file (default: input_grooved.mid).

    Raises:
        ValueError: If neither groove nor genre is provided.
        FileNotFoundError: If input MIDI file doesn't exist.
        ImportError: If mido package is not installed.

    Examples:
        >>> # Apply funk groove
        >>> output = apply_groove("drums.mid", genre="funk", intensity=0.7)
        >>> print(output)
        drums_grooved.mid

        >>> # Use custom template
        >>> template = extract_groove("reference.mid")
        >>> output = apply_groove("track.mid", groove=template)

    Note:
        Requires mido package: pip install mido
    """
```

---

## Sprint 4: Performance & Advanced Testing (🟢 LOW PRIORITY)

### TODO-10: Performance Benchmarks
**Priority:** 🟢 LOW | **Effort:** M (4 hours) | **Risk:** Low - Optimization

**Description:**
Establish performance benchmarks and regression detection for critical operations.

**Performance Targets:**
- Harmony generation: < 100ms for typical progression
- MIDI processing: > 1000 notes/second
- Audio analysis: < 5 seconds for 3-minute song
- Groove application: < 500ms for typical file
- Memory usage: < 500MB for typical operations

**Acceptance Criteria:**
- [ ] Benchmark suite created
- [ ] Baseline measurements recorded
- [ ] Regression detection in CI/CD
- [ ] Memory profiling setup
- [ ] Performance targets documented

**Benchmark Tests:**
```python
@pytest.mark.benchmark
def test_harmony_generation_performance(benchmark):
    """Benchmark: Harmony generation < 100ms"""
    generator = HarmonyGenerator()
    result = benchmark(
        generator.generate_basic_progression,
        key="C",
        mode="major",
        pattern="I-V-vi-IV"
    )
    assert benchmark.stats.mean < 0.1  # 100ms
```

**CI/CD Integration:**
```yaml
- name: Run Performance Benchmarks
  run: pytest tests/ --benchmark-only --benchmark-autosave

- name: Compare to Baseline
  run: pytest-benchmark compare --fail-on-regression
```

---

### TODO-11: Property-Based Testing with Hypothesis
**Priority:** 🟢 LOW | **Effort:** M (6 hours) | **Risk:** Low - Robustness

**Description:**
Use Hypothesis for property-based testing to discover edge cases automatically.

**Acceptance Criteria:**
- [ ] Hypothesis integrated
- [ ] Key invariants tested
- [ ] Fuzzing reveals no crashes
- [ ] Edge cases auto-discovered

**Property Tests:**
```python
from hypothesis import given, strategies as st

@given(st.lists(st.integers(0, 127), min_size=1, max_size=100))
def test_midi_notes_always_in_range(notes):
    """Property: MIDI notes always remain 0-127"""
    processed = process_notes(notes)
    assert all(0 <= n <= 127 for n in processed)

@given(st.floats(0.0, 1.0))
def test_groove_intensity_idempotent(intensity):
    """Property: apply(intensity) ≈ apply(intensity/2) twice"""
    # Test mathematical properties
    pass
```

---

### TODO-12: Voice & Text Module Tests
**Priority:** 🟢 LOW | **Effort:** M (4 hours) | **Risk:** Low - Future features

**Description:**
Tests for voice synthesis and lyric generation modules.

**Acceptance Criteria:**
- [ ] Voice synthesis basic tests
- [ ] Auto-tune application tested
- [ ] Lyric generation from intent
- [ ] Syllable-to-melody mapping
- [ ] Mock audio processing for CI/CD

---

## Tracking & Metrics

### Coverage Targets by Module

| Module | Current | Sprint 1 Target | Final Target |
|--------|---------|----------------|--------------|
| `harmony.py` | 0% → 95% | 95% | 95% |
| `emotion_api.py` | 0% → 95% | 95% | 95% |
| `groove/` | 60% | 85% | 90% |
| `structure/` | 30% | 85% | 90% |
| `audio/` | 15% | 75% | 85% |
| `daw/` | 25% | 80% | 85% |
| `session/` | 65% | 80% | 85% |
| **Overall** | 55% → 70% | 85% | 90% |

### Sprint Planning

**Sprint 1 (Week 1-2): Critical Coverage**
- TODO-1, TODO-2, TODO-3, TODO-4
- Goal: 85% coverage on HIGH priority modules
- Deliverable: Structure, Audio, Error handling tested

**Sprint 2 (Week 3-4): Integration & DAW**
- TODO-5, TODO-6, TODO-7
- Goal: End-to-end workflows validated
- Deliverable: DAW integrations tested, Kelly song workflow

**Sprint 3 (Week 5): Compliance**
- TODO-8, TODO-9
- Goal: 100% type hints and docs on public APIs
- Deliverable: API reference, mypy passes

**Sprint 4 (Week 6): Performance**
- TODO-10, TODO-11, TODO-12
- Goal: Performance benchmarks established
- Deliverable: Regression detection, property tests

---

## Definition of Done

A TODO is complete when:
- [ ] All test cases written and passing
- [ ] Coverage target achieved
- [ ] No regressions in existing tests
- [ ] Documentation updated
- [ ] PR reviewed and approved
- [ ] CI/CD pipeline passing

---

## Notes

- Prioritize based on risk: Critical path items first
- Mock external dependencies (librosa, DAWs) for CI/CD
- Use small test fixtures to keep tests fast
- Document expected failures and edge cases
- Maintain test performance: Full suite < 5 minutes
