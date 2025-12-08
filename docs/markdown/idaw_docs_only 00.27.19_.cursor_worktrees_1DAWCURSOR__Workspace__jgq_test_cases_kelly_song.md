# Kelly Song End-to-End Test

## Test Case: "When I Found You Sleeping" - Grief/Intense/Grief

### Objective

Verify the canonical use case: generating music for the emotion combination **sad → intense → grief** produces musically appropriate output matching the Kelly song reference.

### Test Steps

1. **Open Side B (Therapeutic Interface)**
   - Navigate to Side B in the iDAW interface
   - Verify emotion wheel is visible

2. **Load Emotions**
   - Click "Load Emotions" button
   - Verify 6×6×6 emotion thesaurus loads successfully
   - Verify emotion wheel displays all base emotions

3. **Select Emotion: sad → intense → grief**
   - Click on "sad" base emotion
   - Select "intense" intensity level
   - Select "grief" specific emotion
   - Verify selection displays: "sad → intense → grief"

4. **Generate Music**
   - Click "Generate Music" button
   - Verify API call is made with correct emotion parameters:

     ```json
     {
       "intent": {
         "base_emotion": "sad",
         "intensity": "intense",
         "specific_emotion": "grief"
       }
     }
     ```

5. **Verify MIDI Generation**
   - Verify MIDI file is generated
   - Verify MIDI data is returned as base64
   - Verify music_config is returned with:
     - Key: F (or F minor)
     - Mode: Aeolian (minor)
     - Tempo: ~130 BPM (intense maps to 130, but Kelly example uses 82 - can be overridden)
     - Progression: ["i", "VI", "III", "VII"] (F-C-Am-Dm in F minor)

6. **Download MIDI**
   - Click "Download MIDI" button
   - Verify .mid file downloads
   - Verify filename includes timestamp

7. **Open in Logic Pro**
   - Import downloaded MIDI file into Logic Pro
   - Verify:
     - Tempo is correct (82 BPM if overridden, or 130 BPM default)
     - Key signature is F minor
     - Chord progression matches: F-C-Am-Dm (i-VI-III-VII in F minor)
     - Notes are playable and sound appropriate

8. **Audio Preview (Browser)**
   - Click "Play" button in Audio Preview
   - Verify MIDI plays in browser
   - Verify playback controls work (play/stop)
   - Verify progress bar updates

### Expected Results

#### Music Parameters

- **Key**: F minor (F Aeolian)
- **Progression**: i-VI-III-VII (F-C-Am-Dm)
- **Tempo**: 82 BPM (Kelly reference) or 130 BPM (intense default)
- **Mode**: Aeolian (natural minor)
- **Dynamics**: ff (fortissimo - very loud, for intense)
- **Articulation**: legato (smooth, connected - appropriate for grief)

#### Emotional Accuracy

- Music should feel melancholic and emotionally appropriate for grief
- Progression should have the characteristic minor key with borrowed major chords
- Tempo should support the emotional weight (slower for grief, but can be intense)

#### Technical Validation

- MIDI file opens correctly in DAW
- All notes are in correct key
- Chord voicings are appropriate
- No audio glitches or errors

### Test Results

**Date**: [To be filled]
**Tester**: [To be filled]
**Status**: [PASS / FAIL]

#### Actual Results

- [ ] Emotion selection works correctly
- [ ] API call contains correct parameters
- [ ] MIDI file generated successfully
- [ ] Music config matches expected values
- [ ] MIDI downloads correctly
- [ ] MIDI opens in Logic Pro
- [ ] Audio preview works in browser
- [ ] Emotional accuracy validated

#### Notes

- Tempo mapping: "intense" maps to 130 BPM by default, but Kelly example uses 82 BPM
  - Solution: Can override via technical.bpm parameter
  - Or adjust intensity mapping for grief specifically

### Screenshots

[Add screenshots of UI, Logic Pro, etc.]

### MIDI Analysis

[Add analysis of generated MIDI file]

### Emotional Accuracy Assessment

[Add assessment of whether the music captures the intended emotion]
