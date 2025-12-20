# Vocal Synthesis Guide

## Overview

The vocal synthesis system in Kelly MIDI Companion generates realistic vocal audio using formant-based synthesis. It supports multiple voice types, expression control, and real-time synthesis for integration with MIDI generation.

## Architecture

```
VoiceSynthesizer
    ├── VocoderEngine - Formant-based synthesis
    ├── PitchPhonemeAligner - Aligns lyrics to melody
    ├── ExpressionEngine - Applies vocal expression
    └── PhonemeConverter - Converts text to phonemes
```

## Basic Usage

### Setting Up the Synthesizer

```cpp
#include "voice/VoiceSynthesizer.h"

// Create synthesizer
VoiceSynthesizer synthesizer;
synthesizer.setEnabled(true);
synthesizer.prepare(44100.0);  // Sample rate
synthesizer.setBPM(120.0f);    // Tempo
```

### Generating Vocal Melody

```cpp
// Define emotion
EmotionNode emotion;
emotion.valence = 0.8f;
emotion.arousal = 0.7f;
emotion.dominance = 0.6f;
emotion.intensity = 0.8f;

// Generate MIDI context
GeneratedMidi midiContext;
midiContext.lengthInBeats = 16.0;
midiContext.bpm = 120.0f;

// Generate vocal melody (optionally with lyrics)
LyricStructure* lyrics = nullptr;  // Can provide lyrics here
auto vocalNotes = synthesizer.generateVocalMelody(
    emotion,
    midiContext,
    lyrics
);
```

### Synthesizing Audio

#### Offline Synthesis

```cpp
// Synthesize entire audio buffer
auto audio = synthesizer.synthesizeAudio(
    vocalNotes,
    44100.0,  // Sample rate
    &emotion  // Optional: emotion for vocal characteristics
);

// audio is now a vector<float> containing audio samples
```

#### Real-Time Synthesis

```cpp
// In audio callback
float outputBuffer[512];
int numSamples = 512;
int64_t currentSample = getCurrentPlaybackSample();

synthesizer.synthesizeBlock(
    vocalNotes,
    outputBuffer,
    numSamples,
    currentSample,
    &emotion  // Optional: emotion for vocal characteristics
);
```

## Voice Types

The system supports four voice types with different formant characteristics:

### Setting Voice Type

```cpp
synthesizer.setVoiceType(VoiceType::Female);
```

### Available Voice Types

- **Neutral**: Default voice with no formant shifts (pitch range: C3-C6)
- **Male**: Lower formants, deeper pitch range (C3-F#5)
- **Female**: Higher formants, higher pitch range (C4-F#6)
- **Child**: Highest formants, highest pitch range (C5-C7)

Each voice type applies formant frequency shifts:

- Male: ~85% of base formants
- Female: ~115% of base formants
- Child: ~130% of base formants

## Expression Control

### Vocal Expression Parameters

The `VocalExpression` struct contains:

- **dynamics**: Overall dynamics (0.0 to 1.0)
- **articulation**: Legato (0.0) to staccato (1.0)
- **vibratoDepth**: Vibrato depth variation (0.0 to 1.0)
- **vibratoRate**: Vibrato rate in Hz (typically 3-7 Hz)
- **breathiness**: Breathiness amount (0.0 to 1.0)
- **brightness**: Brightness modulation (0.0 to 1.0)
- **crescendo**: Crescendo amount (0.0 to 1.0)
- **diminuendo**: Diminuendo amount (0.0 to 1.0)

### Emotion-Based Expression

Expression is automatically applied based on emotion:

```cpp
// Expression is automatically mapped from emotion
// Valence → Brightness
// Arousal → Vibrato Rate
// Dominance → Dynamics
// Intensity → Vibrato Depth
```

## Phoneme Conversion

### Converting Text to Phonemes

```cpp
#include "voice/PhonemeConverter.h"

PhonemeConverter converter;

// Convert text to phonemes
std::string text = "hello world";
auto phonemes = converter.textToPhonemes(text);

// Convert single word
auto wordPhonemes = converter.wordToPhonemes("love");
```

### Syllable Analysis

```cpp
// Split word into syllables
auto syllables = converter.splitIntoSyllables("beautiful");

// Detect stress pattern
auto stress = converter.detectStress("beautiful");
// Returns: [2, 0, 0] (primary stress on first syllable)

// Count syllables
int count = converter.countSyllables("hello");  // Returns 2
```

## Pitch-Phoneme Alignment

### Aligning Lyrics to Melody

```cpp
#include "voice/PitchPhonemeAligner.h"

PitchPhonemeAligner aligner;
aligner.setBPM(120.0f);
aligner.setAllowMelisma(true);  // Allow multiple notes per syllable
aligner.setPortamentoTime(0.05);  // 50ms portamento between phonemes

// Align lyrics to vocal notes
auto alignmentResult = aligner.alignLyricsToMelody(
    lyricStructure,
    vocalNotes,
    &midiContext
);

// Access aligned phonemes
for (const auto& alignedPhoneme : alignmentResult.alignedPhonemes) {
    // Each phoneme has: pitch, timing, syllable boundaries
    int pitch = alignedPhoneme.midiPitch;
    double startBeat = alignedPhoneme.startBeat;
    double duration = alignedPhoneme.duration;
}
```

### Melisma Handling

Melisma (multiple notes per syllable) is supported:

```cpp
aligner.setAllowMelisma(true);  // Enable melisma
```

When enabled, longer notes can contain multiple phonemes, creating smoother vocal lines.

## Formant Synthesis

### How It Works

The vocoder uses formant synthesis, which models the vocal tract as a series of resonators:

1. **Glottal Pulse Generation**: Creates the source signal (vocal cord vibration)
2. **Formant Filtering**: Shapes the signal through 4 formant filters (F1-F4)
3. **Vibrato**: Adds pitch modulation for naturalness
4. **Breathiness**: Mixes in noise for breathy sounds
5. **Brightness**: Applies high-frequency emphasis

### Formant Parameters

Each phoneme has:

- **F1-F4**: Formant frequencies in Hz
- **B1-B4**: Formant bandwidths in Hz
- **Duration**: Typical phoneme duration in milliseconds
- **Voicing**: Whether the phoneme is voiced or unvoiced

### Formant Interpolation

Smooth transitions between phonemes:

```cpp
// Interpolate between two phonemes
auto interpolated = PhonemeConverter::interpolatePhonemes(
    phoneme1,
    phoneme2,
    0.5f  // 50% interpolation
);
```

## Expression Curves

### Generating Expression Curves

```cpp
#include "voice/ExpressionEngine.h"

ExpressionEngine expressionEngine;

VocalExpression expression;
expression.dynamics = 0.7f;
expression.crescendo = 0.5f;
expression.diminuendo = 0.3f;

// Generate dynamics curve over time
auto dynamicsCurve = expressionEngine.generateDynamicsCurve(
    4.0,  // Duration in beats
    0.5f, // Crescendo amount
    0.3f  // Diminuendo amount
);

// Generate vibrato curve
auto vibratoCurve = expressionEngine.generateVibratoCurve(
    4.0,  // Duration
    0.3f, // Base depth
    0.1f  // Variation amount
);
```

## Emotion-to-Vocal Mapping

The system maps emotions to vocal characteristics:

### Valence → Brightness

- Positive emotions → Brighter vocal tone
- Negative emotions → Darker vocal tone

### Arousal → Vibrato Rate

- High arousal → Faster vibrato (up to 7 Hz)
- Low arousal → Slower vibrato (down to 3 Hz)

### Intensity → Vibrato Depth

- High intensity → Deeper vibrato
- Low intensity → Subtle vibrato

### Dominance → Dynamics & Breathiness

- High dominance → Louder, less breathy
- Low dominance → Softer, more breathy

## MIDI Integration

### Generating MIDI with Lyrics

```cpp
// Generate vocal notes with lyrics
auto vocalNotes = synthesizer.generateVocalMelody(
    emotion,
    midiContext,
    &lyrics  // Lyric structure
);

// Export MIDI lyric events
auto lyricEvents = synthesizer.generateMidiLyricEvents(vocalNotes, 480);
// These can be included in MIDI file export as text events (0xFF 05)
```

## Performance Considerations

### Real-Time Requirements

- **CPU Usage**: ~5-10% on modern hardware
- **Memory**: Minimal (~1KB per voice instance)
- **Latency**: Zero-latency processing (suitable for real-time)

### Optimization Tips

1. Pre-generate phoneme databases
2. Cache formant calculations
3. Use appropriate buffer sizes for real-time synthesis
4. Profile critical paths (formant filtering, phoneme conversion)

## Troubleshooting

### No Audio Output

- Check that `setEnabled(true)` was called
- Verify sample rate is set correctly
- Ensure vocal notes are generated
- Check that notes are within voice type's pitch range

### Distorted/Clipping Audio

- Reduce overall gain
- Check formant filter stability
- Verify bandwidth values are positive

### Unrealistic Sound

- Adjust formant frequencies for different voice types
- Increase breathiness for softer sounds
- Modify glottal pulse shape for different timbres
- Try different voice types

### Timing Issues

- Verify BPM is set correctly
- Check beat-to-sample conversion
- Ensure portamento time is appropriate

## Advanced Topics

### Custom Formant Sets

You can extend the phoneme database with custom formants:

```cpp
// Load phoneme database from JSON
converter.loadPhonemeDatabase("/path/to/phonemes.json");
```

### Voice Cloning

Future versions will support importing formants from recorded voice samples for voice cloning.

### Multi-Phoneme Synthesis

The system supports smooth transitions between phonemes using formant interpolation, creating natural-sounding vocal synthesis.

## API Reference

See individual class documentation:

- `VoiceSynthesizer`: Main synthesis interface
- `VocoderEngine`: Formant synthesis engine
- `PhonemeConverter`: Text-to-phoneme conversion
- `PitchPhonemeAligner`: Lyric-melody alignment
- `ExpressionEngine`: Vocal expression control
- `LyriSync`: Lyric-vocal synchronization
