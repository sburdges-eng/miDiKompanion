# Lyric Generation Guide

## Overview

The lyric generation system in Kelly MIDI Companion generates structured, emotionally authentic lyrics based on user input emotions and wound descriptions. The system uses semantic expansion, prosody analysis, and rhyme detection to create lyrics that match the emotional tone of the music.

## Architecture

```
LyricGenerator
    ├── SemanticExpander - Maps emotions to vocabulary
    ├── ProsodyAnalyzer - Handles meter and stress patterns
    └── RhymeEngine - Generates rhyming word pairs
```

## Basic Usage

### Generating Lyrics

```cpp
#include "voice/LyricGenerator.h"

// Create generator
LyricGenerator generator;

// Set parameters
generator.setStructureType("verse_chorus");
generator.setRhymeScheme("ABAB");
generator.setLineLength(8);  // 8 syllables per line

// Define emotion and wound
EmotionNode emotion;
emotion.name = "joy";
emotion.category = EmotionCategory::Joy;
emotion.valence = 0.8f;
emotion.arousal = 0.7f;
emotion.dominance = 0.6f;
emotion.intensity = 0.8f;

Wound wound;
wound.description = "feeling happy and free";

// Generate lyrics
auto result = generator.generateLyrics(emotion, wound, &midiContext);

// Access generated lyrics
for (const auto& section : result.structure.sections) {
    for (const auto& line : section.lines) {
        std::cout << line.text << std::endl;
    }
}
```

## Structure Types

The lyric generator supports several structure types:

- **verse_chorus**: Standard verse-chorus pattern (V-C-V-C-B-C)
- **ballad**: Multiple verses with choruses (V-V-C-V-V-C)
- **pop**: Includes pre-chorus sections
- **verse_only**: Verse-only structure

### Setting Structure Type

```cpp
generator.setStructureType("verse_chorus");
```

## Rhyme Schemes

Supported rhyme schemes:

- **ABAB**: Alternating rhyme (lines 1 & 3, 2 & 4 rhyme)
- **AABB**: Couplet rhyme (lines 1 & 2, 3 & 4 rhyme)
- **ABBA**: Enclosed rhyme (lines 1 & 4, 2 & 3 rhyme)
- **ABCB**: Second and fourth lines rhyme
- **AAAA**: All lines rhyme
- **ABAC**: First and third lines rhyme
- **AABA**: First, second, and fourth lines rhyme

### Setting Rhyme Scheme

```cpp
generator.setRhymeScheme("ABAB");
```

## Line Lengths

Set the target number of syllables per line:

```cpp
generator.setLineLength(8);  // 8 syllables per line
```

Common syllable counts: 4, 6, 8, 10, 12

## Lyric Styles

You can customize the lyric style:

```cpp
generator.setLyricStyle("poetic");  // or "conversational", "metaphorical"
```

## Emotion-to-Vocabulary Mapping

The system maps emotions to vocabulary using:

1. **Emotion Categories**: Each category (Joy, Sadness, Anger, etc.) has associated word lists
2. **VAD Values**: Valence, Arousal, and Dominance values influence word choice
3. **Wound Keywords**: Keywords extracted from wound descriptions are incorporated

### Valence Mapping

- High valence (>0.5) → Positive imagery, bright metaphors
- Low valence (<-0.5) → Darker imagery, introspective language

### Arousal Mapping

- High arousal (>0.7) → Action verbs, dynamic words
- Low arousal (<0.3) → Contemplative, slow-paced words

### Dominance Mapping

- High dominance (>0.7) → Powerful, assertive words
- Low dominance (<0.3) → Soft, vulnerable words

## Prosody and Meter

The system analyzes prosody (rhythm and stress):

- **Stress Detection**: Identifies stressed and unstressed syllables (0=unstressed, 1=secondary, 2=primary)
- **Meter Matching**: Supports iambic, trochaic, anapestic, and dactylic meters
- **Line Validation**: Validates lines against target syllable counts and meter patterns

## Rhyme Generation

The rhyme engine:

- Detects perfect and slant rhymes phonetically
- Generates rhyming word pairs following rhyme schemes
- Builds rhyme databases from phoneme sequences

## Output Structure

The generated lyrics include:

- **LyricStructure**: Overall structure with sections (verse, chorus, bridge)
- **LyricLine**: Individual lines with text, syllables, stress patterns, and meter
- **Syllable**: Breakdown of each syllable with phonemes and stress levels
- **RhymeScheme**: Applied rhyme pattern

## Advanced Usage

### Custom Vocabulary

You can load custom emotion vocabulary:

```cpp
generator.loadEmotionVocabulary("/path/to/emotion/data");
```

### Loading Templates

Load custom lyric templates:

```cpp
generator.loadTemplates("/path/to/lyric_templates.json");
```

## Integration with Vocal Synthesis

Generated lyrics can be integrated with vocal synthesis:

```cpp
// Generate lyrics
auto lyricResult = generator.generateLyrics(emotion, wound, &midiContext);

// Generate vocal melody with lyrics
auto vocalNotes = voiceSynthesizer.generateVocalMelody(
    emotion,
    midiContext,
    &lyricResult.structure
);
```

The lyrics are automatically aligned to the vocal melody, mapping syllables to notes.

## Best Practices

1. **Emotion Specificity**: More specific emotion descriptions yield better lyric quality
2. **Wound Descriptions**: Detailed wound descriptions provide more keywords for lyric generation
3. **Structure Selection**: Choose structure types that match your musical style
4. **Rhyme Scheme**: Simpler schemes (AABB) are easier to generate naturally than complex ones
5. **Line Length**: 6-8 syllables per line works well for most styles

## Limitations

- Current implementation uses rule-based G2P (grapheme-to-phoneme) conversion
- Full CMU dictionary integration is planned for future versions
- Rhyme quality depends on available vocabulary
- Meter matching is simplified; full prosodic analysis is a future enhancement

## Future Enhancements

- Full CMU Pronouncing Dictionary integration
- More sophisticated prosody analysis
- Machine learning-based word selection
- Multi-language support
- Rhyme quality scoring and optimization
