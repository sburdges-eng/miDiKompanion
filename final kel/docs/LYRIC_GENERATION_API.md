# Lyric Generator API Documentation

## LyricGenerator

### Overview

The `LyricGenerator` class generates structured lyrics based on emotional input and wound descriptions. It uses semantic expansion, prosody analysis, and rhyme detection to create emotionally authentic lyrics.

### Public API

#### Constructor

```cpp
LyricGenerator();
```

Creates a new lyric generator with default settings.

#### Generate Lyrics

```cpp
LyricResult generateLyrics(
    const EmotionNode& emotion,
    const Wound& wound,
    const GeneratedMidi* midiContext = nullptr
);
```

Generates structured lyrics based on emotion, wound description, and optional MIDI context.

**Parameters:**

- `emotion`: The emotion node with VAD values and category
- `wound`: Wound description providing context for lyrics
- `midiContext`: Optional MIDI context for timing/rhythm information

**Returns:** `LyricResult` containing lyric structure and lines

#### Configuration

```cpp
void setLyricStyle(const std::string& style);
void setStructureType(const std::string& type);
void setRhymeScheme(const std::string& scheme);
void setLineLength(int syllables);
```

Configure lyric generation:

- `style`: "poetic", "conversational", or "metaphorical"
- `type`: "verse_chorus", "ballad", "pop", etc.
- `scheme`: "ABAB", "AABB", "ABBA", etc.
- `syllables`: Target syllables per line (4, 6, 8, 10, 12)

#### Data Loading

```cpp
bool loadTemplates(const std::string& filePath);
bool loadEmotionVocabulary(const std::string& emotionDataPath);
```

Load custom templates and emotion vocabulary from JSON files.

### LyricResult Structure

```cpp
struct LyricResult {
    LyricStructure structure;      // Overall structure with sections
    std::vector<LyricLine> lines;  // All lyric lines
};
```

## PhonemeConverter

### Overview

Converts text to phonemes (IPA symbols) for vocal synthesis.

### Public API

#### Text Conversion

```cpp
std::vector<Phoneme> textToPhonemes(const std::string& text);
std::vector<std::string> wordToPhonemes(const std::string& word);
```

Convert text or words to phoneme sequences.

#### Syllable Analysis

```cpp
std::vector<std::string> splitIntoSyllables(const std::string& word);
std::vector<int> detectStress(const std::string& word);
int countSyllables(const std::string& word);
```

Analyze word structure:

- `splitIntoSyllables`: Split word into syllable strings
- `detectStress`: Get stress pattern (0=unstressed, 1=secondary, 2=primary)
- `countSyllables`: Count number of syllables

#### Phoneme Access

```cpp
Phoneme getPhonemeFromIPA(const std::string& ipa);
std::pair<std::array<float, 4>, std::array<float, 4>> getFormants(const Phoneme& phoneme);
std::pair<std::array<float, 4>, std::array<float, 4>> interpolatePhonemes(
    const Phoneme& p1,
    const Phoneme& p2,
    float t
);
```

Access phoneme data and formant information.

## PitchPhonemeAligner

### Overview

Aligns MIDI pitches to phoneme sequences, handling melisma and timing.

### Public API

#### Alignment

```cpp
AlignmentResult alignLyricsToMelody(
    const LyricStructure& lyrics,
    const std::vector<VoiceSynthesizer::VocalNote>& vocalNotes,
    const GeneratedMidi* midiContext = nullptr
);
```

Aligns lyrics to vocal melody, returning aligned phonemes and updated vocal notes.

#### Configuration

```cpp
void setBPM(float bpm);
void setAllowMelisma(bool allowMelisma);
void setPortamentoTime(double portamentoTime);
```

Configure alignment behavior:

- `bpm`: Beats per minute for timing
- `allowMelisma`: Allow multiple notes per syllable
- `portamentoTime`: Time for portamento between phonemes (in beats)

#### AlignedPhoneme Structure

```cpp
struct AlignedPhoneme {
    Phoneme phoneme;              // Phoneme data
    int midiPitch;                // MIDI pitch
    double startBeat;             // Start position in beats
    double duration;              // Duration in beats
    bool isStartOfSyllable;       // Syllable boundary flag
    bool isEndOfSyllable;         // Syllable boundary flag
};
```

## ExpressionEngine

### Overview

Applies vocal expression and dynamics to synthesis.

### Public API

#### Expression Application

```cpp
VocalExpression applyEmotionExpression(
    const VocalExpression& baseExpression,
    const EmotionNode& emotion
);
```

Applies emotion-based expression to base expression parameters.

#### Curve Generation

```cpp
std::vector<float> generateExpressionCurve(
    double duration,
    const VocalExpression& expression,
    int numPoints = 100
);

std::vector<float> generateDynamicsCurve(
    double duration,
    float crescendoAmount,
    float diminuendoAmount,
    int numPoints = 100
);

std::vector<float> generateVibratoCurve(
    double duration,
    float baseDepth,
    float variationAmount,
    int numPoints = 100
);
```

Generate expression curves over time for smooth parameter changes.

#### Real-Time Expression

```cpp
VocalExpression getExpressionAtPosition(
    const VocalExpression& expression,
    float position  // 0.0 = start, 1.0 = end
);
```

Get expression values at a specific position in a note for real-time synthesis.

## ProsodyAnalyzer

### Overview

Analyzes prosody (rhythm, meter, stress patterns) in lyrics.

### Public API

#### Stress Detection

```cpp
std::vector<int> detectStress(const std::string& word);
std::vector<int> detectStressPattern(const std::vector<std::string>& words);
```

Detect stress patterns (0=unstressed, 1=secondary, 2=primary).

#### Meter Analysis

```cpp
float matchMeter(const std::vector<int>& stressPattern, MeterType meterType);
MeterType detectMeter(const std::vector<int>& stressPattern);
MeterPattern getMeterPattern(MeterType meterType, int numSyllables);
```

Analyze and match meter patterns (iambic, trochaic, anapestic, dactylic).

#### Validation

```cpp
bool validateLineLength(const LyricLine& line, int targetSyllables);
int countSyllables(const std::string& word);
int countSyllables(const std::vector<std::string>& words);
```

Validate line lengths and count syllables.

## RhymeEngine

### Overview

Detects and generates rhymes for lyrics.

### Public API

#### Rhyme Detection

```cpp
RhymeMatch checkRhyme(const std::string& word1, const std::string& word2);
std::vector<RhymeMatch> findRhymes(
    const std::string& targetWord,
    const std::vector<std::string>& vocabulary,
    int maxResults = 10
);
```

Check if words rhyme and find rhyming words from vocabulary.

#### Rhyme Generation

```cpp
std::map<int, std::vector<std::string>> generateRhymeWords(
    const std::vector<int>& scheme,
    const std::vector<std::string>& vocabulary,
    const std::map<int, std::string>& existingWords = {}
);
```

Generate rhyming words following a rhyme scheme pattern.

#### Phoneme Analysis

```cpp
std::vector<std::string> extractEndPhonemes(const std::string& word, int numPhonemes = 3);
float comparePhonemeSequences(
    const std::vector<std::string>& phonemes1,
    const std::vector<std::string>& phonemes2
);
std::vector<RhymeMatch> detectInternalRhymes(const std::vector<std::string>& words);
```

Analyze phonemes for rhyme detection.

## Usage Examples

### Complete Lyric Generation and Synthesis

```cpp
// 1. Generate lyrics
LyricGenerator lyricGen;
lyricGen.setStructureType("verse_chorus");
lyricGen.setRhymeScheme("ABAB");
lyricGen.setLineLength(8);

auto lyricResult = lyricGen.generateLyrics(emotion, wound, &midiContext);

// 2. Generate vocal melody with lyrics
VoiceSynthesizer voiceSynth;
voiceSynth.setEnabled(true);
voiceSynth.prepare(44100.0);
voiceSynth.setVoiceType(VoiceType::Female);

auto vocalNotes = voiceSynth.generateVocalMelody(
    emotion,
    midiContext,
    &lyricResult.structure
);

// 3. Synthesize audio
auto audio = voiceSynth.synthesizeAudio(vocalNotes, 44100.0, &emotion);
```

### Phoneme Analysis

```cpp
PhonemeConverter converter;

// Analyze word
std::string word = "beautiful";
auto syllables = converter.splitIntoSyllables(word);  // ["beau", "ti", "ful"]
auto stress = converter.detectStress(word);            // [2, 0, 0]
int count = converter.countSyllables(word);            // 3

// Convert to phonemes
auto phonemes = converter.wordToPhonemes(word);
for (const auto& phoneme : phonemes) {
    std::cout << phoneme.ipa << " ";
}
```

### Expression Control

```cpp
ExpressionEngine exprEngine;

// Apply emotion to expression
VocalExpression baseExpr;
auto expression = exprEngine.applyEmotionExpression(baseExpr, emotion);

// Generate dynamics curve
auto dynamicsCurve = exprEngine.generateDynamicsCurve(
    4.0,    // 4 beats
    0.5f,   // 50% crescendo
    0.3f    // 30% diminuendo
);
```
