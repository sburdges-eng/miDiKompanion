# Example Datasets for Kelly MIDI Companion ML Training

This guide shows examples of how to structure your datasets for training.

## 1. Emotion Recognition Dataset

### Directory Structure

```
datasets/audio/
├── labels.csv
├── happy_001.wav
├── happy_002.wav
├── sad_001.wav
├── sad_002.wav
├── calm_001.wav
├── angry_001.wav
└── ...
```

### labels.csv Format

```csv
filename,valence,arousal
happy_001.wav,0.8,0.9
happy_002.wav,0.85,0.92
sad_001.wav,-0.6,0.3
sad_002.wav,-0.7,0.25
calm_001.wav,0.2,-0.5
angry_001.wav,-0.5,0.95
fearful_001.wav,-0.7,0.8
content_001.wav,0.6,0.2
```

**Emotion Quadrants**:
- **Happy**: High valence (+), high arousal (+)
- **Sad**: Low valence (-), low arousal (-)
- **Angry**: Low valence (-), high arousal (+)
- **Calm**: Moderate valence, low arousal (-)

### Example Emotion Labels

| Emotion | Valence | Arousal | Description |
|---------|---------|---------|-------------|
| Excited | 0.9 | 0.95 | Very positive, very energetic |
| Happy | 0.7 | 0.7 | Positive, energetic |
| Content | 0.6 | 0.2 | Positive, calm |
| Calm | 0.2 | 0.1 | Neutral, very calm |
| Sad | -0.6 | 0.3 | Negative, somewhat active |
| Depressed | -0.8 | 0.1 | Very negative, low energy |
| Angry | -0.5 | 0.9 | Negative, very energetic |
| Fearful | -0.7 | 0.8 | Negative, tense |

## 2. Melody Generation Dataset

### Directory Structure

```
datasets/midi/
├── emotion_labels.json
├── melody_001.mid
├── melody_002.mid
├── melody_003.mid
└── ...
```

### emotion_labels.json Format

```json
{
  "melody_001.mid": {
    "valence": 0.8,
    "arousal": 0.7,
    "genre": "pop",
    "key": "C major",
    "tempo": 120
  },
  "melody_002.mid": {
    "valence": -0.6,
    "arousal": 0.3,
    "genre": "ballad",
    "key": "A minor",
    "tempo": 70
  },
  "melody_003.mid": {
    "valence": 0.5,
    "arousal": 0.9,
    "genre": "electronic",
    "key": "E minor",
    "tempo": 128
  }
}
```

### MIDI File Requirements

- **Format**: Standard MIDI File (SMF)
- **Tracks**: Single melody track preferred
- **Length**: 4-32 bars
- **Notes**: Clear melodic content
- **Tempo**: Any (will be normalized)

## 3. Chord Progression Dataset

### Directory Structure

```
datasets/chords/
├── chord_progressions.json
└── emotion_chords.json
```

### chord_progressions.json Format

```json
{
  "progressions": [
    {
      "name": "I-V-vi-IV",
      "key": "C major",
      "chords": ["C", "G", "Am", "F"],
      "roman_numerals": ["I", "V", "vi", "IV"],
      "emotion": {
        "valence": 0.7,
        "arousal": 0.6,
        "description": "Uplifting pop progression"
      },
      "examples": [
        "Let It Be - The Beatles",
        "Someone Like You - Adele"
      ]
    },
    {
      "name": "ii-V-I",
      "key": "C major",
      "chords": ["Dm7", "G7", "Cmaj7"],
      "roman_numerals": ["ii7", "V7", "Imaj7"],
      "emotion": {
        "valence": 0.5,
        "arousal": 0.4,
        "description": "Classic jazz resolution"
      },
      "examples": ["Standard jazz"]
    },
    {
      "name": "i-VII-VI-V",
      "key": "A minor",
      "chords": ["Am", "G", "F", "E"],
      "roman_numerals": ["i", "VII", "VI", "V"],
      "emotion": {
        "valence": -0.3,
        "arousal": 0.7,
        "description": "Andalusian cadence - dramatic"
      },
      "examples": ["Hit the Road Jack"]
    }
  ]
}
```

## 4. Dynamics Dataset

Uses the same MIDI files from `datasets/midi/` but extracts velocity information.

### Example MIDI with Good Dynamics

**Properties**:
- Velocity range: 40-120
- Dynamic variation within phrases
- Crescendos and diminuendos
- Accent notes (higher velocity)

**Bad Example** (avoid):
- All notes at velocity 64
- No variation
- Robotic feel

## 5. Drum Pattern Dataset

### Directory Structure

```
datasets/drums/
├── drum_labels.json
├── drum_001.mid
├── drum_002.mid
└── ...
```

### drum_labels.json Format

```json
{
  "drum_001.mid": {
    "style": "rock_beat",
    "tempo": 120,
    "feel": "straight",
    "complexity": 0.3,
    "emotion": {
      "valence": 0.6,
      "arousal": 0.7
    }
  },
  "drum_002.mid": {
    "style": "jazz_swing",
    "tempo": 160,
    "feel": "swing",
    "complexity": 0.6,
    "emotion": {
      "valence": 0.5,
      "arousal": 0.6
    }
  },
  "drum_003.mid": {
    "style": "latin_clave",
    "tempo": 110,
    "feel": "syncopated",
    "complexity": 0.7,
    "emotion": {
      "valence": 0.8,
        "arousal": 0.8
    }
  }
}
```

### Drum MIDI Requirements

- **GM Mapping**: Use General MIDI drum mapping
  - Kick: Note 36 (C1)
  - Snare: Note 38 (D1)
  - Hi-hat: Note 42 (F#1), 44, 46
  - Toms: Notes 41, 43, 45, 47, 48, 50
  - Cymbals: Notes 49, 51, 57

- **Length**: 1-4 bar patterns
- **Quantization**: Avoid over-quantization
- **Humanization**: Include slight timing/velocity variations

## Dataset Sizes

### Minimum for Training
- **EmotionRecognizer**: 100 labeled audio clips
- **MelodyTransformer**: 50 MIDI files with emotion labels
- **HarmonyPredictor**: 20 chord progressions
- **DynamicsEngine**: Same as MelodyTransformer
- **GroovePredictor**: 30 drum patterns

### Recommended for Good Results
- **EmotionRecognizer**: 1,000+ audio clips
- **MelodyTransformer**: 500+ MIDI files
- **HarmonyPredictor**: 100+ progressions
- **DynamicsEngine**: 500+ MIDI files
- **GroovePredictor**: 200+ drum patterns

### Professional Quality
- **EmotionRecognizer**: 10,000+ (e.g., DEAM dataset)
- **MelodyTransformer**: 5,000+ (e.g., Lakh subset)
- **HarmonyPredictor**: 1,000+ progressions
- **DynamicsEngine**: 5,000+ expressive performances
- **GroovePredictor**: 1,000+ (e.g., Groove MIDI)

## Data Collection Tips

### 1. Diverse Coverage

**Emotion Space**:
- Cover all quadrants (happy, sad, angry, calm)
- Include edge cases (very high/low valence/arousal)
- Mix genres and styles

**Musical Variety**:
- Different keys (major, minor, modes)
- Various tempos (slow to fast)
- Multiple genres (pop, jazz, classical, electronic)

### 2. Quality Over Quantity

**Audio Quality**:
- Clean recordings (minimal noise)
- Consistent volume levels
- Good spectral balance

**MIDI Quality**:
- Musical content (not random)
- Proper quantization (not too rigid)
- Realistic velocity curves

**Label Quality**:
- Consistent labeling scheme
- Multiple annotators (inter-rater reliability)
- Clear emotion descriptions

### 3. Balanced Datasets

Ensure roughly equal representation of:
- Positive vs. negative emotions
- High vs. low arousal
- Different musical genres
- Various complexity levels

## Synthetic Data Generation

For initial testing, generate synthetic data:

```python
# Example: Generate synthetic emotion data
import numpy as np

emotions = []
for _ in range(100):
    valence = np.random.uniform(-1, 1)
    arousal = np.random.uniform(0, 1)
    emotions.append((valence, arousal))
```

But **always train final models on real data** for best results.

## Data Augmentation

Expand your dataset through augmentation:

### Audio Augmentation
- Pitch shift: ±2 semitones
- Time stretch: 0.9x - 1.1x
- Add noise: Low level (SNR > 30dB)
- EQ changes: ±3dB

### MIDI Augmentation
- Transpose: ±6 semitones
- Tempo change: 0.8x - 1.2x
- Velocity randomization: ±10
- Timing humanization: ±10ms

## Next Steps

1. Gather or create your datasets following these examples
2. Run the preparation script: `python scripts/prepare_datasets.py`
3. Verify dataset structure and labels
4. Start training: `python scripts/train_all_models.py`

---

**Tip**: Start small with synthetic data to verify the pipeline works, then gradually add real data for better results.
