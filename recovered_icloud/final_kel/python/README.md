# Kelly MIDI Companion - Python-C++ Bridge

Python interface for the Kelly MIDI Companion C++ engine, enabling advanced features and scripting capabilities.

## Overview

The Python-C++ bridge exposes the full Kelly MIDI Companion functionality to Python, allowing you to:

- Process emotional wounds and generate MIDI
- Query the 216-node emotion thesaurus
- Generate MIDI from emotions, VAI values, or text descriptions
- Export MIDI files
- Build custom workflows and scripts

## Installation

### Prerequisites

- Python 3.8 or higher
- CMake 3.22 or higher
- C++20 compiler (Clang, GCC, or MSVC)
- pybind11 (automatically fetched by CMake)

### Building the Bridge

```bash
# Configure with Python bridge enabled
cmake -B build -DBUILD_PYTHON_BRIDGE=ON

# Build
cmake --build build

# The module will be in python/kelly_bridge.*.so (Linux/Mac) or python/kelly_bridge.*.pyd (Windows)
```

### Python Package Setup

```bash
cd python
pip install -e .
```

Or install dependencies manually:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from kelly import Kelly

# Initialize
kelly = Kelly(tempo=120)

# Generate MIDI from emotional description
result, midi = kelly.generate("feeling of loss", intensity=0.8, bars=4)

print(f"Emotion: {result.emotion.name}")
print(f"Generated {len(midi)} MIDI notes")

# Export to MIDI file
kelly.export_midi(midi, "output.mid")
```

## API Reference

### Kelly Class

Main high-level interface:

```python
kelly = Kelly(tempo=120, seed=None)

# Generate from description
result, midi = kelly.generate(description, intensity=0.7, source="internal", bars=4)

# Generate from emotion name
result, midi = kelly.generate_from_emotion("grief", bars=4)

# Generate from VAI values
midi = kelly.generate_from_vai(valence, arousal, intensity, key="C", mode="Aeolian", bars=4)

# Find emotions
emotion = kelly.find_emotion(valence, arousal, intensity)
emotion = kelly.find_emotion_by_name("grief")
emotions = kelly.get_emotions_by_category(EmotionCategory.Sadness)

# Export MIDI
kelly.export_midi(notes, "output.mid", tempo=120)
```

### Low-Level API

Direct access to C++ classes:

```python
from kelly import KellyBrain, IntentPipeline, EmotionThesaurus, Wound

# KellyBrain - Core engine
brain = KellyBrain(tempo=120)
result = brain.process_wound("feeling lost", intensity=0.8)
midi = brain.generate_midi(result, bars=4)

# IntentPipeline - Advanced processing
pipeline = IntentPipeline()
result = pipeline.process(wound)
journey_result = pipeline.process_journey(side_a, side_b)

# EmotionThesaurus - Query emotions
thesaurus = pipeline.thesaurus()
emotion = thesaurus.find_by_name("grief")
emotions = thesaurus.get_nearby(valence, arousal, intensity, threshold=0.5)
```

## Examples

See `python/examples/` for complete examples:

- `basic_usage.py` - Core functionality
- `advanced_features.py` - Advanced features and workflows

Run examples:

```bash
cd python/examples
python basic_usage.py
python advanced_features.py
```

## Data Types

### EmotionNode

```python
emotion.id              # int
emotion.name            # str
emotion.category        # EmotionCategory
emotion.valence         # float (-1.0 to 1.0)
emotion.arousal         # float (0.0 to 1.0)
emotion.intensity       # float (0.0 to 1.0)
emotion.tempoModifier   # float
emotion.preferredMode   # str
```

### MidiNote

```python
note.pitch        # int (0-127)
note.velocity     # int (0-127)
note.startBeat    # float
note.duration     # float
```

### IntentResult

```python
result.wound           # Wound
result.emotion         # EmotionNode
result.ruleBreaks      # List[RuleBreak]
result.musicalParams   # MusicalParameters
```

## Advanced Usage

### Emotional Journey (Side A → Side B)

```python
from kelly import IntentPipeline, SideA, SideB

pipeline = IntentPipeline()

side_a = SideA()
side_a.description = "feeling lost"
side_a.intensity = 0.8

side_b = SideB()
side_b.description = "finding peace"
side_b.intensity = 0.6

journey = pipeline.process_journey(side_a, side_b)
midi = kelly.brain.generate_midi(journey, bars=8)
```

### Custom Wound Processing

```python
from kelly import Wound, IntentPipeline

pipeline = IntentPipeline()

wound = Wound()
wound.description = "complex mix of emotions"
wound.intensity = 0.7
wound.context = "transition period"
wound.triggers = ["change", "uncertainty"]

result = pipeline.process(wound)
```

### Batch Generation

```python
emotions = ["grief", "joy", "anger", "fear"]
results = []

for emotion_name in emotions:
    result, midi = kelly.generate_from_emotion(emotion_name, bars=4)
    results.append((emotion_name, result, midi))
```

## Troubleshooting

### Import Error

If you get `ImportError: Could not import kelly_bridge`:

1. Make sure the bridge is built: `cmake --build build`
2. Check that `python/kelly_bridge.*.so` (or `.pyd` on Windows) exists
3. Add the `python/` directory to `PYTHONPATH` if needed

### Build Errors

- **Python not found**: Install Python development headers
  - macOS: `brew install python3`
  - Ubuntu: `sudo apt-get install python3-dev`
  - Windows: Install Python from python.org with "Development" option

- **pybind11 errors**: CMake should fetch it automatically, but you can install manually:
  ```bash
  pip install pybind11
  ```

## Performance Notes

- The C++ engine is optimized for performance
- Python overhead is minimal (mainly data conversion)
- For batch processing, consider using the C++ API directly
- MIDI generation is fast (< 10ms for 4 bars typically)

## License

MIT License - See main project LICENSE file

## Support

For issues or questions:
1. Check the main project README
2. Review example code in `python/examples/`
3. Check build logs in `build/`
