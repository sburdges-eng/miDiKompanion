# Python Bridge - Quick Start

## 5-Minute Setup

```bash
# 1. Build the bridge
cmake -B build -DBUILD_PYTHON_BRIDGE=ON
cmake --build build

# 2. Test it works
cd python
python3 -c "import kelly_bridge; print('Success!')"

# 3. Run an example
cd examples
python3 basic_usage.py
```

## Basic Usage

```python
from kelly import Kelly

# Create instance
kelly = Kelly(tempo=120)

# Generate MIDI from emotion
result, midi = kelly.generate("feeling of loss", intensity=0.8)

# Check result
print(f"Emotion: {result.emotion.name}")
print(f"Notes: {len(midi)}")

# Export MIDI
kelly.export_midi(midi, "output.mid")
```

## Common Tasks

### Generate from emotion name
```python
result, midi = kelly.generate_from_emotion("grief", bars=4)
```

### Generate from VAI values
```python
midi = kelly.generate_from_vai(
    valence=-0.7,    # Negative emotion
    arousal=0.3,     # Low energy
    intensity=0.8,   # High intensity
    bars=4
)
```

### Find emotion
```python
emotion = kelly.find_emotion_by_name("joy")
emotion = kelly.find_emotion(valence=0.8, arousal=0.7, intensity=0.6)
```

### Get emotions by category
```python
from kelly import EmotionCategory
sad_emotions = kelly.get_emotions_by_category(EmotionCategory.Sadness)
```

## Troubleshooting

**Import error?**
- Check module exists: `ls python/kelly_bridge*`
- Add to path: `export PYTHONPATH=$PWD/python:$PYTHONPATH`

**Build error?**
- Install Python dev headers: `brew install python3` (macOS)
- See `BUILD_PYTHON_BRIDGE.md` for details

## More Info

- Full docs: `python/README.md`
- Examples: `python/examples/`
- Build guide: `BUILD_PYTHON_BRIDGE.md`
