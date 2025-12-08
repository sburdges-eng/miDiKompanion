# Voice Synthesis Research

## Objective

Research and prototype voice synthesis options for generating spoken/sung lyrics in iDAW.

## Options Evaluated

### 1. macOS AVSpeechSynthesizer (Built-in TTS)

**Pros:**

- Built into macOS, no external dependencies
- Free and immediately available
- Good quality for spoken text
- Easy to integrate

**Cons:**

- Limited to spoken voice, not singing
- Robotic/mechanical sound
- No emotional control
- macOS only

**Implementation:**

```python
import subprocess

def speak_text(text):
    subprocess.run(['say', text])
```

**Recommendation:** Good for prototyping, not for production singing.

---

### 2. ElevenLabs API (Cloud-based)

**Pros:**

- Very high quality, natural-sounding voices
- Emotional control and voice cloning
- Singing voice synthesis available
- Multiple voice options

**Cons:**

- Requires API key and paid subscription
- Cloud-based (requires internet)
- Rate limits and costs per character
- Privacy concerns (data sent to cloud)

**API Example:**

```python
import requests

def synthesize_elevenlabs(text, voice_id, api_key):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key}
    data = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
    response = requests.post(url, json=data, headers=headers)
    return response.content
```

**Recommendation:** Best quality, but requires subscription. Good for production if budget allows.

---

### 3. Coqui TTS (Open Source, Local)

**Pros:**

- Open source, free
- Runs locally (privacy)
- Good quality voices
- Can be fine-tuned
- Supports multiple languages

**Cons:**

- Requires model download (large files)
- GPU recommended for best performance
- More complex setup
- Singing synthesis requires additional models

**Installation:**

```bash
pip install TTS
```

**Example:**

```python
from TTS.api import TTS

tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
tts.tts_to_file("When I found you sleeping, everything felt right", file_path="output.wav")
```

**Recommendation:** Good balance of quality and privacy. Best for local deployment.

---

### 4. AWS Polly (Cloud-based)

**Pros:**

- High quality voices
- Neural TTS available
- Pay-per-use pricing
- Reliable infrastructure

**Cons:**

- Cloud-based (requires internet)
- Costs per character
- No built-in singing synthesis
- AWS account required

**Recommendation:** Good alternative to ElevenLabs, but similar limitations.

---

### 5. DiffSinger / NNSVS / OpenUTAU (Singing Synthesis)

**Pros:**

- Specifically designed for singing
- Can generate realistic singing voices
- Open source options available
- Supports pitch/note control

**Cons:**

- Complex setup and training
- Requires MIDI input with lyrics
- Large model files
- Steeper learning curve

**Recommendation:** Best for actual singing synthesis, but most complex to implement.

---

## Prototype Implementation

### Simple macOS TTS Prototype

```python
# music_brain/voice_synthesis.py
import subprocess
import tempfile
from pathlib import Path

def synthesize_speech_macos(text: str, output_path: str = None) -> str:
    """
    Use macOS 'say' command to synthesize speech.

    Args:
        text: Text to speak
        output_path: Optional path to save audio file

    Returns:
        Path to audio file
    """
    if output_path is None:
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.aiff')
        output_path = output_file.name
        output_file.close()

    # Use macOS say command with AIFF output
    subprocess.run([
        'say',
        '-o', output_path,
        text
    ])

    return output_path

# Test with Kelly song lyrics
if __name__ == '__main__':
    lyrics = "When I found you sleeping, everything felt right"
    output = synthesize_speech_macos(lyrics)
    print(f"Audio saved to: {output}")
```

---

## Recommendation for Production

### Phase 1: Prototype (Current)

- Use macOS `say` command for quick prototyping
- Generate spoken lyrics for testing
- Evaluate emotional appropriateness

### Phase 2: Local Production

- Implement Coqui TTS for local, privacy-preserving synthesis
- Fine-tune models for emotional expression
- Generate spoken lyrics with emotional control

### Phase 3: Singing Synthesis (Future)

- Research DiffSinger or NNSVS for actual singing
- Train models on emotional singing datasets
- Integrate with MIDI generation for pitch-synchronized singing

---

## Sample Audio Files

[To be generated during testing]

## Next Steps

1. Test macOS TTS with Kelly song lyrics
2. Evaluate Coqui TTS quality
3. Research singing synthesis integration
4. Create voice synthesis API endpoint
5. Integrate with music generation pipeline
