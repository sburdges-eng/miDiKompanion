# Critical Bugs and Fixes - Kelly MIDI

**Using Python DAiW-Music-Brain as Reference Implementation**

---

## Overview

The C++ JUCE plugin has several critical bugs that the Python backend already solves correctly. This document catalogs all issues and provides fixes based on the working Python implementation.

---

## Bug #1: WoundProcessor Emotion ID Mismatch

### Problem
```cpp
// WoundProcessor.cpp (WRONG)
{{"happy", "happiness", "joy"...}, 10, 0.9f}  // ID 10

// But EmotionThesaurus.cpp defines:
addNode({60, "Ecstasy", EmotionCategory::Joy...});  // ID 60
```

The WoundProcessor uses arbitrary IDs (10, 20, 30) but EmotionThesaurus starts at ID 60. This causes lookups to fail silently.

### Python Solution (CORRECT)
```python
# 1DAW1/music_brain/emotion_mapper.py
BASE_EMOTIONS = {
    "joy": {"valence": 0.8, "arousal": 0.7, "category": "joy"},
    "sad": {"valence": -0.7, "arousal": 0.3, "category": "sadness"},
    "anger": {"valence": -0.6, "arousal": 0.9, "category": "anger"},
    # ...
}

def map_wound_to_emotion(wound_text: str) -> EmotionNode:
    # Uses string matching, returns actual emotion nodes
    # Never uses hardcoded IDs
```

### C++ Fix
```cpp
// WoundProcessor.cpp - Use emotion names, not IDs
struct EmotionClue {
    std::vector<std::string> keywords;
    std::string emotionName;  // NOT ID
    float intensity;
};

static const std::vector<EmotionClue> emotionClues = {
    {{"happy", "happiness", "joy"}, "Ecstasy", 0.9f},
    {{"sad", "grief", "loss"}, "Grief", 0.8f},
    // Match by name, then look up in thesaurus
};

EmotionNode WoundProcessor::process(const std::string& description) {
    // Find matching clue
    for (const auto& clue : emotionClues) {
        if (matches(description, clue.keywords)) {
            // Look up by NAME in thesaurus
            return thesaurus_->findByName(clue.emotionName);
        }
    }
}
```

---

## Bug #2: Hardcoded Path in EmotionThesaurus

### Problem
```cpp
// EmotionThesaurus.cpp (WRONG)
juce::File dataDir = juce::File::getSpecialLocation(juce::File::currentExecutableFile)
    .getParentDirectory()
    .getChildFile("Kelly_MIDI_Project/kellymidicompanion/kellymidicompanion_data");
```

This path doesn't exist in the plugin bundle. Data files are never found.

### Python Solution (CORRECT)
```python
# 1DAW1/music_brain/emotion_mapper.py
import os
from pathlib import Path

# Multiple fallback paths
DATA_DIR = None
for path in [
    Path(__file__).parent / "data",  # Next to source
    Path.home() / ".kelly" / "data",  # User directory
    Path("/usr/local/share/kelly/data"),  # System install
]:
    if path.exists():
        DATA_DIR = path
        break

if DATA_DIR is None:
    # Embedded fallback data
    EMOTIONS = load_embedded_data()
```

### C++ Fix
```cpp
// EmotionThesaurus.cpp
juce::File EmotionThesaurus::findDataDirectory() {
    // 1. Try bundle resources (macOS .app/Contents/Resources)
    auto bundleResources = juce::File::getSpecialLocation(
        juce::File::currentApplicationFile)
        .getChildFile("Contents/Resources/data");
    if (bundleResources.isDirectory()) return bundleResources;

    // 2. Try user data directory
    auto userData = juce::File::getSpecialLocation(
        juce::File::userApplicationDataDirectory)
        .getChildFile("Kelly MIDI/data");
    if (userData.isDirectory()) return userData;

    // 3. Try next to executable (development)
    auto exeDir = juce::File::getSpecialLocation(
        juce::File::currentExecutableFile)
        .getParentDirectory()
        .getChildFile("data");
    if (exeDir.isDirectory()) return exeDir;

    // 4. Fall back to embedded defaults
    juce::Logger::writeToLog("EmotionThesaurus: Using embedded data");
    return juce::File();  // Use hardcoded defaults
}

void EmotionThesaurus::loadData() {
    auto dataDir = findDataDirectory();

    if (dataDir != juce::File()) {
        // Load from files
        loadFromJsonFiles(dataDir);
    } else {
        // Use embedded defaults
        loadEmbeddedDefaults();
    }
}

void EmotionThesaurus::loadEmbeddedDefaults() {
    // Hardcode base 6 emotions so plugin always works
    addNode({1, "Joy", EmotionCategory::Joy, 0.5f, 0.7f, 0.6f, {}});
    addNode({2, "Sadness", EmotionCategory::Sadness, 0.5f, -0.6f, 0.3f, {}});
    addNode({3, "Anger", EmotionCategory::Anger, 0.5f, -0.5f, 0.8f, {}});
    addNode({4, "Fear", EmotionCategory::Fear, 0.5f, -0.7f, 0.7f, {}});
    addNode({5, "Surprise", EmotionCategory::Surprise, 0.5f, 0.0f, 0.8f, {}});
    addNode({6, "Disgust", EmotionCategory::Disgust, 0.5f, -0.6f, 0.5f, {}});
}
```

---

## Bug #3: Global Static (Thread Safety)

### Problem
```cpp
// EmotionThesaurusLoader.cpp (WRONG)
static int g_nextEmotionId = 1;  // Global state, not thread-safe
```

If two threads load thesaurus simultaneously, IDs collide.

### Python Solution (CORRECT)
```python
# 1DAW1/music_brain/emotion_mapper.py
class EmotionThesaurus:
    def __init__(self):
        self._next_id = 1  # Instance variable
        self._lock = threading.Lock()  # Thread-safe

    def add_emotion(self, name, ...):
        with self._lock:
            emotion_id = self._next_id
            self._next_id += 1
            self.emotions[emotion_id] = Emotion(...)
```

### C++ Fix
```cpp
// EmotionThesaurus.h
class EmotionThesaurus {
private:
    std::atomic<int> nextEmotionId_{1};  // Thread-safe atomic
    // OR if needing more complex operations:
    mutable std::mutex mutex_;
    int nextEmotionId_ = 1;
};

// EmotionThesaurus.cpp
void EmotionThesaurus::addNode(const EmotionNode& node) {
    std::lock_guard<std::mutex> lock(mutex_);  // Thread-safe
    nodes_[node.id] = node;
}

int EmotionThesaurus::getNextId() {
    return nextEmotionId_.fetch_add(1);  // Atomic increment
}
```

---

## Bug #4: Raw Pointers Without Ownership

### Problem
```cpp
// EmotionWheel.h (WRONG)
const EmotionThesaurus* thesaurus_ = nullptr;  // Who owns this?

// What if thesaurus is deleted before EmotionWheel?
```

### Python Solution (CORRECT)
```python
# Python has garbage collection, but still uses clear ownership
class EmotionWheel:
    def __init__(self, thesaurus: EmotionThesaurus):
        self.thesaurus = thesaurus  # Strong reference, keeps alive
```

### C++ Fix
```cpp
// Option 1: Non-owning observer (if guaranteed lifetime)
// EmotionWheel.h
class EmotionWheel : public juce::Component {
public:
    void setThesaurus(const EmotionThesaurus& thesaurus) {
        thesaurusRef_ = &thesaurus;  // Non-owning, lifetime guaranteed by parent
    }

private:
    const EmotionThesaurus* thesaurusRef_ = nullptr;
};

// Option 2: Shared ownership (if lifetime unclear)
class EmotionWheel : public juce::Component {
public:
    void setThesaurus(std::shared_ptr<EmotionThesaurus> thesaurus) {
        thesaurus_ = thesaurus;  // Shared ownership
    }

private:
    std::shared_ptr<EmotionThesaurus> thesaurus_;
};

// Recommended: Option 1 (non-owning) with clear lifetime contract
// PluginProcessor owns thesaurus, EmotionWheel just observes it
```

---

## Bug #5: No Thread Safety (UI/Audio Thread)

### Problem
```cpp
// PluginProcessor accesses IntentPipeline from:
// - Audio thread (processBlock)
// - UI thread (parameter changes, button clicks)
// NO LOCKING = data races
```

### Python Solution (CORRECT)
```python
# 1DAW1/music_brain/api.py
import asyncio
from threading import Lock

class MusicBrain:
    def __init__(self):
        self._lock = Lock()
        self._intent_pipeline = IntentPipeline()

    async def process_intent(self, intent):
        with self._lock:  # Thread-safe access
            return self._intent_pipeline.process(intent)
```

### C++ Fix
```cpp
// PluginProcessor.h
class PluginProcessor : public juce::AudioProcessor {
private:
    IntentPipeline intentPipeline_;
    mutable std::mutex intentMutex_;  // Protects intentPipeline_

    // Audio thread uses tryLock, UI thread can block
};

// PluginProcessor.cpp
void PluginProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                   juce::MidiBuffer& midiMessages) {
    // Audio thread: NEVER block
    if (auto lock = std::unique_lock<std::mutex>(intentMutex_, std::try_to_lock)) {
        // Safe to access intentPipeline_
        auto midi = intentPipeline_.getScheduledNotes();
        // ... process MIDI
    }
    // If lock failed, skip this frame (don't block audio)
}

void PluginProcessor::parameterChanged(const juce::String& paramID, float value) {
    // UI thread: can block
    std::lock_guard<std::mutex> lock(intentMutex_);
    intentPipeline_.updateParameter(paramID, value);
}
```

---

## Bug #6: Magic Numbers

### Problem
```cpp
// ChordGenerator.cpp (WRONG)
int rootNote = 48;  // Why 48? C3? Undocumented
if (valence < -0.5f) rootNote = 45;  // Magic threshold
```

### Python Solution (CORRECT)
```python
# 1DAW1/music_brain/harmony.py
# Named constants
MIDI_C3 = 48
MIDI_A2 = 45
VALENCE_SAD_THRESHOLD = -0.5

def generate_root_note(emotion):
    if emotion.valence < VALENCE_SAD_THRESHOLD:
        return MIDI_A2  # Lower register for sadness
    return MIDI_C3  # Default middle C
```

### C++ Fix
```cpp
// common/MusicConstants.h
namespace kelly {
namespace MusicConstants {

// MIDI note numbers
constexpr int MIDI_C0 = 12;
constexpr int MIDI_C3 = 48;
constexpr int MIDI_A2 = 45;
constexpr int MIDI_C4 = 60;  // Middle C

// Emotion thresholds
constexpr float VALENCE_VERY_NEGATIVE = -0.7f;
constexpr float VALENCE_NEGATIVE = -0.3f;
constexpr float VALENCE_NEUTRAL = 0.0f;
constexpr float VALENCE_POSITIVE = 0.3f;
constexpr float VALENCE_VERY_POSITIVE = 0.7f;

constexpr float AROUSAL_LOW = 0.3f;
constexpr float AROUSAL_MODERATE = 0.5f;
constexpr float AROUSAL_HIGH = 0.7f;

// Timing (in beats)
constexpr double BEATS_PER_BAR = 4.0;
constexpr double MINIMUM_NOTE_LENGTH = 0.25;  // 16th note

} // namespace MusicConstants
} // namespace kelly

// ChordGenerator.cpp
using namespace kelly::MusicConstants;

int rootNote = MIDI_C3;
if (valence < VALENCE_NEGATIVE) {
    rootNote = MIDI_A2;  // Lower register for sadness
}
```

---

## Bug #7: No Actual MIDI Output

### Problem
The plugin generates MIDI data but never sends it to the DAW.

### Python Solution (Reference)
```python
# 1DAW1/music_brain/daw/logic.py
def export_to_logic(midi_data, output_path):
    track = mido.MidiTrack()

    for note in midi_data.notes:
        track.append(mido.Message('note_on',
                                 note=note.pitch,
                                 velocity=note.velocity,
                                 time=note.start_ticks))
        track.append(mido.Message('note_off',
                                 note=note.pitch,
                                 time=note.end_ticks))

    midi_file = mido.MidiFile()
    midi_file.tracks.append(track)
    midi_file.save(output_path)
```

### C++ Fix
```cpp
// PluginProcessor.cpp
void PluginProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                   juce::MidiBuffer& midiMessages) {
    // Get playback position from host
    auto playHead = getPlayHead();
    if (playHead == nullptr) return;

    juce::AudioPlayHead::CurrentPositionInfo posInfo;
    playHead->getCurrentPosition(posInfo);

    if (!posInfo.isPlaying) return;

    // Convert beat position to samples
    double currentBeat = posInfo.ppqPosition;

    // Lock-free read of scheduled notes
    for (const auto& note : scheduledNotes_) {
        // Check if note should start this block
        if (note.startBeat >= currentBeat &&
            note.startBeat < currentBeat + beatsPerBlock) {

            // Calculate sample offset within block
            int sampleOffset = static_cast<int>(
                (note.startBeat - currentBeat) * samplesPerBeat);

            // Add note on
            midiMessages.addEvent(
                juce::MidiMessage::noteOn(note.channel, note.pitch,
                                         static_cast<uint8_t>(note.velocity)),
                sampleOffset);
        }

        // Check if note should end this block
        if (note.endBeat >= currentBeat &&
            note.endBeat < currentBeat + beatsPerBlock) {

            int sampleOffset = static_cast<int>(
                (note.endBeat - currentBeat) * samplesPerBeat);

            // Add note off
            midiMessages.addEvent(
                juce::MidiMessage::noteOff(note.channel, note.pitch),
                sampleOffset);
        }
    }
}
```

---

## Bug #8: APVTS Not Connected

### Problem
AudioProcessorValueTreeState exists but isn't connected to actual functionality.

### Python Solution (Reference)
```python
# 1DAW1/music_brain/api.py
class MusicBrainAPI:
    def __init__(self):
        self.parameters = {
            "intensity": 0.5,
            "complexity": 0.5,
            "humanize": 0.3,
        }

    def set_parameter(self, name, value):
        self.parameters[name] = value
        self.regenerate()  # Trigger update
```

### C++ Fix
```cpp
// PluginProcessor.cpp
PluginProcessor::PluginProcessor()
    : AudioProcessor(...),
      parameters(*this, nullptr, "PARAMETERS", createParameterLayout()) {

    // Listen to parameter changes
    parameters.addParameterListener(PARAM_INTENSITY, this);
    parameters.addParameterListener(PARAM_COMPLEXITY, this);
    parameters.addParameterListener(PARAM_HUMANIZE, this);
}

void PluginProcessor::parameterChanged(const juce::String& paramID, float value) {
    std::lock_guard<std::mutex> lock(intentMutex_);

    if (paramID == PARAM_INTENSITY) {
        // Update intent pipeline
        auto& intent = intentPipeline_.getCurrentIntent();
        intent.intensity = value;
        regenerateIfNeeded();
    }
    else if (paramID == PARAM_COMPLEXITY) {
        auto& intent = intentPipeline_.getCurrentIntent();
        intent.complexity = value;
        regenerateIfNeeded();
    }
    // etc...
}

// Make parameters automatable
juce::AudioProcessorValueTreeState::ParameterLayout
PluginProcessor::createParameterLayout() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        PARAM_INTENSITY,
        "Intensity",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f),
        0.5f,  // default
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value, 2); }
    ));

    // ... more parameters

    return { params.begin(), params.end() };
}
```

---

## Complete Fix Priority

### Critical (Fix Immediately)
1. ✅ Fix emotion ID mismatch (causes silent failures)
2. ✅ Fix hardcoded paths (plugin can't find data)
3. ✅ Add thread safety (prevents crashes)

### High Priority (Fix This Week)
4. ✅ Implement MIDI output (core feature)
5. ✅ Connect APVTS (enables automation)
6. ✅ Fix raw pointers (prevents memory errors)

### Medium Priority (Fix This Month)
7. ✅ Replace magic numbers (improves maintainability)
8. Implement BiometricInput (v2.0 feature)
9. Implement VoiceSynthesizer (v2.0 feature)

---

## Using Python as Reference

**Best Practice**: Port algorithms from Python to C++

1. **Emotion Mapping**: `/Users/seanburdges/Desktop/1DAW1/music_brain/emotion_mapper.py`
   - Correct valence/arousal calculations
   - Proper emotion matching

2. **Chord Generation**: `/Users/seanburdges/Desktop/1DAW1/music_brain/harmony.py`
   - Voice leading rules
   - Progression families

3. **Groove Engine**: `/Users/seanburdges/Desktop/1DAW1/music_brain/groove/engine.py`
   - Humanization algorithms
   - Genre pocket maps

4. **Intent Processing**: `/Users/seanburdges/Desktop/1DAW1/music_brain/session/intent_processor.py`
   - Three-phase workflow
   - Rule-breaking logic

---

## Next Steps

1. Apply all critical fixes to `/Users/seanburdges/Desktop/final kel/`
2. Port Python algorithms to C++
3. Add comprehensive unit tests
4. Rebuild and test in Logic Pro

**All fixes ready to implement!**
