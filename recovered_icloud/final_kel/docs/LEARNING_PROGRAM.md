# Kelly MIDI Companion - Advanced ML/DSP Learning Program

## 🎯 **Program Overview**

This learning program integrates cutting-edge ML/DSP techniques with your existing Kelly MIDI Companion codebase to create a comprehensive therapeutic music generation system.

**Current Status**: ✅ Production-ready plugin with emotion-to-music formulas
**Goal**: Add real-time neural inference, timbre transfer, and biometric feedback

---

## 📚 **Phase 1: Foundation (Weeks 1-2)**

### **Module 1.1: Understanding Current Architecture**

**Objective**: Deep dive into existing Kelly codebase before adding ML

#### **Study Materials**:
1. **Emotion Engine** (`src/engine/`)
   - `EmotionMusicMapper.h` - Core formulas (already implemented)
   - `EmotionThesaurus.cpp` - 72-emotion PAD model
   - `WoundProcessor.cpp` - Text-to-emotion conversion
   - `IntentPipeline.cpp` - Emotional intent processing

2. **MIDI Generation** (`src/midi/`, `src/engines/`)
   - `MidiGenerator.cpp` - Main orchestrator (14 engines)
   - `ChordGenerator.cpp` - Harmony generation
   - `MelodyEngine.cpp` - Melodic contour generation
   - `BassEngine.cpp` - Bass line patterns
   - `GrooveEngine.cpp` - Humanization & swing

3. **Plugin Infrastructure** (`src/plugin/`)
   - `PluginProcessor.cpp` - JUCE audio processor
   - `PluginEditor.cpp` - UI components
   - `PluginState.cpp` - Parameter management

#### **Hands-On Exercises**:

**Exercise 1.1.1: Emotion Flow Tracing**
```bash
# Trace the path of an emotion through the system
# Start: User inputs "I feel grief"
# End: MIDI notes generated

# Key checkpoints:
1. WoundProcessor::processWound() -> EmotionNode
2. IntentPipeline::process() -> IntentResult
3. EmotionMusicMapper::mapEmotion() -> MusicalParameters
4. MidiGenerator::generate() -> GeneratedMidi
```

**Exercise 1.1.2: Formula Verification**
```cpp
// File: tests/verify_formulas.cpp
#include "engine/EmotionMusicMapper.h"
#include <cassert>

void testTempoFormula() {
    // Test: tempo = 60 + 120 * arousal
    assert(EmotionMusicMapper::calculateTempo(0.0f) == 60);   // Min
    assert(EmotionMusicMapper::calculateTempo(1.0f) == 180);  // Max
    assert(EmotionMusicMapper::calculateTempo(0.5f) == 120);  // Mid
}

void testVelocityFormula() {
    // Test: velocity = 60 + 67 * dominance
    assert(EmotionMusicMapper::calculateVelocity(0.0f) == 60);
    assert(EmotionMusicMapper::calculateVelocity(1.0f) == 127);
}

// TODO: Run these tests and understand the mappings
```

**Exercise 1.1.3: Build & Test Current System**
```bash
cd "/Users/seanburdges/Desktop/final kel"
cmake --build build --target KellyTests
./build/tests/KellyTests

# Expected: 29/29 tests passing
# If not, debug and understand why
```

#### **Learning Checkpoints**:
- [ ] Can explain the PAD (Pleasure-Arousal-Dominance) emotion model
- [ ] Can trace emotion input → MIDI output path
- [ ] Understand all 5 emotion-to-music formulas
- [ ] Can modify emotion mappings and rebuild plugin

---

### **Module 1.2: Audio Processing Fundamentals**

**Objective**: Master real-time audio concepts needed for ML integration

#### **Study Materials**:

**1.2.1: Sample Rates & Buffer Sizes**
```cpp
// Key concepts from PluginProcessor::prepareToPlay()
void prepareToPlay(double sampleRate, int samplesPerBlock) {
    // sampleRate: 44100, 48000, 96000 Hz
    // samplesPerBlock: 64-2048 samples
    // Processing time budget = samplesPerBlock / sampleRate

    // Example: 512 samples @ 48kHz
    // Budget = 512 / 48000 = 10.67ms per processBlock() call
}
```

**1.2.2: Lock-Free Programming**
```cpp
// Study this existing pattern in your codebase
// From: src/midi/ChordGenerator.h

mutable std::mt19937 rng_;  // Mutable for const methods

// Why mutable?
// - RNG needs state mutation even in const methods
// - Alternative: std::atomic for lock-free state
```

**1.2.3: Latency & Delay Compensation**
```cpp
// Your plugin already reports latency:
int getLatencySamples() const override;

// ML models add latency:
// - Feature extraction: ~100-500 samples
// - Neural inference: ~500-2000 samples
// - Total: Need lookahead buffering
```

#### **Hands-On Exercises**:

**Exercise 1.2.1: Measure Current Latency**
```cpp
// Add to PluginProcessor.cpp
void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) {
    auto start = std::chrono::high_resolution_clock::now();

    // ... existing processing ...

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Log to console
    DBG("Process time: " << duration.count() << "us, Budget: "
        << (buffer.getNumSamples() * 1000000.0 / getSampleRate()) << "us");
}
```

**Exercise 1.2.2: Implement Simple Lookahead Buffer**
```cpp
// File: src/common/LookaheadBuffer.h
#pragma once
#include <juce_audio_basics/juce_audio_basics.h>

class LookaheadBuffer {
public:
    void prepare(int numChannels, int lookaheadSamples, int blockSize) {
        buffer.setSize(numChannels, lookaheadSamples + blockSize);
        buffer.clear();
        writePos = 0;
        readPos = 0;
    }

    void write(const juce::AudioBuffer<float>& input) {
        // TODO: Write input to circular buffer
    }

    void read(juce::AudioBuffer<float>& output) {
        // TODO: Read delayed audio from buffer
    }

private:
    juce::AudioBuffer<float> buffer;
    int writePos = 0;
    int readPos = 0;
};
```

#### **Learning Checkpoints**:
- [ ] Understand audio thread constraints (no allocations, no locks)
- [ ] Can calculate latency budgets for different buffer sizes
- [ ] Implemented working lookahead buffer
- [ ] Understand circular buffer mechanics

---

## 🧠 **Phase 2: Neural Network Integration (Weeks 3-6)**

### **Module 2.1: RTNeural Inference Engine**

**Objective**: Add real-time neural emotion recognition to your plugin

#### **Architecture Integration**:

```
Current Flow:
Text Input → WoundProcessor → EmotionNode → MidiGenerator

New ML-Enhanced Flow:
Audio Input ──┬→ Feature Extraction → RTNeural Model → Emotion Vector ──┐
              │                                                          ├→ Fusion → Enhanced EmotionNode
Text Input ───┴→ WoundProcessor ──────────────────────────────────────→ ┘
```

#### **Implementation Steps**:

**Step 2.1.1: Add RTNeural Dependency**

```cmake
# File: CMakeLists.txt (add after JUCE)
FetchContent_Declare(
    RTNeural
    GIT_REPOSITORY https://github.com/jatinchowdhury18/RTNeural.git
    GIT_TAG v1.2.0
)
FetchContent_MakeAvailable(RTNeural)

target_link_libraries(KellyMidiCompanion PRIVATE RTNeural)
```

**Step 2.1.2: Create Neural Processor**

```cpp
// File: src/engine/NeuralEmotionProcessor.h
#pragma once
#include <RTNeural/RTNeural.h>
#include "common/Types.h"
#include <array>

namespace kelly {

class NeuralEmotionProcessor {
public:
    // Model: 128 input features → 64 output (V/A/D vectors)
    using EmotionModel = RTNeural::ModelT<float, 128, 64,
        RTNeural::DenseT<float, 128, 256>,
        RTNeural::TanhActivationT<float, 256>,
        RTNeural::LSTMLayerT<float, 256, 128>,
        RTNeural::DenseT<float, 128, 64>>;

    NeuralEmotionProcessor();

    bool loadModel(const juce::File& modelPath);

    /**
     * Infer emotion from audio features.
     * @param features Spectral/temporal features (128-dim)
     * @return Emotion vector [valence, arousal, dominance, confidence...]
     */
    std::array<float, 64> inferEmotion(const std::array<float, 128>& features);

    /**
     * Extract features from audio buffer.
     * Features: MFCCs, spectral centroid, RMS, ZCR, etc.
     */
    std::array<float, 128> extractFeatures(const juce::AudioBuffer<float>& audio);

private:
    EmotionModel model_;
    bool modelLoaded_ = false;

    // Feature extraction state
    juce::dsp::FFT fft_{10};  // 1024-point FFT
    std::vector<float> fftBuffer_;
};

} // namespace kelly
```

**Step 2.1.3: Implement Feature Extraction**

```cpp
// File: src/engine/NeuralEmotionProcessor.cpp
#include "NeuralEmotionProcessor.h"
#include <juce_dsp/juce_dsp.h>

namespace kelly {

std::array<float, 128> NeuralEmotionProcessor::extractFeatures(
    const juce::AudioBuffer<float>& audio
) {
    std::array<float, 128> features{};

    const int numSamples = audio.getNumSamples();
    const int numChannels = audio.getNumChannels();

    // 1. RMS Energy (1 feature)
    float rms = 0.0f;
    for (int ch = 0; ch < numChannels; ++ch) {
        rms += audio.getRMSLevel(ch, 0, numSamples);
    }
    features[0] = rms / numChannels;

    // 2. Zero Crossing Rate (1 feature)
    int zeroCrossings = 0;
    for (int ch = 0; ch < numChannels; ++ch) {
        const float* data = audio.getReadPointer(ch);
        for (int i = 1; i < numSamples; ++i) {
            if ((data[i-1] >= 0.0f && data[i] < 0.0f) ||
                (data[i-1] < 0.0f && data[i] >= 0.0f)) {
                zeroCrossings++;
            }
        }
    }
    features[1] = static_cast<float>(zeroCrossings) / (numSamples * numChannels);

    // 3. Spectral Features (MFCCs, centroid, etc.) - 126 features
    // TODO: Implement full spectral analysis
    // For now, placeholder implementation

    return features;
}

std::array<float, 64> NeuralEmotionProcessor::inferEmotion(
    const std::array<float, 128>& features
) {
    if (!modelLoaded_) {
        return {};  // Return zeros if no model
    }

    std::array<float, 64> output;
    const float* result = model_.forward(features.data());
    std::copy(result, result + 64, output.begin());

    return output;
}

bool NeuralEmotionProcessor::loadModel(const juce::File& modelPath) {
    if (!modelPath.existsAsFile()) return false;

    auto stream = juce::FileInputStream(modelPath);
    if (!stream.openedOk()) return false;

    auto jsonStr = stream.readEntireStreamAsString();
    std::istringstream iss(jsonStr.toStdString());

    try {
        model_.parseJson(iss);
        modelLoaded_ = true;
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace kelly
```

**Step 2.1.4: Integrate with Plugin**

```cpp
// File: src/plugin/PluginProcessor.h (add member)
class KellyPluginProcessor : public juce::AudioProcessor {
    // ... existing members ...

    kelly::NeuralEmotionProcessor neuralProcessor_;

    // Lock-free communication
    std::atomic<float> neuralValence_{0.0f};
    std::atomic<float> neuralArousal_{0.0f};
    std::atomic<float> neuralDominance_{0.0f};
    std::atomic<float> neuralConfidence_{0.0f};
};

// File: src/plugin/PluginProcessor.cpp
void KellyPluginProcessor::processBlock(
    juce::AudioBuffer<float>& buffer,
    juce::MidiBuffer& midiMessages
) {
    // Extract features (non-blocking)
    auto features = neuralProcessor_.extractFeatures(buffer);

    // Infer emotion (fast, ~1-2ms)
    auto emotionVector = neuralProcessor_.inferEmotion(features);

    // Update atomic state (lock-free)
    if (emotionVector[63] > 0.5f) {  // Confidence threshold
        neuralValence_.store(emotionVector[0]);
        neuralArousal_.store(emotionVector[1]);
        neuralDominance_.store(emotionVector[2]);
        neuralConfidence_.store(emotionVector[63]);
    }

    // Continue with MIDI generation using enhanced emotion data
    // ... existing code ...
}
```

#### **Hands-On Exercises**:

**Exercise 2.1.1: Train Simple Emotion Model**

```python
# File: ml_training/train_emotion_model.py
import torch
import torch.nn as nn
import json

class SimpleEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x, _ = self.lstm(x.unsqueeze(1))
        x = self.fc2(x.squeeze(1))
        return x

# Train on your emotion-labeled audio dataset
# Export to JSON for RTNeural:

model = SimpleEmotionModel()
# ... training code ...

# Export
weights = {}
for name, param in model.named_parameters():
    weights[name] = param.detach().cpu().numpy().tolist()

with open('emotion_model.json', 'w') as f:
    json.dump(weights, f)
```

**Exercise 2.1.2: Benchmark Inference Speed**

```cpp
// Test: Can we run inference in real-time?
#include <chrono>

void benchmarkInference() {
    NeuralEmotionProcessor proc;
    proc.loadModel(juce::File("emotion_model.json"));

    std::array<float, 128> dummyFeatures{};

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; ++i) {
        auto result = proc.inferEmotion(dummyFeatures);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Avg inference time: " << duration.count() / 1000.0 << "us\n";
    // Target: < 2000us (2ms) per inference
}
```

#### **Learning Checkpoints**:
- [ ] RTNeural compiles and links with your project
- [ ] Can load and run a simple neural model
- [ ] Inference time < 2ms on your machine
- [ ] Feature extraction produces reasonable values
- [ ] Neural emotion values update in real-time

---

### **Module 2.2: Lock-Free Threading Architecture**

**Objective**: Run ML inference on separate thread without blocking audio

#### **Problem**: Audio Thread Constraints

```cpp
// ❌ NEVER DO THIS IN AUDIO THREAD:
void processBlock(...) {
    auto emotion = runNeuralInference(buffer);  // Blocks for 50ms!
    // Buffer underrun! Audio glitches!
}

// ✅ CORRECT APPROACH: Async inference
void processBlock(...) {
    submitToInferenceThread(buffer);  // Fire and forget
    auto emotion = getLatestResult();  // Non-blocking read
    // Audio thread never blocks
}
```

#### **Implementation**:

**Step 2.2.1: Lock-Free Ring Buffer**

```cpp
// File: src/common/LockFreeRingBuffer.h
#pragma once
#include <atomic>
#include <array>
#include <cstring>

namespace kelly {

template<typename T, size_t Capacity>
class LockFreeRingBuffer {
public:
    bool push(const T* data, size_t count) {
        const size_t currentWrite = writePos_.load(std::memory_order_relaxed);
        const size_t currentRead = readPos_.load(std::memory_order_acquire);

        const size_t available = Capacity - (currentWrite - currentRead);
        if (count > available) return false;

        const size_t writeIndex = currentWrite % Capacity;
        const size_t firstPart = std::min(count, Capacity - writeIndex);

        std::memcpy(&buffer_[writeIndex], data, firstPart * sizeof(T));
        if (count > firstPart) {
            std::memcpy(&buffer_[0], data + firstPart, (count - firstPart) * sizeof(T));
        }

        writePos_.store(currentWrite + count, std::memory_order_release);
        return true;
    }

    bool pop(T* data, size_t count) {
        const size_t currentRead = readPos_.load(std::memory_order_relaxed);
        const size_t currentWrite = writePos_.load(std::memory_order_acquire);

        const size_t available = currentWrite - currentRead;
        if (count > available) return false;

        const size_t readIndex = currentRead % Capacity;
        const size_t firstPart = std::min(count, Capacity - readIndex);

        std::memcpy(data, &buffer_[readIndex], firstPart * sizeof(T));
        if (count > firstPart) {
            std::memcpy(data + firstPart, &buffer_[0], (count - firstPart) * sizeof(T));
        }

        readPos_.store(currentRead + count, std::memory_order_release);
        return true;
    }

    size_t availableToRead() const {
        return writePos_.load(std::memory_order_acquire) -
               readPos_.load(std::memory_order_relaxed);
    }

private:
    std::array<T, Capacity> buffer_;
    std::atomic<size_t> writePos_{0};
    std::atomic<size_t> readPos_{0};
};

} // namespace kelly
```

**Step 2.2.2: Inference Thread Manager**

```cpp
// File: src/engine/InferenceThreadManager.h
#pragma once
#include "NeuralEmotionProcessor.h"
#include "common/LockFreeRingBuffer.h"
#include <thread>
#include <atomic>

namespace kelly {

struct InferenceRequest {
    std::array<float, 128> features;
    int64_t timestamp;
};

struct InferenceResult {
    std::array<float, 64> emotionVector;
    int64_t timestamp;
};

class InferenceThreadManager {
public:
    static constexpr size_t BUFFER_SIZE = 256;

    InferenceThreadManager() : running_(false) {}
    ~InferenceThreadManager() { stop(); }

    void start(const juce::File& modelPath) {
        processor_.loadModel(modelPath);
        running_ = true;
        inferenceThread_ = std::thread(&InferenceThreadManager::inferenceLoop, this);
    }

    void stop() {
        running_ = false;
        if (inferenceThread_.joinable()) {
            inferenceThread_.join();
        }
    }

    // Called from audio thread - never blocks
    bool submitRequest(const InferenceRequest& request) {
        return requestBuffer_.push(&request, 1);
    }

    // Called from audio thread - never blocks
    bool getResult(InferenceResult& result) {
        return resultBuffer_.pop(&result, 1);
    }

private:
    void inferenceLoop() {
        InferenceRequest request;
        while (running_) {
            if (requestBuffer_.pop(&request, 1)) {
                InferenceResult result;
                result.emotionVector = processor_.inferEmotion(request.features);
                result.timestamp = request.timestamp;
                resultBuffer_.push(&result, 1);
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
    }

    NeuralEmotionProcessor processor_;
    LockFreeRingBuffer<InferenceRequest, BUFFER_SIZE> requestBuffer_;
    LockFreeRingBuffer<InferenceResult, BUFFER_SIZE> resultBuffer_;
    std::thread inferenceThread_;
    std::atomic<bool> running_;
};

} // namespace kelly
```

**Step 2.2.3: Integrate with Plugin (Final)**

```cpp
// File: src/plugin/PluginProcessor.h
class KellyPluginProcessor : public juce::AudioProcessor {
public:
    // ... existing ...

    void prepareToPlay(double sampleRate, int samplesPerBlock) override {
        // Start inference thread
        inferenceManager_.start(getModelFile());
    }

    void releaseResources() override {
        inferenceManager_.stop();
    }

    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) override {
        // 1. Extract features (fast, ~0.5ms)
        auto features = neuralProcessor_.extractFeatures(buffer);

        // 2. Submit to inference thread (non-blocking, ~0.001ms)
        InferenceRequest request{features, sampleCounter_};
        inferenceManager_.submitRequest(request);

        // 3. Get any completed results (non-blocking, ~0.001ms)
        InferenceResult result;
        while (inferenceManager_.getResult(result)) {
            applyEmotionUpdate(result.emotionVector);
        }

        // 4. Continue with MIDI generation using latest emotion state
        // ... existing code ...

        sampleCounter_ += buffer.getNumSamples();
    }

private:
    InferenceThreadManager inferenceManager_;
    NeuralEmotionProcessor neuralProcessor_;
    int64_t sampleCounter_ = 0;

    juce::File getModelFile() {
        return juce::File::getSpecialLocation(juce::File::currentApplicationFile)
            .getChildFile("Contents/Resources/emotion_model.json");
    }

    void applyEmotionUpdate(const std::array<float, 64>& emotionVector) {
        // Smoothly update emotion state
        float alpha = 0.1f;  // Smoothing factor
        currentValence_ = alpha * emotionVector[0] + (1.0f - alpha) * currentValence_;
        currentArousal_ = alpha * emotionVector[1] + (1.0f - alpha) * currentArousal_;
        currentDominance_ = alpha * emotionVector[2] + (1.0f - alpha) * currentDominance_;
    }

    std::atomic<float> currentValence_{0.0f};
    std::atomic<float> currentArousal_{0.0f};
    std::atomic<float> currentDominance_{0.0f};
};
```

#### **Hands-On Exercises**:

**Exercise 2.2.1: Test Lock-Free Buffer**

```cpp
// File: tests/test_lockfree_buffer.cpp
#include "common/LockFreeRingBuffer.h"
#include <thread>
#include <atomic>
#include <cassert>

void testConcurrentAccess() {
    LockFreeRingBuffer<int, 1024> buffer;
    std::atomic<bool> writerDone{false};
    std::atomic<int> itemsWritten{0};
    std::atomic<int> itemsRead{0};

    // Writer thread
    std::thread writer([&]() {
        for (int i = 0; i < 10000; ++i) {
            while (!buffer.push(&i, 1)) {
                std::this_thread::yield();
            }
            itemsWritten++;
        }
        writerDone = true;
    });

    // Reader thread
    std::thread reader([&]() {
        int value;
        while (!writerDone || buffer.availableToRead() > 0) {
            if (buffer.pop(&value, 1)) {
                itemsRead++;
            } else {
                std::this_thread::yield();
            }
        }
    });

    writer.join();
    reader.join();

    assert(itemsWritten == 10000);
    assert(itemsRead == 10000);
    std::cout << "✅ Lock-free buffer test passed!\n";
}
```

**Exercise 2.2.2: Measure Thread Overhead**

```cpp
void benchmarkThreadLatency() {
    InferenceThreadManager manager;
    manager.start(juce::File("emotion_model.json"));

    auto start = std::chrono::high_resolution_clock::now();

    // Submit 1000 requests
    for (int i = 0; i < 1000; ++i) {
        InferenceRequest req;
        std::fill(req.features.begin(), req.features.end(), 0.5f);
        req.timestamp = i;
        manager.submitRequest(req);
    }

    // Wait for all results
    int received = 0;
    InferenceResult result;
    while (received < 1000) {
        if (manager.getResult(result)) {
            received++;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Total time: " << duration.count() << "ms\n";
    std::cout << "Avg latency: " << duration.count() / 1000.0 << "ms per inference\n";
}
```

#### **Learning Checkpoints**:
- [ ] Lock-free ring buffer passes concurrent tests
- [ ] Inference thread starts/stops cleanly
- [ ] No audio glitches when inference is running
- [ ] Can process 100+ inferences per second
- [ ] Understand memory ordering (acquire/release semantics)

---

## 🎸 **Phase 3: MIDI Generation with Transformers (Weeks 7-10)**

### **Module 3.1: Compound Word Transformer**

**Objective**: Generate emotion-conditioned MIDI using state-of-the-art transformers

#### **Why Compound Words?**

```
Traditional Token-per-Note:
[PITCH_60] [VEL_80] [DUR_480] [PITCH_64] [VEL_75] [DUR_240] ...
↓ Problem: Long sequences, slow generation

Compound Word Tokens:
[NOTE_60_80_480_0] [NOTE_64_75_240_1] ...
↓ Solution: 4x shorter, faster inference, better coherence
```

#### **Architecture**:

```python
# Current System (Rule-Based):
EmotionNode → ChordGenerator → MelodyEngine → Bass/Pad/etc.
↓ Pros: Fast, predictable, therapeutic
↓ Cons: Limited creativity, fixed patterns

# ML-Enhanced System:
EmotionNode ─┬→ Rule-Based (for structure) ──→ Merge → Final MIDI
             └→ Transformer (for variation) ──┘
```

#### **Implementation**:

**Step 3.1.1: Training Data Preparation**

```python
# File: ml_training/prepare_midi_dataset.py
import mido
import json
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class CompoundToken:
    pitch: int       # 0-127
    velocity: int    # 0-31 (quantized from 0-127)
    duration: int    # 0-7 (quantized duration index)
    position: int    # 0-15 (16th note grid)

    DURATION_BINS = [30, 60, 120, 240, 480, 960, 1920]

    def to_index(self) -> int:
        """Encode as single vocabulary index (0-65535)"""
        return (self.pitch +
                self.velocity * 128 +
                self.duration * 128 * 32 +
                self.position * 128 * 32 * 8)

    @classmethod
    def from_midi_note(cls, note, start_tick, ppq=480):
        velocity_quantized = min(note.velocity // 4, 31)
        duration_quantized = cls._quantize_duration(note.duration)
        position = (start_tick % (ppq * 4)) // (ppq // 4)  # 16th note position

        return cls(
            pitch=note.note,
            velocity=velocity_quantized,
            duration=duration_quantized,
            position=position
        )

    @classmethod
    def _quantize_duration(cls, duration_ticks: int) -> int:
        for i, bin_val in enumerate(cls.DURATION_BINS):
            if duration_ticks <= bin_val:
                return i
        return len(cls.DURATION_BINS) - 1

def prepare_dataset(midi_folder: str, output_file: str):
    """Convert MIDI files to compound token sequences"""
    dataset = []

    for midi_file in Path(midi_folder).glob("*.mid"):
        mid = mido.MidiFile(midi_file)

        # Extract emotion label from filename or metadata
        # Format: "grief_001.mid" or use emotion classifier
        emotion_label = extract_emotion(midi_file)

        tokens = midi_to_tokens(mid)

        dataset.append({
            'tokens': tokens,
            'valence': emotion_label['valence'],
            'arousal': emotion_label['arousal'],
            'filename': str(midi_file)
        })

    with open(output_file, 'w') as f:
        json.dump(dataset, f)

def midi_to_tokens(mid: mido.MidiFile) -> List[int]:
    """Convert MIDI to compound token sequence"""
    tokens = [0]  # Start token
    current_time = 0

    for msg in mid.tracks[0]:
        current_time += msg.time

        if msg.type == 'note_on' and msg.velocity > 0:
            token = CompoundToken.from_midi_note(
                msg, current_time, mid.ticks_per_beat
            )
            tokens.append(token.to_index() + 2)  # Offset for special tokens

    tokens.append(1)  # End token
    return tokens
```

**Step 3.1.2: Transformer Model**

```python
# File: ml_training/emotion_transformer.py
import torch
import torch.nn as nn
from typing import Optional

class EmotionEmbedding(nn.Module):
    """Continuous emotion conditioning"""
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.valence_proj = nn.Linear(1, embed_dim // 2)
        self.arousal_proj = nn.Linear(1, embed_dim // 2)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, valence: torch.Tensor, arousal: torch.Tensor) -> torch.Tensor:
        v = self.valence_proj(valence.unsqueeze(-1))
        a = self.arousal_proj(arousal.unsqueeze(-1))
        combined = torch.cat([v, a], dim=-1)
        return self.combine(combined)

class CompoundWordTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 8192,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.emotion_embed = EmotionEmbedding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        valence: torch.Tensor,
        arousal: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)

        # Token + position embeddings
        x = self.token_embed(tokens) + self.pos_embed(positions)

        # Prepend emotion conditioning
        emotion = self.emotion_embed(valence, arousal).unsqueeze(1)
        x = torch.cat([emotion, x], dim=1)

        # Causal mask
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len + 1)
            mask = mask.to(tokens.device)

        x = self.transformer(x, mask=mask)
        return self.output_proj(x[:, 1:, :])  # Remove emotion token

    @torch.no_grad()
    def generate(
        self,
        valence: float,
        arousal: float,
        num_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> List[int]:
        """Generate MIDI sequence conditioned on emotion"""
        self.eval()
        device = next(self.parameters()).device

        v = torch.tensor([valence], device=device)
        a = torch.tensor([arousal], device=device)
        tokens = torch.tensor([[0]], device=device)  # Start token

        generated = [0]

        for _ in range(num_tokens):
            logits = self(tokens, v, a)[:, -1, :]
            logits = logits / temperature

            # Top-k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)
            generated.append(next_token.item())

            if next_token.item() == 1:  # End token
                break

        return generated
```

**Step 3.1.3: Training Script**

```python
# File: ml_training/train_transformer.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm

class MIDIEmotionDataset(Dataset):
    def __init__(self, data_file: str, max_len: int = 512):
        with open(data_file) as f:
            self.data = json.load(f)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens'][:self.max_len]

        # Pad if needed
        if len(tokens) < self.max_len:
            tokens = tokens + [0] * (self.max_len - len(tokens))

        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'valence': torch.tensor([item['valence']], dtype=torch.float),
            'arousal': torch.tensor([item['arousal']], dtype=torch.float)
        }

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        tokens = batch['tokens'].to(device)
        valence = batch['valence'].to(device)
        arousal = batch['arousal'].to(device)

        # Input: all tokens except last
        input_tokens = tokens[:, :-1]
        # Target: all tokens except first
        target_tokens = tokens[:, 1:]

        optimizer.zero_grad()
        logits = model(input_tokens, valence, arousal)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_tokens.reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = MIDIEmotionDataset('midi_emotion_dataset.json')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create model
    model = CompoundWordTransformer(
        vocab_size=8192,
        embed_dim=512,
        num_heads=8,
        num_layers=6
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

    # Training loop
    for epoch in range(100):
        loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch}.pt')

            # Test generation
            with torch.no_grad():
                generated = model.generate(
                    valence=-0.7,  # Grief
                    arousal=0.3,
                    num_tokens=128
                )
                print(f"Generated {len(generated)} tokens")

if __name__ == '__main__':
    main()
```

**Step 3.1.4: Export for C++ Plugin**

```python
# File: ml_training/export_transformer.py
import torch
from emotion_transformer import CompoundWordTransformer

def export_to_onnx(model_path: str, output_path: str):
    model = CompoundWordTransformer()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Dummy inputs
    dummy_tokens = torch.randint(0, 100, (1, 64))
    dummy_v = torch.tensor([0.0])
    dummy_a = torch.tensor([0.0])

    torch.onnx.export(
        model,
        (dummy_tokens, dummy_v, dummy_a),
        output_path,
        input_names=['tokens', 'valence', 'arousal'],
        output_names=['logits'],
        dynamic_axes={
            'tokens': {1: 'seq_len'},
            'logits': {1: 'seq_len'}
        },
        opset_version=14
    )
    print(f"✅ Exported to {output_path}")

if __name__ == '__main__':
    export_to_onnx(
        'checkpoints/model_epoch_99.pt',
        'transformer_model.onnx'
    )
```

#### **Integration with Kelly Plugin**:

```cpp
// File: src/engine/TransformerMIDIGenerator.h
#pragma once
#include <onnxruntime_cxx_api.h>
#include "common/Types.h"

namespace kelly {

class TransformerMIDIGenerator {
public:
    TransformerMIDIGenerator();

    bool loadModel(const juce::File& onnxFile);

    /**
     * Generate MIDI using transformer model.
     * @param valence Emotion valence (-1 to 1)
     * @param arousal Emotion arousal (0 to 1)
     * @param numTokens Number of tokens to generate
     * @return Generated MIDI notes
     */
    std::vector<MidiNote> generate(
        float valence,
        float arousal,
        int numTokens = 256,
        float temperature = 0.9f
    );

private:
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Env> env_;
    bool modelLoaded_ = false;

    std::vector<int> runInference(
        const std::vector<int>& inputTokens,
        float valence,
        float arousal
    );

    std::vector<MidiNote> tokensToMIDI(const std::vector<int>& tokens);
};

} // namespace kelly
```

#### **Hands-On Exercises**:

**Exercise 3.1.1: Create Small Training Dataset**

```bash
# 1. Collect 100 MIDI files for each emotion
mkdir -p midi_dataset/{grief,joy,anger,fear,peaceful}

# 2. Label them manually or use existing labels
python ml_training/prepare_midi_dataset.py \
    --input midi_dataset/ \
    --output midi_emotion_dataset.json

# 3. Verify dataset
python -c "
import json
with open('midi_emotion_dataset.json') as f:
    data = json.load(f)
print(f'Dataset size: {len(data)} sequences')
print(f'Avg tokens per sequence: {sum(len(d[\"tokens\"]) for d in data) / len(data)}')
"
```

**Exercise 3.1.2: Train Mini Model**

```bash
# Train on subset for testing
python ml_training/train_transformer.py \
    --data midi_emotion_dataset.json \
    --epochs 10 \
    --batch-size 8 \
    --embed-dim 256 \
    --num-layers 4

# Expected output:
# Epoch 0: Loss = 3.2154
# Epoch 1: Loss = 2.8943
# ...
# Epoch 9: Loss = 1.9832
```

**Exercise 3.1.3: Test Generation**

```python
# File: test_generation.py
from emotion_transformer import CompoundWordTransformer
from midiutil import MIDIFile

model = CompoundWordTransformer(embed_dim=256, num_layers=4)
model.load_state_dict(torch.load('checkpoints/model_epoch_9.pt'))

# Generate grief melody
tokens = model.generate(
    valence=-0.7,
    arousal=0.3,
    num_tokens=128,
    temperature=0.9
)

# Convert to MIDI
midi = MIDIFile(1)
midi.addTempo(0, 0, 82)

for i, token in enumerate(tokens[1:-1]):  # Skip start/end
    # Decode compound token
    pitch = token % 128
    velocity = ((token // 128) % 32) * 4
    duration = 0.5  # Simplified

    midi.addNote(0, 0, pitch, i * 0.5, duration, velocity)

with open('generated_grief.mid', 'wb') as f:
    midi.writeFile(f)

print("✅ Generated MIDI saved!")
```

#### **Learning Checkpoints**:
- [ ] Can prepare MIDI dataset with emotion labels
- [ ] Transformer trains without errors
- [ ] Can generate emotion-conditioned MIDI
- [ ] Generated sequences sound musical (subjective)
- [ ] Understand attention mechanism basics

---

## 🎨 **Phase 4: DDSP Timbre Transfer (Weeks 11-14)**

### **Module 4.1: Differentiable Digital Signal Processing**

**Objective**: Neural synthesis for emotionally expressive timbres

#### **What is DDSP?**

```
Traditional Synthesis:
Parameters → Oscillators → Filters → Audio
↓ Problem: Parameters hard to control, not differentiable

DDSP:
Neural Network → Synthesis Parameters → Additive/Subtractive Synthesis → Audio
↓ Solution: End-to-end training, natural control
```

#### **Architecture for Kelly**:

```
Current: MIDI → Software Synth (external)
↓
Enhanced: MIDI + Emotion → DDSP Model → Expressive Audio
                           ↓
         Controls: Harmonic distribution, noise filtering, vibrato, etc.
```

#### **Implementation**:

**Step 4.1.1: Harmonic Synthesizer**

```python
# File: ml_training/ddsp_synth.py
import torch
import torch.nn as nn
import numpy as np

class HarmonicSynthesizer(nn.Module):
    """Additive synthesizer with neural control"""

    def __init__(self, sample_rate: int = 44100, n_harmonics: int = 64):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics

    def forward(
        self,
        f0: torch.Tensor,           # Fundamental frequency (batch, time)
        amplitudes: torch.Tensor,    # Harmonic amplitudes (batch, time, n_harmonics)
        noise_mags: torch.Tensor     # Noise filter magnitudes
    ) -> torch.Tensor:
        batch_size, n_frames = f0.shape
        hop_size = self.sample_rate // 50  # 20ms frames
        n_samples = n_frames * hop_size

        # Upsample to audio rate
        f0_upsampled = torch.nn.functional.interpolate(
            f0.unsqueeze(1), size=n_samples, mode='linear'
        ).squeeze(1)

        amps_upsampled = torch.nn.functional.interpolate(
            amplitudes.permute(0, 2, 1), size=n_samples, mode='linear'
        ).permute(0, 2, 1)

        # Generate harmonics
        harmonic_freqs = f0_upsampled.unsqueeze(-1) * torch.arange(
            1, self.n_harmonics + 1, device=f0.device
        )

        # Phase accumulation (key for proper synthesis)
        phases = torch.cumsum(
            2 * np.pi * harmonic_freqs / self.sample_rate, dim=1
        )

        # Synthesize harmonics
        harmonic_signal = (amps_upsampled * torch.sin(phases)).sum(dim=-1)

        # Add filtered noise
        noise = torch.randn(batch_size, n_samples, device=f0.device)
        noise_signal = self._filter_noise(noise, noise_mags, hop_size)

        return harmonic_signal + noise_signal

    def _filter_noise(self, noise, magnitudes, hop_size):
        """STFT-based noise filtering"""
        n_fft = 1024
        stft = torch.stft(noise, n_fft, hop_length=hop_size, return_complex=True)

        # Interpolate magnitudes to match STFT frames
        mags_interp = torch.nn.functional.interpolate(
            magnitudes.permute(0, 2, 1),
            size=stft.shape[2],
            mode='linear'
        ).permute(0, 2, 1)

        # Apply filter
        filtered = stft * mags_interp[:, :stft.shape[1], :].permute(0, 2, 1)

        return torch.istft(filtered, n_fft, hop_length=hop_size)
```

**Step 4.1.2: Neural Encoder**

```python
# File: ml_training/ddsp_encoder.py
import torch
import torch.nn as nn

class DDSPEncoder(nn.Module):
    """Predict synthesis parameters from audio features"""

    def __init__(
        self,
        n_harmonics: int = 64,
        n_noise_filters: int = 65
    ):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_noise_filters = n_noise_filters

        # Feature encoding
        self.gru = nn.GRU(
            input_size=2,  # f0 + loudness
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Harmonic distribution head
        self.harmonic_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, n_harmonics),
            nn.Softmax(dim=-1)
        )

        # Noise filter head
        self.noise_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, n_noise_filters),
            nn.Sigmoid()
        )

        # Overall amplitude head
        self.amplitude_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, f0, loudness):
        # Stack features
        features = torch.stack([f0, loudness], dim=-1)

        # Encode with GRU
        hidden, _ = self.gru(features)

        # Predict parameters
        harmonic_dist = self.harmonic_head(hidden)
        noise_mags = self.noise_head(hidden)
        amplitude = self.amplitude_head(hidden)

        # Scale harmonic distribution by amplitude
        amplitudes = harmonic_dist * amplitude

        return amplitudes, noise_mags

class DDSPModel(nn.Module):
    """Complete DDSP timbre transfer model"""

    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.encoder = DDSPEncoder()
        self.synthesizer = HarmonicSynthesizer(sample_rate)

    def forward(self, f0, loudness):
        amplitudes, noise_mags = self.encoder(f0, loudness)
        audio = self.synthesizer(f0, amplitudes, noise_mags)
        return audio
```

**Step 4.1.3: Emotion-Conditioned DDSP**

```python
# File: ml_training/emotion_ddsp.py
class EmotionDDSP(nn.Module):
    """DDSP with emotion conditioning for Kelly Project"""

    def __init__(self):
        super().__init__()
        self.ddsp = DDSPModel()

        # Emotion conditioning network
        self.emotion_proj = nn.Sequential(
            nn.Linear(3, 128),  # V, A, D
            nn.ReLU(),
            nn.Linear(128, 512)
        )

    def forward(self, f0, loudness, valence, arousal, dominance):
        # Create emotion vector
        emotion = torch.stack([valence, arousal, dominance], dim=-1)
        emotion_features = self.emotion_proj(emotion)

        # Condition encoder on emotion
        amplitudes, noise_mags = self.ddsp.encoder(f0, loudness)

        # Modulate based on emotion
        # High arousal → more noise, sharper harmonics
        # Low valence → darker timbre (lower harmonics)
        # High dominance → louder, more aggressive

        arousal_mod = arousal.unsqueeze(-1).unsqueeze(-1)
        valence_mod = valence.unsqueeze(-1).unsqueeze(-1)

        # Arousal affects noise level
        noise_mags = noise_mags * (0.5 + 0.5 * arousal_mod)

        # Valence affects harmonic distribution
        # Negative valence → emphasize lower harmonics
        harmonic_weights = torch.linspace(1.0, 0.5, amplitudes.shape[-1], device=f0.device)
        harmonic_weights = harmonic_weights ** (1.0 - valence_mod)
        amplitudes = amplitudes * harmonic_weights

        # Synthesize
        audio = self.ddsp.synthesizer(f0, amplitudes, noise_mags)
        return audio
```

**Step 4.1.4: Training with Multi-Scale Loss**

```python
# File: ml_training/train_ddsp.py
class MultiScaleSpectralLoss(nn.Module):
    """Sum of spectral losses at multiple FFT sizes"""

    def __init__(self, fft_sizes: list = [2048, 1024, 512, 256]):
        super().__init__()
        self.fft_sizes = fft_sizes

    def forward(self, predicted, target):
        loss = 0.0
        for fft_size in self.fft_sizes:
            hop = fft_size // 4

            pred_spec = torch.stft(
                predicted, fft_size, hop_length=hop, return_complex=True
            ).abs()
            target_spec = torch.stft(
                target, fft_size, hop_length=hop, return_complex=True
            ).abs()

            # Log magnitude loss
            loss += torch.mean(torch.abs(
                torch.log(pred_spec + 1e-7) - torch.log(target_spec + 1e-7)
            ))

            # Linear magnitude loss
            loss += torch.mean(torch.abs(pred_spec - target_spec))

        return loss / len(self.fft_sizes)

def train_ddsp():
    model = EmotionDDSP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    spectral_loss = MultiScaleSpectralLoss()

    # Training loop
    for epoch in range(100):
        for batch in dataloader:
            # Extract f0 and loudness from target audio
            f0 = extract_f0(batch['audio'])
            loudness = extract_loudness(batch['audio'])

            # Get emotion labels
            v, a, d = batch['valence'], batch['arousal'], batch['dominance']

            # Generate audio
            predicted = model(f0, loudness, v, a, d)

            # Compute loss
            loss = spectral_loss(predicted, batch['audio'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

#### **Integration with Plugin**:

```cpp
// File: src/voice/DDSPVoice.h
#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <onnxruntime_cxx_api.h>

namespace kelly {

class DDSPVoice : public juce::SynthesiserVoice {
public:
    DDSPVoice();

    bool loadModel(const juce::File& modelPath);

    void setEmotionParameters(float valence, float arousal, float dominance) {
        valence_ = valence;
        arousal_ = arousal;
        dominance_ = dominance;
    }

    void startNote(int midiNoteNumber, float velocity,
                   juce::SynthesiserSound*, int) override;

    void stopNote(float velocity, bool allowTailOff) override;

    void renderNextBlock(juce::AudioBuffer<float>& output,
                        int startSample, int numSamples) override;

    bool canPlaySound(juce::SynthesiserSound*) override { return true; }

private:
    std::unique_ptr<Ort::Session> session_;

    float valence_ = 0.0f;
    float arousal_ = 0.5f;
    float dominance_ = 0.5f;

    float currentFrequency_ = 440.0f;
    float currentLoudness_ = 0.5f;
    bool isPlaying_ = false;

    std::vector<float> synthesize(int numSamples);
};

} // namespace kelly
```

#### **Hands-On Exercises**:

**Exercise 4.1.1: Extract Audio Features**

```python
# File: utils/extract_audio_features.py
import librosa
import numpy as np

def extract_f0_and_loudness(audio_file, sr=44100):
    """Extract pitch and loudness for DDSP training"""
    y, sr = librosa.load(audio_file, sr=sr)

    # Extract f0 using CREPE or pYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )

    # Fill unvoiced regions
    f0 = np.nan_to_num(f0, nan=0.0)

    # Extract RMS loudness
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    loudness = librosa.amplitude_to_db(rms)
    loudness = (loudness + 80) / 80  # Normalize to [0, 1]

    return f0, loudness
```

**Exercise 4.1.2: Test Harmonic Synthesis**

```python
# Test that harmonic synthesizer works
import torch
import soundfile as sf

synth = HarmonicSynthesizer(sample_rate=44100)

# Create test inputs
n_frames = 100
f0 = torch.ones(1, n_frames) * 440.0  # A4
amplitudes = torch.rand(1, n_frames, 64)
amplitudes = amplitudes / amplitudes.sum(dim=-1, keepdim=True)
noise_mags = torch.ones(1, n_frames, 65) * 0.01

# Synthesize
audio = synth(f0, amplitudes, noise_mags)

# Save
sf.write('test_harmonic.wav', audio[0].numpy(), 44100)
print("✅ Saved test_harmonic.wav - listen for A440!")
```

#### **Learning Checkpoints**:
- [ ] Understand additive synthesis principles
- [ ] Can extract f0 and loudness from audio
- [ ] Harmonic synthesizer produces recognizable tones
- [ ] Multi-scale spectral loss converges during training
- [ ] Generated audio matches target timbre

---

## 🚀 **Phase 5: Production Integration (Weeks 15-16)**

### **Module 5.1: Tauri Companion App**

**Objective**: Build desktop app for model training and plugin control

#### **Architecture**:

```
Tauri App (Rust + React)
├── Frontend (React + TypeScript)
│   ├── Training UI
│   ├── Model Management
│   └── Plugin Control
│
└── Backend (Rust)
    ├── IPC with Plugin (Unix sockets)
    ├── Training Process Manager
    └── File System Access
```

#### **Implementation**:

**Step 5.1.1: Create Tauri Project**

```bash
# Install Tauri
npm install -g @tauri-apps/cli

# Create project
npm create tauri-app@latest kelly-companion

# Template choices:
# - Package manager: npm
# - UI framework: React + TypeScript
# - Bundler: Vite
```

**Step 5.1.2: Training Panel Component**

```typescript
// File: kelly-companion/src/components/TrainingPanel.tsx
import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';

interface TrainingProgress {
  epoch: number;
  total_epochs: number;
  loss: number;
  val_loss: number | null;
  eta_seconds: number;
}

export function TrainingPanel() {
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState<TrainingProgress | null>(null);

  useEffect(() => {
    const unlisten = listen<TrainingProgress>('training-progress', (event) => {
      setProgress(event.payload);
    });
    return () => { unlisten.then(fn => fn()); };
  }, []);

  const startTraining = async () => {
    setIsTraining(true);
    try {
      await invoke('train_emotion_model', {
        config: {
          dataset_path: '/path/to/midi/dataset',
          model_type: 'compound_word_transformer',
          epochs: 100,
          batch_size: 32,
          learning_rate: 0.0001
        }
      });
    } catch (e) {
      console.error('Training failed:', e);
    }
    setIsTraining(false);
  };

  return (
    <div className="training-panel">
      <h2>Train Emotion Model</h2>
      {progress && (
        <div className="progress">
          <p>Epoch {progress.epoch}/{progress.total_epochs}</p>
          <p>Loss: {progress.loss.toFixed(4)}</p>
          <progress value={progress.epoch} max={progress.total_epochs} />
          <p>ETA: {Math.floor(progress.eta_seconds / 60)}m</p>
        </div>
      )}
      <button onClick={startTraining} disabled={isTraining}>
        {isTraining ? 'Training...' : 'Start Training'}
      </button>
    </div>
  );
}
```

**Step 5.1.3: IPC Server (Rust Backend)**

```rust
// File: kelly-companion/src-tauri/src/ipc.rs
use serde::{Deserialize, Serialize};
use tokio::net::UnixListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Debug, Serialize, Deserialize)]
pub enum PluginMessage {
    LoadModel { path: String },
    SetEmotion { valence: f32, arousal: f32, dominance: f32 },
    GenerateMIDI { bars: u32 },
}

#[derive(Debug, Serialize, Deserialize)]
pub enum PluginResponse {
    Ok,
    ModelLoaded { name: String },
    MIDIGenerated { path: String },
    Error { message: String },
}

pub struct IPCServer {
    socket_path: String,
}

impl IPCServer {
    pub fn new() -> Self {
        Self {
            socket_path: "/tmp/kelly_plugin.sock".to_string(),
        }
    }

    pub async fn send(&self, message: PluginMessage) -> Result<PluginResponse, String> {
        let stream = UnixStream::connect(&self.socket_path)
            .await
            .map_err(|e| e.to_string())?;

        let msg_bytes = serde_json::to_vec(&message)
            .map_err(|e| e.to_string())?;

        // Send length-prefixed message
        let len = (msg_bytes.len() as u32).to_le_bytes();
        stream.write_all(&len).await.map_err(|e| e.to_string())?;
        stream.write_all(&msg_bytes).await.map_err(|e| e.to_string())?;

        // Read response
        let mut len_bytes = [0u8; 4];
        stream.read_exact(&mut len_bytes).await.map_err(|e| e.to_string())?;
        let resp_len = u32::from_le_bytes(len_bytes) as usize;

        let mut resp_bytes = vec![0u8; resp_len];
        stream.read_exact(&mut resp_bytes).await.map_err(|e| e.to_string())?;

        serde_json::from_slice(&resp_bytes).map_err(|e| e.to_string())
    }
}
```

#### **Learning Checkpoints**:
- [ ] Tauri app builds and runs
- [ ] Can communicate with plugin via IPC
- [ ] Training progress updates in real-time
- [ ] Model files managed correctly

---

## 📋 **Complete Implementation Checklist**

### **Phase 1: Foundation** ✅
- [x] Understand current Kelly architecture
- [x] Verify all 5 emotion formulas
- [x] Pass all 29 unit tests
- [x] Build plugin successfully

### **Phase 2: Neural Integration** 🔄
- [ ] RTNeural dependency added
- [ ] Feature extraction implemented
- [ ] Inference < 2ms per call
- [ ] Lock-free buffers working
- [ ] Async inference thread stable

### **Phase 3: Transformer MIDI** 🔄
- [ ] MIDI dataset prepared (1000+ files)
- [ ] Transformer trains successfully
- [ ] Generated MIDI sounds musical
- [ ] ONNX export working
- [ ] C++ integration complete

### **Phase 4: DDSP Timbre** 🔄
- [ ] Harmonic synthesizer working
- [ ] F0 extraction accurate
- [ ] Multi-scale loss converging
- [ ] Emotion conditioning effective
- [ ] Real-time synthesis in plugin

### **Phase 5: Production** 🔄
- [ ] Tauri app functional
- [ ] IPC communication stable
- [ ] Model training from UI
- [ ] Plugin control working
- [ ] Full system integrated

---

## 🎓 **Resources & References**

### **Papers**:
1. **DDSP**: "DDSP: Differentiable Digital Signal Processing" (Engel et al., 2020)
2. **Compound Word Transformer**: "MuseNet" (OpenAI, 2019)
3. **RTNeural**: "Real-time neural audio" (Chowdhury, 2021)
4. **Emotion Models**: "Pleasure-Arousal-Dominance: A General Framework" (Mehrabian, 1996)

### **Code Examples**:
- DDSP GitHub: https://github.com/magenta/ddsp
- RTNeural: https://github.com/jatinchowdhury18/RTNeural
- Tauri: https://tauri.app/

### **Your Codebase**:
- Emotion Formulas: `src/engine/EmotionMusicMapper.h`
- MIDI Generation: `src/midi/MidiGenerator.cpp`
- Plugin Core: `src/plugin/PluginProcessor.cpp`

---

## ⏱️ **Timeline**

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1-2 | Foundation | Understand codebase, pass all tests |
| 3-4 | RTNeural | Working emotion inference |
| 5-6 | Threading | Lock-free async inference |
| 7-8 | Transformer Prep | Dataset + training code |
| 9-10 | Transformer Training | Trained model |
| 11-12 | DDSP Prep | Harmonic synth working |
| 13-14 | DDSP Training | Trained timbre model |
| 15-16 | Integration | Full system working |

---

**Next Step**: Choose starting point based on your priority:
- **Want fastest results?** → Start with Phase 2 (RTNeural)
- **Want best music quality?** → Start with Phase 3 (Transformer)
- **Want expressive sound?** → Start with Phase 4 (DDSP)
