# Kelly ML/DSP Quick Start Guide

## 🚀 **Getting Started in 5 Minutes**

### **Option 1: Add Real-Time Emotion Recognition (Easiest)**

```bash
# 1. Install RTNeural
cd "/Users/seanburdges/Desktop/final kel"
git clone https://github.com/jatinchowdhury18/RTNeural.git external/RTNeural

# 2. Add to CMakeLists.txt
echo "add_subdirectory(external/RTNeural)" >> CMakeLists.txt
echo "target_link_libraries(KellyMidiCompanion PRIVATE RTNeural)" >> CMakeLists.txt

# 3. Create placeholder model
python3 -c "import json; json.dump({'layers': []}, open('emotion_model.json', 'w'))"

# 4. Rebuild
cmake --build build
```

**Time investment**: 2-3 weeks
**Benefit**: Real-time audio → emotion detection

---

### **Option 2: Add AI MIDI Generation (Most Musical)**

```bash
# 1. Install PyTorch
pip3 install torch torchvision torchaudio

# 2. Prepare small dataset
mkdir -p ml_training/midi_data
# Copy 50-100 MIDI files to ml_training/midi_data

# 3. Train mini model
cd ml_training
python3 train_transformer.py --epochs 10 --batch-size 8

# 4. Export to ONNX
python3 export_transformer.py

# 5. Integrate with plugin (C++)
# See LEARNING_PROGRAM.md Module 3.1
```

**Time investment**: 4-6 weeks
**Benefit**: AI-generated melodies conditioned on emotion

---

### **Option 3: Add Timbre Transfer (Most Expressive)**

```bash
# 1. Install audio processing libs
pip3 install librosa soundfile

# 2. Collect audio samples
mkdir -p ml_training/audio_samples/violin
mkdir -p ml_training/audio_samples/voice
# Add 20-30 samples each

# 3. Train DDSP model
python3 ml_training/train_ddsp.py

# 4. Export for plugin
python3 ml_training/export_ddsp.py
```

**Time investment**: 6-8 weeks
**Benefit**: Neural synthesis with emotional timbre control

---

## 📊 **Current System Status**

### **✅ What's Already Working**:
```
✓ 72-emotion PAD model (Pleasure-Arousal-Dominance)
✓ 5 emotion-to-music formulas (tempo, velocity, mode, reward, resonance)
✓ 14 MIDI generation engines (melody, bass, chords, pads, etc.)
✓ Full plugin build (AU + VST3)
✓ 29/29 unit tests passing
✓ Installed in Logic Pro
```

### **🎯 What We're Adding**:
```
→ Real-time neural emotion recognition from audio
→ AI-generated MIDI sequences (transformer)
→ Neural synthesis with timbre transfer (DDSP)
→ Desktop companion app for training models
→ Lock-free threading for real-time ML
```

---

## 🧠 **Architecture Overview**

```
CURRENT (Working):
┌──────────────┐
│ Text Input   │ → WoundProcessor → EmotionNode → MidiGenerator → MIDI Out
└──────────────┘         ↓
                   PAD Coordinates (V/A/D)
                         ↓
                   Formulas (tempo/velocity/mode)

ENHANCED (Adding):
┌──────────────┐
│ Audio Input  │ → Feature Extract → RTNeural → Emotion Vector ┐
└──────────────┘                                                 │
                                                                 ├→ Fusion
┌──────────────┐                                                 │
│ Text Input   │ → WoundProcessor → EmotionNode ────────────────┘
└──────────────┘                         ↓
                                    Enhanced VAD
                    ┌───────────────────┴────────────────────┐
                    ↓                                         ↓
         Rule-Based Generator                    Transformer Generator
           (Fast, Therapeutic)                    (Creative, Varied)
                    ↓                                         ↓
                    └─────────────→ Merge ←──────────────────┘
                                      ↓
                                 MIDI Notes
                                      ↓
                                 DDSP Voice
                                      ↓
                               Expressive Audio
```

---

## 📁 **File Structure**

```
final kel/
├── src/
│   ├── engine/
│   │   ├── EmotionMusicMapper.h          # ✅ Core formulas (done)
│   │   ├── NeuralEmotionProcessor.h      # 🔄 Add RTNeural (new)
│   │   └── TransformerMIDIGenerator.h    # 🔄 Add transformer (new)
│   ├── midi/
│   │   └── MidiGenerator.cpp             # ✅ Main orchestrator (done)
│   ├── voice/
│   │   └── DDSPVoice.h                   # 🔄 Add DDSP synth (new)
│   └── common/
│       └── LockFreeRingBuffer.h          # 🔄 Add threading (new)
│
├── ml_training/                           # 🔄 New folder
│   ├── train_transformer.py
│   ├── train_ddsp.py
│   ├── export_transformer.py
│   └── datasets/
│
├── external/
│   └── RTNeural/                         # 🔄 Clone from GitHub
│
├── LEARNING_PROGRAM.md                   # 📚 Full curriculum
└── QUICK_START_GUIDE.md                  # 📋 This file
```

---

## 🎓 **Learning Path Decision Tree**

**Question 1**: Do you have MIDI files with emotion labels?
- **YES** → Start with **Phase 3: Transformer** (best music quality)
- **NO** → Start with **Phase 2: RTNeural** (works with any audio)

**Question 2**: Is your priority therapeutic accuracy or creative variety?
- **Therapeutic** → Enhance current rule-based system with RTNeural
- **Creative** → Add Transformer for AI-generated variations

**Question 3**: Do you need real-time synthesis or external DAW?
- **Real-time** → Add **Phase 4: DDSP** voice
- **External** → Just export MIDI (current system works)

---

## 💡 **Recommended Starting Point**

### **For Most Users: Phase 2 (RTNeural)**

**Why?**
1. Fastest to implement (2-3 weeks)
2. Works with any audio input
3. Enhances existing system (no replacement needed)
4. Real-time performance

**What You Get:**
```
Before: Text → Emotion → MIDI
After:  Audio → Neural Emotion Detection → Enhanced MIDI
        Text  →        ↓
```

**First Steps:**
1. Read `LEARNING_PROGRAM.md` Module 1.1 (understand current system)
2. Read `LEARNING_PROGRAM.md` Module 2.1 (RTNeural integration)
3. Complete Exercise 2.1.1 (train simple model)
4. Test in plugin

---

## 🔧 **Development Environment Setup**

```bash
# 1. Python environment
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio librosa soundfile onnx

# 2. C++ dependencies (already have JUCE)
brew install cmake ninja

# 3. Optional: ONNX Runtime (for transformer/DDSP)
brew install onnxruntime

# 4. Verify current build still works
cd "/Users/seanburdges/Desktop/final kel"
cmake --build build --target KellyTests
./build/tests/KellyTests
# Should see: [  PASSED  ] 29 tests
```

---

## 📞 **Support & Resources**

### **Existing Documentation:**
- `EMOTION_TO_MUSIC_FORMULAS.md` - Current formula implementation
- `LEARNING_PROGRAM.md` - Complete ML/DSP curriculum
- `tests/` - 29 unit tests showing how everything works

### **External Resources:**
- **RTNeural**: https://github.com/jatinchowdhury18/RTNeural
- **DDSP**: https://github.com/magenta/ddsp
- **PyTorch**: https://pytorch.org/tutorials/
- **ONNX**: https://onnx.ai/

### **Your Codebase Highlights:**
- **Best starting point**: `src/engine/EmotionMusicMapper.h:34-45`
  - See how tempo/velocity formulas work
  - Add neural predictions alongside formulas

- **MIDI generation entry**: `src/midi/MidiGenerator.cpp:generate()`
  - Line ~38: Where emotions become music
  - Perfect place to insert transformer output

- **Plugin audio thread**: `src/plugin/PluginProcessor.cpp:processBlock()`
  - Where audio flows
  - Add feature extraction here

---

## ⚡ **Quick Wins**

### **Win #1: Add Emotion Smoothing (10 minutes)**

```cpp
// File: src/plugin/PluginProcessor.cpp
// Add after line ~50 (in processBlock)

// Smooth emotion updates
float alpha = 0.1f;  // Smoothing factor
currentValence = alpha * newValence + (1.0f - alpha) * currentValence;
currentArousal = alpha * newArousal + (1.0f - alpha) * currentArousal;

// Now use smoothed values for MIDI generation
```

**Effect**: Smoother emotion transitions, less jarring changes

---

### **Win #2: Add Emotion Visualization (30 minutes)**

```cpp
// File: src/plugin/PluginEditor.cpp
// Add to paint() method

void paint(juce::Graphics& g) override {
    // ... existing code ...

    // Draw emotion circle
    float x = (valence + 1.0f) * 0.5f * getWidth();   // -1 to 1 → 0 to width
    float y = (1.0f - arousal) * getHeight();          // 0 to 1 → height to 0

    g.setColour(juce::Colours::red);
    g.fillEllipse(x - 5, y - 5, 10, 10);
}
```

**Effect**: Real-time visualization of emotion state

---

### **Win #3: Log Emotion Stats (5 minutes)**

```cpp
// File: src/plugin/PluginProcessor.cpp

void processBlock(...) {
    // ... existing code ...

    static int logCounter = 0;
    if (++logCounter % 100 == 0) {  // Every 100 blocks
        DBG("Emotion: V=" << currentValence << " A=" << currentArousal
            << " → Tempo=" << calculatedTempo << " BPM");
    }
}
```

**Effect**: See emotion-to-music mappings in console

---

## 🎯 **Next Actions**

### **This Week:**
- [ ] Read `LEARNING_PROGRAM.md` Phase 1 (Foundation)
- [ ] Run all existing tests: `./build/tests/KellyTests`
- [ ] Trace one emotion through the system (Exercise 1.1.1)
- [ ] Decide which ML feature to add first

### **Next Week:**
- [ ] Set up Python environment
- [ ] Choose: RTNeural OR Transformer OR DDSP
- [ ] Complete Module 2.1, 3.1, or 4.1 from learning program
- [ ] Build first prototype

### **Month 1 Goal:**
- [ ] One ML feature fully integrated and working
- [ ] Can demonstrate emotion → ML → music pipeline
- [ ] Plugin still stable (all tests passing)

---

**Remember**: The current system already works beautifully! These ML additions are enhancements, not replacements. Start small, test often, and build incrementally.

**Need help?** Reference the detailed `LEARNING_PROGRAM.md` for step-by-step instructions on any module.
