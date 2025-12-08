# JUCE Bridge Plan

Design notes for a native C++/JUCE wrapper that exposes DAiW’s voice + backing-track engines inside traditional DAW/plugin workflows.

---

## 1. Goals

1. **Voice everywhere** – Auto-tune, modulation, and text-to-talk should run inside a JUCE plugin or standalone app with minimal latency.
2. **Intent-aware UX** – Keep DAiW’s “interrogate before generate” flow by piping intent metadata into the C++ layer (presets, rule breaks, naming prompts).
3. **Cross-platform** – macOS (ARM/Intel) + Windows (AMD/Intel) builds, optional Linux standalone.
4. **Scalable architecture** – Allow both embedded Python (pybind11) and out-of-process gRPC/HTTP bridges so heavy ML workloads stay in Python land when needed.

---

## 2. High-Level Architecture

```
┌───────────────────────────┐       ┌─────────────────────────┐
│  JUCE Host (Plugin/App)   │       │   DAiW Python Runtime   │
│                           │  RPC  │  (existing CLI/API)     │
│  • Audio/MIDI Device I/O  ├──────►│  • Voice engines        │
│  • UI (presets, intent)   │◄──────┤  • Backing generator    │
│  • Offline render passes  │       │  • Intent metadata      │
└───────────────────────────┘       └─────────────────────────┘
```

### Components

| Module | Responsibility |
|--------|----------------|
| **JUCEHostApp** | Audio device handling, UI/UX, preset browsing. Builds standalone + VST3/AU/AAX from same codebase. |
| **BridgeClient** | Handles communication with DAiW Python services. Two modes: embedded (pybind11) or remote (gRPC/HTTP). |
| **VoiceProcessor** | C++ wrappers for voice features. Streams audio blocks to DAiW or uses native DSP when available. |
| **BackingTrackManager** | Requests backing tracks from DAiW, handles caching, renders previews. |
| **PresetSync** | Loads/stores JSON presets, synchronizes with DAiW’s metadata (intent, rule breaks, naming). |

---

## 3. Data Flow Details

### A. Voice Auto-Tune / Modulation

1. JUCE captures incoming vocal buffer.
2. Buffer + key/mode metadata sent to `BridgeClient`.
3. Python `AutoTuneProcessor` returns processed buffer.
4. Optional local preview using streaming chunks; offline rendering for accuracy.

### B. Text-to-Talk “Name Announcer”

1. User enters alias/title in JUCE UI.
2. Request hits `VoiceSynthesizer.speak_text`.
3. WAV returned and cached; plugin can trigger sample via MIDI or automation.

### C. Backing Track Requests

1. User selects intent preset or imports existing JSON.
2. JUCE requests `backing_generator` plan; receives stems + metadata.
3. Stems loaded into JUCE’s internal sampler/player; metadata shown (key, rule break, tempo).

---

## 4. Technical Considerations

### 4.1 IPC vs Embedded Python

| Mode | Pros | Cons |
|------|------|------|
| **Embedded (pybind11)** | Lower latency, no external service. | Packaging complexity, Python env mgmt per OS. |
| **gRPC/HTTP Service** | Easy to scale, reuses existing CLI. | Requires background service; potential latency. |

Initial plan: start with gRPC (easier to iterate), then add optional embedded mode for offline/standalone builds.

### 4.2 Project Structure

```
cpp/
  juce_bridge/
    CMakeLists.txt (or Projucer file)
    Source/
      App.cpp
      PluginProcessor.cpp
      UI/
        VoicePanel.cpp
        IntentPanel.cpp
      Bridge/
        BridgeClient.h/.cpp
        Proto/ (if gRPC)
      DSP/
        VoiceBufferQueue.h
        OfflineRenderJob.h
    Resources/
      Presets/
      Icons/
```

### 4.3 AMD Optimization Hooks

- Build with AOCC where available; enable AVX2/AVX512 flags for FFT-heavy routines.
- Provide toggle to offload FFT/pitch detection to native C++ (e.g., via `AudioAnalyzer` port) to minimize round trips.

---

## 5. Phased Implementation

| Phase | Deliverables |
|-------|--------------|
| **Phase A – Skeleton** | JUCE project + `BridgeClient` stub calling REST endpoint; placeholder UI showing connection status. |
| **Phase B – Voice Loop** | Real-time streaming for auto-tune/modulate; fallback to offline render when latency too high. |
| **Phase C – Text-to-Talk & Naming** | UI flow for entering aliases, previewing spoken prompts, and saving WAV/MIDI triggers. |
| **Phase D – Backing Tracks** | Request/receive stems, display arrangement timeline, allow drag-drop to DAW. |
| **Phase E – Embedded Mode** | Bundle Python runtime or convert key DSP blocks to C++. |

---

## 6. Naming Workflow Integration

1. **Name Suggestion** – expose DAiW’s naming helper (based on intent) inside JUCE.
2. **Speak Preview** – automatically call `speak_text` for each suggested name.
3. **Macro** – user selects favorite name; plugin inserts spoken intro clip and updates metadata.

---

## 7. Next Actions (for follow-up tasks)

1. Create `cpp/juce_bridge` folder with CMake + JUCE module references.
2. Define `BridgeClient` interface (pure virtual) and implement HTTP version using cURL or Boost.Beast.
3. Add gRPC proto definitions aligning with `DAiWAPI` endpoints (`auto_tune`, `modulate`, `speak_text`, `generate_backing`).
4. Draft UI mockups (Figma or markdown) for voice + naming panels.
5. Document build steps (macOS: Projucer/Xcode; Windows: Visual Studio; Linux: CMake).

This plan satisfies “begin C++/JUCE wrap” by providing the architectural blueprint. Next todos (`cpp2`, `cpp3`, `cpp4`) will implement the skeleton, stub interfaces, and docs.
