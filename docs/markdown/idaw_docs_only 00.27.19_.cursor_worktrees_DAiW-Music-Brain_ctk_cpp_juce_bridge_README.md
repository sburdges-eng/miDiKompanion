# DAiW JUCE Bridge (Skeleton)

This folder bootstraps a JUCE-based desktop/plugin project that will eventually wrap DAiW’s voice/backing engines. The current state is a placeholder GUI + stub bridge client.

## Dependencies

- CMake ≥ 3.18
- A C++17 compiler (Xcode/Clang on macOS, MSVC on Windows, GCC/Clang on Linux)
- JUCE 7.0.9 (automatically fetched by default)

## Building (Standalone App)

```bash
cd cpp/juce_bridge
cmake -B build -S . -G "Unix Makefiles"
cmake --build build
```

Executables are created under `build/DAiWJuceBridge_artefacts/`.

### Using a Local JUCE Checkout

```bash
cmake -B build -S . -DDAIW_FETCH_JUCE=OFF -DJUCE_DIR=/path/to/JUCE
```

## Structure

- `Source/Main.cpp` – JUCE application entry point
- `Source/MainComponent.*` – placeholder UI
- `Source/Bridge/BridgeClient.*` – stub communication layer (will call DAiW Python API)
- `Source/Voice/VoiceProcessor.*` – future audio processor wrapping auto-tune/modulation

## Next Steps

1. Implement `BridgeClient` using HTTP/gRPC to talk to DAiW’s Python services.
2. Replace `VoiceProcessor::processBlock` with streaming audio to/from the bridge.
3. Add plugin targets (VST3/AU/AAX) once the standalone flow stabilizes.
4. Expose name generator + text-to-talk previews in the JUCE UI.
