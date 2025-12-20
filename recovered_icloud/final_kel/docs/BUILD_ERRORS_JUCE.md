# JUCE Build System Error Documentation

## Error Summary

During the CMake configuration phase, the JUCE build system fails when attempting to build `juceaide`, a required build tool for JUCE plugins.

## Error Details

### Error Message

```
CMake Error at external/JUCE/extras/Build/juceaide/CMakeLists.txt:142 (message):
  Failed to build juceaide

error: error opening
'CMakeFiles/juceaide.dir/__/__/__/modules/juce_graphics/juce_graphics_Harfbuzz.cpp.o.d':
CMakeFiles/juceaide.dir/__/__/__/modules/juce_graphics/juce_graphics_Harfbuzz.cpp.o.d:
No such file or directory
```

### Environment

- **OS**: macOS 25.1.0 (darwin)
- **Compiler**: Apple Clang 17.0.0.17000603
- **CMake**: 4.2.1
- **JUCE Version**: 8.0.4 (from `external/JUCE`)
- **Build System**: Unix Makefiles

### Root Cause

The build system attempts to write dependency files (`.d` files) to directories that don't exist. This is a known issue with:

1. Parallel builds where directory creation races occur
2. JUCE 8.0.4 build system on certain macOS configurations
3. CMake's dependency file generation in deeply nested paths

### Warnings Observed

```
warning: variable 'target' set but not used [-Wunused-but-set-variable]
warning: enumeration value 'NSEventTypeMouseCancelled' not explicitly handled in switch [-Wswitch-enum]
```

These are non-critical warnings from JUCE's macOS-specific code.

## Workarounds Attempted

1. ✅ **Single-threaded build**: Not attempted yet (recommended next step)
2. ✅ **Clean build directory**: Attempted - issue persists
3. ✅ **Manual directory creation**: Attempted - CMake recreates structure
4. ❌ **Ninja generator**: Not attempted yet (may help with parallel builds)

## Recommended Solutions

### Solution 1: Single-threaded Build (Recommended First)

```bash
cd "/Users/seanburdges/Desktop/final kel"
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DENABLE_RTNEURAL=OFF
cmake --build build --config Release -j1
```

### Solution 2: Use Ninja Generator

```bash
cd "/Users/seanburdges/Desktop/final kel"
rm -rf build
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DENABLE_RTNEURAL=OFF
cmake --build build --config Release
```

### Solution 3: Pre-build juceaide Manually

```bash
cd external/JUCE/extras/Build/juceaide
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j1
```

### Solution 4: Update JUCE Version

Consider upgrading to JUCE 8.0.10 (latest stable) which may have fixes for this issue:

```bash
cd external/JUCE
git checkout 8.0.10
```

### Solution 5: Disable Tests Temporarily

Build without tests to avoid Google Test dependency issues:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DENABLE_RTNEURAL=OFF
cmake --build build --config Release -j1
```

## Related Issues

- **RTNeural Fetch Error**: Network issue fetching RTNeural from GitHub
  - Workaround: Clone manually or disable with `-DENABLE_RTNEURAL=OFF`

## Status

- **Phase 1**: ✅ Complete (Prerequisites verified)
- **Phase 2**: ✅ Complete (All Python environments set up)
- **Phase 3**: ⚠️ Blocked (JUCE build system error)
- **Phase 4**: ⏳ Pending (Waiting for Phase 3)

## Next Steps

1. Try single-threaded build (Solution 1)
2. If that fails, try Ninja generator (Solution 2)
3. Continue with Python tests (Phase 4.2) while C++ build is resolved
4. Document successful workaround for future builds
