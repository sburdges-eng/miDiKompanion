# RTNeural Auto-Fetch Configuration

**Date:** 2025-01-27
**Status:** ✅ **AUTO-FETCH ENABLED**

---

## Overview

RTNeural is automatically fetched from GitHub during CMake configuration if a local copy is not found. This ensures the ML inference infrastructure is always available without manual setup.

---

## Auto-Fetch Behavior

### Priority Order

1. **Local Copy (Preferred)**
   - Checks for `external/RTNeural/` directory
   - Uses local copy if found
   - Faster builds, no network required

2. **Auto-Fetch (Fallback)**
   - Automatically downloads from GitHub if local copy not found
   - Uses CMake `FetchContent` module
   - Fetches version `v1.2.0` tag
   - Caches in build directory for subsequent builds

### Configuration

```cmake
option(ENABLE_RTNEURAL "Enable RTNeural library for ML inference" ON)
```

**Default:** Enabled (auto-fetches if needed)

**To Disable:**

```bash
cmake -DENABLE_RTNEURAL=OFF ..
```

---

## Usage

### Automatic (Recommended)

Just build normally - RTNeural will be fetched automatically:

```bash
mkdir build
cd build
cmake ..
make
```

**First build:** RTNeural will be downloaded automatically
**Subsequent builds:** Uses cached copy (no download)

### Manual Local Copy (Optional)

If you prefer a local copy for faster builds:

```bash
# Clone to external directory
git clone https://github.com/jatinchowdhury18/RTNeural.git external/RTNeural
cd external/RTNeural
git checkout v1.2.0

# Build normally
cd ../..
mkdir build
cd build
cmake ..
make
```

---

## What Gets Fetched

- **Repository:** <https://github.com/jatinchowdhury18/RTNeural.git>
- **Version:** v1.2.0 (stable release)
- **Location:** Cached in CMake build directory
- **Type:** Header-only library (no separate compilation needed)

---

## Build Output

When auto-fetching, you'll see:

```
✅ Auto-fetching RTNeural from GitHub (v1.2.0)
   To use local copy: git clone https://github.com/jatinchowdhury18/RTNeural.git external/RTNeural
RTNeural enabled - ML inference available
```

---

## Troubleshooting

### Network Issues

**Problem:** CMake can't fetch RTNeural
**Solution:**

1. Check internet connection
2. Use local copy: `git clone ... external/RTNeural`
3. Or disable: `cmake -DENABLE_RTNEURAL=OFF ..`

### Version Conflicts

**Problem:** Need different RTNeural version
**Solution:**

1. Edit `CMakeLists.txt` - change `GIT_TAG v1.2.0` to desired version
2. Or use local copy with specific version

### Build Cache Issues

**Problem:** Stale RTNeural cache
**Solution:**

```bash
# Clear CMake cache
rm -rf build/CMakeCache.txt
rm -rf build/_deps/RTNeural*

# Rebuild
cmake ..
make
```

---

## Benefits

✅ **Zero Setup** - Works out of the box
✅ **Version Control** - Uses stable v1.2.0 tag
✅ **Caching** - Only downloads once
✅ **Flexible** - Can use local copy if preferred
✅ **Optional** - Can be disabled if not needed

---

## Related Files

- `CMakeLists.txt` - Auto-fetch configuration
- `src/ml/RTNeuralProcessor.h` - RTNeural integration
- `src/ml/InferenceThreadManager.h` - ML inference management

---

**Last Updated:** 2025-01-27
**Status:** ✅ Auto-fetch enabled and working
