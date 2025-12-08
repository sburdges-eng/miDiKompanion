# C++ Testing Guide for DAiW Bridge Plugin

## Overview

This guide explains how to build, run, and extend the C++ test suite for the DAiW Bridge JUCE plugin.

## Test Structure

### Test Files

1. **`PluginProcessorTest.cpp`** - Core processor functionality
   - Initialization
   - Connection management
   - Audio processing
   - OSC communication
   - JSON parsing
   - MIDI event handling
   - Error handling
   - Performance

2. **`PluginEditorTest.cpp`** - UI component tests
   - Editor initialization
   - Resize handling
   - Paint operations
   - Component lifecycle

3. **`OSCCommunicationTest.cpp`** - OSC protocol tests
   - Message format validation
   - JSON response parsing
   - Error handling
   - Timing calculations

## Building Tests

### Option 1: CMake (Recommended)

```bash
cd cpp/DAiWBridge
mkdir build && cd build
cmake ..
make
./DAiWBridgeTests
```

### Option 2: JUCE Projucer

1. Open `DAiWBridge.jucer` in Projucer
2. Add test files to project
3. Generate Xcode/Visual Studio project
4. Build test target
5. Run tests

### Option 3: Manual Compilation

```bash
# macOS
clang++ -std=c++17 \
    -I/path/to/JUCE/modules \
    -framework CoreFoundation \
    -framework CoreAudio \
    PluginProcessorTest.cpp PluginProcessor.cpp \
    -o DAiWBridgeTests

# Linux
g++ -std=c++17 \
    -I/path/to/JUCE/modules \
    PluginProcessorTest.cpp PluginProcessor.cpp \
    -ljuce_core -ljuce_audio_basics -ljuce_audio_processors \
    -o DAiWBridgeTests
```

## Running Tests

### All Tests
```bash
./DAiWBridgeTests
```

### Filter by Tag
```bash
./DAiWBridgeTests "[processor]"  # Processor tests only
./DAiWBridgeTests "[osc]"         # OSC tests only
./DAiWBridgeTests "[json]"       # JSON parsing tests
./DAiWBridgeTests "[error]"      # Error handling tests
```

### Verbose Output
```bash
./DAiWBridgeTests -s  # Show success messages
./DAiWBridgeTests -v  # Verbose output
```

## Test Categories

### Basic Functionality (`[processor]`)
- Plugin initialization
- Connection status
- Prepare to play
- Process block

### OSC Communication (`[osc]`)
- Message format validation
- Send generate request
- Send ping
- Receive responses

### JSON Parsing (`[json]`)
- Valid JSON parsing
- Invalid JSON handling
- Missing properties
- Boundary values

### Error Handling (`[error]`)
- Error message handling
- Unknown OSC messages
- Invalid data

### Performance (`[performance]`)
- Process block performance
- Memory management

## Writing New Tests

### Test Case Template

```cpp
CATCH_TEST_CASE("Test Name", "[tag]")
{
    PluginProcessorTestFixture fixture;
    
    // Setup
    fixture.processor->prepareToPlay(44100.0, 512);
    
    // Test
    REQUIRE(condition);
    
    // Cleanup
    fixture.processor->releaseResources();
}
```

### Test Fixtures

Use `PluginProcessorTestFixture` for processor tests:
```cpp
PluginProcessorTestFixture fixture;
auto* processor = fixture.processor.get();
```

Use `PluginEditorTestFixture` for editor tests:
```cpp
PluginEditorTestFixture fixture;
auto* editor = fixture.editor.get();
```

### Assertions

- `REQUIRE(condition)` - Fails test if false
- `REQUIRE_NOTHROW(expression)` - Ensures no exception
- `REQUIRE_FALSE(condition)` - Requires false
- `REQUIRE_THROWS(expression)` - Requires exception

## Test Coverage

### Current Coverage

✅ **PluginProcessor**
- Initialization
- Connection management
- Audio processing
- OSC message sending
- JSON parsing
- MIDI event scheduling
- Error handling
- State management

✅ **PluginEditor**
- UI initialization
- Resize handling
- Paint operations

✅ **OSC Communication**
- Message format validation
- Response parsing
- Error handling
- Timing calculations

### Missing Coverage

- [ ] Real OSC server integration tests
- [ ] Multi-threaded MIDI buffer access
- [ ] Plugin state persistence
- [ ] UI interaction tests
- [ ] Performance benchmarks

## Debugging Failed Tests

### Common Issues

1. **JUCE modules not found**
   - Verify JUCE path is correct
   - Check all required modules are linked

2. **OSC connection failures**
   - Tests don't require actual OSC connection
   - Verify OSC modules are available

3. **JSON parsing errors**
   - Check `juce::JSON` is working
   - Verify JSON format matches expected structure

4. **Memory issues**
   - Use AddressSanitizer: `-fsanitize=address`
   - Use Valgrind on Linux

### Debug Build

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
gdb ./DAiWBridgeTests
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: C++ Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [macos-latest, ubuntu-latest]
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Install JUCE
        run: |
          git clone https://github.com/juce-framework/JUCE.git
          cd JUCE/extras/Projucer/Builds/MacOSX
          xcodebuild -project Projucer.xcodeproj
      
      - name: Build tests
        run: |
          cd cpp/DAiWBridge
          mkdir build && cd build
          cmake ..
          make
      
      - name: Run tests
        run: ./DAiWBridgeTests
```

## Integration with Python Tests

The C++ tests focus on plugin functionality. For end-to-end testing:

1. **Python Server Tests** - `tests/test_osc_server.py`
2. **Python Client Tests** - `examples/test_osc_client.py`
3. **Integration Tests** - Manual testing with both running

## Best Practices

1. **Isolation** - Each test should be independent
2. **Cleanup** - Always release resources in teardown
3. **Boundary Testing** - Test edge cases (0, 127, -1, etc.)
4. **Error Cases** - Test both success and failure paths
5. **Performance** - Include performance tests for critical paths
6. **Documentation** - Comment complex test logic

## See Also

- `TEST_README.md` - Quick reference
- `docs/JUCE_PLUGIN_GUIDE.md` - Plugin building guide
- `docs/OSC_SERVER_GUIDE.md` - OSC protocol documentation
- `docs/PHASE_3_REFINEMENTS.md` - Implementation details

