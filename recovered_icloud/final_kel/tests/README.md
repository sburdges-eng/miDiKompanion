# Kelly MIDI Companion - Unit Tests

Comprehensive unit test suite for all engines and components.

## Overview

This test suite provides comprehensive coverage for:
- **13 Algorithm Engines**: Melody, Bass, Rhythm, Arrangement, Dynamics, Pad, Tension, CounterMelody, Fill, String, Transition, Variation, VoiceLeading
- **Core Components**: EmotionThesaurus, IntentPipeline, WoundProcessor, RuleBreakEngine, EmotionMapper
- **MIDI Components**: MidiGenerator, ChordGenerator, GrooveEngine, MidiBuilder, InstrumentSelector
- **Utilities**: EQPresetManager
- **Integration Tests**: Cross-engine functionality

## Building Tests

Tests are built automatically when `BUILD_TESTS` is enabled (default: ON).

```bash
mkdir build
cd build
cmake .. -DBUILD_TESTS=ON
make
```

## Running Tests

### Using CTest (Recommended)
```bash
cd build
ctest
ctest --verbose  # For detailed output
ctest --output-on-failure  # Show output only on failures
```

### Running Directly
```bash
./build/tests/KellyTests
./build/tests/KellyTests --gtest_filter=MelodyEngineTest.*  # Run specific tests
```

### Running Specific Test Suites
```bash
# All engine tests
./build/tests/KellyTests --gtest_filter=*EngineTest*

# Core component tests
./build/tests/KellyTests --gtest_filter=*ThesaurusTest*

# MIDI component tests
./build/tests/KellyTests --gtest_filter=*MidiTest*
```

## Test Structure

```
tests/
├── CMakeLists.txt              # Test build configuration
├── test_main.cpp               # Test entry point
├── engines/                    # Engine tests
│   ├── test_melody_engine.cpp
│   ├── test_bass_engine.cpp
│   ├── test_rhythm_engine.cpp
│   ├── test_arrangement_engine.cpp
│   ├── test_dynamics_engine.cpp
│   ├── test_pad_engine.cpp
│   ├── test_tension_engine.cpp
│   ├── test_counter_melody_engine.cpp
│   ├── test_fill_engine.cpp
│   ├── test_string_engine.cpp
│   ├── test_transition_engine.cpp
│   ├── test_variation_engine.cpp
│   └── test_voice_leading.cpp
├── core/                       # Core component tests
│   ├── test_emotion_thesaurus.cpp
│   ├── test_intent_pipeline.cpp
│   ├── test_wound_processor.cpp
│   ├── test_rule_break_engine.cpp
│   └── test_emotion_mapper.cpp
├── midi/                       # MIDI component tests
│   ├── test_midi_generator.cpp
│   ├── test_chord_generator.cpp
│   ├── test_groove_engine.cpp
│   ├── test_midi_builder.cpp
│   └── test_instrument_selector.cpp
├── utils/                      # Utility tests
│   └── test_eq_preset_manager.cpp
└── integration/                # Integration tests
    └── test_engine_integration.cpp
```

## Test Coverage

### Engine Tests
Each engine test suite covers:
- Basic generation functionality
- Different emotion inputs
- Parameter validation (keys, modes, tempos, bar counts)
- Output validity (pitch ranges, velocity ranges, timing)
- Seed reproducibility
- Configuration overrides
- Section-specific generation (where applicable)

### Core Component Tests
- EmotionThesaurus: Lookup, categorization, nearest neighbor search
- IntentPipeline: Wound processing, emotion mapping, rule break generation
- WoundProcessor: Text analysis, emotion extraction
- RuleBreakEngine: Rule break generation for different emotions
- EmotionMapper: V/A/I to musical parameter mapping

### MIDI Component Tests
- MidiGenerator: Full MIDI generation from intent
- ChordGenerator: Chord progression generation
- GrooveEngine: Humanization and groove templates
- MidiBuilder: MIDI file construction
- InstrumentSelector: Instrument selection by emotion/category

## Writing New Tests

When adding new components:

1. Create test file in appropriate directory (`engines/`, `core/`, `midi/`, etc.)
2. Follow naming convention: `test_<component_name>.cpp`
3. Use Google Test framework:
   ```cpp
   #include <gtest/gtest.h>
   
   class ComponentTest : public ::testing::Test {
   protected:
       void SetUp() override {
           component = std::make_unique<Component>();
       }
       std::unique_ptr<Component> component;
   };
   
   TEST_F(ComponentTest, BasicFunctionality) {
       // Test code
   }
   ```
4. Add test file to `tests/CMakeLists.txt`
5. Add source files to test target sources

## Continuous Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Build and Test
  run: |
    mkdir build && cd build
    cmake .. -DBUILD_TESTS=ON
    make
    ctest --output-on-failure
```

## Test Metrics

- **Total Test Cases**: ~150+
- **Coverage**: All public APIs of engines and components
- **Test Types**: Unit tests, integration tests, parameter validation
- **Execution Time**: ~5-10 seconds (depending on system)

## Troubleshooting

### Tests Fail to Build
- Ensure Google Test is available (fetched automatically via CMake)
- Check that all source files are included in test target
- Verify include paths are correct

### Tests Fail at Runtime
- Check that data files are accessible (emotion JSON files, etc.)
- Verify initialization code runs correctly
- Check for missing dependencies

### Specific Test Failures
- Run with `--gtest_filter` to isolate failing tests
- Use `--gtest_repeat=N` to check for flaky tests
- Enable verbose output with `--gtest_print_time=1`

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain or improve test coverage
4. Update this README if adding new test categories
