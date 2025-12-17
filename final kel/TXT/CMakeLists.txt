cmake_minimum_required(VERSION 3.22)
project(KellyMidiCompanion VERSION 2.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# JUCE - Fetch from GitHub
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.4
)
FetchContent_MakeAvailable(JUCE)

# Plugin target
juce_add_plugin(KellyMidiCompanion
    COMPANY_NAME "Kelly Project"
    PLUGIN_MANUFACTURER_CODE Klly
    PLUGIN_CODE Kmcp
    
    # Plugin formats
    FORMATS AU VST3 Standalone
    
    # This is a MIDI effect, not an audio processor
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT TRUE
    IS_MIDI_EFFECT TRUE
    
    PRODUCT_NAME "Kelly MIDI Companion"
    PLUGIN_DESCRIPTION "Therapeutic MIDI generation through emotion mapping. Transforms emotional states (valence, arousal, intensity) into musical MIDI patterns using a 216-node emotion thesaurus and intentional rule-breaking system."
    
    # Bundle IDs
    BUNDLE_ID "com.kellyproject.midicompanion"
    AU_MAIN_TYPE "kAudioUnitType_MIDIProcessor"
)

# Source files
target_sources(KellyMidiCompanion
    PRIVATE
        # Plugin core
        src/plugin/PluginProcessor.cpp
        src/plugin/PluginEditor.cpp
        src/plugin/PluginState.cpp
        
        # Emotion engine
        src/engine/EmotionThesaurus.cpp
        src/engine/EmotionThesaurusLoader.cpp
        src/engine/WoundProcessor.cpp
        src/engine/RuleBreakEngine.cpp
        src/engine/IntentPipeline.cpp
        src/engine/VADCalculator.cpp
        src/engine/QuantumEmotionalField.cpp
        src/engine/EmotionToMusicMapper.cpp
        
        # MIDI generation
        src/midi/ChordGenerator.cpp
        src/midi/GrooveEngine.cpp
        src/midi/MidiBuilder.cpp
        src/midi/MidiGenerator.cpp
        src/midi/InstrumentSelector.cpp

        # Common utilities
        src/common/EQPresetManager.cpp
        src/common/PathResolver.cpp
        
        # UI (VERSION 3.0.00 components)
        src/ui/EmotionWorkstation.cpp
        src/ui/GenerateButton.cpp
        src/ui/KellyLookAndFeel.cpp
        src/ui/EmotionWheel.cpp
        src/ui/AIGenerationDialog.cpp
        src/ui/ChordDisplay.cpp
        src/ui/EmotionRadar.cpp
        src/ui/MusicTheoryPanel.cpp
        src/ui/PianoRollPreview.cpp
        src/ui/TooltipComponent.cpp
        src/ui/WorkstationPanel.cpp

        # Algorithm engines (VERSION 3.0.00)
        src/engines/ArrangementEngine.cpp
        src/engines/BassEngine.cpp
        src/engines/CounterMelodyEngine.cpp
        src/engines/DynamicsEngine.cpp
        src/engines/FillEngine.cpp
        src/engines/MelodyEngine.cpp
        src/engines/PadEngine.cpp
        src/engines/RhythmEngine.cpp
        src/engines/StringEngine.cpp
        src/engines/TensionEngine.cpp
        src/engines/TransitionEngine.cpp
        src/engines/VariationEngine.cpp
        src/engines/VoiceLeading.cpp
        src/engines/GrooveEngine.cpp
        # Note: src/engines/GrooveEngine.cpp implements GroovePatternEngine (drum patterns)
        # src/midi/GrooveEngine.cpp implements GrooveEngine (MIDI note processing)
        # Both are needed - different namespaces prevent conflict

        # Voice synthesis (v2.0)
        src/voice/VoiceSynthesizer.cpp
        src/voice/VocoderEngine.cpp

        # Biometric input (v2.0)
        src/biometric/BiometricInput.cpp
)

# Include directories
target_include_directories(KellyMidiCompanion
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# JUCE modules
target_compile_definitions(KellyMidiCompanion
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_DISPLAY_SPLASH_SCREEN=0
)

target_link_libraries(KellyMidiCompanion
    PRIVATE
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_plugin_client
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_core
        juce::juce_data_structures
        juce::juce_events
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_gui_extra
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# Note: Code signing is handled by build_and_install.sh script
# We don't sign during build to avoid Gatekeeper issues
# The install script properly removes quarantine and signs after installation

# =============================================================================
# Python-C++ Bridge (Optional - requires pybind11)
# =============================================================================

option(BUILD_PYTHON_BRIDGE "Build Python-C++ bridge module" OFF)

if(BUILD_PYTHON_BRIDGE)
    # Find Python
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    
    # Fetch pybind11
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
    
    # Python module target
    pybind11_add_module(kelly_bridge
        src/bridge/kelly_bridge.cpp
        
        # Emotion engine (shared with plugin)
        src/engine/EmotionThesaurus.cpp
        src/engine/EmotionThesaurusLoader.cpp
        src/engine/WoundProcessor.cpp
        src/engine/RuleBreakEngine.cpp
        src/engine/IntentPipeline.cpp
        
        # MIDI generation (shared with plugin)
        src/midi/ChordGenerator.cpp
        src/midi/GrooveEngine.cpp
        src/midi/MidiBuilder.cpp
        src/midi/MidiGenerator.cpp
        src/midi/InstrumentSelector.cpp
        
        # Common utilities
        src/common/EQPresetManager.cpp
    )
    
    # Include directories
    target_include_directories(kelly_bridge
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_INCLUDE_DIRS}
    )
    
    # Link Python
    target_link_libraries(kelly_bridge
        PRIVATE
            ${Python3_LIBRARIES}
    )
    
    # Compile definitions
    target_compile_definitions(kelly_bridge
        PRIVATE
            VERSION_INFO="${PROJECT_VERSION}"
    )
    
    # Set output directory to python/ directory
    set_target_properties(kelly_bridge PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
        OUTPUT_NAME kelly_bridge
    )
    
    message(STATUS "Python-C++ bridge enabled")
    message(STATUS "  Python executable: ${Python3_EXECUTABLE}")
    message(STATUS "  Python version: ${Python3_VERSION}")
    message(STATUS "  Module will be built to: ${CMAKE_CURRENT_SOURCE_DIR}/python")
endif()

# =============================================================================
# Unit Tests (Optional - requires Google Test)
# =============================================================================

option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    add_subdirectory(tests)
    message(STATUS "Unit tests enabled")
    message(STATUS "  Run tests with: ctest or ./tests/KellyTests")
endif()
