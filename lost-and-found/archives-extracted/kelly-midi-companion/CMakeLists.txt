cmake_minimum_required(VERSION 3.22)
project(KellyMidiCompanion VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.4
)
FetchContent_MakeAvailable(JUCE)

juce_add_plugin(KellyMidiCompanion
    COMPANY_NAME "Kelly Project"
    PLUGIN_MANUFACTURER_CODE Klly
    PLUGIN_CODE Kmcp
    FORMATS Standalone
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT TRUE
    IS_MIDI_EFFECT TRUE
    PRODUCT_NAME "Kelly MIDI Companion"
)

target_sources(KellyMidiCompanion
    PRIVATE
        src/plugin/PluginProcessor.cpp
        src/plugin/PluginEditor.cpp
        src/plugin/PluginState.cpp
        src/engine/EmotionThesaurus.cpp
        src/engine/WoundProcessor.cpp
        src/engine/RuleBreakEngine.cpp
        src/engine/IntentPipeline.cpp
        src/midi/ChordGenerator.cpp
        src/midi/GrooveEngine.cpp
        src/midi/MidiBuilder.cpp
        src/ui/CassetteView.cpp
        src/ui/SidePanel.cpp
        src/ui/GenerateButton.cpp
        src/ui/KellyLookAndFeel.cpp
)

target_include_directories(KellyMidiCompanion
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

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
