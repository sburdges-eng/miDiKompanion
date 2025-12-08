cmake_minimum_required(VERSION 3.27)
project(Kelly VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_PLUGINS "Build VST3 and CLAP plugins" ON)
option(BUILD_TESTS "Build tests" ON)
option(ENABLE_TRACY "Enable Tracy profiling" OFF)

# Find packages
find_package(Qt6 COMPONENTS Core Widgets REQUIRED)

# JUCE setup
add_subdirectory(external/JUCE EXCLUDE_FROM_ALL)

# Main Kelly library
add_library(KellyCore STATIC
    src/core/emotion_engine.cpp
    src/core/emotion_thesaurus.cpp
    src/core/groove_templates.cpp
    src/core/chord_diagnostics.cpp
    src/core/midi_pipeline.cpp
    src/core/intent_processor.cpp
)

target_include_directories(KellyCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(KellyCore PUBLIC
    Qt6::Core
    Qt6::Widgets
    juce::juce_audio_basics
    juce::juce_audio_devices
    juce::juce_audio_formats
    juce::juce_audio_processors
    juce::juce_core
    juce::juce_data_structures
    juce::juce_events
    juce::juce_graphics
    juce::juce_gui_basics
)

target_compile_options(KellyCore PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

# Kelly GUI Application
add_executable(KellyApp
    src/gui/main.cpp
    src/gui/main_window.cpp
)

target_link_libraries(KellyApp PRIVATE
    KellyCore
)

# Plugin builds
if(BUILD_PLUGINS)
    # VST3 Plugin
    juce_add_plugin(KellyPlugin
        COMPANY_NAME "Kelly"
        PLUGIN_MANUFACTURER_CODE Klly
        PLUGIN_CODE Klp1
        FORMATS VST3 CLAP
        PRODUCT_NAME "Kelly Emotion Processor"
        VST3_CATEGORIES Fx Synth
        CLAP_ID com.kelly.emotion-processor
    )

    target_sources(KellyPlugin PRIVATE
        src/plugin/plugin_processor.cpp
        src/plugin/plugin_editor.cpp
    )

    target_link_libraries(KellyPlugin PRIVATE
        KellyCore
        juce::juce_audio_plugin_client
    )
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    
    # Catch2 for C++ tests
    add_subdirectory(external/Catch2 EXCLUDE_FROM_ALL)
    
    add_executable(KellyTests
        tests/cpp/test_emotion_engine.cpp
        tests/cpp/test_midi_pipeline.cpp
        tests/cpp/test_chord_diagnostics.cpp
    )
    
    target_link_libraries(KellyTests PRIVATE
        KellyCore
        Catch2::Catch2WithMain
    )
    
    include(CTest)
    include(Catch2)
    catch_discover_tests(KellyTests)
endif()

# Tracy profiling
if(ENABLE_TRACY)
    add_subdirectory(external/tracy EXCLUDE_FROM_ALL)
    target_link_libraries(KellyCore PUBLIC TracyClient)
    target_compile_definitions(KellyCore PUBLIC TRACY_ENABLE)
endif()

# Installation
install(TARGETS KellyApp KellyCore
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY src/
    DESTINATION include/kelly
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
