# Penta-Core C++ Library
# Real-time audio analysis engines

cmake_minimum_required(VERSION 3.22)
project(PentaCore VERSION 1.0.0)

# Set C++ standard (C++20 for modern features)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PENTA_CORE_SOURCES
    # Common utilities
    common/RTMemoryPool.cpp
    common/RTLogger.cpp

    # Harmony engine
    harmony/ChordAnalyzer.cpp
    harmony/ChordAnalyzerSIMD.cpp
    harmony/ScaleDetector.cpp
    harmony/VoiceLeading.cpp
    harmony/HarmonyEngine.cpp

    # Groove analysis
    groove/OnsetDetector.cpp
    groove/TempoEstimator.cpp
    groove/RhythmQuantizer.cpp
    groove/GrooveEngine.cpp

    # Diagnostics
    diagnostics/PerformanceMonitor.cpp
    diagnostics/AudioAnalyzer.cpp
    diagnostics/DiagnosticsEngine.cpp

    # OSC communication
    osc/OSCServer.cpp
    osc/OSCClient.cpp
    osc/OSCMessage.cpp
    osc/OSCHub.cpp
    osc/RTMessageQueue.cpp
)

set(PENTA_CORE_HEADERS
    ${PROJECT_SOURCE_DIR}/include/penta/common/RTMemoryPool.h
    ${PROJECT_SOURCE_DIR}/include/penta/common/RTLogger.h
    ${PROJECT_SOURCE_DIR}/include/penta/common/RTTypes.h

    ${PROJECT_SOURCE_DIR}/include/penta/harmony/ChordAnalyzer.h
    ${PROJECT_SOURCE_DIR}/include/penta/harmony/ScaleDetector.h
    ${PROJECT_SOURCE_DIR}/include/penta/harmony/VoiceLeading.h
    ${PROJECT_SOURCE_DIR}/include/penta/harmony/HarmonyEngine.h

    ${PROJECT_SOURCE_DIR}/include/penta/groove/OnsetDetector.h
    ${PROJECT_SOURCE_DIR}/include/penta/groove/TempoEstimator.h
    ${PROJECT_SOURCE_DIR}/include/penta/groove/RhythmQuantizer.h
    ${PROJECT_SOURCE_DIR}/include/penta/groove/GrooveEngine.h

    ${PROJECT_SOURCE_DIR}/include/penta/diagnostics/PerformanceMonitor.h
    ${PROJECT_SOURCE_DIR}/include/penta/diagnostics/AudioAnalyzer.h
    ${PROJECT_SOURCE_DIR}/include/penta/diagnostics/DiagnosticsEngine.h

    ${PROJECT_SOURCE_DIR}/include/penta/osc/OSCServer.h
    ${PROJECT_SOURCE_DIR}/include/penta/osc/OSCClient.h
    ${PROJECT_SOURCE_DIR}/include/penta/osc/OSCMessage.h
    ${PROJECT_SOURCE_DIR}/include/penta/osc/RTMessageQueue.h
    ${PROJECT_SOURCE_DIR}/include/penta/osc/OSCHub.h
)

add_library(penta_core STATIC
    ${PENTA_CORE_SOURCES}
    ${PENTA_CORE_HEADERS}
)

set_target_properties(penta_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(penta_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# External dependencies (oscpack, readerwriterqueue)
# Use stubs from external/ directory or fetch real libraries
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../external/CMakeLists.txt")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../external ${CMAKE_BINARY_DIR}/external EXCLUDE_FROM_ALL)
    message(STATUS "Using external dependencies from external/ directory")
else()
    # Fetch real libraries if stubs not available
    include(FetchContent)

    # readerwriterqueue (header-only)
    FetchContent_Declare(
        readerwriterqueue
        GIT_REPOSITORY https://github.com/cameron314/readerwriterqueue.git
        GIT_TAG v1.0.6
    )
    FetchContent_MakeAvailable(readerwriterqueue)

    # oscpack (requires building)
    # Note: oscpack doesn't have a standard CMake setup, may need manual integration
    message(WARNING "oscpack needs manual integration - using stub if available")
endif()

target_link_libraries(penta_core PUBLIC
    oscpack
    readerwriterqueue
)

# Enable AVX2 SIMD optimizations if available
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    target_compile_options(penta_core PRIVATE -mavx2 -mpopcnt)
    message(STATUS "AVX2 SIMD optimizations enabled")
else()
    message(STATUS "AVX2 not available, using scalar fallback")
endif()

# Platform-specific threading
if(UNIX)
    target_link_libraries(penta_core PUBLIC pthread)
endif()
