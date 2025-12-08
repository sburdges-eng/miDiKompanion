/**
 * @file AudioEngine.h
 * @brief Real-time audio I/O engine with cross-platform support
 *
 * Uses RtAudio as the backend for cross-platform audio I/O:
 * - CoreAudio on macOS
 * - WASAPI/DirectSound on Windows
 * - ALSA/PulseAudio/JACK on Linux
 *
 * All audio callback methods are RT-safe (noexcept, no allocations).
 */

#pragma once

#include "penta/common/RTTypes.h"
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace penta::audio {

/**
 * @brief Audio API backend selection
 */
enum class AudioApi {
    Unspecified,    // Let RtAudio choose the best available
    CoreAudio,      // macOS
    WASAPI,         // Windows Vista+
    DirectSound,    // Windows (legacy)
    ASIO,           // Windows (low-latency)
    ALSA,           // Linux
    PulseAudio,     // Linux
    JACK,           // Linux/macOS (pro audio)
    Dummy           // No actual I/O (for testing)
};

/**
 * @brief Sample format for audio I/O
 */
enum class SampleFormat {
    Float32,        // 32-bit floating point (preferred)
    Float64,        // 64-bit floating point
    Int16,          // 16-bit signed integer
    Int24,          // 24-bit signed integer (packed)
    Int32           // 32-bit signed integer
};

/**
 * @brief Information about an audio device
 */
struct AudioDeviceInfo {
    int deviceId = -1;
    std::string name;
    uint32_t inputChannels = 0;
    uint32_t outputChannels = 0;
    uint32_t duplexChannels = 0;
    bool isDefaultInput = false;
    bool isDefaultOutput = false;
    std::vector<uint32_t> sampleRates;
    uint32_t preferredSampleRate = 0;

    [[nodiscard]] bool hasInput() const noexcept { return inputChannels > 0; }
    [[nodiscard]] bool hasOutput() const noexcept { return outputChannels > 0; }
    [[nodiscard]] bool isDuplex() const noexcept { return duplexChannels > 0; }
};

/**
 * @brief Stream configuration for audio I/O
 */
struct StreamConfig {
    int inputDeviceId = -1;         // -1 = no input
    int outputDeviceId = -1;        // -1 = no output
    uint32_t inputChannels = 0;
    uint32_t outputChannels = 2;
    uint32_t sampleRate = 48000;
    uint32_t bufferFrames = 512;    // Frames per buffer (latency control)
    SampleFormat format = SampleFormat::Float32;
    AudioApi api = AudioApi::Unspecified;

    // Optional: first channel offset for multi-channel interfaces
    uint32_t inputFirstChannel = 0;
    uint32_t outputFirstChannel = 0;

    [[nodiscard]] double getLatencyMs() const noexcept {
        return (static_cast<double>(bufferFrames) / sampleRate) * 1000.0;
    }
};

/**
 * @brief Stream status for audio callbacks
 */
struct StreamStatus {
    bool inputOverflow = false;     // Input buffer overflowed
    bool outputUnderflow = false;   // Output buffer underflowed
    double streamTime = 0.0;        // Current stream time in seconds
};

/**
 * @brief Audio callback function type
 *
 * @param outputBuffer  Buffer to write output samples (interleaved)
 * @param inputBuffer   Buffer with input samples (interleaved), may be nullptr
 * @param numFrames     Number of frames to process
 * @param status        Stream status flags
 *
 * @return 0 to continue, non-zero to stop the stream
 *
 * IMPORTANT: This callback runs on the audio thread.
 * Must be RT-safe: no allocations, no locks, no blocking operations.
 */
using AudioCallback = std::function<int(
    float* outputBuffer,
    const float* inputBuffer,
    uint32_t numFrames,
    const StreamStatus& status
)>;

/**
 * @brief Audio stream state
 */
enum class StreamState {
    Closed,
    Stopped,
    Running
};

/**
 * @brief Real-time audio I/O engine
 *
 * Provides cross-platform audio input/output using RtAudio as the backend.
 * Designed for low-latency, real-time audio processing.
 *
 * Usage:
 * @code
 *   AudioEngine engine;
 *
 *   // Enumerate devices
 *   auto devices = engine.getAvailableDevices();
 *
 *   // Configure stream
 *   StreamConfig config;
 *   config.outputDeviceId = engine.getDefaultOutputDevice();
 *   config.sampleRate = 48000;
 *   config.bufferFrames = 256;
 *
 *   // Set callback
 *   engine.setCallback([](float* out, const float* in, uint32_t frames, auto& status) {
 *       // Process audio here (RT-safe!)
 *       return 0;
 *   });
 *
 *   // Open and start
 *   engine.openStream(config);
 *   engine.startStream();
 * @endcode
 */
class AudioEngine {
public:
    /**
     * @brief Construct audio engine
     * @param api Preferred audio API (Unspecified = auto-detect)
     */
    explicit AudioEngine(AudioApi api = AudioApi::Unspecified);

    /**
     * @brief Destructor - closes stream if open
     */
    ~AudioEngine();

    // Non-copyable
    AudioEngine(const AudioEngine&) = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;

    // Movable
    AudioEngine(AudioEngine&&) noexcept;
    AudioEngine& operator=(AudioEngine&&) noexcept;

    // =========================================================================
    // Device Enumeration
    // =========================================================================

    /**
     * @brief Get list of available audio APIs on this platform
     */
    static std::vector<AudioApi> getAvailableApis();

    /**
     * @brief Get human-readable name for an API
     */
    static std::string getApiName(AudioApi api);

    /**
     * @brief Get current API in use
     */
    [[nodiscard]] AudioApi getCurrentApi() const noexcept;

    /**
     * @brief Enumerate available audio devices
     */
    [[nodiscard]] std::vector<AudioDeviceInfo> getAvailableDevices() const;

    /**
     * @brief Get default input device ID
     * @return Device ID, or -1 if no input device available
     */
    [[nodiscard]] int getDefaultInputDevice() const;

    /**
     * @brief Get default output device ID
     * @return Device ID, or -1 if no output device available
     */
    [[nodiscard]] int getDefaultOutputDevice() const;

    /**
     * @brief Get device info by ID
     * @param deviceId Device ID
     * @return Device info, or empty struct if not found
     */
    [[nodiscard]] AudioDeviceInfo getDeviceInfo(int deviceId) const;

    // =========================================================================
    // Stream Management
    // =========================================================================

    /**
     * @brief Set the audio processing callback
     *
     * Must be called before openStream().
     * The callback must be RT-safe (no allocations, no blocking).
     */
    void setCallback(AudioCallback callback);

    /**
     * @brief Open an audio stream
     * @param config Stream configuration
     * @return true if successful
     */
    bool openStream(const StreamConfig& config);

    /**
     * @brief Close the audio stream
     */
    void closeStream();

    /**
     * @brief Start the audio stream
     * @return true if successful
     */
    bool startStream();

    /**
     * @brief Stop the audio stream
     * @return true if successful
     */
    bool stopStream();

    /**
     * @brief Abort the stream immediately (may cause audio glitches)
     * @return true if successful
     */
    bool abortStream();

    /**
     * @brief Get current stream state
     */
    [[nodiscard]] StreamState getStreamState() const noexcept;

    /**
     * @brief Check if stream is open
     */
    [[nodiscard]] bool isStreamOpen() const noexcept;

    /**
     * @brief Check if stream is running
     */
    [[nodiscard]] bool isStreamRunning() const noexcept;

    /**
     * @brief Get current stream configuration
     */
    [[nodiscard]] const StreamConfig& getStreamConfig() const noexcept;

    /**
     * @brief Get current stream time in seconds
     *
     * RT-safe: can be called from audio callback.
     */
    [[nodiscard]] double getStreamTime() const noexcept;

    /**
     * @brief Get measured stream latency in samples
     */
    [[nodiscard]] uint32_t getStreamLatency() const;

    /**
     * @brief Get actual sample rate (may differ from requested)
     */
    [[nodiscard]] uint32_t getActualSampleRate() const;

    /**
     * @brief Get actual buffer size (may differ from requested)
     */
    [[nodiscard]] uint32_t getActualBufferSize() const;

    // =========================================================================
    // Error Handling
    // =========================================================================

    /**
     * @brief Get last error message
     */
    [[nodiscard]] std::string getLastError() const;

    /**
     * @brief Check if last operation had an error
     */
    [[nodiscard]] bool hasError() const noexcept;

    /**
     * @brief Clear error state
     */
    void clearError() noexcept;

    // =========================================================================
    // Statistics (RT-safe reads)
    // =========================================================================

    /**
     * @brief Get CPU load estimate (0.0 - 1.0)
     *
     * RT-safe: uses atomic reads.
     */
    [[nodiscard]] float getCpuLoad() const noexcept;

    /**
     * @brief Get number of buffer underruns since stream start
     */
    [[nodiscard]] uint64_t getUnderrunCount() const noexcept;

    /**
     * @brief Get number of buffer overruns since stream start
     */
    [[nodiscard]] uint64_t getOverrunCount() const noexcept;

    /**
     * @brief Reset statistics counters
     */
    void resetStatistics() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace penta::audio
