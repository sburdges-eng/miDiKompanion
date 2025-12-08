/**
 * @file AudioEngine.cpp
 * @brief Real-time audio I/O engine implementation
 *
 * This implementation provides cross-platform audio I/O using RtAudio.
 * Currently implemented as a functional stub that maintains proper state
 * tracking and provides the full interface structure.
 *
 * For production use, link against RtAudio library:
 * - macOS: CoreAudio framework
 * - Windows: WASAPI/DirectSound/ASIO
 * - Linux: ALSA/PulseAudio/JACK
 *
 * Build with: -DUSE_RTAUDIO=ON to enable full RtAudio backend
 */

#include "penta/audio/AudioEngine.h"
#include <chrono>
#include <cstring>
#include <thread>

namespace penta::audio {

// =============================================================================
// Implementation Details
// =============================================================================

struct AudioEngine::Impl {
    // Configuration
    AudioApi currentApi = AudioApi::Unspecified;
    StreamConfig streamConfig;
    AudioCallback callback;

    // State
    std::atomic<StreamState> state{StreamState::Closed};
    std::atomic<double> streamTime{0.0};
    std::atomic<float> cpuLoad{0.0f};
    std::atomic<uint64_t> underrunCount{0};
    std::atomic<uint64_t> overrunCount{0};
    std::atomic<bool> hasError{false};
    std::string lastError;

    // Stream properties (set after open)
    uint32_t actualSampleRate = 0;
    uint32_t actualBufferSize = 0;
    uint32_t streamLatency = 0;

    // Background thread for stub implementation
    std::thread audioThread;
    std::atomic<bool> threadRunning{false};

    explicit Impl(AudioApi api) : currentApi(api) {
        // Auto-detect best API for this platform
        if (currentApi == AudioApi::Unspecified) {
#if defined(__APPLE__)
            currentApi = AudioApi::CoreAudio;
#elif defined(_WIN32)
            currentApi = AudioApi::WASAPI;
#elif defined(__linux__)
            currentApi = AudioApi::PulseAudio;
#else
            currentApi = AudioApi::Dummy;
#endif
        }
    }

    ~Impl() {
        stopAudioThread();
    }

    void startAudioThread() {
        if (threadRunning.load()) {
            return;
        }

        threadRunning.store(true);
        audioThread = std::thread([this]() {
            runAudioLoop();
        });
    }

    void stopAudioThread() {
        threadRunning.store(false);
        if (audioThread.joinable()) {
            audioThread.join();
        }
    }

    void runAudioLoop() {
        // Stub audio loop - simulates audio callback timing
        // In production, RtAudio handles the actual audio thread

        const uint32_t frames = streamConfig.bufferFrames;
        const uint32_t outputChannels = streamConfig.outputChannels;
        const uint32_t inputChannels = streamConfig.inputChannels;
        const double sampleRate = static_cast<double>(actualSampleRate);

        // Allocate buffers (done once, before RT loop)
        std::vector<float> outputBuffer(frames * outputChannels, 0.0f);
        std::vector<float> inputBuffer(frames * inputChannels, 0.0f);

        // Calculate sleep duration per buffer
        const double bufferDurationMs = (static_cast<double>(frames) / sampleRate) * 1000.0;
        const auto sleepDuration = std::chrono::microseconds(
            static_cast<int64_t>(bufferDurationMs * 1000.0 * 0.9)  // 90% of buffer time
        );

        double currentTime = 0.0;
        const double timeIncrement = static_cast<double>(frames) / sampleRate;

        while (threadRunning.load() && state.load() == StreamState::Running) {
            auto startTime = std::chrono::high_resolution_clock::now();

            // Prepare status
            StreamStatus status;
            status.streamTime = currentTime;
            status.inputOverflow = false;
            status.outputUnderflow = false;

            // Call user callback if set
            if (callback) {
                float* outPtr = outputChannels > 0 ? outputBuffer.data() : nullptr;
                const float* inPtr = inputChannels > 0 ? inputBuffer.data() : nullptr;

                int result = callback(outPtr, inPtr, frames, status);

                if (result != 0) {
                    // User requested stop
                    state.store(StreamState::Stopped);
                    break;
                }
            }

            // Update stream time
            currentTime += timeIncrement;
            streamTime.store(currentTime);

            // Calculate CPU load
            auto endTime = std::chrono::high_resolution_clock::now();
            auto processingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
            float load = static_cast<float>(processingTime / bufferDurationMs);
            cpuLoad.store(std::min(1.0f, load));

            // Sleep to simulate audio timing
            std::this_thread::sleep_for(sleepDuration);
        }
    }

    void setError(const std::string& error) {
        lastError = error;
        hasError.store(true);
    }

    void clearErrorState() {
        lastError.clear();
        hasError.store(false);
    }
};

// =============================================================================
// AudioEngine Public Interface
// =============================================================================

AudioEngine::AudioEngine(AudioApi api)
    : impl_(std::make_unique<Impl>(api))
{
}

AudioEngine::~AudioEngine() {
    closeStream();
}

AudioEngine::AudioEngine(AudioEngine&&) noexcept = default;
AudioEngine& AudioEngine::operator=(AudioEngine&&) noexcept = default;

// =============================================================================
// Device Enumeration
// =============================================================================

std::vector<AudioApi> AudioEngine::getAvailableApis() {
    std::vector<AudioApi> apis;

#if defined(__APPLE__)
    apis.push_back(AudioApi::CoreAudio);
    apis.push_back(AudioApi::JACK);
#elif defined(_WIN32)
    apis.push_back(AudioApi::WASAPI);
    apis.push_back(AudioApi::DirectSound);
    apis.push_back(AudioApi::ASIO);
#elif defined(__linux__)
    apis.push_back(AudioApi::ALSA);
    apis.push_back(AudioApi::PulseAudio);
    apis.push_back(AudioApi::JACK);
#endif

    apis.push_back(AudioApi::Dummy);

    return apis;
}

std::string AudioEngine::getApiName(AudioApi api) {
    switch (api) {
        case AudioApi::Unspecified: return "Unspecified";
        case AudioApi::CoreAudio:   return "CoreAudio";
        case AudioApi::WASAPI:      return "WASAPI";
        case AudioApi::DirectSound: return "DirectSound";
        case AudioApi::ASIO:        return "ASIO";
        case AudioApi::ALSA:        return "ALSA";
        case AudioApi::PulseAudio:  return "PulseAudio";
        case AudioApi::JACK:        return "JACK";
        case AudioApi::Dummy:       return "Dummy";
    }
    return "Unknown";
}

AudioApi AudioEngine::getCurrentApi() const noexcept {
    return impl_->currentApi;
}

std::vector<AudioDeviceInfo> AudioEngine::getAvailableDevices() const {
    // Stub implementation: Return simulated devices
    // Full implementation would query RtAudio for actual devices
    std::vector<AudioDeviceInfo> devices;

    // Default output device (simulated)
    AudioDeviceInfo defaultOutput;
    defaultOutput.deviceId = 0;
    defaultOutput.name = "Default Audio Output";
    defaultOutput.outputChannels = 2;
    defaultOutput.inputChannels = 0;
    defaultOutput.isDefaultOutput = true;
    defaultOutput.sampleRates = {44100, 48000, 88200, 96000};
    defaultOutput.preferredSampleRate = 48000;
    devices.push_back(defaultOutput);

    // Default input device (simulated)
    AudioDeviceInfo defaultInput;
    defaultInput.deviceId = 1;
    defaultInput.name = "Default Audio Input";
    defaultInput.outputChannels = 0;
    defaultInput.inputChannels = 2;
    defaultInput.isDefaultInput = true;
    defaultInput.sampleRates = {44100, 48000, 88200, 96000};
    defaultInput.preferredSampleRate = 48000;
    devices.push_back(defaultInput);

    return devices;
}

int AudioEngine::getDefaultInputDevice() const {
    // Stub: Return simulated default input
    return 1;
}

int AudioEngine::getDefaultOutputDevice() const {
    // Stub: Return simulated default output
    return 0;
}

AudioDeviceInfo AudioEngine::getDeviceInfo(int deviceId) const {
    auto devices = getAvailableDevices();
    for (const auto& device : devices) {
        if (device.deviceId == deviceId) {
            return device;
        }
    }
    return AudioDeviceInfo{};
}

// =============================================================================
// Stream Management
// =============================================================================

void AudioEngine::setCallback(AudioCallback callback) {
    impl_->callback = std::move(callback);
}

bool AudioEngine::openStream(const StreamConfig& config) {
    if (impl_->state.load() != StreamState::Closed) {
        impl_->setError("Stream already open");
        return false;
    }

    // Validate configuration
    if (config.outputDeviceId < 0 && config.inputDeviceId < 0) {
        impl_->setError("No input or output device specified");
        return false;
    }

    if (config.sampleRate == 0) {
        impl_->setError("Invalid sample rate");
        return false;
    }

    if (config.bufferFrames == 0) {
        impl_->setError("Invalid buffer size");
        return false;
    }

    // Store configuration
    impl_->streamConfig = config;
    impl_->actualSampleRate = config.sampleRate;
    impl_->actualBufferSize = config.bufferFrames;
    impl_->streamLatency = config.bufferFrames;  // Single buffer latency in stub

    // In production, would call RtAudio::openStream() here
    // For now, just update state

    impl_->state.store(StreamState::Stopped);
    impl_->clearErrorState();

    return true;
}

void AudioEngine::closeStream() {
    if (impl_->state.load() == StreamState::Closed) {
        return;
    }

    // Stop if running
    if (impl_->state.load() == StreamState::Running) {
        stopStream();
    }

    // In production, would call RtAudio::closeStream() here

    impl_->state.store(StreamState::Closed);
    impl_->streamTime.store(0.0);
    impl_->clearErrorState();
}

bool AudioEngine::startStream() {
    StreamState expected = StreamState::Stopped;
    if (!impl_->state.compare_exchange_strong(expected, StreamState::Running)) {
        if (expected == StreamState::Closed) {
            impl_->setError("Stream not open");
        } else {
            impl_->setError("Stream already running");
        }
        return false;
    }

    // Reset statistics
    resetStatistics();

    // In production, would call RtAudio::startStream() here
    // For stub, start our simulation thread
    impl_->startAudioThread();

    impl_->clearErrorState();
    return true;
}

bool AudioEngine::stopStream() {
    StreamState expected = StreamState::Running;
    if (!impl_->state.compare_exchange_strong(expected, StreamState::Stopped)) {
        if (expected == StreamState::Closed) {
            impl_->setError("Stream not open");
        } else {
            impl_->setError("Stream not running");
        }
        return false;
    }

    // In production, would call RtAudio::stopStream() here
    impl_->stopAudioThread();

    impl_->clearErrorState();
    return true;
}

bool AudioEngine::abortStream() {
    if (impl_->state.load() == StreamState::Closed) {
        impl_->setError("Stream not open");
        return false;
    }

    // Force stop regardless of current state
    impl_->stopAudioThread();
    impl_->state.store(StreamState::Stopped);

    impl_->clearErrorState();
    return true;
}

StreamState AudioEngine::getStreamState() const noexcept {
    return impl_->state.load();
}

bool AudioEngine::isStreamOpen() const noexcept {
    return impl_->state.load() != StreamState::Closed;
}

bool AudioEngine::isStreamRunning() const noexcept {
    return impl_->state.load() == StreamState::Running;
}

const StreamConfig& AudioEngine::getStreamConfig() const noexcept {
    return impl_->streamConfig;
}

double AudioEngine::getStreamTime() const noexcept {
    return impl_->streamTime.load();
}

uint32_t AudioEngine::getStreamLatency() const {
    return impl_->streamLatency;
}

uint32_t AudioEngine::getActualSampleRate() const {
    return impl_->actualSampleRate;
}

uint32_t AudioEngine::getActualBufferSize() const {
    return impl_->actualBufferSize;
}

// =============================================================================
// Error Handling
// =============================================================================

std::string AudioEngine::getLastError() const {
    return impl_->lastError;
}

bool AudioEngine::hasError() const noexcept {
    return impl_->hasError.load();
}

void AudioEngine::clearError() noexcept {
    impl_->clearErrorState();
}

// =============================================================================
// Statistics
// =============================================================================

float AudioEngine::getCpuLoad() const noexcept {
    return impl_->cpuLoad.load();
}

uint64_t AudioEngine::getUnderrunCount() const noexcept {
    return impl_->underrunCount.load();
}

uint64_t AudioEngine::getOverrunCount() const noexcept {
    return impl_->overrunCount.load();
}

void AudioEngine::resetStatistics() noexcept {
    impl_->cpuLoad.store(0.0f);
    impl_->underrunCount.store(0);
    impl_->overrunCount.store(0);
}

} // namespace penta::audio
