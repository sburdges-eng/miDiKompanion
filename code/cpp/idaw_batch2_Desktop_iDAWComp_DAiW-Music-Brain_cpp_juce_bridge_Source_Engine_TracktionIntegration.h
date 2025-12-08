#pragma once

/**
 * Tracktion Engine Integration for DAiW
 *
 * This header provides integration points for Tracktion Engine,
 * enabling full DAW sequencing capabilities within the DAiW ecosystem.
 *
 * Tracktion Engine Features Used:
 * - Edit/Timeline management
 * - Audio clip handling
 * - Plugin hosting
 * - Transport control
 * - Real-time audio engine
 *
 * Requirements:
 * - Clone tracktion_engine with submodules
 * - Add tracktion_engine module to CMakeLists.txt
 *
 * Example CMakeLists.txt addition:
 *   add_subdirectory(${TRACKTION_ENGINE_DIR}/modules)
 *   target_link_libraries(YourTarget PRIVATE tracktion::tracktion_engine)
 */

// NOTE: This is a header-only interface definition.
// Implementation requires Tracktion Engine to be properly linked.

#ifdef TRACKTION_ENGINE_AVAILABLE

#include <tracktion_engine/tracktion_engine.h>

namespace te = tracktion;

/**
 * DAiW Engine wrapper around Tracktion Engine
 */
class DAiWEngine
{
public:
    DAiWEngine();
    ~DAiWEngine();

    //==========================================================================
    // Engine lifecycle

    /** Initialize the engine */
    bool initialize();

    /** Shutdown the engine */
    void shutdown();

    //==========================================================================
    // Edit/Project management

    /** Create a new empty edit */
    te::Edit* createEdit(const juce::String& name);

    /** Load an edit from file */
    te::Edit* loadEdit(const juce::File& file);

    /** Save the current edit */
    bool saveEdit(te::Edit* edit, const juce::File& file);

    /** Get the current edit */
    te::Edit* getCurrentEdit() const { return currentEdit_.get(); }

    //==========================================================================
    // Track management

    /** Add an audio track to the edit */
    te::AudioTrack* addAudioTrack(te::Edit* edit, const juce::String& name);

    /** Add a MIDI track to the edit */
    te::AudioTrack* addMidiTrack(te::Edit* edit, const juce::String& name);

    /** Remove a track */
    void removeTrack(te::Edit* edit, te::Track* track);

    //==========================================================================
    // Clip management

    /** Add an audio clip to a track */
    te::AudioClipBase* addAudioClip(
        te::AudioTrack* track,
        const juce::File& audioFile,
        te::TimePosition startTime,
        te::TimeDuration duration
    );

    /** Add a MIDI clip to a track */
    te::MidiClip* addMidiClip(
        te::AudioTrack* track,
        te::TimePosition startTime,
        te::TimeDuration duration
    );

    /** Add MIDI notes to a clip */
    void addMidiNotes(
        te::MidiClip* clip,
        const std::vector<std::tuple<int, int, double, double>>& notes  // pitch, velocity, start, duration
    );

    //==========================================================================
    // Plugin management

    /** Add a plugin to a track */
    te::Plugin* addPlugin(
        te::AudioTrack* track,
        const juce::String& pluginId
    );

    /** Load a VST3/AU plugin */
    te::Plugin* loadExternalPlugin(
        te::AudioTrack* track,
        const juce::File& pluginFile
    );

    /** Get list of available plugins */
    juce::StringArray getAvailablePlugins() const;

    //==========================================================================
    // Transport control

    /** Start playback */
    void play();

    /** Stop playback */
    void stop();

    /** Pause playback */
    void pause();

    /** Set playback position */
    void setPosition(te::TimePosition position);

    /** Get current playback position */
    te::TimePosition getPosition() const;

    /** Set tempo */
    void setTempo(double bpm);

    /** Get tempo */
    double getTempo() const;

    /** Set loop region */
    void setLoopRegion(te::TimeRange range);

    /** Enable/disable looping */
    void setLooping(bool enabled);

    //==========================================================================
    // Recording

    /** Arm a track for recording */
    void armTrack(te::AudioTrack* track, bool arm);

    /** Start recording */
    void startRecording();

    /** Stop recording */
    void stopRecording();

    //==========================================================================
    // Rendering

    /** Render the edit to a file */
    bool renderToFile(
        te::Edit* edit,
        const juce::File& outputFile,
        te::TimeRange range,
        const te::Renderer::Parameters& params
    );

    /** Render to memory buffer */
    juce::AudioBuffer<float> renderToBuffer(
        te::Edit* edit,
        te::TimeRange range,
        int sampleRate
    );

    //==========================================================================
    // Voice synthesis integration

    /** Create a voice synthesis track */
    te::AudioTrack* createVoiceTrack(te::Edit* edit, const juce::String& name);

    /** Add synthesized voice audio to track */
    void addVoiceAudio(
        te::AudioTrack* track,
        const juce::AudioBuffer<float>& audio,
        int sampleRate,
        te::TimePosition startTime
    );

private:
    std::unique_ptr<te::Engine> engine_;
    std::unique_ptr<te::Edit> currentEdit_;

    // Plugin scanning
    void scanForPlugins();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DAiWEngine)
};


/**
 * Voice Track Processor
 *
 * Custom Tracktion plugin for integrating DAiW voice synthesis
 * directly into the Tracktion timeline.
 */
class VoiceTrackPlugin : public te::Plugin
{
public:
    VoiceTrackPlugin(te::PluginCreationInfo info);
    ~VoiceTrackPlugin() override;

    //==========================================================================
    // Plugin identification

    static const char* getPluginName() { return "DAiW Voice"; }
    static const char* xmlTypeName;

    juce::String getName() const override { return getPluginName(); }
    juce::String getPluginType() override { return xmlTypeName; }
    bool producesAudioWhenNoAudioInput() override { return true; }

    //==========================================================================
    // Audio processing

    void initialise(const PluginInitialisationInfo& info) override;
    void deinitialise() override;
    void applyToBuffer(const PluginRenderContext& context) override;

    //==========================================================================
    // Voice control

    /** Set the voice model JSON */
    void setVoiceModel(const juce::String& modelJson);

    /** Queue text for synthesis at a specific time */
    void queueText(const juce::String& text, te::TimePosition startTime);

    /** Set the current vowel for real-time control */
    void setVowel(int vowelIndex);

    /** Set pitch in Hz */
    void setPitch(float pitch);

    //==========================================================================
    // State

    void restorePluginStateFromValueTree(const juce::ValueTree& state) override;
    void flushPluginStateToValueTree() override;

private:
    // Voice synthesis (uses VoiceProcessor internally)
    class Impl;
    std::unique_ptr<Impl> impl_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VoiceTrackPlugin)
};

#else // TRACKTION_ENGINE_AVAILABLE

// Stub definitions when Tracktion Engine is not available
#warning "Tracktion Engine not available. Using stub definitions."

class DAiWEngine
{
public:
    bool initialize() { return false; }
    void shutdown() {}
};

#endif // TRACKTION_ENGINE_AVAILABLE


/**
 * Integration Guide:
 *
 * 1. Clone Tracktion Engine:
 *    git clone --recursive https://github.com/Tracktion/tracktion_engine.git
 *
 * 2. Add to CMakeLists.txt:
 *    set(TRACKTION_ENGINE_DIR "${CMAKE_SOURCE_DIR}/tracktion_engine")
 *    add_subdirectory(${TRACKTION_ENGINE_DIR}/modules)
 *    target_link_libraries(DAiWBridge PRIVATE tracktion::tracktion_engine)
 *    target_compile_definitions(DAiWBridge PRIVATE TRACKTION_ENGINE_AVAILABLE=1)
 *
 * 3. Usage example:
 *
 *    DAiWEngine engine;
 *    engine.initialize();
 *
 *    auto* edit = engine.createEdit("My Song");
 *    auto* voiceTrack = engine.createVoiceTrack(edit, "Lead Vocal");
 *
 *    // Synthesize voice and add to track
 *    juce::AudioBuffer<float> voiceAudio = synthesizeVoice("Hello world");
 *    engine.addVoiceAudio(voiceTrack, voiceAudio, 44100, 0.0);
 *
 *    // Render to file
 *    engine.renderToFile(edit, outputFile, {0.0, 10.0}, {});
 */
