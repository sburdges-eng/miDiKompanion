#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

class BridgeClient;

/**
 * Placeholder voice processor that will wrap DAiW voice engines.
 */
class VoiceProcessor : public juce::AudioProcessor
{
public:
    explicit VoiceProcessor(BridgeClient* client);
    ~VoiceProcessor() override = default;

    // AudioProcessor overrides
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    const juce::String getName() const override { return "DAiWVoiceProcessor"; }
    bool hasEditor() const override { return false; }
    juce::AudioProcessorEditor* createEditor() override { return nullptr; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock&) override {}
    void setStateInformation(const void*, int) override {}

private:
    BridgeClient* bridgeClient;
};

