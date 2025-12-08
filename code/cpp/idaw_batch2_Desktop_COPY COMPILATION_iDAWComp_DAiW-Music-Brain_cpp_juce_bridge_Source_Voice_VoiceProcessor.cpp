#include "VoiceProcessor.h"
#include "../Bridge/BridgeClient.h"

VoiceProcessor::VoiceProcessor(BridgeClient* client)
    : bridgeClient(client)
{
}

void VoiceProcessor::prepareToPlay(double, int)
{
}

void VoiceProcessor::releaseResources()
{
}

void VoiceProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    // TODO: Send audio to DAiW voice pipeline.
    buffer.clear(); // silence for now
}

