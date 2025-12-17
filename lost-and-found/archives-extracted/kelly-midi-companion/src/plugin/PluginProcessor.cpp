#include "plugin/PluginProcessor.h"
#include "plugin/PluginEditor.h"
#include "midi/ChordGenerator.h"
#include "midi/MidiBuilder.h"

namespace kelly {

PluginProcessor::PluginProcessor()
    : AudioProcessor(BusesProperties()
          .withInput("Input", juce::AudioChannelSet::stereo(), true)
          .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters(*this, nullptr, "KellyParams", createParameterLayout())
{
}

juce::AudioProcessorValueTreeState::ParameterLayout PluginProcessor::createParameterLayout() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_INTENSITY, 1},
        "Intensity",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f),
        0.7f
    ));
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_HUMANIZE, 1},
        "Humanize",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f),
        0.4f
    ));
    
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{PARAM_TEMPO_LOCK, 1},
        "Lock to DAW Tempo",
        true
    ));
    
    return {params.begin(), params.end()};
}

void PluginProcessor::prepareToPlay(double sampleRate, int /*samplesPerBlock*/) {
    currentSampleRate_ = sampleRate;
    
    // Get tempo from host if available
    if (auto* playHead = getPlayHead()) {
        if (auto position = playHead->getPosition()) {
            if (auto bpm = position->getBpm()) {
                currentBpm_ = static_cast<float>(*bpm);
            }
        }
    }
}

void PluginProcessor::releaseResources() {
    pendingMidiOutput_.clear();
}

void PluginProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) {
    juce::ScopedNoDenormals noDenormals;
    
    // Clear audio (we're MIDI-only)
    buffer.clear();
    
    // Update tempo from host
    if (auto* playHead = getPlayHead()) {
        if (auto position = playHead->getPosition()) {
            if (auto bpm = position->getBpm()) {
                currentBpm_ = static_cast<float>(*bpm);
            }
        }
    }
    
    // If we have pending MIDI to output, add it
    if (hasPendingMidi_.exchange(false)) {
        midiMessages.addEvents(pendingMidiOutput_, 0, buffer.getNumSamples(), 0);
        pendingMidiOutput_.clear();
    }
}

GeneratedMidi PluginProcessor::generateFromWound(const std::string& description, float intensity) {
    Wound wound{description, intensity, "user_input"};
    IntentResult intent = intentPipeline_.process(wound);
    
    // Generate MIDI using the chord and groove generators
    ChordGenerator chordGen;
    GeneratedMidi result;
    
    result.bpm = parameters.getRawParameterValue(PARAM_TEMPO_LOCK)->load() > 0.5f 
        ? currentBpm_ 
        : currentBpm_ * intent.tempo;
    
    // Generate chord progression based on intent
    result.chords = chordGen.generate(intent);
    result.lengthInBeats = 16.0;  // 4 bars default
    
    lastGenerated_ = result;
    return result;
}

GeneratedMidi PluginProcessor::generateFromJourney(const SideA& current, const SideB& desired) {
    IntentResult intent = intentPipeline_.processJourney(current, desired);
    
    ChordGenerator chordGen;
    GeneratedMidi result;
    
    result.bpm = parameters.getRawParameterValue(PARAM_TEMPO_LOCK)->load() > 0.5f 
        ? currentBpm_ 
        : currentBpm_ * intent.tempo;
    
    result.chords = chordGen.generate(intent);
    result.lengthInBeats = 16.0;
    
    lastGenerated_ = result;
    return result;
}

bool PluginProcessor::exportMidiToFile(const juce::File& file) {
    if (lastGenerated_.chords.empty()) {
        return false;
    }
    
    MidiBuilder builder;
    auto midiFile = builder.buildMidiFile(lastGenerated_);
    
    juce::FileOutputStream stream(file);
    if (stream.openedOk()) {
        midiFile.writeTo(stream);
        return true;
    }
    return false;
}

void PluginProcessor::queueMidiForOutput(const GeneratedMidi& midi) {
    MidiBuilder builder;
    pendingMidiOutput_ = builder.buildMidiBuffer(midi, currentSampleRate_, currentBpm_);
    hasPendingMidi_ = true;
}

juce::AudioProcessorEditor* PluginProcessor::createEditor() {
    return new PluginEditor(*this);
}

void PluginProcessor::getStateInformation(juce::MemoryBlock& destData) {
    auto state = parameters.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void PluginProcessor::setStateInformation(const void* data, int sizeInBytes) {
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
    if (xml && xml->hasTagName(parameters.state.getType())) {
        parameters.replaceState(juce::ValueTree::fromXml(*xml));
    }
}

} // namespace kelly

// Plugin entry point
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new kelly::PluginProcessor();
}
