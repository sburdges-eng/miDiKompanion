#include "plugin/PluginEditor.h"

namespace kelly {

PluginEditor::PluginEditor(PluginProcessor& p)
    : AudioProcessorEditor(p), processor_(p)
{
    setSize(500, 400);
    
    // =========================================================================
    // SIDE A - "Where you are"
    // =========================================================================
    
    sideALabel_.setText("SIDE A - Where You Are", juce::dontSendNotification);
    sideALabel_.setFont(juce::Font(16.0f, juce::Font::bold));
    sideALabel_.setColour(juce::Label::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(sideALabel_);
    
    sideAInput_.setMultiLine(false);
    sideAInput_.setTextToShowWhenEmpty("Describe your current feeling...", juce::Colours::grey);
    addAndMakeVisible(sideAInput_);
    
    sideAIntensity_.setRange(0.0, 1.0, 0.01);
    sideAIntensity_.setValue(0.7);
    sideAIntensity_.setSliderStyle(juce::Slider::LinearHorizontal);
    sideAIntensity_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
    addAndMakeVisible(sideAIntensity_);
    
    // =========================================================================
    // SIDE B - "Where you want to go"
    // =========================================================================
    
    sideBLabel_.setText("SIDE B - Where You Want To Go", juce::dontSendNotification);
    sideBLabel_.setFont(juce::Font(16.0f, juce::Font::bold));
    sideBLabel_.setColour(juce::Label::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(sideBLabel_);
    
    sideBInput_.setMultiLine(false);
    sideBInput_.setTextToShowWhenEmpty("Describe your desired state...", juce::Colours::grey);
    addAndMakeVisible(sideBInput_);
    
    sideBIntensity_.setRange(0.0, 1.0, 0.01);
    sideBIntensity_.setValue(0.5);
    sideBIntensity_.setSliderStyle(juce::Slider::LinearHorizontal);
    sideBIntensity_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
    addAndMakeVisible(sideBIntensity_);
    
    // =========================================================================
    // GENERATE BUTTON
    // =========================================================================
    
    generateButton_.setButtonText("GENERATE");
    generateButton_.onClick = [this] { onGenerate(); };
    addAndMakeVisible(generateButton_);
    
    // =========================================================================
    // EXPORT BUTTON
    // =========================================================================
    
    exportButton_.setButtonText("Export MIDI");
    exportButton_.onClick = [this] { onExport(); };
    exportButton_.setEnabled(false);
    addAndMakeVisible(exportButton_);
    
    // =========================================================================
    // STATUS DISPLAY
    // =========================================================================
    
    statusLabel_.setText("Enter your emotions and click Generate", juce::dontSendNotification);
    statusLabel_.setFont(juce::Font(12.0f));
    statusLabel_.setColour(juce::Label::textColourId, juce::Colours::grey);
    addAndMakeVisible(statusLabel_);
    
    emotionDisplay_.setText("", juce::dontSendNotification);
    emotionDisplay_.setFont(juce::Font(14.0f));
    emotionDisplay_.setColour(juce::Label::textColourId, juce::Colours::darkblue);
    addAndMakeVisible(emotionDisplay_);
}

void PluginEditor::paint(juce::Graphics& g) {
    // Background gradient
    g.fillAll(juce::Colour(0xFFF5F5F5));
    
    // Cassette visual divider
    g.setColour(juce::Colours::lightgrey);
    g.drawLine(0, getHeight() / 2.0f, static_cast<float>(getWidth()), getHeight() / 2.0f, 2.0f);
    
    // Header
    g.setColour(juce::Colours::darkgrey);
    g.setFont(juce::Font(20.0f, juce::Font::bold));
    g.drawText("KELLY", getLocalBounds().removeFromTop(30), juce::Justification::centred);
}

void PluginEditor::resized() {
    auto bounds = getLocalBounds().reduced(20);
    
    // Header space
    bounds.removeFromTop(30);
    
    // SIDE A (top half)
    auto sideAArea = bounds.removeFromTop(bounds.getHeight() / 2 - 30);
    sideALabel_.setBounds(sideAArea.removeFromTop(25));
    sideAInput_.setBounds(sideAArea.removeFromTop(30).reduced(0, 2));
    sideAIntensity_.setBounds(sideAArea.removeFromTop(30));
    
    // Gap
    bounds.removeFromTop(20);
    
    // SIDE B (bottom portion)
    auto sideBArea = bounds.removeFromTop(100);
    sideBLabel_.setBounds(sideBArea.removeFromTop(25));
    sideBInput_.setBounds(sideBArea.removeFromTop(30).reduced(0, 2));
    sideBIntensity_.setBounds(sideBArea.removeFromTop(30));
    
    // Buttons
    bounds.removeFromTop(10);
    auto buttonArea = bounds.removeFromTop(40);
    generateButton_.setBounds(buttonArea.removeFromLeft(buttonArea.getWidth() / 2).reduced(5));
    exportButton_.setBounds(buttonArea.reduced(5));
    
    // Status
    bounds.removeFromTop(10);
    emotionDisplay_.setBounds(bounds.removeFromTop(25));
    statusLabel_.setBounds(bounds.removeFromTop(20));
}

void PluginEditor::onGenerate() {
    SideA current{
        sideAInput_.getText().toStdString(),
        static_cast<float>(sideAIntensity_.getValue()),
        std::nullopt
    };
    
    SideB desired{
        sideBInput_.getText().toStdString(),
        static_cast<float>(sideBIntensity_.getValue()),
        std::nullopt
    };
    
    // Use Side A alone if Side B is empty
    if (sideBInput_.getText().isEmpty()) {
        auto midi = processor_.generateFromWound(current.description, current.intensity);
        
        // Get intent result for display
        Wound wound{current.description, current.intensity, "ui"};
        auto result = processor_.getIntentPipeline().process(wound);
        updateEmotionDisplay(result);
    } else {
        auto midi = processor_.generateFromJourney(current, desired);
        
        // Display journey info
        auto result = processor_.getIntentPipeline().processJourney(current, desired);
        updateEmotionDisplay(result);
    }
    
    exportButton_.setEnabled(true);
    statusLabel_.setText("MIDI generated! Drag to DAW or click Export.", 
                         juce::dontSendNotification);
}

void PluginEditor::onExport() {
    juce::FileChooser chooser("Save MIDI File",
                               juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
                               "*.mid");
    
    if (chooser.browseForFileToSave(true)) {
        auto file = chooser.getResult();
        if (processor_.exportMidiToFile(file)) {
            statusLabel_.setText("Saved: " + file.getFileName(), juce::dontSendNotification);
        } else {
            statusLabel_.setText("Export failed!", juce::dontSendNotification);
        }
    }
}

void PluginEditor::updateEmotionDisplay(const IntentResult& result) {
    juce::String display;
    display << "Emotion: " << result.emotion.name;
    display << " | Mode: " << result.mode;
    display << " | Tempo: " << juce::String(result.tempo, 2) << "x";
    
    if (!result.ruleBreaks.empty()) {
        display << " | Rules broken: " << static_cast<int>(result.ruleBreaks.size());
    }
    
    emotionDisplay_.setText(display, juce::dontSendNotification);
}

} // namespace kelly
