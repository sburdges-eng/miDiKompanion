#pragma once

#include <juce_gui_extra/juce_gui_extra.h>

class BridgeClient;

class MainComponent : public juce::Component,
                      private juce::Button::Listener
{
public:
    explicit MainComponent(BridgeClient* client);
    ~MainComponent() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    void buttonClicked(juce::Button* button) override;

    BridgeClient* bridgeClient;
    juce::TextButton connectButton { "Connect to DAiW" };
    juce::Label statusLabel { {}, "Disconnected" };
    juce::TextEditor promptEditor;
    juce::TextEditor chatHistory;
    juce::TextEditor chatInput;
    juce::TextButton sendButton { "Send" };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainComponent)
};

