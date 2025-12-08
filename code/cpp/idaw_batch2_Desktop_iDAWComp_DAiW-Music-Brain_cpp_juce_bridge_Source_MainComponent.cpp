#include "MainComponent.h"
#include "Bridge/BridgeClient.h"

MainComponent::MainComponent(BridgeClient* client)
    : bridgeClient(client)
{
    addAndMakeVisible(connectButton);
    addAndMakeVisible(statusLabel);
    addAndMakeVisible(promptEditor);
    addAndMakeVisible(chatHistory);
    addAndMakeVisible(chatInput);
    addAndMakeVisible(sendButton);

    connectButton.addListener(this);
    sendButton.addListener(this);

    promptEditor.setMultiLine(true);
    promptEditor.setText("Enter intent or vocal prompt...");
    chatHistory.setMultiLine(true);
    chatHistory.setReadOnly(true);
    chatHistory.setText("Chat history will appear here.\n");
    chatInput.setMultiLine(false);
    chatInput.setText("");

    statusLabel.setJustificationType(juce::Justification::centred);
    setSize(480, 320);
}

void MainComponent::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::dimgrey);
    g.setColour(juce::Colours::white);
    g.setFont(20.0f);
    g.drawFittedText("DAiW JUCE Bridge (prototype)", getLocalBounds().removeFromTop(40), juce::Justification::centred, 1);
}

void MainComponent::resized()
{
    auto bounds = getLocalBounds().reduced(20);
    connectButton.setBounds(bounds.removeFromTop(40));
    statusLabel.setBounds(bounds.removeFromTop(30));
    bounds.removeFromTop(10);

    auto upper = bounds.removeFromTop(bounds.getHeight() / 2);
    promptEditor.setBounds(upper);

    chatHistory.setBounds(bounds.removeFromTop(bounds.getHeight() - 40));
    auto inputArea = bounds.removeFromTop(30);
    chatInput.setBounds(inputArea.removeFromLeft(inputArea.getWidth() - 80));
    sendButton.setBounds(inputArea);
}

void MainComponent::buttonClicked(juce::Button* button)
{
    if (button == &connectButton && bridgeClient != nullptr)
    {
        statusLabel.setText("Connecting...", juce::dontSendNotification);
        auto result = bridgeClient->ping();
        statusLabel.setText(result ? "Connected to DAiW" : "Connection failed", juce::dontSendNotification);
    }
    else if (button == &sendButton && bridgeClient != nullptr)
    {
        auto text = chatInput.getText();
        if (text.isNotEmpty())
        {
            chatHistory.moveCaretToEnd();
            chatHistory.insertTextAtCaret("you> " + text + "\n");
            auto reply = bridgeClient->sendChatMessage(text);
            chatHistory.insertTextAtCaret("daiw> " + reply + "\n");
            chatInput.clear();
        }
    }
}

