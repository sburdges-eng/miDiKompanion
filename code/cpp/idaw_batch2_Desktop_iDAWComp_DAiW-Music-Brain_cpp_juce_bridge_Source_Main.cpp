#include <juce_gui_extra/juce_gui_extra.h>

#include "MainComponent.h"
#include "Bridge/BridgeClient.h"

class DAiWBridgeApplication : public juce::JUCEApplication
{
public:
    const juce::String getApplicationName() override    { return "DAiW Juce Bridge"; }
    const juce::String getApplicationVersion() override { return "0.1.0"; }
    bool moreThanOneInstanceAllowed() override          { return true; }

    void initialise(const juce::String&) override
    {
        bridgeClient = std::make_unique<BridgeClient>();

        mainWindow = std::make_unique<MainWindow>(
            getApplicationName(),
            juce::Colours::black,
            bridgeClient.get()
        );
    }

    void shutdown() override
    {
        mainWindow = nullptr;
        bridgeClient = nullptr;
    }

    void systemRequestedQuit() override { quit(); }
    void anotherInstanceStarted(const juce::String&) override {}

    class MainWindow : public juce::DocumentWindow
    {
    public:
        MainWindow(juce::String name, juce::Colour bg, BridgeClient* client)
            : DocumentWindow(name,
                             bg,
                             DocumentWindow::allButtons)
        {
            setUsingNativeTitleBar(true);
            setResizable(true, true);
            setContentOwned(new MainComponent(client), true);
            centreWithSize(getWidth(), getHeight());
            setVisible(true);
        }

        void closeButtonPressed() override
        {
            juce::JUCEApplication::getInstance()->systemRequestedQuit();
        }
    };

private:
    std::unique_ptr<BridgeClient> bridgeClient;
    std::unique_ptr<MainWindow> mainWindow;
};

START_JUCE_APPLICATION(DAiWBridgeApplication)

