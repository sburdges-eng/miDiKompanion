#include "BridgeClient.h"

BridgeClient::BridgeClient() = default;
BridgeClient::~BridgeClient() = default;

bool BridgeClient::ping()
{
    // TODO: Replace with HTTP/gRPC call to DAiW API.
    juce::Thread::sleep(150);
    return true;
}

bool BridgeClient::requestAutoTune(const juce::File&, juce::File&)
{
    // TODO: Implement auto-tune RPC pipeline.
    return false;
}

juce::String BridgeClient::sendChatMessage(const juce::String& message)
{
    // TODO: Replace with offline chatbot service call.
    return "Echo: " + message;
}

