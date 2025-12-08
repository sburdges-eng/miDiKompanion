# JUCE Bridge Integration Guide

This guide explains how to integrate DAiW's Python brain with JUCE plugins for DAW integration.

## Overview

The JUCE bridge enables DAiW to:
- Generate music in real-time from within your DAW
- Receive emotional intent from plugin UI
- Send MIDI notes back to the plugin for playback
- Control plugin parameters via OSC

## Architecture

```
┌─────────────────┐         OSC (UDP)          ┌──────────────────┐
│  JUCE Plugin    │ ◄─────────────────────────► │  Python Brain    │
│  (DAW)          │    Port 9000/9001          │  (DAiW Server)   │
│                 │                             │                  │
│  - UI           │    /daiw/generate          │  - Intent Proc   │
│  - OSC Client   │    /daiw/notes             │  - Harmony Gen   │
│  - MIDI Output  │    /daiw/status             │  - OSC Server   │
└─────────────────┘                             └──────────────────┘
       │                                                  │
       │ MIDI                                            │
       ▼                                                  │
┌─────────────────┐                                     │
│  DAW (Logic)    │                                     │
│  - Instruments  │                                     │
│  - Effects      │                                     │
└─────────────────┘                                     │
```

## Python Side: OSC Transport

The `OscTransport` class in `music_brain.realtime.transport` sends events to JUCE plugins:

```python
from music_brain.realtime import RealtimeEngine, OscTransport

# Create engine with OSC transport
engine = RealtimeEngine(tempo_bpm=120, ppq=960)
osc_transport = OscTransport(
    host="127.0.0.1",
    port=9001,
    osc_address="/daiw/notes",  # OSC address pattern
    auto_reconnect=True,         # Auto-reconnect on failures
)
engine.add_transport(osc_transport)

# Check connection status
if osc_transport.is_connected:
    print("Connected to OSC receiver")

# Load and play
engine.load_note_events(note_events)
engine.start()

# Engine will send events via OSC to plugin
# Monitor connection health
if osc_transport.last_error:
    print(f"OSC error: {osc_transport.last_error}")
```

See `docs/OSC_TRANSPORT_GUIDE.md` for complete documentation.

### OSC Message Format

Events are sent as JSON to `/daiw/notes`:

```json
[
  {
    "pitch": 60,
    "velocity": 80,
    "channel": 0,
    "start_tick": 0,
    "duration_ticks": 480,
    "event_id": "0",
    "metadata": {}
  }
]
```

## JUCE Side: Plugin Implementation

### 1. OSC Receiver Setup

```cpp
#include <juce_osc/juce_osc.h>

class DAiWPluginProcessor : public juce::AudioProcessor,
                            private juce::OSCReceiver::Listener<
                                juce::OSCReceiver::MessageLoopCallback>
{
public:
    DAiWPluginProcessor()
    {
        // Listen for notes from Python
        if (oscReceiver.connect(9001))
        {
            oscReceiver.addListener(this);
            DBG("OSC Receiver listening on port 9001");
        }
    }

    void oscMessageReceived(const juce::OSCMessage& message) override
    {
        auto address = message.getAddressPattern().toString();
        
        if (address == "/daiw/notes")
        {
            // Parse JSON and schedule MIDI
            juce::String json = message[0].getString();
            parseAndScheduleNotes(json);
        }
        else if (address == "/daiw/status")
        {
            juce::String status = message[0].getString();
            updateStatus(status);
        }
    }

private:
    juce::OSCReceiver oscReceiver;
    juce::OSCSender oscSender;
};
```

### 2. MIDI Scheduling

```cpp
void DAiWPluginProcessor::processBlock(
    juce::AudioBuffer<float>& buffer,
    juce::MidiBuffer& midiMessages)
{
    // Merge scheduled notes from Python
    midiMessages.addEvents(pendingMidiBuffer, 0, buffer.getNumSamples(), 0);
    pendingMidiBuffer.clear();
}

void DAiWPluginProcessor::parseAndScheduleNotes(const juce::String& json)
{
    // Parse JSON array of notes
    auto jsonObj = juce::JSON::parse(json);
    auto notesArray = jsonObj.getArray();
    
    for (auto* noteObj : *notesArray)
    {
        int pitch = noteObj->getProperty("pitch");
        int velocity = noteObj->getProperty("velocity");
        int channel = noteObj->getProperty("channel", 0);
        int startTick = noteObj->getProperty("start_tick");
        int durationTicks = noteObj->getProperty("duration_ticks");
        
        // Convert ticks to sample position
        int startSample = ticksToSamples(startTick);
        int durationSamples = ticksToSamples(durationTicks);
        
        // Schedule note on
        juce::MidiMessage noteOn = juce::MidiMessage::noteOn(
            channel + 1, pitch, (juce::uint8)velocity);
        pendingMidiBuffer.addEvent(noteOn, startSample);
        
        // Schedule note off
        juce::MidiMessage noteOff = juce::MidiMessage::noteOff(
            channel + 1, pitch);
        pendingMidiBuffer.addEvent(noteOff, startSample + durationSamples);
    }
}
```

### 3. OSC Sender (Request Generation)

```cpp
void DAiWPluginProcessor::requestGeneration(
    const juce::String& intentText,
    float chaos,
    float vulnerability)
{
    if (!oscSender.isConnected())
    {
        oscSender.connect("127.0.0.1", 9000);
    }
    
    juce::OSCMessage msg("/daiw/generate");
    msg.addString(intentText);
    msg.addFloat32(chaos);
    msg.addFloat32(vulnerability);
    
    oscSender.send(msg);
}
```

## Complete Integration Example

### Python Server

```python
from music_brain.realtime import RealtimeEngine, OscTransport
from music_brain.harmony import HarmonyGenerator

# Create engine with OSC output
engine = RealtimeEngine(tempo_bpm=120)
osc = OscTransport(port=9001)
engine.add_transport(osc)

# Generate harmony
generator = HarmonyGenerator()
result = generator.generate(
    key="C",
    mode="major",
    pattern="I-V-vi-IV",
    bars=4
)

# Load and play
engine.load_note_events(result.note_events)
engine.start()

# Process in loop
import time
while engine.is_running():
    engine.process_tick()
    time.sleep(0.01)  # ~100Hz update rate
```

### JUCE Plugin

```cpp
// In PluginProcessor.h
class DAiWPluginProcessor : public juce::AudioProcessor,
                            private juce::OSCReceiver::Listener<>
{
    void requestGeneration();
    void oscMessageReceived(const juce::OSCMessage&) override;
    
    juce::OSCReceiver oscReceiver;
    juce::OSCSender oscSender;
    juce::MidiBuffer pendingMidi;
};

// In PluginEditor.cpp (button click handler)
void DAiWPluginEditor::generateButtonClicked()
{
    auto* processor = getAudioProcessor();
    processor->requestGeneration(
        textEditor.getText(),
        chaosSlider.getValue(),
        vulnerabilitySlider.getValue()
    );
}
```

## Testing

### 1. Test OSC Communication

```python
# Python side
python examples/test_osc_client.py
```

### 2. Test Plugin

1. Start Python server: `python brain_server.py`
2. Load plugin in DAW
3. Click "Generate" in plugin UI
4. Verify MIDI plays in DAW

### 3. Debug OSC

```bash
# macOS/Linux
oscdump 9000  # Listen to plugin → Python
oscdump 9001  # Listen to Python → plugin

# Send test message
oscsend localhost 9000 /daiw/ping
```

## Performance Considerations

- **Latency**: OSC adds ~1-5ms network latency
- **Jitter**: Use lookahead scheduling (2 bars recommended)
- **Threading**: OSC callbacks run on message thread, not audio thread
- **Buffering**: Schedule MIDI in `processBlock()`, not in OSC callback

## Next Steps

1. ✅ OSC transport implemented (Python side)
2. ⏳ JUCE plugin skeleton (see `cpp/DAiWBridge/README.md`)
3. ⏳ Full UI implementation
4. ⏳ Parameter automation
5. ⏳ Error handling and recovery

## Resources

- [JUCE OSC Tutorial](https://docs.juce.com/master/tutorial_osc_message.html)
- [OSC Protocol Spec](http://opensoundcontrol.org/)
- [DAiW Realtime Engine Plan](docs/REALTIME_ENGINE_PLAN.md)

