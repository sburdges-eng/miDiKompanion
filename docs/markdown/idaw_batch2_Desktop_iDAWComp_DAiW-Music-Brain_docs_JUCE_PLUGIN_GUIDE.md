# JUCE Plugin Integration Guide

## Overview

The DAiW Bridge JUCE plugin connects your DAW to the Python DAiW brain server, allowing you to generate MIDI from emotional input directly within your DAW.

## Architecture

```
┌─────────────────┐         OSC          ┌──────────────────┐
│  JUCE Plugin    │ ◄──────────────────► │  Python Brain    │
│  (C++ in DAW)   │   Port 9000/9001     │  (brain_server)  │
└─────────────────┘                       └──────────────────┘
     │                                           │
     │                                           │
     └──────────► MIDI Output to DAW ◄───────────┘
```

## Files Created

### C++ Plugin Files

- `cpp/DAiWBridge/PluginProcessor.h` - Audio processor with OSC communication
- `cpp/DAiWBridge/PluginProcessor.cpp` - Implementation
- `cpp/DAiWBridge/PluginEditor.h` - UI components
- `cpp/DAiWBridge/PluginEditor.cpp` - UI implementation
- `cpp/DAiWBridge/README.md` - Plugin documentation

## Features

### Plugin Processor (`PluginProcessor.h/cpp`)

- **OSC Sender**: Sends requests to Python brain server (port 9000)
- **OSC Receiver**: Receives MIDI events from brain server (port 9001)
- **MIDI Buffer**: Thread-safe MIDI event scheduling
- **JSON Parsing**: Converts brain server JSON to MIDI events

### Plugin Editor (`PluginEditor.h/cpp`)

- **Text Input**: Multi-line text editor for emotional input
- **Parameter Sliders**: Motivation, Chaos, Vulnerability (1-10)
- **Generate Button**: Triggers MIDI generation
- **Status Indicator**: Shows connection status to brain server

## Building the Plugin

### Step 1: Install JUCE

1. Download JUCE from https://juce.com/get-juce/download
2. Extract to a location (e.g., `~/JUCE`)
3. Build Projucer:
   ```bash
   cd ~/JUCE/extras/Projucer/Builds/MacOSX
   xcodebuild -project Projucer.xcodeproj
   ```

### Step 2: Create Projucer Project

1. Open Projucer
2. Create new project → Audio Plug-In
3. Configure:
   - **Name**: DAiWBridge
   - **Plugin Formats**: AU, VST3
   - **Plugin Type**: MIDI Effect
4. Add required modules:
   - `juce_osc` (for OSC communication)
   - `juce_core`
   - `juce_audio_basics`
   - `juce_audio_processors`
   - `juce_gui_basics`
5. Copy our plugin files:
   - Replace generated `PluginProcessor.h/cpp` with our versions
   - Replace generated `PluginEditor.h/cpp` with our versions

### Step 3: Build

1. Open generated Xcode project
2. Select target: "DAiWBridge - All"
3. Build (Cmd+B)
4. Plugin will be installed to:
   - **AU**: `~/Library/Audio/Plug-Ins/Components/`
   - **VST3**: `~/Library/Audio/Plug-Ins/VST3/`

## Usage

### 1. Start Brain Server

```bash
python brain_server.py
```

Server should show:
```
[INFO] DAiW Brain Server starting...
[INFO] Listening on port 9000
[INFO] Sending to port 9001
```

### 2. Load Plugin in DAW

1. Open your DAW (Logic Pro, Ableton, etc.)
2. Create a new MIDI track
3. Add DAiW Bridge as a MIDI effect
4. Plugin UI should show "Connected" status (green)

### 3. Generate MIDI

1. Type emotional text: "I feel deep grief and longing"
2. Adjust sliders:
   - **Motivation**: 7 (how complete the piece should be)
   - **Chaos**: 5 (chaos tolerance)
   - **Vulnerability**: 6 (emotional exposure)
3. Click "Generate MIDI"
4. Status changes to "Generating..."
5. MIDI events appear in your DAW timeline

## OSC Protocol

### Plugin → Brain Server

#### `/daiw/generate`
Generate MIDI from emotional input.

**Arguments:**
- `text` (string) - User's emotional input
- `motivation` (float) - 1-10, how complete
- `chaos` (float) - 1-10, chaos tolerance
- `vulnerability` (float) - 1-10, emotional exposure

**Example:**
```cpp
juce::OSCMessage msg("/daiw/generate");
msg.addString("I feel deep grief");
msg.addFloat32(7.0f);
msg.addFloat32(5.0f);
msg.addFloat32(6.0f);
oscSender.send(msg);
```

#### `/daiw/ping`
Health check.

**Response:** Server sends `/daiw/pong`

### Brain Server → Plugin

#### `/daiw/result`
Generation result with MIDI events.

**Format:**
```json
{
  "status": "success",
  "affect": {
    "primary": "grief",
    "intensity": 0.85
  },
  "plan": {
    "tempo_bpm": 82,
    "key": "C",
    "mode": "aeolian",
    "chords": ["Cm", "Ab", "Fm", "Cm"]
  },
  "midi_events": [
    {
      "type": "note_on",
      "pitch": 60,
      "velocity": 80,
      "channel": 1,
      "tick": 0
    }
  ]
}
```

#### `/daiw/error`
Error response.

**Format:**
```json
{
  "status": "error",
  "message": "Error description"
}
```

## MIDI Event Scheduling

The plugin converts JSON MIDI events to JUCE MIDI messages:

```cpp
void parseMidiEventsFromJSON(const juce::String& jsonString)
{
    auto json = juce::JSON::parse(jsonString);
    auto midiEventsArray = json.getProperty("midi_events");
    
    for (int i = 0; i < midiEventsArray.size(); ++i)
    {
        auto event = midiEventsArray[i];
        int pitch = (int)event.getProperty("pitch");
        int velocity = (int)event.getProperty("velocity");
        int tick = (int)event.getProperty("tick");
        
        // Convert tick to sample position
        int sampleOffset = (int)((tick / 480.0) * sampleRate);
        
        pendingMidi.addEvent(
            juce::MidiMessage::noteOn(1, pitch, (juce::uint8)velocity),
            sampleOffset
        );
    }
}
```

## Testing

### Test OSC Communication

1. Start brain server: `python brain_server.py`
2. Run test client: `python examples/test_osc_client.py`
3. Verify messages are received

### Test Plugin

1. Build plugin in Xcode
2. Load in DAW
3. Check connection status (should be green)
4. Send test generation request
5. Verify MIDI appears in DAW

## Troubleshooting

### Plugin shows "Not Connected"

- Verify `brain_server.py` is running
- Check ports 9000/9001 are not blocked
- Check firewall settings
- Try restarting plugin

### MIDI not appearing

- Ensure plugin is on a MIDI track (not audio)
- Check MIDI channel settings
- Verify JSON parsing (check console logs)
- Check sample rate matches DAW

### Build errors

- Ensure JUCE modules are included (`juce_osc`)
- Check C++ standard (C++17 or later)
- Verify all JUCE paths are correct

## Next Steps

1. **Projucer Project**: Create `.jucer` file for easier building
2. **MIDI Timing**: Improve tick-to-sample conversion
3. **UI Polish**: Better visual design and feedback
4. **Error Handling**: More robust error messages
5. **Presets**: Save/load parameter presets

## See Also

- `docs/OSC_SERVER_GUIDE.md` - Python brain server documentation
- `cpp/DAiWBridge/README.md` - Plugin-specific documentation
- `brain_server.py` - Python brain server implementation
- `examples/test_osc_client.py` - OSC test client

