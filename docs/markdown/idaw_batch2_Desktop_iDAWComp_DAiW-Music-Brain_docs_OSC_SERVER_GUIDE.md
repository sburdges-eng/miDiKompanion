# DAiW OSC Server Guide

## Overview

The DAiW Brain Server (`brain_server.py`) is an OSC (Open Sound Control) server that bridges the Python DAiW brain with C++ JUCE plugins. This enables DAW integration where the plugin UI runs in your DAW, but the heavy processing happens in Python.

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

## Installation

Install the OSC dependency:

```bash
pip install python-osc
# Or with optional dependencies:
pip install -e ".[osc]"
```

## Running the Server

### Basic Usage

```bash
python brain_server.py
```

The server will:
- Listen on port **9000** for incoming OSC messages (from plugin)
- Send responses on port **9001** (to plugin)

### Custom Ports

```bash
python brain_server.py --listen-port 8000 --send-port 8001
```

## OSC Protocol

### Messages Received (from Plugin)

#### `/daiw/generate`
Generate MIDI from emotional input.

**Arguments:**
- `text` (string) - User's emotional input
- `motivation` (float, 1-10) - How complete the piece should be
- `chaos` (float, 1-10) - Chaos tolerance level
- `vulnerability` (float, 1-10) - Vulnerability level

**Example:**
```python
client.send_message(
    "/daiw/generate",
    ["I feel deep grief", 7.0, 5.0, 6.0]
)
```

#### `/daiw/ping`
Health check from plugin.

**Response:** Server sends `/daiw/pong` with `["alive"]`

#### `/daiw/set_intent`
Update intent context (for future use).

**Arguments:**
- `intent_json` (string) - JSON string with intent data

### Messages Sent (to Plugin)

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
    "time_signature": "4/4",
    "chords": ["Cm", "Ab", "Fm", "Cm"],
    "length_bars": 32
  },
  "midi_events": [
    {
      "type": "note_on",
      "pitch": 60,
      "velocity": 80,
      "channel": 1,
      "tick": 0,
      "bar": 0,
      "chord": "Cm"
    },
    ...
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

#### `/daiw/pong`
Response to ping.

**Arguments:**
- `["alive"]`

## Testing

### Test Client

A test client is provided to verify the server works:

```bash
# Terminal 1: Start the server
python brain_server.py

# Terminal 2: Run the test client
python examples/test_osc_client.py
```

The test client will:
1. Send a ping to verify connectivity
2. Send a generation request
3. Display the received result

### Manual Testing with Python

```python
from pythonosc import udp_client

client = udp_client.SimpleUDPClient("127.0.0.1", 9000)

# Send generation request
client.send_message(
    "/daiw/generate",
    ["I feel deep grief and longing", 7.0, 5.0, 6.0]
)
```

## Integration with JUCE Plugin

### Plugin Side (C++)

```cpp
// Send generation request
juce::OSCMessage msg("/daiw/generate");
msg.addString("I feel deep grief");
msg.addFloat32(7.0f);  // motivation
msg.addFloat32(5.0f);  // chaos
msg.addFloat32(6.0f);  // vulnerability
oscSender.send(msg);

// Receive result
void oscMessageReceived(const juce::OSCMessage& message) override
{
    if (message.getAddressPattern() == "/daiw/result")
    {
        juce::String json = message[0].getString();
        // Parse JSON and schedule MIDI events
    }
}
```

## Server Statistics

The server tracks statistics:
- Requests received
- Requests processed
- Errors
- Uptime

View stats by sending SIGINT (Ctrl+C) to stop the server.

## Error Handling

The server handles:
- Invalid arguments (sends error response)
- Empty text input (sends error response)
- Processing errors (sends error response with details)
- Missing dependencies (raises RuntimeError on startup)

## Next Steps

1. **JUCE Plugin Skeleton** - Create basic plugin UI
2. **OSC Bridge Wiring** - Connect plugin to server
3. **MIDI Scheduling** - Parse JSON and schedule MIDI in plugin
4. **Real-time Processing** - Add streaming support for live generation

## See Also

- `vault/Production_Workflows/osc_bridge_python_cpp.md` - Detailed OSC protocol
- `vault/Production_Workflows/juce_getting_started.md` - JUCE setup guide
- `vault/Production_Workflows/hybrid_development_roadmap.md` - Full roadmap

