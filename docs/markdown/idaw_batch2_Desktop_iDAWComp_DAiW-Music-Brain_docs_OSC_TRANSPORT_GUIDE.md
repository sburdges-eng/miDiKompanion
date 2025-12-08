# OSC Transport Guide

The `OscTransport` class enables DAiW's realtime engine to send MIDI events to JUCE plugins or other OSC receivers over UDP.

## Overview

`OscTransport` is a transport implementation for the `RealtimeEngine` that:
- Serializes scheduled MIDI events as JSON
- Sends events via OSC (Open Sound Control) protocol
- Supports automatic reconnection on failures
- Provides connection status monitoring

## Basic Usage

```python
from music_brain.realtime import RealtimeEngine, OscTransport
from music_brain.structure.comprehensive_engine import NoteEvent

# Create engine
engine = RealtimeEngine(tempo_bpm=120, ppq=960)

# Create OSC transport (sends to JUCE plugin on port 9001)
osc = OscTransport(host="127.0.0.1", port=9001)
engine.add_transport(osc)

# Load and play notes
notes = [NoteEvent(pitch=60, velocity=80, start_tick=0, duration_ticks=960)]
engine.load_note_events(notes)
engine.start()

# Process events (they'll be sent via OSC)
while engine.is_running():
    engine.process_tick()
    time.sleep(0.01)
```

## Configuration Options

### Constructor Parameters

```python
OscTransport(
    host="127.0.0.1",              # OSC receiver host
    port=9001,                     # OSC receiver port
    osc_address="/daiw/notes",     # OSC address pattern
    auto_reconnect=True,            # Auto-reconnect on failures
    max_reconnect_attempts=3,        # Max reconnection attempts
)
```

### OSC Address Patterns

- **`/daiw/notes`** (default) - Direct note events to plugin
- **`/daiw/result`** - Brain server format (compatible with `brain_server.py`)

## OSC Message Format

Events are sent as a JSON array in the OSC message payload:

```json
[
  {
    "pitch": 60,
    "velocity": 80,
    "channel": 0,
    "start_tick": 0,
    "duration_ticks": 960,
    "event_id": "0",
    "metadata": {}
  },
  {
    "pitch": 64,
    "velocity": 75,
    "channel": 0,
    "start_tick": 480,
    "duration_ticks": 480,
    "event_id": "1",
    "metadata": {}
  }
]
```

## Connection Management

### Check Connection Status

```python
if osc.is_connected:
    print("Connected to OSC receiver")
else:
    print("Not connected")
```

### Check for Errors

```python
if osc.last_error:
    print(f"Last error: {osc.last_error}")
```

### Manual Reconnection

```python
# Force reconnection attempt
osc._connect()
```

## Integration with brain_server.py

The `brain_server.py` expects messages on `/daiw/result`. To use with the brain server:

```python
osc = OscTransport(
    host="127.0.0.1",
    port=9001,
    osc_address="/daiw/result",  # Use brain server format
)
```

Note: The brain server also expects a different JSON structure. For direct plugin communication, use `/daiw/notes`.

## Error Handling

The transport handles network errors gracefully:

- **Automatic reconnection**: If `auto_reconnect=True`, failed sends trigger reconnection attempts
- **Silent failures**: Network errors don't crash the engine (failures are logged via `last_error`)
- **Connection state**: Check `is_connected` to monitor connection health

### Example with Error Handling

```python
osc = OscTransport(host="127.0.0.1", port=9001)

# Monitor connection
if not osc.is_connected:
    print("Warning: OSC transport not connected")
    if osc.last_error:
        print(f"Error: {osc.last_error}")

# Send events (will attempt reconnection if needed)
engine.process_tick()

# Check status after sending
if osc.last_error:
    print(f"Send error: {osc.last_error}")
```

## Performance Considerations

- **UDP Protocol**: OSC uses UDP, so messages may be lost (no guaranteed delivery)
- **Network Latency**: Typical latency is 1-5ms on localhost
- **Thread Safety**: The transport uses locks for thread-safe operation
- **Batching**: Events are batched per `emit()` call (all events in one OSC message)

## Troubleshooting

### Messages Not Received

1. **Check receiver is listening**:
   ```bash
   # macOS/Linux
   oscdump 9001
   ```

2. **Check connection status**:
   ```python
   print(f"Connected: {osc.is_connected}")
   print(f"Last error: {osc.last_error}")
   ```

3. **Verify port and host**:
   ```python
   # Use correct host/port for your receiver
   osc = OscTransport(host="127.0.0.1", port=9001)
   ```

### Connection Failures

- **Firewall**: Ensure UDP ports are not blocked
- **Port conflicts**: Check if another process is using the port
- **Network interface**: Use `127.0.0.1` for localhost, or actual IP for remote

### JSON Parsing Errors

- Ensure receiver expects JSON string in OSC message
- Check JSON structure matches receiver's expectations
- Verify all required fields are present (pitch, velocity, etc.)

## Examples

See `examples/realtime_osc_example.py` for a complete working example.

## See Also

- `docs/JUCE_BRIDGE_GUIDE.md` - Complete JUCE integration guide
- `brain_server.py` - Python OSC server implementation
- `music_brain/realtime/engine.py` - RealtimeEngine documentation

