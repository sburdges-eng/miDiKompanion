# DAiW iOS App

iOS app for DAiW Music Brain - Emotional intent to MIDI generation.

## Overview

The iOS app provides:
- **UI**: Text input for emotional intent, parameter sliders
- **OSC Communication**: Connects to Python brain server
- **MIDI Playback**: Real-time MIDI event scheduling and playback
- **Integration**: Full workflow from intent to music

## Architecture

```
┌─────────────┐         OSC          ┌──────────────┐
│  iOS App    │ ◄─────────────────► │ Python Brain │
│  (Swift)    │   Port 9000/9001     │  (Server)    │
└─────────────┘                       └──────────────┘
     │                                       │
     │                                       │
     └──────► MIDI Playback ◄────────────────┘
```

## Project Structure

```
ios/
├── DAiW/                    # Main app
│   ├── DAiWOSCClient.swift  # OSC communication
│   ├── DAiWRealtimeEngine.swift  # MIDI scheduling
│   └── DAiWBrainServerConnection.swift  # Server connection
├── DAiWTests/               # Test suite
│   ├── DAiWOSCTransportTests.swift
│   ├── DAiWRealtimeEngineTests.swift
│   └── DAiWIntegrationTests.swift
└── README.md                # This file
```

## Setup

### Prerequisites

1. **Xcode 14+** with iOS 15+ deployment target
2. **Python brain server** running (see main project README)
3. **Network framework** (included in iOS)

### Installation

1. Open `DAiW.xcodeproj` in Xcode
2. Select your development team in Signing & Capabilities
3. Build and run on simulator or device

### Dependencies

- **Network Framework** (built-in) - For OSC/UDP communication
- **AVFoundation** (built-in) - For MIDI playback
- **Combine** (built-in) - For reactive programming

## Running Tests

```bash
# In Xcode
⌘U  # Run all tests

# Or via command line
xcodebuild test -scheme DAiW -destination 'platform=iOS Simulator,name=iPhone 15'
```

## Test Coverage

### OSC Transport Tests
- Connection establishment
- Message sending/receiving
- Error handling
- Network timeouts

### Realtime Engine Tests
- Event scheduling
- Tempo management
- Playback control
- MIDI output

### Integration Tests
- End-to-end workflows
- Server communication
- Error recovery

## Usage

### Basic Workflow

```swift
// 1. Connect to brain server
let brainServer = BrainServerConnection()
brainServer.connect()

// 2. Generate MIDI from intent
brainServer.generateMIDI(
    intent: "I feel deep grief",
    motivation: 7.0,
    chaos: 5.0,
    vulnerability: 6.0
)
.sink { events in
    // 3. Play MIDI events
    playMIDI(events)
}
```

### Real-time Playback

```swift
let engine = RealtimeEngine(tempoBPM: 120, ppq: 960)

// Schedule events
for event in midiEvents {
    engine.scheduleNote(event, channel: 0)
}

// Start playback
engine.start()

// Process in loop
Timer.scheduledTimer(withTimeInterval: 0.01, repeats: true) { _ in
    engine.processTick()
}
```

## Development

### Adding Features

1. **OSC Communication**: Edit `DAiWOSCClient.swift`
2. **MIDI Scheduling**: Edit `DAiWRealtimeEngine.swift`
3. **UI Components**: Add SwiftUI views
4. **Tests**: Add test cases in `DAiWTests/`

### Testing

Run tests with:
```bash
xcodebuild test -scheme DAiW
```

Or use Xcode's test navigator (⌘6).

## Troubleshooting

### Tests Fail to Connect

- Ensure Python brain server is running: `python brain_server.py`
- Check ports 9000/9001 are not blocked
- Verify network permissions in Info.plist

### MIDI Not Playing

- Check AVAudioSession configuration
- Verify MIDI output routing
- Check device/simulator MIDI capabilities

### Build Errors

- Ensure iOS 15+ deployment target
- Check all frameworks are linked
- Verify code signing settings

## Next Steps

- [ ] Complete OSC library integration (SwiftOSC)
- [ ] Add SwiftUI interface
- [ ] Implement Core MIDI output
- [ ] Add audio engine integration
- [ ] Performance optimizations

## See Also

- `docs/JUCE_BRIDGE_GUIDE.md` - JUCE plugin integration
- `docs/OSC_TRANSPORT_GUIDE.md` - OSC protocol documentation
- `brain_server.py` - Python brain server

