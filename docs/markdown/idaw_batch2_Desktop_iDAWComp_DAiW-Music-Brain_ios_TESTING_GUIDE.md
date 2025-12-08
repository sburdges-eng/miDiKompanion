# iOS Testing Guide

Complete guide for testing the DAiW iOS app.

## Test Structure

The iOS test suite is organized into three main categories:

### 1. OSC Transport Tests (`DAiWOSCTransportTests.swift`)

Tests OSC communication with the Python brain server:

- **Connection Tests**: Verify OSC client can connect/disconnect
- **Message Sending**: Test sending `/daiw/generate`, `/daiw/ping` messages
- **Message Receiving**: Test receiving `/daiw/result`, `/daiw/pong` responses
- **Error Handling**: Test network failures, timeouts, invalid messages

### 2. Realtime Engine Tests (`DAiWRealtimeEngineTests.swift`)

Tests MIDI event scheduling and playback:

- **Engine Initialization**: Tempo, PPQ, lookahead configuration
- **Event Scheduling**: Single/multiple events, ordering
- **Playback Control**: Start/stop, tick processing
- **MIDI Integration**: Note on/off, channel routing
- **Performance**: High event rates, processing speed

### 3. Integration Tests (`DAiWIntegrationTests.swift`)

End-to-end workflow tests:

- **Full Workflow**: Intent → Generation → MIDI Playback
- **Real-time Playback**: Live MIDI scheduling
- **Multiple Generations**: Sequential requests
- **Error Recovery**: Server unavailable, invalid data

## Running Tests

### In Xcode

1. Open `ios/DAiW.xcodeproj` in Xcode
2. Press `⌘U` to run all tests
3. Or use Test Navigator (`⌘6`) to run specific tests

### Command Line

```bash
# Run all tests
./ios/run_tests.sh

# Run on specific simulator
./ios/run_tests.sh --simulator "iPhone 15 Pro"

# Run on physical device
./ios/run_tests.sh --device "Sean's iPhone"
```

### CI/CD

```bash
xcodebuild test \
    -project ios/DAiW.xcodeproj \
    -scheme DAiW \
    -destination 'platform=iOS Simulator,name=iPhone 15' \
    -only-testing:DAiWTests
```

## Prerequisites

### 1. Python Brain Server

The integration tests require the Python brain server to be running:

```bash
# Start brain server
python brain_server.py

# Server should be listening on:
# - Port 9000 (receive from iOS)
# - Port 9001 (send to iOS)
```

### 2. Network Permissions

iOS requires network permissions in `Info.plist`:

```xml
<key>NSLocalNetworkUsageDescription</key>
<string>DAiW needs network access to communicate with the Python brain server.</string>
```

### 3. Test Environment

- **Xcode 14+** required
- **iOS 15+** deployment target
- **Simulator or Device** for running tests

## Test Coverage

### Current Coverage

- ✅ OSC connection/disconnection
- ✅ Message sending/receiving
- ✅ Engine initialization and configuration
- ✅ Event scheduling and ordering
- ✅ Playback control
- ✅ Error handling
- ✅ Integration workflows

### Missing Coverage

- ⏳ MIDI output verification (requires Core MIDI setup)
- ⏳ Audio engine integration
- ⏳ UI component tests
- ⏳ Performance benchmarks
- ⏳ Memory leak detection

## Writing New Tests

### Test Template

```swift
func testFeatureName() {
    // Arrange
    let expectation = XCTestExpectation(description: "Feature works")
    
    // Act
    feature.doSomething { result in
        // Assert
        XCTAssertNotNil(result)
        expectation.fulfill()
    }
    
    wait(for: [expectation], timeout: 10.0)
}
```

### Best Practices

1. **Use Expectations**: All async operations should use `XCTestExpectation`
2. **Set Timeouts**: Always specify reasonable timeouts
3. **Clean Up**: Use `tearDown()` to clean resources
4. **Isolate Tests**: Each test should be independent
5. **Mock Dependencies**: Use mocks for external services when possible

## Debugging Tests

### View Test Logs

In Xcode:
1. Open Report Navigator (`⌘9`)
2. Select test run
3. View detailed logs and screenshots

### Breakpoints

Set breakpoints in test code to debug:
- Test execution flow
- Variable values
- Network communication

### Network Debugging

Use `oscdump` to monitor OSC traffic:

```bash
# Terminal 1: Monitor iOS → Python
oscdump 9000

# Terminal 2: Monitor Python → iOS
oscdump 9001
```

## Continuous Integration

### GitHub Actions Example

```yaml
- name: Run iOS Tests
  run: |
    xcodebuild test \
      -project ios/DAiW.xcodeproj \
      -scheme DAiW \
      -destination 'platform=iOS Simulator,name=iPhone 15' \
      -only-testing:DAiWTests
```

### Test Reports

Tests generate JUnit XML reports:

```bash
xcodebuild test \
    -project ios/DAiW.xcodeproj \
    -scheme DAiW \
    -resultBundlePath TestResults.xcresult
```

## Troubleshooting

### Tests Fail to Connect

**Problem**: OSC connection tests fail

**Solutions**:
- Ensure `brain_server.py` is running
- Check firewall settings
- Verify ports 9000/9001 are available
- Check network permissions in Info.plist

### Simulator Issues

**Problem**: Simulator won't start or tests hang

**Solutions**:
- Reset simulator: `xcrun simctl erase all`
- Restart Xcode
- Check available disk space

### Build Errors

**Problem**: Tests won't compile

**Solutions**:
- Clean build folder: `⌘⇧K`
- Delete derived data
- Check deployment target matches test target
- Verify all frameworks are linked

## Next Steps

- [ ] Add UI tests with XCTest UI Testing
- [ ] Add performance tests with `measure` blocks
- [ ] Add memory leak detection
- [ ] Set up CI/CD pipeline
- [ ] Add code coverage reporting

## See Also

- [XCTest Documentation](https://developer.apple.com/documentation/xctest)
- `ios/README.md` - iOS app overview
- `docs/JUCE_BRIDGE_GUIDE.md` - OSC protocol reference

