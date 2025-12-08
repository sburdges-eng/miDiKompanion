//
//  DAiWIntegrationTests.swift
//  DAiWTests
//
//  End-to-end integration tests for iOS app with Python brain server
//

import XCTest
import Combine
@testable import DAiW

class DAiWIntegrationTests: XCTestCase {
    
    var brainServer: BrainServerConnection?
    var cancellables: Set<AnyCancellable> = []
    
    override func setUp() {
        super.setUp()
        brainServer = BrainServerConnection(host: "127.0.0.1", port: 9000)
    }
    
    override func tearDown() {
        brainServer?.disconnect()
        brainServer = nil
        cancellables.removeAll()
        super.tearDown()
    }
    
    // MARK: - Full Workflow Tests
    
    func testGenerateMIDIFromIntent() {
        // Test complete workflow: intent → generation → MIDI playback
        let expectation = XCTestExpectation(description: "MIDI generated and played")
        
        var receivedMIDI: [MIDINoteEvent] = []
        
        brainServer?.generateMIDI(
            intent: "I feel deep grief and loss",
            motivation: 7.0,
            chaos: 5.0,
            vulnerability: 6.0
        )
        .sink(
            receiveCompletion: { completion in
                if case .failure(let error) = completion {
                    XCTFail("Generation failed: \(error)")
                }
            },
            receiveValue: { midiEvents in
                receivedMIDI = midiEvents
                XCTAssertGreaterThan(midiEvents.count, 0, "Should receive MIDI events")
                expectation.fulfill()
            }
        )
        .store(in: &cancellables)
        
        wait(for: [expectation], timeout: 30.0)
    }
    
    func testRealTimePlayback() {
        // Test real-time playback of generated MIDI
        let expectation = XCTestExpectation(description: "MIDI played in real-time")
        
        var playbackStarted = false
        var playbackCompleted = false
        
        brainServer?.generateMIDI(
            intent: "test",
            motivation: 5.0,
            chaos: 5.0,
            vulnerability: 5.0
        )
        .flatMap { events -> AnyPublisher<Bool, Error> in
            // Start real-time playback
            playbackStarted = true
            return self.playMIDIEvents(events)
        }
        .sink(
            receiveCompletion: { completion in
                if case .failure(let error) = completion {
                    XCTFail("Playback failed: \(error)")
                } else {
                    playbackCompleted = true
                    expectation.fulfill()
                }
            },
            receiveValue: { _ in }
        )
        .store(in: &cancellables)
        
        wait(for: [expectation], timeout: 30.0)
        XCTAssertTrue(playbackStarted, "Playback should start")
        XCTAssertTrue(playbackCompleted, "Playback should complete")
    }
    
    func testMultipleGenerations() {
        // Test generating multiple pieces in sequence
        let expectation = XCTestExpectation(description: "Multiple generations")
        expectation.expectedFulfillmentCount = 3
        
        let intents = [
            "I feel deep grief",
            "I feel angry and frustrated",
            "I feel hopeful and tender"
        ]
        
        for intent in intents {
            brainServer?.generateMIDI(
                intent: intent,
                motivation: 5.0,
                chaos: 5.0,
                vulnerability: 5.0
            )
            .sink(
                receiveCompletion: { _ in },
                receiveValue: { events in
                    XCTAssertGreaterThan(events.count, 0)
                    expectation.fulfill()
                }
            )
            .store(in: &cancellables)
        }
        
        wait(for: [expectation], timeout: 60.0)
    }
    
    // MARK: - Error Handling Tests
    
    func testServerUnavailable() {
        // Test behavior when brain server is not running
        let invalidServer = BrainServerConnection(host: "127.0.0.1", port: 9999)
        let expectation = XCTestExpectation(description: "Error handled gracefully")
        
        invalidServer.generateMIDI(
            intent: "test",
            motivation: 5.0,
            chaos: 5.0,
            vulnerability: 5.0
        )
        .sink(
            receiveCompletion: { completion in
                if case .failure = completion {
                    expectation.fulfill()
                }
            },
            receiveValue: { _ in
                XCTFail("Should not receive value when server unavailable")
            }
        )
        .store(in: &cancellables)
        
        wait(for: [expectation], timeout: 10.0)
    }
    
    func testInvalidIntent() {
        // Test handling of invalid intent data
        let expectation = XCTestExpectation(description: "Invalid intent handled")
        
        brainServer?.generateMIDI(
            intent: "",  // Empty intent
            motivation: 5.0,
            chaos: 5.0,
            vulnerability: 5.0
        )
        .sink(
            receiveCompletion: { completion in
                // Should either succeed with default or fail gracefully
                expectation.fulfill()
            },
            receiveValue: { _ in
                expectation.fulfill()
            }
        )
        .store(in: &cancellables)
        
        wait(for: [expectation], timeout: 15.0)
    }
    
    // MARK: - Helper Methods
    
    private func playMIDIEvents(_ events: [MIDINoteEvent]) -> AnyPublisher<Bool, Error> {
        return Future { promise in
            // Simulate playback
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                promise(.success(true))
            }
        }
        .eraseToAnyPublisher()
    }
}

