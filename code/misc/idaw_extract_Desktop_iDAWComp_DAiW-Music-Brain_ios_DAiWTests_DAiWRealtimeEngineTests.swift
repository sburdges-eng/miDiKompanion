//
//  DAiWRealtimeEngineTests.swift
//  DAiWTests
//
//  Tests for realtime MIDI event scheduling and playback
//

import XCTest
import AVFoundation
@testable import DAiW

class DAiWRealtimeEngineTests: XCTestCase {
    
    var engine: RealtimeEngine?
    var midiScheduler: MIDIScheduler?
    
    override func setUp() {
        super.setUp()
        engine = RealtimeEngine(tempoBPM: 120, ppq: 960)
        midiScheduler = MIDIScheduler()
    }
    
    override func tearDown() {
        engine?.stop()
        engine = nil
        midiScheduler = nil
        super.tearDown()
    }
    
    // MARK: - Engine Initialization Tests
    
    func testEngineInitialization() {
        XCTAssertNotNil(engine, "Engine should be initialized")
        XCTAssertEqual(engine?.tempoBPM, 120.0, accuracy: 0.1)
        XCTAssertEqual(engine?.ppq, 960)
    }
    
    func testEngineTempoChange() {
        engine?.setTempo(140.0)
        XCTAssertEqual(engine?.tempoBPM, 140.0, accuracy: 0.1)
    }
    
    func testEngineInvalidTempo() {
        // Test that invalid tempo is rejected
        let initialTempo = engine?.tempoBPM
        engine?.setTempo(-10.0)  // Invalid
        XCTAssertEqual(engine?.tempoBPM, initialTempo, "Tempo should not change")
    }
    
    // MARK: - Event Scheduling Tests
    
    func testScheduleNoteEvent() {
        let note = MIDINoteEvent(
            pitch: 60,
            velocity: 80,
            startTick: 0,
            durationTicks: 960
        )
        
        engine?.scheduleNote(note, channel: 0)
        XCTAssertEqual(engine?.scheduledEventCount, 1, "Should have one scheduled event")
    }
    
    func testScheduleMultipleEvents() {
        let notes = [
            MIDINoteEvent(pitch: 60, velocity: 80, startTick: 0, durationTicks: 960),
            MIDINoteEvent(pitch: 64, velocity: 75, startTick: 480, durationTicks: 480),
            MIDINoteEvent(pitch: 67, velocity: 70, startTick: 960, durationTicks: 960),
        ]
        
        for note in notes {
            engine?.scheduleNote(note, channel: 0)
        }
        
        XCTAssertEqual(engine?.scheduledEventCount, 3, "Should have three scheduled events")
    }
    
    func testEventOrdering() {
        // Schedule events out of order
        engine?.scheduleNote(
            MIDINoteEvent(pitch: 67, velocity: 70, startTick: 960, durationTicks: 960),
            channel: 0
        )
        engine?.scheduleNote(
            MIDINoteEvent(pitch: 60, velocity: 80, startTick: 0, durationTicks: 960),
            channel: 0
        )
        
        // Events should be ordered by startTick
        let firstEvent = engine?.nextScheduledEvent
        XCTAssertEqual(firstEvent?.startTick, 0, "First event should be at tick 0")
    }
    
    // MARK: - Playback Tests
    
    func testEngineStartStop() {
        XCTAssertFalse(engine?.isRunning ?? true, "Engine should not be running initially")
        
        engine?.start()
        XCTAssertTrue(engine?.isRunning ?? false, "Engine should be running after start")
        
        engine?.stop()
        XCTAssertFalse(engine?.isRunning ?? true, "Engine should not be running after stop")
    }
    
    func testProcessTick() {
        let note = MIDINoteEvent(
            pitch: 60,
            velocity: 80,
            startTick: 0,
            durationTicks: 960
        )
        
        engine?.scheduleNote(note, channel: 0)
        engine?.start()
        
        let eventsEmitted = engine?.processTick()
        XCTAssertGreaterThan(eventsEmitted ?? 0, 0, "Should emit events when due")
    }
    
    func testLookaheadWindow() {
        // Schedule event beyond lookahead window
        let futureNote = MIDINoteEvent(
            pitch: 60,
            velocity: 80,
            startTick: 5000,  // Far in future
            durationTicks: 960
        )
        
        engine?.scheduleNote(futureNote, channel: 0)
        engine?.start()
        
        let eventsEmitted = engine?.processTick()
        XCTAssertEqual(eventsEmitted, 0, "Should not emit events beyond lookahead")
    }
    
    // MARK: - MIDI Integration Tests
    
    func testMIDIOutput() {
        // Test that scheduled events are converted to MIDI messages
        let note = MIDINoteEvent(
            pitch: 60,
            velocity: 80,
            startTick: 0,
            durationTicks: 960
        )
        
        engine?.scheduleNote(note, channel: 0)
        engine?.start()
        
        let expectation = XCTestExpectation(description: "MIDI message received")
        
        midiScheduler?.onMIDIMessage = { message in
            if message.status == 0x90 {  // Note On
                XCTAssertEqual(message.data1, 60, "Pitch should match")
                XCTAssertEqual(message.data2, 80, "Velocity should match")
                expectation.fulfill()
            }
        }
        
        engine?.processTick()
        
        wait(for: [expectation], timeout: 2.0)
    }
    
    func testMIDINoteOff() {
        // Test that note_off is scheduled after duration
        let note = MIDINoteEvent(
            pitch: 60,
            velocity: 80,
            startTick: 0,
            durationTicks: 960
        )
        
        engine?.scheduleNote(note, channel: 0)
        engine?.start()
        
        var noteOnReceived = false
        var noteOffReceived = false
        
        midiScheduler?.onMIDIMessage = { message in
            if message.status == 0x90 {  // Note On
                noteOnReceived = true
            } else if message.status == 0x80 {  // Note Off
                noteOffReceived = true
            }
        }
        
        // Process initial tick (note on)
        engine?.processTick()
        XCTAssertTrue(noteOnReceived, "Note On should be sent")
        
        // Advance time and process (note off)
        engine?.advanceTime(ticks: 960)
        engine?.processTick()
        XCTAssertTrue(noteOffReceived, "Note Off should be sent after duration")
    }
    
    // MARK: - Performance Tests
    
    func testHighEventRate() {
        // Test scheduling many events quickly
        measure {
            for i in 0..<1000 {
                let note = MIDINoteEvent(
                    pitch: 60 + (i % 12),
                    velocity: 80,
                    startTick: Int64(i * 480),
                    durationTicks: 480
                )
                engine?.scheduleNote(note, channel: 0)
            }
        }
    }
    
    func testProcessTickPerformance() {
        // Schedule many events
        for i in 0..<100 {
            let note = MIDINoteEvent(
                pitch: 60,
                velocity: 80,
                startTick: Int64(i * 480),
                durationTicks: 480
            )
            engine?.scheduleNote(note, channel: 0)
        }
        
        engine?.start()
        
        measure {
            for _ in 0..<1000 {
                _ = engine?.processTick()
            }
        }
    }
}

