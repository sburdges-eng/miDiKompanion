//
//  DAiWRealtimeEngine.swift
//  DAiW
//
//  Realtime MIDI event scheduling engine for iOS
//

import Foundation
import AVFoundation

/// MIDI note event
struct MIDINoteEvent {
    let pitch: Int
    let velocity: Int
    let startTick: Int64
    let durationTicks: Int64
}

/// Realtime engine for scheduling and playing MIDI events
class RealtimeEngine {
    private(set) var tempoBPM: Double
    private(set) var ppq: Int
    private(set) var isRunning: Bool = false
    
    private var scheduledEvents: [ScheduledMIDIEvent] = []
    private var currentTick: Int64 = 0
    private var lookaheadTicks: Int64
    
    init(tempoBPM: Double = 120.0, ppq: Int = 960, lookaheadBeats: Double = 2.0) {
        self.tempoBPM = tempoBPM
        self.ppq = ppq
        self.lookaheadTicks = Int64(lookaheadBeats * Double(ppq))
    }
    
    func setTempo(_ bpm: Double) {
        guard bpm > 0 else { return }
        tempoBPM = bpm
    }
    
    func scheduleNote(_ note: MIDINoteEvent, channel: Int) {
        let event = ScheduledMIDIEvent(
            note: note,
            channel: channel,
            startTick: note.startTick
        )
        scheduledEvents.append(event)
        scheduledEvents.sort { $0.startTick < $1.startTick }
    }
    
    func start() {
        isRunning = true
        currentTick = 0
    }
    
    func stop() {
        isRunning = false
    }
    
    func processTick() -> Int {
        guard isRunning else { return 0 }
        
        let windowEnd = currentTick + lookaheadTicks
        var emitted = 0
        
        scheduledEvents.removeAll { event in
            if event.startTick <= windowEnd {
                emitMIDIEvent(event)
                emitted += 1
                return true
            }
            return false
        }
        
        return emitted
    }
    
    func advanceTime(ticks: Int64) {
        currentTick += ticks
    }
    
    var scheduledEventCount: Int {
        scheduledEvents.count
    }
    
    var nextScheduledEvent: ScheduledMIDIEvent? {
        scheduledEvents.first
    }
    
    private func emitMIDIEvent(_ event: ScheduledMIDIEvent) {
        // Emit MIDI note on
        // In production, send to MIDI output or audio engine
    }
}

private struct ScheduledMIDIEvent {
    let note: MIDINoteEvent
    let channel: Int
    let startTick: Int64
}

