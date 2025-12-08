#!/usr/bin/env python3
"""
Example: Using OSC Transport with Realtime Engine

This example demonstrates how to use the OscTransport to send MIDI events
from DAiW's realtime engine to a JUCE plugin (or any OSC receiver).

Usage:
    python examples/realtime_osc_example.py

Requirements:
    pip install python-osc mido

Note: This example sends events to port 9001 (default JUCE plugin OSC port).
Make sure your JUCE plugin is listening on this port.
"""

import time
from music_brain.realtime import RealtimeEngine, OscTransport
from music_brain.structure.comprehensive_engine import NoteEvent


def create_test_notes() -> list[NoteEvent]:
    """Create a simple test pattern: C major chord progression."""
    notes = []
    
    # C major chord (C-E-G) - 1 bar
    for pitch in [60, 64, 67]:  # C4, E4, G4
        notes.append(NoteEvent(
            pitch=pitch,
            velocity=80,
            start_tick=0,
            duration_ticks=960  # 1 bar at 960 PPQ
        ))
    
    # F major chord (F-A-C) - 1 bar
    for pitch in [65, 69, 72]:  # F4, A4, C5
        notes.append(NoteEvent(
            pitch=pitch,
            velocity=80,
            start_tick=960,
            duration_ticks=960
        ))
    
    # G major chord (G-B-D) - 1 bar
    for pitch in [67, 71, 74]:  # G4, B4, D5
        notes.append(NoteEvent(
            pitch=pitch,
            velocity=80,
            start_tick=1920,
            duration_ticks=960
        ))
    
    # C major chord again - 1 bar
    for pitch in [60, 64, 67]:  # C4, E4, G4
        notes.append(NoteEvent(
            pitch=pitch,
            velocity=80,
            start_tick=2880,
            duration_ticks=960
        ))
    
    return notes


def main():
    print("DAiW Realtime Engine - OSC Transport Example")
    print("=" * 50)
    
    # Create realtime engine
    print("\n1. Creating RealtimeEngine...")
    engine = RealtimeEngine(tempo_bpm=120, ppq=960, lookahead_beats=2.0)
    
    # Create OSC transport (sends to JUCE plugin on port 9001)
    print("2. Creating OscTransport (host=127.0.0.1, port=9001)...")
    try:
        osc_transport = OscTransport(
            host="127.0.0.1",
            port=9001,
            osc_address="/daiw/notes",
            auto_reconnect=True,
        )
        engine.add_transport(osc_transport)
        if osc_transport.is_connected:
            print("   ✓ OSC transport connected")
        else:
            print("   ⚠ OSC transport created but not yet connected")
            print("   (Connection will be established on first send)")
    except RuntimeError as e:
        print(f"   ✗ Error: {e}")
        print("   Install python-osc: pip install python-osc")
        return
    
    # Create test notes
    print("3. Creating test note pattern (C-F-G-C progression)...")
    notes = create_test_notes()
    print(f"   ✓ Created {len(notes)} notes")
    
    # Load notes into engine
    print("4. Loading notes into engine...")
    engine.load_note_events(notes)
    print("   ✓ Notes loaded")
    
    # Start engine
    print("5. Starting engine...")
    engine.start()
    print("   ✓ Engine started")
    print("\n   Sending events via OSC to port 9001...")
    print("   (Make sure your JUCE plugin is listening on this port)")
    print("\n   Press Ctrl+C to stop\n")
    
    # Process events in a loop
    try:
        start_time = time.time()
        last_status_check = start_time
        
        while True:
            events_emitted = engine.process_tick()
            if events_emitted > 0:
                elapsed = time.time() - start_time
                status = "✓" if osc_transport.is_connected else "⚠"
                print(f"   [{elapsed:.2f}s] {status} Emitted {events_emitted} events")
                
                # Check for connection errors
                if not osc_transport.is_connected and osc_transport.last_error:
                    print(f"   ⚠ Connection error: {osc_transport.last_error}")
            
            # Periodic status check
            now = time.time()
            if now - last_status_check > 2.0:  # Every 2 seconds
                if not osc_transport.is_connected:
                    print("   ⚠ OSC transport disconnected - attempting reconnect...")
                    osc_transport._connect()
                last_status_check = now
            
            # Check if all events have been sent
            if len(engine.scheduler) == 0:
                print("\n   ✓ All events sent!")
                break
            
            time.sleep(0.01)  # ~100Hz update rate
    
    except KeyboardInterrupt:
        print("\n\n   Stopping engine...")
    
    finally:
        engine.stop()
        engine.close()
        print("   ✓ Engine stopped and cleaned up")
        
        # Final status
        if osc_transport.last_error:
            print(f"   Last OSC error: {osc_transport.last_error}")
        
        print("\nDone!")


if __name__ == "__main__":
    main()

