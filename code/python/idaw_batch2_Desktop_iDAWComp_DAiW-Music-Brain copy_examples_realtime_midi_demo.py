"""
Real-time MIDI Processing Demo

Example usage of the real-time MIDI processor for:
- Live chord detection
- Groove analysis
- MIDI transformation
"""

import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from music_brain.realtime import (
    RealtimeMidiProcessor,
    MidiProcessorConfig,
    create_transpose_transformer,
    create_velocity_scale_transformer,
    create_humanize_transformer,
)


def chord_callback(chord, notes):
    """Callback for chord detection events."""
    if chord:
        print(f"🎹 Chord detected: {chord.name} (notes: {notes})")
    else:
        print(f"🎵 Notes played: {notes} (no clear chord)")


def groove_callback(groove_data):
    """Callback for groove analysis updates."""
    print(
        f"🥁 Groove: deviation={groove_data['mean_deviation_ms']:.1f}ms, "
        f"vel_mean={groove_data['velocity_mean']:.0f}, "
        f"notes={groove_data['note_count']}"
    )


def main():
    """Main demo function."""
    print("=" * 60)
    print("DAiW Real-time MIDI Processing Demo")
    print("=" * 60)
    print()
    
    # List available ports
    config = MidiProcessorConfig()
    processor = RealtimeMidiProcessor(config)
    
    input_ports = processor.list_input_ports()
    output_ports = processor.list_output_ports()
    
    print("Available MIDI Input Ports:")
    if input_ports:
        for i, port in enumerate(input_ports):
            print(f"  {i}: {port}")
    else:
        print("  (none available)")
        return
    
    print()
    print("Available MIDI Output Ports:")
    if output_ports:
        for i, port in enumerate(output_ports):
            print(f"  {i}: {port}")
    else:
        print("  (none available)")
    
    print()
    
    # Configure processor
    config = MidiProcessorConfig(
        input_port_name=None,  # Auto-select first available
        output_port_name=None,  # No output (or set to specific port)
        enable_chord_detection=True,
        chord_window_ms=200.0,
        enable_groove_analysis=True,
        enable_transformation=False,  # Set to True to enable transformations
    )
    
    processor = RealtimeMidiProcessor(config)
    processor.set_chord_callback(chord_callback)
    processor.set_groove_callback(groove_callback)
    
    # Example: Add transformation (uncomment to enable)
    # transpose = create_transpose_transformer(semitones=5)  # Transpose up 5 semitones
    # processor.set_transform_callback(transpose)
    
    # velocity_scale = create_velocity_scale_transformer(scale=0.8)  # Reduce velocity
    # processor.set_transform_callback(velocity_scale)
    
    # humanize = create_humanize_transformer(velocity_variation=10)
    # processor.set_transform_callback(humanize)
    
    print("Starting real-time MIDI processing...")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        # Start processing
        if not processor.start():
            print("Failed to start MIDI processing")
            return
        
        # Keep running
        while True:
            time.sleep(0.1)
            
            # Optional: Print active notes periodically
            active = processor.get_active_notes()
            if active:
                print(f"Active notes: {active}")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        processor.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()

