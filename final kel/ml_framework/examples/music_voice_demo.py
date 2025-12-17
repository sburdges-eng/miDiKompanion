"""
Music and Voice Generation Demo

Demonstrates music and voice synthesis from emotional models.
"""

import numpy as np
import matplotlib.pyplot as plt
from cif_las_qef.emotion_models import (
    VADState, EmotionBasis,
    EmotionToMusicMapper, EmotionalMusicField,
    EmotionToVoiceMapper, QuantumVoiceField, VoiceMorphing
)


def demo_music_generation():
    """Demonstrate music generation from emotions."""
    print("=== Music Generation Demo ===\n")
    
    # Create VAD state
    vad = VADState(valence=0.6, arousal=0.7, dominance=0.4)
    
    # Music mapper
    mapper = EmotionToMusicMapper()
    
    # Get chord
    chord = mapper.vad_to_chord(vad)
    print(f"VAD {vad.valence:.2f}, {vad.arousal:.2f}, {vad.dominance:.2f}")
    print(f"Generated Chord (MIDI notes): {chord}")
    
    # Get tempo and volume
    tempo = mapper.vad_to_tempo(vad)
    volume = mapper.vad_to_volume(vad)
    print(f"Tempo: {tempo:.1f} BPM")
    print(f"Volume: {volume:.2f}\n")
    
    # Generate music
    music_field = EmotionalMusicField()
    waveform, params = music_field.generate_from_vad(vad, duration=2.0)
    
    print(f"Generated waveform: {len(waveform)} samples")
    print(f"Frequencies: {[f'{f:.1f} Hz' for f in params.frequencies]}")
    print(f"Mode: {params.mode}\n")
    
    # Plot waveform
    try:
        t = np.linspace(0, 2.0, len(waveform))
        plt.figure(figsize=(12, 4))
        plt.plot(t[:1000], waveform[:1000])  # First 1000 samples
        plt.title("Generated Music Waveform (first 1000 samples)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("music_waveform.png", dpi=150)
        print("Saved: music_waveform.png\n")
    except Exception as e:
        print(f"Plotting failed: {e}\n")


def demo_voice_synthesis():
    """Demonstrate voice synthesis from emotions."""
    print("=== Voice Synthesis Demo ===\n")
    
    # Create VAD states for different emotions
    joy_vad = VADState(valence=0.8, arousal=0.7, dominance=0.5)
    sadness_vad = VADState(valence=-0.7, arousal=0.3, dominance=-0.2)
    
    # Voice mapper
    mapper = EmotionToVoiceMapper()
    
    # Get voice parameters
    joy_params = mapper.vad_to_voice_parameters(joy_vad)
    sadness_params = mapper.vad_to_voice_parameters(sadness_vad)
    
    print("Joy Voice Parameters:")
    print(f"  Pitch (f0): {joy_params.f0:.1f} Hz")
    print(f"  Volume: {joy_params.amplitude:.2f}")
    print(f"  Formants: {[f'{f:.0f}' for f in joy_params.formants]} Hz")
    print(f"  Vibrato Rate: {joy_params.vibrato_rate:.1f} Hz")
    print(f"  Vibrato Depth: {joy_params.vibrato_depth:.2f} semitones\n")
    
    print("Sadness Voice Parameters:")
    print(f"  Pitch (f0): {sadness_params.f0:.1f} Hz")
    print(f"  Volume: {sadness_params.amplitude:.2f}")
    print(f"  Formants: {[f'{f:.0f}' for f in sadness_params.formants]} Hz")
    print(f"  Vibrato Rate: {sadness_params.vibrato_rate:.1f} Hz")
    print(f"  Vibrato Depth: {sadness_params.vibrato_depth:.2f} semitones\n")
    
    # Generate voice waveforms
    quantum_voice = QuantumVoiceField()
    
    joy_waveform = quantum_voice.generate_voice_waveform(joy_params, duration=1.0)
    sadness_waveform = quantum_voice.generate_voice_waveform(sadness_params, duration=1.0)
    
    print(f"Generated Joy voice: {len(joy_waveform)} samples")
    print(f"Generated Sadness voice: {len(sadness_waveform)} samples\n")
    
    # Plot waveforms
    try:
        t = np.linspace(0, 1.0, len(joy_waveform))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        ax1.plot(t[:2000], joy_waveform[:2000])
        ax1.set_title("Joy Voice Waveform")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(t[:2000], sadness_waveform[:2000])
        ax2.set_title("Sadness Voice Waveform")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("voice_waveforms.png", dpi=150)
        print("Saved: voice_waveforms.png\n")
    except Exception as e:
        print(f"Plotting failed: {e}\n")


def demo_voice_morphing():
    """Demonstrate voice morphing between emotions."""
    print("=== Voice Morphing Demo ===\n")
    
    # Create VAD states
    calm_vad = VADState(valence=0.2, arousal=0.3, dominance=0.1)
    angry_vad = VADState(valence=-0.5, arousal=0.9, dominance=0.7)
    
    # Voice morphing
    morphing = VoiceMorphing()
    
    # Generate morphing sequence
    waveform, param_sequence = morphing.generate_morphing_sequence(
        calm_vad,
        angry_vad,
        duration=3.0,
        morph_frequency=0.3
    )
    
    print(f"Morphing from Calm to Angry")
    print(f"Generated waveform: {len(waveform)} samples")
    print(f"Parameter sequence: {len(param_sequence)} steps")
    print(f"Final pitch: {param_sequence[-1].f0:.1f} Hz")
    print(f"Final volume: {param_sequence[-1].amplitude:.2f}\n")
    
    # Plot morphing
    try:
        t = np.linspace(0, 3.0, len(waveform))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        # Waveform
        ax1.plot(t, waveform, alpha=0.7)
        ax1.set_title("Voice Morphing: Calm → Angry")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)
        
        # Parameter evolution
        pitches = [p.f0 for p in param_sequence[::100]]  # Sample every 100th
        volumes = [p.amplitude for p in param_sequence[::100]]
        t_params = np.linspace(0, 3.0, len(pitches))
        
        ax2_twin = ax2.twinx()
        line1 = ax2.plot(t_params, pitches, 'b-', label='Pitch (f0)')
        line2 = ax2_twin.plot(t_params, volumes, 'r-', label='Volume')
        
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Pitch (Hz)", color='b')
        ax2_twin.set_ylabel("Volume", color='r')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        ax2.set_title("Voice Parameter Evolution")
        ax2.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        plt.savefig("voice_morphing.png", dpi=150)
        print("Saved: voice_morphing.png\n")
    except Exception as e:
        print(f"Plotting failed: {e}\n")


def demo_quantum_voice():
    """Demonstrate quantum voice field."""
    print("=== Quantum Voice Field Demo ===\n")
    
    from cif_las_qef.emotion_models import QuantumEmotionalField, QuantumEmotionalVoiceField
    
    # Create quantum superposition
    qf = QuantumEmotionalField()
    superposition = qf.create_superposition()
    
    # Quantum voice field
    qevf = QuantumEmotionalVoiceField()
    
    # Generate voice from superposition
    waveform, field_params = qevf.generate_voice_field(
        superposition,
        duration=1.5
    )
    
    print("Quantum Voice Field Parameters:")
    print(f"  Pitch (f0): {field_params['f0']:.1f} Hz")
    print(f"  Formants: {[f'{f:.0f}' for f in field_params['formants']]} Hz")
    print(f"  Amplitude: {field_params['amplitude']:.2f}")
    print(f"  Emotion Probabilities: {[f'{p:.2f}' for p in field_params['probabilities'][:4]]}\n")
    
    print(f"Generated waveform: {len(waveform)} samples\n")


def main():
    """Run all demos."""
    print("=" * 60)
    print("Music and Voice Generation from Emotions")
    print("=" * 60)
    print()
    
    demo_music_generation()
    demo_voice_synthesis()
    demo_voice_morphing()
    demo_quantum_voice()
    
    print("=" * 60)
    print("All demos complete!")
    print("=" * 60)
    print("\nNote: To play audio, install sounddevice:")
    print("  pip install sounddevice")
    print("Then use: sd.play(waveform, sample_rate)")


if __name__ == "__main__":
    main()
