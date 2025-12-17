"""
FX-MIDI Integration - Embed and display FX in MIDI tracks.

Integrates FX Engine with MIDI generation pipeline.
Provides DAW-style visualization of FX chains per track.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import mido

from kellymidicompanion_fx_engine import (
    FXType, FXChain, FXInstance, FXFactory, EmotionFXEngine,
    TrackFXState, embed_fx_in_midi, extract_fx_from_midi,
    generate_fx_cc_automation, render_channel_strip, render_mixer_view,
    MIDI_CC_FX_MAP, EMOTION_FX_PRESETS
)


# =============================================================================
# TRACK CONFIGURATION WITH FX
# =============================================================================

@dataclass
class TrackConfig:
    """Complete track configuration including instrument and FX."""
    name: str
    channel: int
    instrument_program: int = 0
    volume: int = 100
    pan: int = 64  # 0-127, 64 = center
    
    # FX configuration
    fx_chain: Optional[FXChain] = None
    emotion_fx: Optional[str] = None  # Use preset if no custom chain
    fx_intensity: float = 0.5
    
    # Display state
    muted: bool = False
    soloed: bool = False
    
    def get_fx_chain(self, engine: EmotionFXEngine) -> FXChain:
        """Get or create FX chain based on config."""
        if self.fx_chain:
            return self.fx_chain
        elif self.emotion_fx:
            return engine.get_chain_for_emotion(self.emotion_fx, self.fx_intensity)
        else:
            return FXChain(name="Default")


@dataclass
class MIDITrackWithFX:
    """MIDI track data with embedded FX information."""
    config: TrackConfig
    notes: List[Dict] = field(default_factory=list)
    cc_automation: List[Dict] = field(default_factory=list)
    
    def to_midi_track(self, ticks_per_beat: int = 480) -> mido.MidiTrack:
        """Convert to MIDO track with FX CC messages."""
        track = mido.MidiTrack()
        
        # Track name
        track.append(mido.MetaMessage('track_name', name=self.config.name))
        
        # Program change (instrument)
        track.append(mido.Message('program_change', 
                                  channel=self.config.channel, 
                                  program=self.config.instrument_program,
                                  time=0))
        
        # Volume and pan
        track.append(mido.Message('control_change',
                                  channel=self.config.channel,
                                  control=7,  # Volume
                                  value=self.config.volume,
                                  time=0))
        track.append(mido.Message('control_change',
                                  channel=self.config.channel,
                                  control=10,  # Pan
                                  value=self.config.pan,
                                  time=0))
        
        # FX CC automation (from chain)
        engine = EmotionFXEngine()
        fx_chain = self.config.get_fx_chain(engine)
        fx_messages = generate_fx_cc_automation(fx_chain, self.config.channel)
        for msg in fx_messages:
            track.append(msg)
        
        # Notes (sorted by time)
        events = []
        for note in self.notes:
            events.append({
                'time': note.get('start_tick', 0),
                'type': 'note_on',
                'note': note['pitch'],
                'velocity': note.get('velocity', 100),
            })
            events.append({
                'time': note.get('start_tick', 0) + note.get('duration_ticks', ticks_per_beat),
                'type': 'note_off',
                'note': note['pitch'],
                'velocity': 0,
            })
        
        # CC automation
        for cc in self.cc_automation:
            events.append({
                'time': cc.get('tick', 0),
                'type': 'cc',
                'control': cc['control'],
                'value': cc['value'],
            })
        
        # Sort and convert to delta time
        events.sort(key=lambda x: x['time'])
        current_time = 0
        
        for event in events:
            delta = event['time'] - current_time
            current_time = event['time']
            
            if event['type'] == 'note_on':
                track.append(mido.Message('note_on',
                                         channel=self.config.channel,
                                         note=event['note'],
                                         velocity=event['velocity'],
                                         time=delta))
            elif event['type'] == 'note_off':
                track.append(mido.Message('note_off',
                                         channel=self.config.channel,
                                         note=event['note'],
                                         velocity=0,
                                         time=delta))
            elif event['type'] == 'cc':
                track.append(mido.Message('control_change',
                                         channel=self.config.channel,
                                         control=event['control'],
                                         value=event['value'],
                                         time=delta))
        
        # End of track
        track.append(mido.MetaMessage('end_of_track', time=0))
        
        return track


# =============================================================================
# FX-AWARE MIDI GENERATOR
# =============================================================================

class FXMIDIGenerator:
    """
    Generate MIDI files with embedded FX metadata.
    
    Usage:
        gen = FXMIDIGenerator()
        
        # Add tracks with emotion-based FX
        gen.add_track("Lead", channel=0, instrument=73, emotion_fx="grief")
        gen.add_track("Bass", channel=1, instrument=32, emotion_fx="melancholy")
        
        # Add notes
        gen.add_note("Lead", pitch=60, start=0, duration=1.0)
        
        # Generate
        gen.generate("output.mid")
    """
    
    def __init__(self, tempo: float = 120, ticks_per_beat: int = 480):
        self.tempo = tempo
        self.ticks_per_beat = ticks_per_beat
        self.tracks: Dict[str, MIDITrackWithFX] = {}
        self.fx_engine = EmotionFXEngine()
    
    def add_track(
        self,
        name: str,
        channel: int,
        instrument: int = 0,
        volume: int = 100,
        pan: int = 64,
        emotion_fx: Optional[str] = None,
        fx_intensity: float = 0.5,
        custom_fx: Optional[List[Tuple[FXType, float]]] = None,
    ) -> TrackConfig:
        """Add a track with FX configuration."""
        
        # Create custom chain if specified
        fx_chain = None
        if custom_fx:
            fx_chain = FXChain(name=f"{name} FX")
            for fx_type, wet in custom_fx:
                fx_chain.add_effect(FXFactory.create(fx_type, wet_dry=wet))
        
        config = TrackConfig(
            name=name,
            channel=channel,
            instrument_program=instrument,
            volume=volume,
            pan=pan,
            emotion_fx=emotion_fx,
            fx_intensity=fx_intensity,
            fx_chain=fx_chain,
        )
        
        self.tracks[name] = MIDITrackWithFX(config=config)
        return config
    
    def add_note(
        self,
        track_name: str,
        pitch: int,
        start: float,  # beats
        duration: float,  # beats
        velocity: int = 100,
    ) -> None:
        """Add a note to a track."""
        if track_name not in self.tracks:
            raise ValueError(f"Track '{track_name}' not found")
        
        self.tracks[track_name].notes.append({
            'pitch': pitch,
            'start_tick': int(start * self.ticks_per_beat),
            'duration_ticks': int(duration * self.ticks_per_beat),
            'velocity': velocity,
        })
    
    def add_chord(
        self,
        track_name: str,
        pitches: List[int],
        start: float,
        duration: float,
        velocity: int = 100,
    ) -> None:
        """Add a chord (multiple simultaneous notes)."""
        for pitch in pitches:
            self.add_note(track_name, pitch, start, duration, velocity)
    
    def add_fx_automation(
        self,
        track_name: str,
        control: int,
        value: int,
        time: float,  # beats
    ) -> None:
        """Add FX CC automation to a track."""
        if track_name not in self.tracks:
            raise ValueError(f"Track '{track_name}' not found")
        
        self.tracks[track_name].cc_automation.append({
            'control': control,
            'value': value,
            'tick': int(time * self.ticks_per_beat),
        })
    
    def set_track_fx(
        self,
        track_name: str,
        fx_chain: FXChain
    ) -> None:
        """Override FX chain for a track."""
        if track_name not in self.tracks:
            raise ValueError(f"Track '{track_name}' not found")
        
        self.tracks[track_name].config.fx_chain = fx_chain
    
    def generate(self, output_path: str) -> mido.MidiFile:
        """Generate MIDI file with embedded FX."""
        mid = mido.MidiFile(ticks_per_beat=self.ticks_per_beat)
        
        # Tempo track
        tempo_track = mido.MidiTrack()
        tempo_track.append(mido.MetaMessage('set_tempo', 
                                            tempo=mido.bpm2tempo(self.tempo)))
        mid.tracks.append(tempo_track)
        
        # Track FX states for embedding
        fx_states = []
        
        # Add each track
        for track_data in self.tracks.values():
            midi_track = track_data.to_midi_track(self.ticks_per_beat)
            mid.tracks.append(midi_track)
            
            # Collect FX state
            fx_chain = track_data.config.get_fx_chain(self.fx_engine)
            fx_states.append(TrackFXState(
                track_name=track_data.config.name,
                channel=track_data.config.channel,
                fx_chain=fx_chain,
            ))
        
        # Embed FX metadata
        mid = embed_fx_in_midi(mid, fx_states)
        
        # Save
        mid.save(output_path)
        return mid
    
    def get_mixer_display(self) -> str:
        """Get DAW-style mixer display of all tracks with FX."""
        track_states = []
        
        for track_data in self.tracks.values():
            fx_chain = track_data.config.get_fx_chain(self.fx_engine)
            track_states.append(TrackFXState(
                track_name=track_data.config.name,
                channel=track_data.config.channel,
                fx_chain=fx_chain,
            ))
        
        return render_mixer_view(track_states)
    
    def get_track_display(self, track_name: str) -> str:
        """Get single track channel strip display."""
        if track_name not in self.tracks:
            raise ValueError(f"Track '{track_name}' not found")
        
        track_data = self.tracks[track_name]
        fx_chain = track_data.config.get_fx_chain(self.fx_engine)
        
        return render_channel_strip(
            track_data.config.name,
            track_data.config.channel,
            fx_chain,
            volume=track_data.config.volume / 127,
            pan=track_data.config.pan / 127,
            muted=track_data.config.muted,
            soloed=track_data.config.soloed,
        )


# =============================================================================
# FX DISPLAY FORMATS
# =============================================================================

def format_fx_chain_compact(chain: FXChain) -> str:
    """Compact one-line FX chain format."""
    if not chain.effects:
        return "No FX"
    
    parts = []
    for fx in chain.effects:
        status = "✓" if fx.enabled else "✗"
        wet = int(fx.wet_dry * 100)
        parts.append(f"{status}{fx.display_name}({wet}%)")
    
    return " → ".join(parts)


def format_fx_for_midi_track_name(chain: FXChain) -> str:
    """Format for embedding in MIDI track name."""
    if not chain.effects:
        return ""
    
    abbrevs = {
        "Hall Reverb": "HRV",
        "Room Reverb": "RRV",
        "Plate Reverb": "PRV",
        "1/4 Delay": "D4",
        "1/8 Delay": "D8",
        "Tape Delay": "TDL",
        "Light Chorus": "CHR",
        "Deep Chorus": "CHR+",
        "Warm Sat": "SAT",
        "Low Pass": "LP",
        "High Pass": "HP",
        "Lo-Fi": "LOFI",
        "Tape": "TAPE",
    }
    
    fx_abbrevs = []
    for fx in chain.effects[:3]:  # Max 3 for readability
        abbrev = abbrevs.get(fx.display_name, fx.display_name[:3].upper())
        fx_abbrevs.append(abbrev)
    
    return f"[{'|'.join(fx_abbrevs)}]"


def generate_fx_report(midi_path: str) -> str:
    """Generate report of FX in a MIDI file."""
    mid = mido.MidiFile(midi_path)
    fx_states = extract_fx_from_midi(mid)
    
    lines = [
        "=" * 50,
        f"FX REPORT: {midi_path}",
        "=" * 50,
        "",
    ]
    
    if not fx_states:
        lines.append("No Kelly FX metadata found in this MIDI file.")
    else:
        for state in fx_states:
            lines.append(f"Track: {state.track_name} (Ch.{state.channel})")
            lines.append(f"FX Chain: {format_fx_chain_compact(state.fx_chain)}")
            lines.append("")
            
            for fx in state.fx_chain.effects:
                lines.append(f"  {fx.display_name}:")
                lines.append(f"    Wet/Dry: {fx.wet_dry*100:.0f}%")
                for param_name, param in fx.parameters.items():
                    lines.append(f"    {param_name}: {param.get_display_value()}")
                lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# EMOTION-TO-FX QUICK API
# =============================================================================

def apply_emotion_fx_to_midi(
    input_path: str,
    output_path: str,
    track_emotions: Dict[int, str],
    intensity: float = 0.5,
) -> None:
    """
    Apply emotion-based FX to an existing MIDI file.
    
    Args:
        input_path: Source MIDI file
        output_path: Output MIDI file with FX
        track_emotions: Dict of {channel: emotion} mappings
        intensity: FX intensity (0.0-1.0)
    """
    mid = mido.MidiFile(input_path)
    engine = EmotionFXEngine()
    
    fx_states = []
    
    for i, track in enumerate(mid.tracks):
        # Get track name from meta
        track_name = f"Track {i}"
        for msg in track:
            if msg.type == 'track_name':
                track_name = msg.name
                break
        
        # Find channel for this track
        channel = None
        for msg in track:
            if hasattr(msg, 'channel'):
                channel = msg.channel
                break
        
        if channel is not None and channel in track_emotions:
            emotion = track_emotions[channel]
            fx_chain = engine.get_chain_for_emotion(emotion, intensity)
            
            fx_states.append(TrackFXState(
                track_name=track_name,
                channel=channel,
                fx_chain=fx_chain,
            ))
            
            # Insert FX CC messages
            cc_messages = generate_fx_cc_automation(fx_chain, channel)
            for j, msg in enumerate(cc_messages):
                track.insert(1 + j, msg)
    
    # Embed metadata
    mid = embed_fx_in_midi(mid, fx_states)
    mid.save(output_path)


# =============================================================================
# CLI-STYLE DISPLAY UTILITIES
# =============================================================================

class FXDisplayMode(Enum):
    COMPACT = "compact"
    CHANNEL_STRIP = "channel_strip"
    MIXER = "mixer"
    FULL = "full"


def display_track_fx(
    track_name: str,
    emotion: str,
    intensity: float = 0.5,
    mode: FXDisplayMode = FXDisplayMode.CHANNEL_STRIP,
) -> str:
    """Display FX for a track in various formats."""
    engine = EmotionFXEngine()
    chain = engine.get_chain_for_emotion(emotion, intensity)
    
    if mode == FXDisplayMode.COMPACT:
        return f"{track_name}: {format_fx_chain_compact(chain)}"
    
    elif mode == FXDisplayMode.CHANNEL_STRIP:
        return render_channel_strip(track_name, 0, chain)
    
    elif mode == FXDisplayMode.FULL:
        lines = [
            f"Track: {track_name}",
            f"Emotion: {emotion.title()}",
            f"Intensity: {intensity:.0%}",
            "",
            chain.to_display_string(),
        ]
        return "\n".join(lines)
    
    return chain.to_display_string(compact=True)


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FX-MIDI Integration Demo")
    print("=" * 60)
    
    # Create generator
    gen = FXMIDIGenerator(tempo=82)
    
    # Add tracks with emotion-based FX
    gen.add_track(
        name="Lead Piano",
        channel=0,
        instrument=0,  # Acoustic Grand Piano
        emotion_fx="grief",
        fx_intensity=0.7,
    )
    
    gen.add_track(
        name="Strings",
        channel=1,
        instrument=48,  # String Ensemble
        emotion_fx="melancholy",
        fx_intensity=0.6,
    )
    
    gen.add_track(
        name="Bass",
        channel=2,
        instrument=32,  # Acoustic Bass
        emotion_fx="peace",
        fx_intensity=0.4,
    )
    
    # Add track with custom FX
    gen.add_track(
        name="Ambient Pad",
        channel=3,
        instrument=89,  # Warm Pad
        custom_fx=[
            (FXType.REVERB_SHIMMER, 0.7),
            (FXType.DELAY_PINGPONG, 0.4),
            (FXType.CHORUS_DEEP, 0.3),
        ],
    )
    
    # Add some notes (F-C-Dm-Bbm at 82 BPM)
    # F major: F3, A3, C4
    gen.add_chord("Lead Piano", [53, 57, 60], start=0, duration=4, velocity=80)
    # C major: C3, E3, G3
    gen.add_chord("Lead Piano", [48, 52, 55], start=4, duration=4, velocity=75)
    # Dm: D3, F3, A3
    gen.add_chord("Lead Piano", [50, 53, 57], start=8, duration=4, velocity=70)
    # Bbm: Bb2, Db3, F3
    gen.add_chord("Lead Piano", [46, 49, 53], start=12, duration=4, velocity=85)
    
    # Bass notes
    gen.add_note("Bass", 29, start=0, duration=4, velocity=90)   # F1
    gen.add_note("Bass", 24, start=4, duration=4, velocity=85)   # C1
    gen.add_note("Bass", 26, start=8, duration=4, velocity=80)   # D1
    gen.add_note("Bass", 22, start=12, duration=4, velocity=95)  # Bb0
    
    # Display mixer view
    print("\n" + "=" * 60)
    print("MIXER VIEW (DAW-Style)")
    print("=" * 60)
    print(gen.get_mixer_display())
    
    # Display individual track
    print("\n" + "=" * 60)
    print("SINGLE TRACK VIEW")
    print("=" * 60)
    print(gen.get_track_display("Lead Piano"))
    
    # Generate MIDI
    output_path = "/home/claude/demo_with_fx.mid"
    gen.generate(output_path)
    print(f"\n✓ Generated: {output_path}")
    
    # Generate FX report
    print("\n" + "=" * 60)
    print("FX REPORT")
    print("=" * 60)
    print(generate_fx_report(output_path))
    
    # Demo different display modes
    print("\n" + "=" * 60)
    print("DISPLAY MODES")
    print("=" * 60)
    
    for mode in FXDisplayMode:
        print(f"\n--- {mode.value.upper()} ---")
        print(display_track_fx("Vocal", "grief", 0.8, mode))
