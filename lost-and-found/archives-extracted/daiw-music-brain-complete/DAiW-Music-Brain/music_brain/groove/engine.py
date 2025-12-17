"""
DAiW Groove Engine
==================

Humanization and groove application for MIDI.
Implements the "Drunken Drummer" concept - systematic deviation from the grid.

Philosophy: "Feel isn't random—it's systematic deviation from perfection."
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

try:
    import mido
    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class TimingFeel(Enum):
    """Where notes sit relative to the beat."""
    BEHIND = "behind"      # Laid back (grief, nostalgia)
    ON = "on"              # Precise, neutral
    AHEAD = "ahead"        # Pushing (anxious, urgent)


@dataclass
class GrooveTemplate:
    """Template for groove characteristics."""
    name: str
    tempo_bpm: int = 120
    swing_percentage: float = 50.0  # 50 = straight, 66 = heavy swing
    push_pull: Dict[str, int] = field(default_factory=dict)  # Instrument -> ms offset
    velocity_map: Dict[str, int] = field(default_factory=dict)  # Instrument -> base velocity
    humanize_amount: float = 0.5  # 0-1 scale
    timing_feel: TimingFeel = TimingFeel.ON


@dataclass
class MidiNoteEvent:
    """Represents a MIDI note event."""
    start_tick: int
    duration_tick: int
    pitch: int
    velocity: int
    channel: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# PRESET GROOVE TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

GROOVE_PRESETS = {
    "funk": GrooveTemplate(
        name="Funk",
        swing_percentage=55,
        push_pull={"kick": 0, "snare": -5, "hihat": -10},
        velocity_map={"kick": 100, "snare": 110, "hihat": 70},
        humanize_amount=0.3,
        timing_feel=TimingFeel.ON
    ),
    "boom_bap": GrooveTemplate(
        name="Boom Bap",
        swing_percentage=58,
        push_pull={"kick": 8, "snare": -3, "hihat": -12},
        velocity_map={"kick": 95, "snare": 105, "hihat": 60},
        humanize_amount=0.4,
        timing_feel=TimingFeel.BEHIND
    ),
    "dilla": GrooveTemplate(
        name="Dilla",
        swing_percentage=62,
        push_pull={"kick": 15, "snare": -8, "hihat": -20},
        velocity_map={"kick": 90, "snare": 100, "hihat": 55},
        humanize_amount=0.6,
        timing_feel=TimingFeel.BEHIND
    ),
    "lofi": GrooveTemplate(
        name="Lo-Fi",
        swing_percentage=54,
        push_pull={"kick": 10, "snare": 5, "hihat": -8},
        velocity_map={"kick": 85, "snare": 90, "hihat": 50},
        humanize_amount=0.7,
        timing_feel=TimingFeel.BEHIND
    ),
    "straight": GrooveTemplate(
        name="Straight",
        swing_percentage=50,
        push_pull={"kick": 0, "snare": 0, "hihat": 0},
        velocity_map={"kick": 100, "snare": 100, "hihat": 80},
        humanize_amount=0.1,
        timing_feel=TimingFeel.ON
    ),
    "driving": GrooveTemplate(
        name="Driving",
        swing_percentage=50,
        push_pull={"kick": -5, "snare": -3, "hihat": -8},
        velocity_map={"kick": 110, "snare": 115, "hihat": 85},
        humanize_amount=0.2,
        timing_feel=TimingFeel.AHEAD
    ),
}

# Emotional presets for the "Drunken Drummer"
EMOTIONAL_PRESETS = {
    "grief": GrooveTemplate(
        name="Grief",
        swing_percentage=52,
        push_pull={"kick": 12, "snare": 8, "hihat": -5},
        velocity_map={"kick": 75, "snare": 80, "hihat": 45},
        humanize_amount=0.5,
        timing_feel=TimingFeel.BEHIND
    ),
    "rage": GrooveTemplate(
        name="Rage",
        swing_percentage=48,
        push_pull={"kick": -8, "snare": -5, "hihat": -10},
        velocity_map={"kick": 120, "snare": 125, "hihat": 100},
        humanize_amount=0.3,
        timing_feel=TimingFeel.AHEAD
    ),
    "anxiety": GrooveTemplate(
        name="Anxiety",
        swing_percentage=51,
        push_pull={"kick": 5, "snare": -8, "hihat": 3},
        velocity_map={"kick": 95, "snare": 90, "hihat": 75},
        humanize_amount=0.6,
        timing_feel=TimingFeel.AHEAD
    ),
    "nostalgia": GrooveTemplate(
        name="Nostalgia",
        swing_percentage=56,
        push_pull={"kick": 8, "snare": 5, "hihat": -3},
        velocity_map={"kick": 80, "snare": 85, "hihat": 55},
        humanize_amount=0.5,
        timing_feel=TimingFeel.BEHIND
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# HUMANIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def drunken_drummer(
    events: List[MidiNoteEvent],
    vulnerability_scale: float,
    max_jitter_ticks: int = 30,
    seed: Optional[int] = None
) -> List[MidiNoteEvent]:
    """
    Apply "drunken drummer" humanization.
    
    High vulnerability → small jitter (seeking control)
    Low vulnerability → bigger jitter (confident looseness)
    
    Args:
        events: List of MIDI note events
        vulnerability_scale: 0.0 (loose) to 1.0 (tight)
        max_jitter_ticks: Maximum timing deviation in ticks
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    
    # Invert vulnerability: high vulnerability = less jitter
    jitter_scale = 1.0 - vulnerability_scale
    actual_jitter = int(max_jitter_ticks * jitter_scale)
    
    humanized = []
    for event in events:
        # Apply timing jitter
        timing_offset = random.randint(-actual_jitter, actual_jitter)
        new_start = max(0, event.start_tick + timing_offset)
        
        # Apply velocity variation (smaller range for high vulnerability)
        vel_range = int(20 * jitter_scale)
        vel_offset = random.randint(-vel_range, vel_range)
        new_velocity = max(1, min(127, event.velocity + vel_offset))
        
        humanized.append(MidiNoteEvent(
            start_tick=new_start,
            duration_tick=event.duration_tick,
            pitch=event.pitch,
            velocity=new_velocity,
            channel=event.channel
        ))
    
    return humanized


def apply_swing(
    events: List[MidiNoteEvent],
    swing_percentage: float = 55.0,
    ppqn: int = 480
) -> List[MidiNoteEvent]:
    """
    Apply swing feel to events.
    
    Args:
        events: MIDI events
        swing_percentage: 50 = straight, 66 = heavy triplet swing
        ppqn: Pulses per quarter note
    """
    if swing_percentage == 50:
        return events
    
    eighth_note = ppqn // 2
    swing_offset = int((swing_percentage - 50) / 50 * eighth_note * 0.5)
    
    swung = []
    for event in events:
        # Check if on an off-beat eighth note
        position_in_beat = event.start_tick % ppqn
        is_offbeat = eighth_note - 20 < position_in_beat < eighth_note + 20
        
        if is_offbeat:
            new_start = event.start_tick + swing_offset
        else:
            new_start = event.start_tick
        
        swung.append(MidiNoteEvent(
            start_tick=new_start,
            duration_tick=event.duration_tick,
            pitch=event.pitch,
            velocity=event.velocity,
            channel=event.channel
        ))
    
    return swung


def apply_push_pull(
    events: List[MidiNoteEvent],
    push_pull_ms: int,
    tempo_bpm: int,
    ppqn: int = 480
) -> List[MidiNoteEvent]:
    """
    Apply push/pull (timing offset) to events.
    
    Args:
        events: MIDI events
        push_pull_ms: Milliseconds to shift (negative = ahead, positive = behind)
        tempo_bpm: Tempo for ms-to-tick conversion
        ppqn: Pulses per quarter note
    """
    if push_pull_ms == 0:
        return events
    
    # Convert ms to ticks
    ms_per_beat = 60000 / tempo_bpm
    ticks_per_ms = ppqn / ms_per_beat
    offset_ticks = int(push_pull_ms * ticks_per_ms)
    
    shifted = []
    for event in events:
        new_start = max(0, event.start_tick + offset_ticks)
        shifted.append(MidiNoteEvent(
            start_tick=new_start,
            duration_tick=event.duration_tick,
            pitch=event.pitch,
            velocity=event.velocity,
            channel=event.channel
        ))
    
    return shifted


# ═══════════════════════════════════════════════════════════════════════════════
# GROOVE APPLICATOR
# ═══════════════════════════════════════════════════════════════════════════════

class GrooveApplicator:
    """Applies groove templates to MIDI data."""
    
    # General MIDI drum map
    GM_DRUM_MAP = {
        36: "kick", 35: "kick",
        38: "snare", 40: "snare",
        42: "hihat", 44: "hihat", 46: "hihat",
        49: "crash", 57: "crash",
        51: "ride", 59: "ride",
    }
    
    def __init__(self, ppqn: int = 480):
        self.ppqn = ppqn
    
    def apply_groove(
        self,
        input_path: str,
        output_path: str,
        template: GrooveTemplate,
        intensity: float = 1.0
    ) -> str:
        """
        Apply a groove template to a MIDI file.
        
        Args:
            input_path: Input MIDI file
            output_path: Output MIDI file
            template: Groove template to apply
            intensity: How much to apply (0-1)
        
        Returns:
            Path to output file
        """
        if not HAS_MIDO:
            raise ImportError("mido required: pip install mido")
        
        mid = mido.MidiFile(input_path)
        new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
        
        for track in mid.tracks:
            new_track = mido.MidiTrack()
            
            for msg in track:
                if msg.type == 'note_on' or msg.type == 'note_off':
                    # Get instrument type from drum map
                    inst_type = self.GM_DRUM_MAP.get(msg.note, "other")
                    
                    # Apply push/pull
                    push_pull = template.push_pull.get(inst_type, 0)
                    push_pull = int(push_pull * intensity)
                    
                    # Apply velocity adjustment
                    if msg.type == 'note_on' and msg.velocity > 0:
                        base_vel = template.velocity_map.get(inst_type, msg.velocity)
                        new_vel = int(msg.velocity * (1 - intensity) + base_vel * intensity)
                        new_vel = max(1, min(127, new_vel))
                        msg = msg.copy(velocity=new_vel)
                    
                    # Apply timing offset (simplified - would need delta time adjustment)
                    new_track.append(msg)
                else:
                    new_track.append(msg)
            
            new_mid.tracks.append(new_track)
        
        new_mid.save(output_path)
        return output_path
    
    def get_preset(self, name: str) -> Optional[GrooveTemplate]:
        """Get a preset groove template by name."""
        name_lower = name.lower()
        
        if name_lower in GROOVE_PRESETS:
            return GROOVE_PRESETS[name_lower]
        if name_lower in EMOTIONAL_PRESETS:
            return EMOTIONAL_PRESETS[name_lower]
        
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# GROOVE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

def extract_groove(midi_path: str) -> Dict[str, Any]:
    """
    Extract groove characteristics from a MIDI file.
    
    Returns timing deviations, velocity patterns, and swing analysis.
    """
    if not HAS_MIDO:
        raise ImportError("mido required: pip install mido")
    
    mid = mido.MidiFile(midi_path)
    ppqn = mid.ticks_per_beat
    
    # Collect note events
    events = []
    current_time = 0
    
    for track in mid.tracks:
        time = 0
        for msg in track:
            time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                events.append({
                    'time': time,
                    'pitch': msg.note,
                    'velocity': msg.velocity,
                    'channel': msg.channel
                })
    
    if not events:
        return {"error": "No note events found"}
    
    # Analyze timing deviations from grid
    eighth_note = ppqn // 2
    timing_deviations = []
    
    for event in events:
        nearest_grid = round(event['time'] / eighth_note) * eighth_note
        deviation = event['time'] - nearest_grid
        timing_deviations.append(deviation)
    
    # Calculate swing (ratio of offbeat timing)
    offbeat_events = [e for e in events if (e['time'] % ppqn) > ppqn * 0.25]
    if offbeat_events:
        avg_offbeat_offset = sum(
            (e['time'] % ppqn) - ppqn // 2 for e in offbeat_events
        ) / len(offbeat_events)
        swing_percentage = 50 + (avg_offbeat_offset / eighth_note * 50)
    else:
        swing_percentage = 50
    
    # Velocity analysis
    velocities = [e['velocity'] for e in events]
    avg_velocity = sum(velocities) / len(velocities) if velocities else 0
    velocity_range = max(velocities) - min(velocities) if velocities else 0
    
    return {
        "num_events": len(events),
        "ppqn": ppqn,
        "swing_percentage": round(swing_percentage, 1),
        "avg_timing_deviation_ticks": round(sum(abs(d) for d in timing_deviations) / len(timing_deviations), 1),
        "avg_velocity": round(avg_velocity, 1),
        "velocity_range": velocity_range,
        "humanization_estimate": min(1.0, len(set(timing_deviations)) / len(timing_deviations)) if timing_deviations else 0
    }
