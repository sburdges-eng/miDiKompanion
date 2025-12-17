"""
Kelly Generator - Emotion-driven MIDI generation.

The central brain that connects:
- EmotionThesaurus → MusicalAttributes
- MusicalAttributes → MIDI output

Usage:
    gen = KellyGenerator()
    midi = gen.from_emotion("devastated", bars=8)
    midi.save("output.mid")
    
    # Or with full intent
    midi = gen.generate(
        emotion="grief",
        intensity=0.8,
        key="F",
        genre="lo_fi_bedroom",
    )
"""

import random
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import mido

# Import the thesaurus (adjust path as needed)
try:
    from emotion_thesaurus import (
        EmotionThesaurus, EmotionNode, MusicalAttributes, Mode,
        get_thesaurus, emotion_to_music
    )
except ImportError:
    # For standalone testing
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from emotion_thesaurus import (
        EmotionThesaurus, EmotionNode, MusicalAttributes, Mode,
        get_thesaurus, emotion_to_music
    )


# =============================================================================
# CONSTANTS
# =============================================================================

TICKS_PER_BEAT = 480
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Mode intervals (semitones from root)
MODE_INTERVALS = {
    Mode.IONIAN: [0, 2, 4, 5, 7, 9, 11],
    Mode.DORIAN: [0, 2, 3, 5, 7, 9, 10],
    Mode.PHRYGIAN: [0, 1, 3, 5, 7, 8, 10],
    Mode.LYDIAN: [0, 2, 4, 6, 7, 9, 11],
    Mode.MIXOLYDIAN: [0, 2, 4, 5, 7, 9, 10],
    Mode.AEOLIAN: [0, 2, 3, 5, 7, 8, 10],
    Mode.LOCRIAN: [0, 1, 3, 5, 6, 8, 10],
}

# Chord types: intervals from root
CHORD_TYPES = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dom7": [0, 4, 7, 10],
    "dim7": [0, 3, 6, 9],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "add9": [0, 4, 7, 14],
    "min9": [0, 3, 7, 10, 14],
    "maj9": [0, 4, 7, 11, 14],
}

# Diatonic chord qualities per mode degree (0-indexed)
MODE_CHORD_QUALITIES = {
    Mode.IONIAN: ["maj", "min", "min", "maj", "maj", "min", "dim"],
    Mode.DORIAN: ["min", "min", "maj", "maj", "min", "dim", "maj"],
    Mode.PHRYGIAN: ["min", "maj", "maj", "min", "dim", "maj", "min"],
    Mode.LYDIAN: ["maj", "maj", "min", "dim", "maj", "min", "min"],
    Mode.MIXOLYDIAN: ["maj", "min", "dim", "maj", "min", "min", "maj"],
    Mode.AEOLIAN: ["min", "dim", "maj", "min", "min", "maj", "maj"],
    Mode.LOCRIAN: ["dim", "maj", "min", "min", "maj", "maj", "min"],
}

# Common progressions by mode (degree indices, 0-based)
MODE_PROGRESSIONS = {
    Mode.IONIAN: [
        [0, 4, 5, 3],      # I-V-vi-IV (pop)
        [0, 3, 4, 0],      # I-IV-V-I (classic)
        [0, 5, 3, 4],      # I-vi-IV-V
        [1, 4, 0, 0],      # ii-V-I-I (jazz)
    ],
    Mode.DORIAN: [
        [0, 3, 6, 0],      # i-IV-bVII-i
        [0, 3, 0, 3],      # i-IV-i-IV (vamp)
        [0, 6, 3, 0],      # i-bVII-IV-i
    ],
    Mode.PHRYGIAN: [
        [0, 1, 0, 1],      # i-bII-i-bII (Spanish)
        [0, 6, 5, 0],      # i-bVII-bVI-i
        [0, 1, 6, 0],      # i-bII-bVII-i
    ],
    Mode.LYDIAN: [
        [0, 1, 0, 1],      # I-II-I-II (floating)
        [0, 1, 4, 0],      # I-II-v-I
    ],
    Mode.MIXOLYDIAN: [
        [0, 6, 3, 0],      # I-bVII-IV-I (rock)
        [0, 6, 0, 6],      # I-bVII-I-bVII
        [0, 3, 6, 3],      # I-IV-bVII-IV
    ],
    Mode.AEOLIAN: [
        [0, 5, 2, 6],      # i-bVI-bIII-bVII
        [0, 6, 5, 6],      # i-bVII-bVI-bVII (andalusian)
        [0, 3, 5, 4],      # i-iv-bVI-v
        [0, 5, 6, 0],      # i-bVI-bVII-i
    ],
    Mode.LOCRIAN: [
        [0, 1, 6, 0],      # i°-bII-bVII-i°
        [0, 5, 6, 0],      # i°-bVI-bVII-i°
    ],
}

# Borrowed chords (for rule-breaking): target mode → chord to borrow
BORROWED_CHORDS = {
    Mode.IONIAN: [
        (3, "min"),   # iv (from parallel minor)
        (5, "maj"),   # bVI (from parallel minor)
        (6, "maj"),   # bVII (from parallel minor)
    ],
    Mode.AEOLIAN: [
        (4, "maj"),   # V (from harmonic minor)
        (6, "dim"),   # vii° (from harmonic minor)
    ],
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class NoteEvent:
    """A single note event."""
    pitch: int          # MIDI note number
    start_ticks: int    # Start time in ticks
    duration_ticks: int # Duration in ticks
    velocity: int       # 0-127
    channel: int = 0


@dataclass
class ChordEvent:
    """A chord (multiple notes)."""
    notes: List[NoteEvent]
    root: int
    quality: str
    degree: int  # Scale degree (0-indexed)
    
    @property
    def start_ticks(self) -> int:
        return self.notes[0].start_ticks if self.notes else 0


@dataclass 
class GeneratedPhrase:
    """A generated musical phrase."""
    events: List[Union[NoteEvent, ChordEvent]]
    tempo: int
    key: str
    mode: Mode
    bars: int
    emotion: str
    rule_breaks_applied: List[str] = field(default_factory=list)


# =============================================================================
# RULE BREAKERS
# =============================================================================

class RuleBreaker:
    """Applies intentional musical violations based on emotion."""
    
    @staticmethod
    def apply_rule_break(
        events: List[ChordEvent],
        rule: str,
        params: MusicalAttributes,
        mode: Mode,
        key_root: int,
    ) -> Tuple[List[ChordEvent], str]:
        """
        Apply a rule-breaking technique.
        
        Returns:
            (modified_events, description)
        """
        if rule == "modal_mixture":
            return RuleBreaker._modal_mixture(events, mode, key_root)
        elif rule == "suspend_resolution":
            return RuleBreaker._suspend_resolution(events)
        elif rule == "add_dissonance":
            return RuleBreaker._add_dissonance(events, params.dissonance_level)
        elif rule == "cluster_voicing":
            return RuleBreaker._cluster_voicing(events)
        elif rule == "tritone_substitution":
            return RuleBreaker._tritone_sub(events, key_root)
        elif rule == "avoid_root":
            return RuleBreaker._avoid_root(events)
        elif rule == "metric_displacement":
            return RuleBreaker._metric_displacement(events)
        elif rule == "delay_resolution":
            return RuleBreaker._delay_resolution(events)
        elif rule == "add_suspension":
            return RuleBreaker._add_suspension(events)
        else:
            return events, f"Unknown rule: {rule}"
    
    @staticmethod
    def _modal_mixture(
        events: List[ChordEvent], 
        mode: Mode, 
        key_root: int
    ) -> Tuple[List[ChordEvent], str]:
        """Borrow chord from parallel mode."""
        if not events or mode not in BORROWED_CHORDS:
            return events, "No mixture applied"
        
        # Pick a random chord to replace
        idx = random.randint(0, len(events) - 1)
        borrowed = random.choice(BORROWED_CHORDS[mode])
        degree, quality = borrowed
        
        # Calculate borrowed chord root
        intervals = MODE_INTERVALS[Mode.AEOLIAN if mode == Mode.IONIAN else Mode.IONIAN]
        borrowed_root = key_root + intervals[degree]
        
        # Build new chord
        chord_intervals = CHORD_TYPES.get(quality, CHORD_TYPES["min"])
        new_notes = []
        for note in events[idx].notes:
            interval_idx = events[idx].notes.index(note)
            if interval_idx < len(chord_intervals):
                new_pitch = borrowed_root + chord_intervals[interval_idx]
                new_notes.append(NoteEvent(
                    pitch=new_pitch,
                    start_ticks=note.start_ticks,
                    duration_ticks=note.duration_ticks,
                    velocity=note.velocity,
                    channel=note.channel,
                ))
        
        if new_notes:
            events[idx] = ChordEvent(
                notes=new_notes,
                root=borrowed_root,
                quality=quality,
                degree=degree,
            )
        
        return events, f"Borrowed {quality} chord on degree {degree+1}"
    
    @staticmethod
    def _suspend_resolution(events: List[ChordEvent]) -> Tuple[List[ChordEvent], str]:
        """Replace 3rd with 4th (sus4) on random chord."""
        if not events:
            return events, "No suspension"
        
        idx = random.randint(0, len(events) - 1)
        chord = events[idx]
        
        for i, note in enumerate(chord.notes):
            # If this is likely the 3rd (4 semitones up from root)
            if (note.pitch - chord.root) % 12 == 4:
                chord.notes[i] = NoteEvent(
                    pitch=note.pitch + 1,  # Raise to 4th
                    start_ticks=note.start_ticks,
                    duration_ticks=note.duration_ticks,
                    velocity=note.velocity,
                    channel=note.channel,
                )
                return events, f"Suspended 3rd→4th on chord {idx+1}"
        
        return events, "No 3rd found to suspend"
    
    @staticmethod
    def _add_dissonance(
        events: List[ChordEvent],
        level: float
    ) -> Tuple[List[ChordEvent], str]:
        """Add dissonant intervals based on level."""
        if not events or level < 0.3:
            return events, "Dissonance too low"
        
        dissonant_intervals = [1, 2, 6, 11, 13]  # m2, M2, tritone, M7, m9
        
        for chord in events:
            if random.random() < level:
                interval = random.choice(dissonant_intervals)
                new_pitch = chord.root + interval
                chord.notes.append(NoteEvent(
                    pitch=new_pitch,
                    start_ticks=chord.start_ticks,
                    duration_ticks=chord.notes[0].duration_ticks,
                    velocity=int(chord.notes[0].velocity * 0.7),
                    channel=0,
                ))
        
        return events, f"Added dissonance (level {level:.1f})"
    
    @staticmethod
    def _cluster_voicing(events: List[ChordEvent]) -> Tuple[List[ChordEvent], str]:
        """Compress voicing into tight cluster."""
        for chord in events:
            if len(chord.notes) < 3:
                continue
            
            # Bring all notes within one octave of root
            root = chord.root
            for i, note in enumerate(chord.notes):
                while note.pitch > root + 12:
                    chord.notes[i] = NoteEvent(
                        pitch=note.pitch - 12,
                        start_ticks=note.start_ticks,
                        duration_ticks=note.duration_ticks,
                        velocity=note.velocity,
                        channel=note.channel,
                    )
                    note = chord.notes[i]
        
        return events, "Applied cluster voicing"
    
    @staticmethod
    def _tritone_sub(
        events: List[ChordEvent],
        key_root: int
    ) -> Tuple[List[ChordEvent], str]:
        """Replace V chord with bII (tritone substitution)."""
        v_root = key_root + 7  # V is 7 semitones up
        
        for i, chord in enumerate(events):
            if chord.root % 12 == v_root % 12:
                # Replace with bII (1 semitone up from root)
                new_root = key_root + 1
                new_notes = []
                for note in chord.notes:
                    interval = (note.pitch - chord.root) % 12
                    new_notes.append(NoteEvent(
                        pitch=new_root + interval,
                        start_ticks=note.start_ticks,
                        duration_ticks=note.duration_ticks,
                        velocity=note.velocity,
                        channel=note.channel,
                    ))
                events[i] = ChordEvent(
                    notes=new_notes,
                    root=new_root,
                    quality="dom7",
                    degree=1,
                )
                return events, "Tritone substitution: V → bII"
        
        return events, "No V chord found for tritone sub"
    
    @staticmethod
    def _avoid_root(events: List[ChordEvent]) -> Tuple[List[ChordEvent], str]:
        """Remove root from voicings (rootless voicing)."""
        for chord in events:
            chord.notes = [n for n in chord.notes if (n.pitch - chord.root) % 12 != 0]
        return events, "Rootless voicings applied"
    
    @staticmethod
    def _metric_displacement(events: List[ChordEvent]) -> Tuple[List[ChordEvent], str]:
        """Shift events by an eighth note."""
        shift = TICKS_PER_BEAT // 2  # Eighth note
        for chord in events:
            for note in chord.notes:
                note.start_ticks += shift
        return events, "Displaced by eighth note"
    
    @staticmethod
    def _delay_resolution(events: List[ChordEvent]) -> Tuple[List[ChordEvent], str]:
        """Extend penultimate chord, delay final."""
        if len(events) < 2:
            return events, "Not enough chords"
        
        # Extend second-to-last
        for note in events[-2].notes:
            note.duration_ticks = int(note.duration_ticks * 1.5)
        
        # Delay last
        delay = TICKS_PER_BEAT // 2
        for note in events[-1].notes:
            note.start_ticks += delay
        
        return events, "Delayed resolution"
    
    @staticmethod
    def _add_suspension(events: List[ChordEvent]) -> Tuple[List[ChordEvent], str]:
        """Add suspended notes that resolve."""
        if not events:
            return events, "No events"
        
        # Add 4th that should resolve to 3rd
        chord = events[0]
        sus_pitch = chord.root + 5  # Perfect 4th
        
        # Add suspension before chord
        sus_note = NoteEvent(
            pitch=sus_pitch,
            start_ticks=chord.start_ticks - TICKS_PER_BEAT // 4,
            duration_ticks=TICKS_PER_BEAT // 2,
            velocity=chord.notes[0].velocity - 10,
            channel=0,
        )
        chord.notes.insert(0, sus_note)
        
        return events, "Added suspension"


# =============================================================================
# MAIN GENERATOR
# =============================================================================

class KellyGenerator:
    """
    The Kelly MIDI Generator.
    
    Converts emotions into MIDI through the thesaurus pipeline.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize generator.
        
        Args:
            data_dir: Path to emotion JSON files
        """
        self.thesaurus = EmotionThesaurus(data_dir)
        self.rule_breaker = RuleBreaker()
    
    def from_emotion(
        self,
        emotion: str,
        bars: int = 4,
        key: str = "C",
        octave: int = 4,
    ) -> mido.MidiFile:
        """
        Generate MIDI directly from an emotion word.
        
        Args:
            emotion: Any emotion word ("devastated", "joyful", etc.)
            bars: Number of bars to generate
            key: Musical key
            octave: Base octave
            
        Returns:
            mido.MidiFile ready to save/play
        """
        node = self.thesaurus.lookup(emotion)
        if not node:
            raise ValueError(f"Unknown emotion: {emotion}")
        
        return self._generate_from_attributes(
            params=node.musical,
            emotion_name=node.name,
            bars=bars,
            key=key,
            octave=octave,
        )
    
    def from_blend(
        self,
        emotions: List[Tuple[str, float]],
        bars: int = 4,
        key: str = "C",
        octave: int = 4,
    ) -> mido.MidiFile:
        """
        Generate MIDI from blended emotions.
        
        Args:
            emotions: List of (emotion_word, weight) tuples
            bars: Number of bars
            key: Musical key
            octave: Base octave
            
        Returns:
            mido.MidiFile
        """
        params = self.thesaurus.blend_emotions(emotions)
        emotion_str = "+".join(f"{e}({w:.1f})" for e, w in emotions)
        
        return self._generate_from_attributes(
            params=params,
            emotion_name=emotion_str,
            bars=bars,
            key=key,
            octave=octave,
        )
    
    def from_transition(
        self,
        from_emotion: str,
        to_emotion: str,
        bars: int = 8,
        key: str = "C",
        octave: int = 4,
    ) -> mido.MidiFile:
        """
        Generate MIDI that transitions between two emotions.
        
        Args:
            from_emotion: Starting emotion
            to_emotion: Ending emotion
            bars: Total bars (divided among transition steps)
            key: Musical key
            octave: Base octave
            
        Returns:
            mido.MidiFile with gradual emotional transition
        """
        start = self.thesaurus.lookup(from_emotion)
        end = self.thesaurus.lookup(to_emotion)
        
        if not start or not end:
            raise ValueError(f"Unknown emotion(s)")
        
        path = self.thesaurus.find_transition_path(start.id, end.id, steps=4)
        bars_per_step = max(1, bars // len(path))
        
        mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Use starting tempo
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(start.musical.tempo_base)))
        
        current_tick = 0
        key_root = self._key_to_midi(key, octave)
        
        for node in path:
            phrase = self._generate_phrase(
                params=node.musical,
                key_root=key_root,
                bars=bars_per_step,
                start_tick=current_tick,
            )
            
            for event in phrase.events:
                if isinstance(event, ChordEvent):
                    for note in event.notes:
                        track.append(mido.Message(
                            'note_on',
                            note=note.pitch,
                            velocity=note.velocity,
                            time=note.start_ticks - current_tick,
                        ))
                        current_tick = note.start_ticks
            
            current_tick = phrase.events[-1].start_ticks + TICKS_PER_BEAT * bars_per_step
        
        # Convert to delta times
        self._convert_to_delta(track)
        
        return mid
    
    def generate(
        self,
        emotion: Optional[str] = None,
        intensity: float = 0.5,
        key: str = "C",
        mode: Optional[Mode] = None,
        tempo: Optional[int] = None,
        bars: int = 4,
        octave: int = 4,
        apply_rules: bool = True,
    ) -> mido.MidiFile:
        """
        Full generation with all parameters.
        
        Args:
            emotion: Emotion word (optional, uses neutral if not provided)
            intensity: Override intensity (0-1)
            key: Musical key
            mode: Override mode
            tempo: Override tempo
            bars: Number of bars
            octave: Base octave
            apply_rules: Whether to apply rule-breaking
            
        Returns:
            mido.MidiFile
        """
        # Get base parameters
        if emotion:
            node = self.thesaurus.lookup(emotion)
            if node:
                params = node.musical
            else:
                params = MusicalAttributes()
        else:
            params = MusicalAttributes()
        
        # Apply overrides
        if mode:
            params.mode = mode
        if tempo:
            params.tempo_base = tempo
        
        # Scale by intensity
        params.velocity_base = int(60 + intensity * 60)
        params.dissonance_level *= intensity
        
        if not apply_rules:
            params.rule_breaks = []
        
        return self._generate_from_attributes(
            params=params,
            emotion_name=emotion or "neutral",
            bars=bars,
            key=key,
            octave=octave,
        )
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _generate_from_attributes(
        self,
        params: MusicalAttributes,
        emotion_name: str,
        bars: int,
        key: str,
        octave: int,
    ) -> mido.MidiFile:
        """Generate MIDI from MusicalAttributes."""
        mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Tempo
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(params.tempo_base)))
        
        # Track name
        track.append(mido.MetaMessage('track_name', name=f"Kelly: {emotion_name}"))
        
        key_root = self._key_to_midi(key, octave)
        
        # Generate phrase
        phrase = self._generate_phrase(
            params=params,
            key_root=key_root,
            bars=bars,
            start_tick=0,
        )
        
        # Flatten to note events
        all_notes = []
        for event in phrase.events:
            if isinstance(event, ChordEvent):
                all_notes.extend(event.notes)
            elif isinstance(event, NoteEvent):
                all_notes.append(event)
        
        # Sort by start time
        all_notes.sort(key=lambda n: n.start_ticks)
        
        # Create note on/off pairs
        note_events = []
        for note in all_notes:
            note_events.append(('on', note.start_ticks, note.pitch, note.velocity))
            note_events.append(('off', note.start_ticks + note.duration_ticks, note.pitch, 0))
        
        # Sort by time
        note_events.sort(key=lambda x: (x[1], 0 if x[0] == 'off' else 1))
        
        # Convert to delta time messages
        current_time = 0
        for event_type, abs_time, pitch, velocity in note_events:
            delta = abs_time - current_time
            if event_type == 'on':
                track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=delta))
            else:
                track.append(mido.Message('note_off', note=pitch, velocity=0, time=delta))
            current_time = abs_time
        
        return mid
    
    def _generate_phrase(
        self,
        params: MusicalAttributes,
        key_root: int,
        bars: int,
        start_tick: int = 0,
    ) -> GeneratedPhrase:
        """Generate a musical phrase."""
        mode = params.mode
        
        # Pick progression
        progressions = MODE_PROGRESSIONS.get(mode, MODE_PROGRESSIONS[Mode.AEOLIAN])
        progression = random.choice(progressions)
        
        # Extend/truncate to match bars
        while len(progression) < bars:
            progression = progression + progression
        progression = progression[:bars]
        
        # Generate chords
        events = []
        ticks_per_bar = TICKS_PER_BEAT * 4
        
        intervals = MODE_INTERVALS[mode]
        qualities = MODE_CHORD_QUALITIES[mode]
        
        for bar_idx, degree in enumerate(progression):
            bar_start = start_tick + bar_idx * ticks_per_bar
            
            # Build chord
            root = key_root + intervals[degree]
            quality = qualities[degree]
            chord_intervals = CHORD_TYPES.get(quality, CHORD_TYPES["min"])
            
            # Determine voicing based on params
            notes = self._voice_chord(
                root=root,
                intervals=chord_intervals,
                params=params,
                bar_start=bar_start,
                bar_duration=ticks_per_bar,
            )
            
            events.append(ChordEvent(
                notes=notes,
                root=root,
                quality=quality,
                degree=degree,
            ))
        
        # Apply rule breaks
        applied_rules = []
        for rule in params.rule_breaks:
            if random.random() < 0.7:  # 70% chance to apply each rule
                events, desc = RuleBreaker.apply_rule_break(
                    events, rule, params, mode, key_root
                )
                applied_rules.append(desc)
        
        return GeneratedPhrase(
            events=events,
            tempo=params.tempo_base,
            key=NOTE_NAMES[key_root % 12],
            mode=mode,
            bars=bars,
            emotion="",
            rule_breaks_applied=applied_rules,
        )
    
    def _voice_chord(
        self,
        root: int,
        intervals: List[int],
        params: MusicalAttributes,
        bar_start: int,
        bar_duration: int,
    ) -> List[NoteEvent]:
        """Create voiced notes for a chord."""
        notes = []
        
        # Base duration (full bar with legato adjustment)
        base_duration = int(bar_duration * params.legato)
        
        # Humanize timing
        timing_spread = int(TICKS_PER_BEAT * params.humanize_timing * 0.25)
        
        for i, interval in enumerate(intervals):
            pitch = root + interval
            
            # Register adjustment
            if params.register == "low":
                pitch -= 12
            elif params.register == "high":
                pitch += 12
            elif params.register == "low-mid":
                pitch -= 6
            elif params.register == "mid-high":
                pitch += 6
            
            # Humanized start time
            start = bar_start
            if params.humanize_timing > 0 and i > 0:
                start += random.randint(-timing_spread, timing_spread)
            
            # Humanized velocity
            vel_base = params.velocity_base
            if params.humanize_velocity > 0:
                vel_spread = int(20 * params.humanize_velocity)
                vel_base += random.randint(-vel_spread, vel_spread)
            
            # Accent on root
            if i == 0:
                vel_base = int(vel_base * (1 + params.accent_strength * 0.3))
            
            velocity = max(1, min(127, vel_base))
            
            # Duration with articulation
            duration = base_duration
            if params.attack_sharpness > 0.7:
                duration = int(duration * 0.6)  # Staccato-ish
            
            notes.append(NoteEvent(
                pitch=pitch,
                start_ticks=max(0, start),
                duration_ticks=duration,
                velocity=velocity,
                channel=0,
            ))
        
        return notes
    
    def _key_to_midi(self, key: str, octave: int) -> int:
        """Convert key name to MIDI note number."""
        key_clean = key.replace('b', '').replace('#', '')
        base = NOTE_NAMES.index(key_clean) if key_clean in NOTE_NAMES else 0
        
        if 'b' in key:
            base -= 1
        elif '#' in key:
            base += 1
        
        return base + (octave + 1) * 12
    
    def _convert_to_delta(self, track: mido.MidiTrack) -> None:
        """Convert absolute times to delta times in place."""
        current = 0
        for msg in track:
            if hasattr(msg, 'time'):
                delta = msg.time - current
                current = msg.time
                msg.time = max(0, delta)


# =============================================================================
# CLI / TEST
# =============================================================================

def main():
    """Test the generator."""
    import sys
    
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    
    gen = KellyGenerator(data_dir)
    
    print("=== KELLY GENERATOR TEST ===\n")
    
    # Test emotions
    test_emotions = ["devastated", "furious", "joyful", "anxious", "content"]
    
    for emotion in test_emotions:
        node = gen.thesaurus.lookup(emotion)
        if node:
            print(f"'{emotion}' → {node.musical.mode.value}, {node.musical.tempo_base} BPM")
            
            # Generate MIDI
            midi = gen.from_emotion(emotion, bars=4, key="F")
            output_path = output_dir / f"kelly_{emotion}.mid"
            midi.save(str(output_path))
            print(f"  Saved: {output_path}")
            
            # Show rule breaks if any
            if node.musical.rule_breaks:
                print(f"  Rule breaks: {node.musical.rule_breaks}")
        else:
            print(f"'{emotion}' → NOT FOUND")
        print()
    
    # Test blend
    print("=== BLEND TEST ===")
    midi = gen.from_blend([("grief", 0.6), ("anger", 0.4)], bars=4, key="Am")
    output_path = output_dir / "kelly_blend_grief_anger.mid"
    midi.save(str(output_path))
    print(f"Saved blend: {output_path}\n")
    
    # Test transition
    print("=== TRANSITION TEST ===")
    midi = gen.from_transition("devastated", "content", bars=8, key="F")
    output_path = output_dir / "kelly_transition.mid"
    midi.save(str(output_path))
    print(f"Saved transition: {output_path}")


if __name__ == "__main__":
    main()
