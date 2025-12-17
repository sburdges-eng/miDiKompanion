"""
DAiW Comprehensive Engine - THE BRAIN
======================================

The central engine that:
1. Analyzes emotional input (AffectAnalyzer)
2. Maintains session state (TherapySession)
3. Maps affect to musical parameters
4. Generates harmony plans
5. Renders to MIDI

This is the core of the "Interrogate Before Generate" philosophy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

try:
    from midiutil import MIDIFile
    HAS_MIDIUTIL = True
except ImportError:
    HAS_MIDIUTIL = False


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AffectResult:
    """Result of affect analysis."""
    primary: str
    secondary: Optional[str]
    scores: Dict[str, float]
    intensity: float


@dataclass
class TherapyState:
    """Current state of the therapy session."""
    core_wound: str = ""
    narrative_entity: str = ""
    primary_affect: str = "neutral"
    secondary_affect: Optional[str] = None
    intensity: float = 0.5
    motivation_scale: int = 5
    vulnerability_scale: float = 0.5
    chaos_tolerance: float = 0.5
    suggested_mode: str = "ionian"


@dataclass
class HarmonyPlan:
    """Plan for harmonic content."""
    mode: str
    tempo_bpm: int
    length_bars: int
    structure_type: str
    mood_profile: str
    chords: List[str] = field(default_factory=list)
    tension_curve: List[float] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# AFFECT ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class AffectAnalyzer:
    """
    Analyzes text input to detect emotional content.
    Uses keyword matching with weighted scoring.
    """
    
    KEYWORDS = {
        "grief": {
            "loss", "gone", "miss", "dead", "died", "funeral", "empty", 
            "heavy", "sleeping", "found", "body", "left", "never", "cold"
        },
        "rage": {
            "angry", "furious", "hate", "betrayed", "burn", "fight", 
            "destroy", "violent", "scream", "kill"
        },
        "awe": {
            "wonder", "beautiful", "infinite", "god", "universe", "light", 
            "vast", "amazing", "transcendent"
        },
        "nostalgia": {
            "remember", "used to", "childhood", "old days", "memory", 
            "home", "before", "once", "way back"
        },
        "fear": {
            "scared", "terrified", "panic", "trapped", "anxious", "dread", 
            "dark", "nightmare", "afraid"
        },
        "dissociation": {
            "numb", "nothing", "floating", "unreal", "detached", "fog", 
            "wall", "static", "empty", "blank"
        },
        "defiance": {
            "won't", "refuse", "stand", "strong", "break", "free", 
            "no more", "enough", "rise"
        },
        "confusion": {
            "why", "lost", "spinning", "chaos", "strange", "question",
            "don't understand", "makes no sense"
        },
        "longing": {
            "wish", "want", "need", "hope", "someday", "if only",
            "waiting", "yearning"
        },
    }
    
    def analyze(self, text: str) -> AffectResult:
        """Analyze text for emotional content."""
        if not text:
            return AffectResult("neutral", None, {}, 0.0)
        
        text_lower = text.lower()
        scores = {affect: 0.0 for affect in self.KEYWORDS}
        
        for affect, words in self.KEYWORDS.items():
            for word in words:
                if word in text_lower:
                    scores[affect] += 1.0
        
        sorted_affects = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_affects or sorted_affects[0][1] == 0:
            return AffectResult("neutral", None, scores, 0.0)
        
        primary, p_score = sorted_affects[0]
        secondary = (
            sorted_affects[1][0]
            if len(sorted_affects) > 1 and sorted_affects[1][1] > 0
            else None
        )
        intensity = min(1.0, p_score / 3.0)
        
        return AffectResult(primary, secondary, scores, intensity)


# ═══════════════════════════════════════════════════════════════════════════════
# THERAPY SESSION
# ═══════════════════════════════════════════════════════════════════════════════

class TherapySession:
    """
    Manages the therapeutic interrogation session.
    Maps emotional input to musical parameters.
    """
    
    AFFECT_TO_MODE = {
        "awe": "lydian",
        "nostalgia": "dorian",
        "rage": "phrygian",
        "fear": "phrygian",
        "dissociation": "locrian",
        "grief": "aeolian",
        "defiance": "mixolydian",
        "confusion": "locrian",
        "longing": "dorian",
        "neutral": "ionian",
    }
    
    AFFECT_TO_TEMPO_RANGE = {
        "grief": (60, 80),
        "nostalgia": (70, 90),
        "awe": (65, 85),
        "rage": (120, 160),
        "fear": (100, 140),
        "dissociation": (50, 70),
        "defiance": (110, 140),
        "confusion": (90, 120),
        "longing": (65, 85),
        "neutral": (100, 120),
    }
    
    def __init__(self) -> None:
        self.state = TherapyState()
        self.analyzer = AffectAnalyzer()
    
    def process_core_input(self, text: str) -> str:
        """
        Process the core wound text and set affect/mode.
        Returns the detected primary affect.
        """
        result = self.analyzer.analyze(text)
        
        self.state.core_wound = text
        self.state.primary_affect = result.primary
        self.state.secondary_affect = result.secondary
        self.state.intensity = result.intensity
        self.state.suggested_mode = self.AFFECT_TO_MODE.get(result.primary, "ionian")
        
        return result.primary
    
    def set_scales(self, motivation: int, chaos: float) -> None:
        """Set motivation and chaos scales."""
        self.state.motivation_scale = max(1, min(10, motivation))
        self.state.chaos_tolerance = max(0.0, min(1.0, chaos))
    
    def generate_plan(self) -> HarmonyPlan:
        """Generate a harmony plan based on current state."""
        affect = self.state.primary_affect
        
        # Determine tempo
        tempo_range = self.AFFECT_TO_TEMPO_RANGE.get(affect, (100, 120))
        tempo = random.randint(tempo_range[0], tempo_range[1])
        
        # Adjust for chaos
        tempo += int((self.state.chaos_tolerance - 0.5) * 20)
        tempo = max(50, min(180, tempo))
        
        # Determine length based on motivation
        if self.state.motivation_scale >= 8:
            length_bars = 64
        elif self.state.motivation_scale >= 5:
            length_bars = 32
        else:
            length_bars = 16
        
        # Determine structure type
        structure_type = choose_structure_type_for_mood(affect)
        
        # Generate tension curve
        from .tension import generate_tension_curve
        tension_curve = generate_tension_curve(length_bars, structure_type).tolist()
        
        return HarmonyPlan(
            mode=self.state.suggested_mode,
            tempo_bpm=tempo,
            length_bars=length_bars,
            structure_type=structure_type,
            mood_profile=affect,
            chords=self._generate_chord_progression(affect),
            tension_curve=tension_curve
        )
    
    def _generate_chord_progression(self, affect: str) -> List[str]:
        """Generate a chord progression based on affect."""
        progressions = {
            "grief": ["i", "VI", "III", "VII"],
            "nostalgia": ["I", "V", "vi", "IV"],
            "rage": ["i", "bVII", "i", "bVI"],
            "fear": ["i", "bII", "i", "bVII"],
            "awe": ["I", "IV", "#IVdim", "I"],
            "defiance": ["I", "V", "IV", "V"],
            "dissociation": ["i", "v", "i", "v"],
            "longing": ["I", "V", "vi", "iii"],
            "confusion": ["i", "bII", "bVII", "bVI"],
        }
        return progressions.get(affect, ["I", "IV", "V", "I"])


def choose_structure_type_for_mood(mood: str) -> str:
    """Map affect/mood into a macro structure type."""
    mood_lower = (mood or "").lower()
    if mood_lower in {"grief", "dissociation", "broken"}:
        return "climb"
    if mood_lower in {"rage", "defiance", "fear"}:
        return "standard"
    if mood_lower in {"awe", "nostalgia"}:
        return "standard"
    return "constant"


# ═══════════════════════════════════════════════════════════════════════════════
# MIDI RENDERING
# ═══════════════════════════════════════════════════════════════════════════════

# Note name to MIDI number mapping
NOTE_MAP = {
    "C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63,
    "E": 64, "F": 65, "F#": 66, "Gb": 66, "G": 67, "G#": 68,
    "Ab": 68, "A": 69, "A#": 70, "Bb": 70, "B": 71
}

# Scale patterns (semitones from root)
SCALE_PATTERNS = {
    "ionian": [0, 2, 4, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
}


def roman_to_semitones(roman: str, mode: str = "major") -> int:
    """Convert roman numeral to semitone offset."""
    base_map = {
        "I": 0, "II": 2, "III": 4, "IV": 5, "V": 7, "VI": 9, "VII": 11,
        "i": 0, "ii": 2, "iii": 4, "iv": 5, "v": 7, "vi": 9, "vii": 11,
    }
    
    clean = roman.replace("b", "").replace("#", "").replace("dim", "")
    offset = base_map.get(clean.upper(), 0)
    
    if roman.startswith("b"):
        offset -= 1
    elif roman.startswith("#"):
        offset += 1
    
    return offset % 12


def build_chord(root_midi: int, quality: str = "major") -> List[int]:
    """Build a chord from root MIDI note."""
    intervals = {
        "major": [0, 4, 7],
        "minor": [0, 3, 7],
        "dim": [0, 3, 6],
        "aug": [0, 4, 8],
        "maj7": [0, 4, 7, 11],
        "min7": [0, 3, 7, 10],
        "dom7": [0, 4, 7, 10],
    }
    pattern = intervals.get(quality, intervals["major"])
    return [root_midi + i for i in pattern]


def render_plan_to_midi(
    plan: HarmonyPlan,
    output_path: str,
    vulnerability: float = 0.5,
    seed: int = 42,
    key_root: str = "C"
) -> str:
    """
    Render a HarmonyPlan to a MIDI file.
    
    Args:
        plan: The harmony plan to render
        output_path: Path for output MIDI file
        vulnerability: Controls timing humanization (higher = tighter)
        seed: Random seed for reproducibility
        key_root: Key root note (e.g., "C", "F", "Bb")
    
    Returns:
        Path to the generated MIDI file
    """
    if not HAS_MIDIUTIL:
        raise ImportError("midiutil required: pip install midiutil")
    
    random.seed(seed)
    
    # Create MIDI file
    midi = MIDIFile(2)  # 2 tracks: chords, bass
    
    # Track 0: Chords
    midi.addTrackName(0, 0, "Chords")
    midi.addTempo(0, 0, plan.tempo_bpm)
    
    # Track 1: Bass
    midi.addTrackName(1, 0, "Bass")
    
    # Get root MIDI note
    root_midi = NOTE_MAP.get(key_root, 60)
    
    # Generate chord events
    bars_per_chord = max(1, plan.length_bars // len(plan.chords)) if plan.chords else 4
    
    for i, roman in enumerate(plan.chords):
        # Calculate timing
        start_beat = i * bars_per_chord * 4  # 4 beats per bar
        duration = bars_per_chord * 4 - 0.5  # Leave slight gap
        
        # Get chord root
        offset = roman_to_semitones(roman, plan.mode)
        chord_root = root_midi + offset
        
        # Determine chord quality
        is_minor = roman.islower() or "i" in roman.lower()
        is_dim = "dim" in roman
        quality = "dim" if is_dim else ("minor" if is_minor else "major")
        
        # Build and add chord notes
        chord_notes = build_chord(chord_root, quality)
        
        # Humanize timing based on vulnerability
        jitter_range = int((1 - vulnerability) * 30)
        
        for note in chord_notes:
            # Add slight timing variation
            time_offset = random.uniform(-jitter_range, jitter_range) / 480
            velocity = random.randint(70, 100)
            
            midi.addNote(
                track=0,
                channel=0,
                pitch=note,
                time=max(0, start_beat + time_offset),
                duration=duration,
                volume=velocity
            )
        
        # Add bass note (octave lower)
        bass_note = chord_root - 12
        midi.addNote(
            track=1,
            channel=1,
            pitch=bass_note,
            time=start_beat,
            duration=duration,
            volume=90
        )
    
    # Write MIDI file
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, "wb") as f:
        midi.writeFile(f)
    
    return str(output)


def select_kit_for_mood(mood: str) -> str:
    """Select appropriate drum kit based on mood."""
    kit_map = {
        "grief": "Lo-Fi Kit",
        "rage": "Aggressive Kit",
        "awe": "Ambient Kit",
        "nostalgia": "Vintage Kit",
        "defiance": "Punchy Kit",
        "fear": "Dark Kit",
        "dissociation": "Minimal Kit",
    }
    return kit_map.get(mood, "Standard Kit")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_cli():
    """Simple CLI for testing the engine."""
    session = TherapySession()
    print("\n--- DAiW: Interrogate Before Generate ---")
    
    txt = input("What is hurting you? > ").strip()
    mood = session.process_core_input(txt)
    print(f"Detected mood: {mood} (Mode: {session.state.suggested_mode})")
    
    try:
        mot = int(input("Motivation (1-10)? > ").strip() or "5")
        chaos = int(input("Chaos (1-10)? > ").strip() or "5")
    except ValueError:
        mot, chaos = 5, 5
    
    session.set_scales(motivation=mot, chaos=chaos / 10.0)
    plan = session.generate_plan()
    print(
        f"Plan: {plan.length_bars} bars @ {plan.tempo_bpm} BPM "
        f"in {plan.mode} ({plan.mood_profile}), structure={plan.structure_type}"
    )
    
    out = "daiw_output.mid"
    render_plan_to_midi(plan, out, vulnerability=0.5, seed=42)
    print(f"✅ Exported: {out}")


if __name__ == "__main__":
    run_cli()
