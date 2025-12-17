"""
DAiW - Data Models (Frozen Schema of Truth)

These dataclasses define the complete contract for the DAiW pipeline.
All functions in the system accept/return data conforming to these models.

Pipeline Flow:
    CoreWoundModel → IntentModel → ConstraintModel + RuleBreakModel → DirectiveModel → FinalPayload

References:
    - Typer CLI: https://github.com/fastapi/typer
    - MIDIUtil: https://github.com/MarkCWirt/MIDIUtil
    - Mingus (music theory): https://github.com/bspaans/python-mingus
"""

from dataclasses import dataclass, field
from typing import Literal, List, Tuple, Dict, Any, Optional
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS - Constrained Choices
# ═══════════════════════════════════════════════════════════════════════════════

class NarrativeArc(str, Enum):
    """Emotional/structural shape of the song"""
    CLIMB_TO_CLIMAX = "Climb-to-Climax"
    SLOW_REVEAL = "Slow Reveal"
    REPETITIVE_DESPAIR = "Repetitive Despair"
    SUDDEN_SHIFT = "Sudden Shift"
    STATIC_REFLECTION = "Static Reflection"


class VulnerabilityScale(str, Enum):
    """How exposed/raw is the emotional content"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class GrooveFeel(str, Enum):
    """Rhythmic character"""
    STRAIGHT = "Straight/Driving"
    SWING = "Swing"
    SHUFFLE = "Shuffle"
    HALFTIME = "Half-Time"
    SYNCOPATED = "Syncopated"
    FLOATING = "Floating/Ambient"


class HarmonicComplexity(str, Enum):
    """Chord density level"""
    SIMPLE = "Simple"        # Triads only (C, Cm, Cdim)
    MODERATE = "Moderate"    # 7ths (Cmaj7, Cm7, Cdom7)
    COMPLEX = "Complex"      # Extensions (Cm9, Cmaj9, C13)
    EXPERIMENTAL = "Experimental"  # Clusters, polytonality


class RuleToBreak(str, Enum):
    """
    Technical rules to intentionally violate.
    Based on rule_breaking_masterpieces.md database.
    """
    NONE = "NONE"
    
    # Harmony rules
    HARMONY_ParallelMotion = "HARMONY_ParallelMotion"
    HARMONY_ModalInterchange = "HARMONY_ModalInterchange"
    HARMONY_UnresolvedDissonance = "HARMONY_UnresolvedDissonance"
    HARMONY_AvoidTonicResolution = "HARMONY_AvoidTonicResolution"
    HARMONY_Polytonality = "HARMONY_Polytonality"
    HARMONY_TritoneSubstitution = "HARMONY_TritoneSubstitution"
    
    # Rhythm rules
    RHYTHM_ConstantDisplacement = "RHYTHM_ConstantDisplacement"
    RHYTHM_MeterAmbiguity = "RHYTHM_MeterAmbiguity"
    RHYTHM_TempoFluctuation = "RHYTHM_TempoFluctuation"
    
    # Structure rules
    STRUCTURE_NonResolution = "STRUCTURE_NonResolution"
    STRUCTURE_AsymmetricForm = "STRUCTURE_AsymmetricForm"
    STRUCTURE_AntiDrop = "STRUCTURE_AntiDrop"
    
    # Production rules
    PRODUCTION_BuriedVocals = "PRODUCTION_BuriedVocals"
    PRODUCTION_LoFiArtifacts = "PRODUCTION_LoFiArtifacts"
    PRODUCTION_PitchImperfection = "PRODUCTION_PitchImperfection"
    
    # Arrangement rules
    ARRANGEMENT_SparseClimax = "ARRANGEMENT_SparseClimax"
    ARRANGEMENT_InstrumentDrop = "ARRANGEMENT_InstrumentDrop"


class OutputTarget(str, Enum):
    """What the generation engine should produce"""
    MIDI_CHORD_PROGRESSION = "MIDI Chord Progression"
    MIDI_RHYTHM_PATTERN = "MIDI Rhythm Pattern"
    FULL_ARRANGEMENT_SKETCH = "Full Arrangement Sketch (MIDI)"
    LYRICAL_CONCEPT = "Lyrical Concept/Outline"
    PRODUCTION_TEMPLATE = "Production/Mixing Template"


class FeedbackLoop(str, Enum):
    """What to preview first"""
    HARMONY_ONLY = "Harmony Only"
    GROOVE_ONLY = "Groove Only"
    STRUCTURE_ONLY = "Structure Only"
    FULL_SKETCH = "Full Sketch"
    HARMONY_AND_RHYTHM = "Harmony and Rhythm"


class OutputFormat(str, Enum):
    """Output file format"""
    MIDI = "MIDI"
    JSON = "JSON"
    OBSIDIAN_NOTE = "Obsidian Note"


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0: CORE WOUND MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CoreWoundModel:
    """
    Phase 0: The Core Wound/Desire.
    Captures the raw, psychological truth behind the song.
    
    Populated via: Therapist interrogation engine prompts
    """
    
    # The actual event/story being processed
    core_event: str
    
    # What part of the user resists writing this?
    core_resistance: str
    
    # What do they wish they could feel instead?
    core_longing: str
    
    # What's at stake if the song isn't written?
    core_stakes: str = ""
    
    # How should they feel when the song is complete?
    core_transformation: str = ""
    
    # Optional: Named entity for the emotion (narrative therapy)
    narrative_entity_name: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: INTENT MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IntentModel:
    """
    Phase 1: The Emotional Intent.
    Maps psychological state to structural decisions.
    
    Populated via: System proposals based on CoreWoundModel + user refinement
    """
    
    # Primary emotional target (e.g., "Defiance and Liberation", "Grief", "Anger")
    mood_primary: str
    
    # Tension level (0.0-1.0) - higher = more unresolved dissonance
    mood_secondary_tension: float = 0.5
    
    # Structural shape of the emotional journey
    narrative_arc: NarrativeArc = NarrativeArc.CLIMB_TO_CLIMAX
    
    # Harmonic density
    harmonic_complexity: HarmonicComplexity = HarmonicComplexity.MODERATE
    
    # How exposed/raw is the content
    vulnerability_scale: VulnerabilityScale = VulnerabilityScale.MEDIUM
    
    # Imagery/texture keywords
    imagery_texture: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: CONSTRAINT MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConstraintModel:
    """
    Phase 2A: Technical parameters for generation.
    Defines the sonic/stylistic boundaries.
    
    Populated via: System proposals based on IntentModel + user selection
    """
    
    # Genre descriptor (e.g., "Industrial Pop/Synthwave", "Lo-Fi Bedroom Emo")
    technical_genre: str
    
    # BPM range as tuple (min, max)
    technical_tempo_range: Tuple[int, int] = (120, 130)
    
    # Rhythmic feel
    technical_groove_feel: GrooveFeel = GrooveFeel.STRAIGHT
    
    # Key center (e.g., "F", "Am", "Bb")
    technical_key: str = "C"
    
    # Mode/scale preference (e.g., "minor", "dorian", "mixolydian")
    technical_mode: str = "minor"


@dataclass
class RuleBreakModel:
    """
    Phase 2B: The intentional "abrasive element".
    Specifies which conventional music theory rule to violate.
    
    Based on: rule_breaking_masterpieces.md database
    """
    
    # Which rule to break
    technical_rule_to_break: RuleToBreak = RuleToBreak.NONE
    
    # Why this rule serves the emotional intent
    rule_breaking_justification: str = ""
    
    # Intensity of rule-breaking (0.0-1.0)
    rule_breaking_intensity: float = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2C: INSTRUMENT PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InstrumentPalette:
    """
    Sound design choices for each role.
    Two presets: C1 (Basic) and C2 (Advanced/Textural)
    """
    
    bass: str = "Sub Bass"
    drums: str = "Classic Drum Machine (909)"
    pad: str = "Warm Analog Pad"
    lead: str = "Aggressive Saw Synth"
    
    # Whether using basic or advanced palette
    palette_type: Literal["C1_Basic", "C2_Advanced"] = "C1_Basic"


# C1 and C2 Presets
PALETTE_C1_BASIC = InstrumentPalette(
    bass="Sub Bass (Clean, low-end support)",
    drums="Classic Drum Machine (909) (Iconic, powerful transients)",
    pad="Warm Analog Pad (Supporting, non-intrusive)",
    lead="Aggressive Saw Synth (Cutting, simple melody focus)",
    palette_type="C1_Basic"
)

PALETTE_C2_ADVANCED = InstrumentPalette(
    bass="Mid-Range Fuzz Bass (Harmonically rich and aggressive)",
    drums="Glitch/Percussive Noise (Disruptive, chaotic texture)",
    pad="Cold Digital Pad (Dissonant, brittle high-end)",
    lead="Plucked Arpeggiator (Busy, complex rhythmic layer)",
    palette_type="C2_Advanced"
)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: DIRECTIVE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DirectiveModel:
    """
    Phase 3: Final output parameters.
    Commands for the generation engine.
    """
    
    # What to generate
    output_target: OutputTarget = OutputTarget.FULL_ARRANGEMENT_SKETCH
    
    # What to preview first
    output_feedback_loop: FeedbackLoop = FeedbackLoop.FULL_SKETCH
    
    # Song length in bars
    song_length_bars: int = 64
    
    # Output file format
    output_format: OutputFormat = OutputFormat.MIDI


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL PAYLOAD - AGGREGATED STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FinalPayload:
    """
    The complete, frozen state passed to the Generation Engine.
    Aggregates all phase models into a single execution unit.
    """
    
    # Phase 0
    wound: CoreWoundModel
    
    # Phase 1
    intent: IntentModel
    
    # Phase 2
    constraints: ConstraintModel
    rule_break: RuleBreakModel
    instrument_palette: InstrumentPalette
    
    # Phase 3
    directive: DirectiveModel
    
    # Metadata
    session_id: str = ""
    created_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Flatten all fields into a single dictionary for the generation engine."""
        return {
            # Phase 0
            "core_event": self.wound.core_event,
            "core_resistance": self.wound.core_resistance,
            "core_longing": self.wound.core_longing,
            "core_stakes": self.wound.core_stakes,
            "core_transformation": self.wound.core_transformation,
            "narrative_entity_name": self.wound.narrative_entity_name,
            
            # Phase 1
            "mood_primary": self.intent.mood_primary,
            "mood_secondary_tension": self.intent.mood_secondary_tension,
            "narrative_arc": self.intent.narrative_arc.value,
            "harmonic_complexity": self.intent.harmonic_complexity.value,
            "vulnerability_scale": self.intent.vulnerability_scale.value,
            "imagery_texture": self.intent.imagery_texture,
            
            # Phase 2A
            "technical_genre": self.constraints.technical_genre,
            "technical_tempo_range": self.constraints.technical_tempo_range,
            "technical_groove_feel": self.constraints.technical_groove_feel.value,
            "technical_key": self.constraints.technical_key,
            "technical_mode": self.constraints.technical_mode,
            
            # Phase 2B
            "technical_rule_to_break": self.rule_break.technical_rule_to_break.value,
            "rule_breaking_justification": self.rule_break.rule_breaking_justification,
            "rule_breaking_intensity": self.rule_break.rule_breaking_intensity,
            
            # Phase 2C
            "instrument_bass": self.instrument_palette.bass,
            "instrument_drums": self.instrument_palette.drums,
            "instrument_pad": self.instrument_palette.pad,
            "instrument_lead": self.instrument_palette.lead,
            "palette_type": self.instrument_palette.palette_type,
            
            # Phase 3
            "output_target": self.directive.output_target.value,
            "output_feedback_loop": self.directive.output_feedback_loop.value,
            "song_length_bars": self.directive.song_length_bars,
            "output_format": self.directive.output_format.value,
            
            # Metadata
            "session_id": self.session_id,
            "created_at": self.created_at,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_example_payload() -> FinalPayload:
    """Create an example payload for the Kelly song."""
    from datetime import datetime
    import uuid
    
    return FinalPayload(
        wound=CoreWoundModel(
            core_event="Finding my friend Kelly after she died by suicide",
            core_resistance="Fear of making her death about my pain",
            core_longing="Permission to grieve without guilt",
            core_stakes="If I don't write this, the grief stays stuck",
            core_transformation="Peace with what I couldn't control",
            narrative_entity_name="The Grey Weight"
        ),
        intent=IntentModel(
            mood_primary="Grief and Longing",
            mood_secondary_tension=0.7,
            narrative_arc=NarrativeArc.SUDDEN_SHIFT,
            harmonic_complexity=HarmonicComplexity.MODERATE,
            vulnerability_scale=VulnerabilityScale.HIGH,
            imagery_texture="grey morning light, cold hands, empty room"
        ),
        constraints=ConstraintModel(
            technical_genre="Lo-Fi Bedroom Emo",
            technical_tempo_range=(78, 86),
            technical_groove_feel=GrooveFeel.FLOATING,
            technical_key="F",
            technical_mode="major"
        ),
        rule_break=RuleBreakModel(
            technical_rule_to_break=RuleToBreak.HARMONY_ModalInterchange,
            rule_breaking_justification="Bbm borrowed chord makes hope feel earned, not given",
            rule_breaking_intensity=0.6
        ),
        instrument_palette=PALETTE_C1_BASIC,
        directive=DirectiveModel(
            output_target=OutputTarget.FULL_ARRANGEMENT_SKETCH,
            output_feedback_loop=FeedbackLoop.HARMONY_ONLY,
            song_length_bars=64,
            output_format=OutputFormat.MIDI
        ),
        session_id=str(uuid.uuid4())[:8],
        created_at=datetime.now().isoformat()
    )
