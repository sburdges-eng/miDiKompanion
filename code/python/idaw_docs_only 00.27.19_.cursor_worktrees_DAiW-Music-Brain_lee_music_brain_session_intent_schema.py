"""
Song Intent Schema - Structured deep interrogation for songwriting.

Implements the three-phase interrogation model:
- Phase 0: Core Wound/Desire (deep interrogation)
- Phase 1: Emotional & Intent (validation)
- Phase 2: Technical Constraints (implementation)

Plus comprehensive rule-breaking enums for intentional creative choices.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# =================================================================
# ENUMS: Rule Breaking Categories
# =================================================================

class HarmonyRuleBreak(Enum):
    """Harmony rules to intentionally break."""
    AVOID_TONIC_RESOLUTION = "HARMONY_AvoidTonicResolution"
    PARALLEL_MOTION = "HARMONY_ParallelMotion"
    MODAL_INTERCHANGE = "HARMONY_ModalInterchange"
    TRITONE_SUBSTITUTION = "HARMONY_TritoneSubstitution"
    POLYTONALITY = "HARMONY_Polytonality"
    UNRESOLVED_DISSONANCE = "HARMONY_UnresolvedDissonance"


class RhythmRuleBreak(Enum):
    """Rhythm rules to intentionally break."""
    CONSTANT_DISPLACEMENT = "RHYTHM_ConstantDisplacement"
    TEMPO_FLUCTUATION = "RHYTHM_TempoFluctuation"
    METRIC_MODULATION = "RHYTHM_MetricModulation"
    POLYRHYTHMIC_LAYERS = "RHYTHM_PolyrhythmicLayers"
    DROPPED_BEATS = "RHYTHM_DroppedBeats"


class ArrangementRuleBreak(Enum):
    """Arrangement rules to intentionally break."""
    UNBALANCED_DYNAMICS = "ARRANGEMENT_UnbalancedDynamics"
    STRUCTURAL_MISMATCH = "ARRANGEMENT_StructuralMismatch"
    BURIED_VOCALS = "ARRANGEMENT_BuriedVocals"
    EXTREME_DYNAMIC_RANGE = "ARRANGEMENT_ExtremeDynamicRange"
    PREMATURE_CLIMAX = "ARRANGEMENT_PrematureClimax"


class ProductionRuleBreak(Enum):
    """Production rules to intentionally break."""
    EXCESSIVE_MUD = "PRODUCTION_ExcessiveMud"
    PITCH_IMPERFECTION = "PRODUCTION_PitchImperfection"
    ROOM_NOISE = "PRODUCTION_RoomNoise"
    DISTORTION = "PRODUCTION_Distortion"
    MONO_COLLAPSE = "PRODUCTION_MonoCollapse"
    LO_FI_DEGRADATION = "PRODUCTION_LoFiDegradation"
    SILENCE_AS_INSTRUMENT = "PRODUCTION_SilenceAsInstrument"
    CLIPPING_PEAKS = "PRODUCTION_ClippingPeaks"


class MelodyRuleBreak(Enum):
    """Melody rules to intentionally break."""
    AVOID_RESOLUTION = "MELODY_AvoidResolution"
    EXCESSIVE_REPETITION = "MELODY_ExcessiveRepetition"
    ANGULAR_INTERVALS = "MELODY_AngularIntervals"
    ANTI_CLIMAX = "MELODY_AntiClimax"
    MONOTONE_DRONE = "MELODY_MonotoneDrone"
    FRAGMENTED_PHRASES = "MELODY_FragmentedPhrases"


class TextureRuleBreak(Enum):
    """Texture rules to intentionally break."""
    FREQUENCY_MASKING = "TEXTURE_FrequencyMasking"
    SPARSE_EMPTINESS = "TEXTURE_SparseEmptiness"
    DENSE_WALL = "TEXTURE_DenseWall"
    CONFLICTING_TIMBRES = "TEXTURE_ConflictingTimbres"
    SINGLE_ELEMENT_FOCUS = "TEXTURE_SingleElementFocus"
    TIMBRAL_DRIFT = "TEXTURE_TimbralDrift"


class TemporalRuleBreak(Enum):
    """Temporal/time rules to intentionally break."""
    EXTENDED_INTRO = "TEMPORAL_ExtendedIntro"
    ABRUPT_ENDING = "TEMPORAL_AbruptEnding"
    TIME_STRETCH = "TEMPORAL_TimeStretch"
    LOOP_HYPNOSIS = "TEMPORAL_LoopHypnosis"
    BREATH_PAUSES = "TEMPORAL_BreathPauses"
    ACCELERANDO_DECAY = "TEMPORAL_AccelerandoDecay"


# =================================================================
# ENUMS: Musical Concept Mappings
# =================================================================

class AffectState(Enum):
    """Emotional states with musical parameter mappings."""
    GRIEF = "grief"
    LONGING = "longing"
    DEFIANCE = "defiance"
    HOPE = "hope"
    RAGE = "rage"
    TENDERNESS = "tenderness"
    ANXIETY = "anxiety"
    EUPHORIA = "euphoria"
    MELANCHOLY = "melancholy"
    NOSTALGIA = "nostalgia"
    CATHARSIS = "catharsis"
    DISSOCIATION = "dissociation"
    DETERMINATION = "determination"
    SURRENDER = "surrender"


class TextureType(Enum):
    """Sonic texture categories."""
    ETHEREAL = "Ethereal"           # Airy, spacious, reverberant
    INTIMATE = "Intimate"           # Close, dry, personal
    MASSIVE = "Massive"             # Full, layered, powerful
    SKELETAL = "Skeletal"           # Minimal, exposed, fragile
    LUSH = "Lush"                   # Rich, warm, enveloping
    HARSH = "Harsh"                 # Aggressive, abrasive, cutting
    MURKY = "Murky"                 # Dark, unclear, submerged
    CRYSTALLINE = "Crystalline"     # Clear, precise, bright


class TensionProfile(Enum):
    """How tension evolves over the song."""
    CONSTANT_HIGH = "Constant High"         # Relentless intensity
    CONSTANT_LOW = "Constant Low"           # Meditative calm
    BUILD_RELEASE = "Build and Release"     # Classic arc
    SAWTOOTH = "Sawtooth"                   # Repeated build-drops
    INVERSE = "Inverse"                     # Start high, end low
    PLATEAU = "Plateau"                     # Build to sustained level
    CHAOTIC = "Chaotic"                     # Unpredictable shifts
    SLOW_BURN = "Slow Burn"                 # Gradual accumulation


class DensityLevel(Enum):
    """Arrangement density."""
    SOLO = "Solo"               # Single voice/instrument
    DUO = "Duo"                 # Two elements in conversation
    SPARSE = "Sparse"           # Few elements, lots of space
    MODERATE = "Moderate"       # Balanced arrangement
    FULL = "Full"               # Complete band/ensemble
    DENSE = "Dense"             # Layered, complex
    OVERWHELMING = "Overwhelming"  # Intentionally too much


class ModalColor(Enum):
    """Mode with emotional associations."""
    IONIAN = "Ionian"           # Bright, happy, resolved
    DORIAN = "Dorian"           # Minor but hopeful, jazzy
    PHRYGIAN = "Phrygian"       # Dark, Spanish, exotic
    LYDIAN = "Lydian"           # Dreamy, floating, ethereal
    MIXOLYDIAN = "Mixolydian"   # Bluesy, earthy, unresolved
    AEOLIAN = "Aeolian"         # Natural minor, sad, serious
    LOCRIAN = "Locrian"         # Unstable, tense, rare


class VulnerabilityScale(Enum):
    """Vulnerability level for emotional exposure."""
    LOW = "Low"       # Guarded, protective
    MEDIUM = "Medium" # Honest but controlled
    HIGH = "High"     # Raw, exposed


class NarrativeArc(Enum):
    """Structural emotional arc."""
    CLIMB_TO_CLIMAX = "Climb-to-Climax"
    SLOW_REVEAL = "Slow Reveal"
    REPETITIVE_DESPAIR = "Repetitive Despair"
    STATIC_REFLECTION = "Static Reflection"
    SUDDEN_SHIFT = "Sudden Shift"
    DESCENT = "Descent"
    RISE_AND_FALL = "Rise and Fall"
    SPIRAL = "Spiral"


class CoreStakes(Enum):
    """What's at stake in the song."""
    PERSONAL = "Personal"         # Individual identity
    RELATIONAL = "Relational"     # Connections
    EXISTENTIAL = "Existential"   # Meaning/purpose
    SURVIVAL = "Survival"         # Life/safety
    CREATIVE = "Creative"         # Expression
    MORAL = "Moral"               # Right/wrong


class GrooveFeel(Enum):
    """Rhythmic feel."""
    STRAIGHT_DRIVING = "Straight/Driving"
    LAID_BACK = "Laid Back"
    SWUNG = "Swung"
    SYNCOPATED = "Syncopated"
    RUBATO_FREE = "Rubato/Free"
    MECHANICAL = "Mechanical"
    ORGANIC_BREATHING = "Organic/Breathing"
    PUSH_PULL = "Push-Pull"


# =================================================================
# RULE BREAKING DEFINITIONS
# =================================================================

RULE_BREAKING_EFFECTS = {
    # Harmony
    "HARMONY_AvoidTonicResolution": {
        "description": "Resolve to IV or VI instead of I",
        "effect": "Unresolved, yearning feeling",
        "use_when": "Song shouldn't feel 'finished' or 'answered'",
        "example_emotions": ["longing", "grief", "uncertainty"],
    },
    "HARMONY_ParallelMotion": {
        "description": "Use forbidden parallel 5ths/octaves",
        "effect": "Vintage, punk, or medieval sound",
        "use_when": "Defiance, raw power, or historical evocation",
        "example_emotions": ["defiance", "anger", "power"],
    },
    "HARMONY_ModalInterchange": {
        "description": "Borrow chord from unrelated key",
        "effect": "Unexpected color, emotional shift",
        "use_when": "Making emotions feel 'earned' or complex",
        "example_emotions": ["bittersweet", "nostalgia", "hope"],
    },
    "HARMONY_TritoneSubstitution": {
        "description": "Replace V7 with bII7",
        "effect": "Jazz sophistication, chromatic movement",
        "use_when": "Adding depth without cliché resolution",
        "example_emotions": ["sophistication", "complexity"],
    },
    "HARMONY_Polytonality": {
        "description": "Stack chords from different keys",
        "effect": "Tension, complexity, disorientation",
        "use_when": "Representing internal conflict or chaos",
        "example_emotions": ["confusion", "conflict", "chaos"],
    },
    "HARMONY_UnresolvedDissonance": {
        "description": "Leave 7ths, 9ths, tritones unresolved",
        "effect": "Lingering tension, incompleteness",
        "use_when": "Grief, longing, or open questions",
        "example_emotions": ["grief", "longing", "uncertainty"],
    },
    
    # Rhythm
    "RHYTHM_ConstantDisplacement": {
        "description": "Shift pattern one 16th note late/early",
        "effect": "Perpetually off-kilter, unsettling",
        "use_when": "Anxiety, instability, before a shift",
        "example_emotions": ["anxiety", "unease", "anticipation"],
    },
    "RHYTHM_TempoFluctuation": {
        "description": "Gradual ±5 BPM drift over phrase",
        "effect": "Organic breathing, tension/release",
        "use_when": "Human feel, emotional intensity",
        "example_emotions": ["intimacy", "vulnerability"],
    },
    "RHYTHM_MetricModulation": {
        "description": "Temporarily change implied time signature",
        "effect": "Disorientation, complexity",
        "use_when": "Representing mental state change",
        "example_emotions": ["confusion", "transformation"],
    },
    "RHYTHM_PolyrhythmicLayers": {
        "description": "Layer conflicting rhythmic patterns",
        "effect": "Complexity, tension, richness",
        "use_when": "Multiple emotions simultaneously",
        "example_emotions": ["complexity", "internal conflict"],
    },
    "RHYTHM_DroppedBeats": {
        "description": "Remove expected beats/hits",
        "effect": "Surprise, space, emphasis",
        "use_when": "Creating impact through absence",
        "example_emotions": ["shock", "emphasis", "breath"],
    },
    
    # Arrangement
    "ARRANGEMENT_UnbalancedDynamics": {
        "description": "Keep element too loud/quiet for standard",
        "effect": "Intentional imbalance, focus shift",
        "use_when": "Drawing attention or creating discomfort",
        "example_emotions": ["obsession", "imbalance"],
    },
    "ARRANGEMENT_StructuralMismatch": {
        "description": "Use unexpected structure for genre",
        "effect": "Subverted expectations, uniqueness",
        "use_when": "Story requires non-standard form",
        "example_emotions": ["defiance", "uniqueness"],
    },
    "ARRANGEMENT_BuriedVocals": {
        "description": "Place vocals below instruments",
        "effect": "Intimacy, dissociation, texture",
        "use_when": "Vulnerability, dream states",
        "example_emotions": ["dissociation", "intimacy", "dreams"],
    },
    "ARRANGEMENT_ExtremeDynamicRange": {
        "description": "Exceed normal dynamic limits",
        "effect": "Dramatic impact, contrast",
        "use_when": "Emotional crescendos, reveals",
        "example_emotions": ["catharsis", "revelation"],
    },
    "ARRANGEMENT_PrematureClimax": {
        "description": "Put peak earlier than expected",
        "effect": "Subversion, reflection time",
        "use_when": "Aftermath is the point",
        "example_emotions": ["aftermath", "reflection"],
    },
    
    # Production
    "PRODUCTION_ExcessiveMud": {
        "description": "Leave 200-400Hz buildup",
        "effect": "Weight, claustrophobia, heaviness",
        "use_when": "Trapped feelings, density",
        "example_emotions": ["trapped", "heavy", "suffocating"],
    },
    "PRODUCTION_PitchImperfection": {
        "description": "Leave natural pitch drift",
        "effect": "Emotional honesty, vulnerability",
        "use_when": "Raw emotional delivery",
        "example_emotions": ["vulnerability", "honesty", "rawness"],
    },
    "PRODUCTION_RoomNoise": {
        "description": "Keep ambient room sound",
        "effect": "Authenticity, intimacy, place",
        "use_when": "Lo-fi aesthetic, presence",
        "example_emotions": ["intimacy", "authenticity"],
    },
    "PRODUCTION_Distortion": {
        "description": "Allow clipping/saturation",
        "effect": "Aggression, urgency, damage",
        "use_when": "Anger, intensity, decay",
        "example_emotions": ["anger", "damage", "intensity"],
    },
    "PRODUCTION_MonoCollapse": {
        "description": "Intentionally narrow stereo field",
        "effect": "Claustrophobia, focus, intimacy",
        "use_when": "Internal monologue, pressure",
        "example_emotions": ["pressure", "focus", "isolation"],
    },
    "PRODUCTION_LoFiDegradation": {
        "description": "Add tape hiss, bit crush, sample rate reduction",
        "effect": "Nostalgia, memory, distance from reality",
        "use_when": "Representing the past, unreliable memory",
        "example_emotions": ["nostalgia", "memory", "dissociation"],
    },
    "PRODUCTION_SilenceAsInstrument": {
        "description": "Use dramatic silence/dropouts mid-phrase",
        "effect": "Shock, breath, emphasis through absence",
        "use_when": "Creating impact, letting words land",
        "example_emotions": ["shock", "contemplation", "weight"],
    },
    "PRODUCTION_ClippingPeaks": {
        "description": "Allow transients to clip intentionally",
        "effect": "Urgency, damage, rawness",
        "use_when": "Anger, desperation, breaking point",
        "example_emotions": ["rage", "desperation", "breaking"],
    },

    # Melody
    "MELODY_AvoidResolution": {
        "description": "End phrases on non-tonic tones",
        "effect": "Incompleteness, searching, yearning",
        "use_when": "Questions without answers",
        "example_emotions": ["longing", "uncertainty", "searching"],
    },
    "MELODY_ExcessiveRepetition": {
        "description": "Repeat melodic cell obsessively",
        "effect": "Hypnotic, obsessive, ritualistic",
        "use_when": "Spiral thoughts, inability to move on",
        "example_emotions": ["obsession", "anxiety", "ritual"],
    },
    "MELODY_AngularIntervals": {
        "description": "Use wide, awkward interval leaps",
        "effect": "Discomfort, unease, otherness",
        "use_when": "Alienation, confusion, discord",
        "example_emotions": ["alienation", "confusion", "discord"],
    },
    "MELODY_AntiClimax": {
        "description": "Build up then resolve downward/weakly",
        "effect": "Disappointment, deflation, anticlimax",
        "use_when": "Failed expectations, letting go",
        "example_emotions": ["disappointment", "resignation", "deflation"],
    },
    "MELODY_MonotoneDrone": {
        "description": "Minimal melodic movement, near-monotone",
        "effect": "Numbness, dissociation, meditation",
        "use_when": "Emotional shutdown, trance states",
        "example_emotions": ["numbness", "dissociation", "meditation"],
    },
    "MELODY_FragmentedPhrases": {
        "description": "Break melody into disconnected fragments",
        "effect": "Fractured thought, interrupted speech",
        "use_when": "Difficulty expressing, trauma",
        "example_emotions": ["trauma", "confusion", "fragmentation"],
    },

    # Texture
    "TEXTURE_FrequencyMasking": {
        "description": "Let elements fight for same frequencies",
        "effect": "Crowded, competitive, overwhelming",
        "use_when": "Internal voices, crowded thoughts",
        "example_emotions": ["overwhelm", "conflict", "noise"],
    },
    "TEXTURE_SparseEmptiness": {
        "description": "Extreme space between elements",
        "effect": "Isolation, exposure, vulnerability",
        "use_when": "Loneliness, nakedness",
        "example_emotions": ["isolation", "vulnerability", "exposure"],
    },
    "TEXTURE_DenseWall": {
        "description": "Stack elements into undifferentiated mass",
        "effect": "Overwhelming force, loss of self",
        "use_when": "Catharsis, surrender, being swept away",
        "example_emotions": ["catharsis", "surrender", "power"],
    },
    "TEXTURE_ConflictingTimbres": {
        "description": "Combine timbres that traditionally clash",
        "effect": "Dissonance, wrongness, tension",
        "use_when": "Things that don't belong together",
        "example_emotions": ["conflict", "dissonance", "wrongness"],
    },
    "TEXTURE_SingleElementFocus": {
        "description": "Strip away all but one element",
        "effect": "Stark truth, nowhere to hide",
        "use_when": "Confession, revelation",
        "example_emotions": ["honesty", "revelation", "nakedness"],
    },
    "TEXTURE_TimbralDrift": {
        "description": "Gradually morph timbre over time",
        "effect": "Transformation, unease, evolution",
        "use_when": "Change, metamorphosis",
        "example_emotions": ["transformation", "unease", "evolution"],
    },

    # Temporal
    "TEMPORAL_ExtendedIntro": {
        "description": "Unusually long intro before main content",
        "effect": "Anticipation, world-building, patience test",
        "use_when": "Setting scene, building dread",
        "example_emotions": ["anticipation", "dread", "patience"],
    },
    "TEMPORAL_AbruptEnding": {
        "description": "End suddenly without resolution",
        "effect": "Shock, incompleteness, cut-off",
        "use_when": "Sudden loss, unexpected ending",
        "example_emotions": ["shock", "loss", "abruptness"],
    },
    "TEMPORAL_TimeStretch": {
        "description": "Stretch or compress time perception",
        "effect": "Altered reality, time distortion",
        "use_when": "Dream states, altered consciousness",
        "example_emotions": ["dissociation", "dream", "unreality"],
    },
    "TEMPORAL_LoopHypnosis": {
        "description": "Loop beyond comfortable repetition",
        "effect": "Hypnotic, meditative, obsessive",
        "use_when": "Trance, circular thoughts",
        "example_emotions": ["trance", "obsession", "meditation"],
    },
    "TEMPORAL_BreathPauses": {
        "description": "Insert pauses like held breath",
        "effect": "Tension, anticipation, human feel",
        "use_when": "Before revelation, gathering courage",
        "example_emotions": ["anticipation", "courage", "hesitation"],
    },
    "TEMPORAL_AccelerandoDecay": {
        "description": "Speed up then decay/fall apart",
        "effect": "Panic, collapse, exhaustion",
        "use_when": "Overwhelm, breaking point, surrender",
        "example_emotions": ["panic", "exhaustion", "surrender"],
    },
}


# =================================================================
# AFFECT → MODE MAPPING
# =================================================================

AFFECT_MODE_MAP = {
    # Maps emotional states to suggested musical modes
    "grief": {"modes": ["Aeolian", "Phrygian"], "tempo_range": (50, 80), "density": "Sparse"},
    "longing": {"modes": ["Dorian", "Mixolydian"], "tempo_range": (60, 90), "density": "Sparse"},
    "defiance": {"modes": ["Phrygian", "Locrian"], "tempo_range": (100, 140), "density": "Full"},
    "hope": {"modes": ["Ionian", "Lydian"], "tempo_range": (90, 120), "density": "Moderate"},
    "rage": {"modes": ["Phrygian", "Locrian"], "tempo_range": (120, 180), "density": "Dense"},
    "tenderness": {"modes": ["Ionian", "Lydian"], "tempo_range": (60, 85), "density": "Sparse"},
    "anxiety": {"modes": ["Locrian", "Phrygian"], "tempo_range": (90, 130), "density": "Moderate"},
    "euphoria": {"modes": ["Ionian", "Lydian"], "tempo_range": (120, 150), "density": "Full"},
    "melancholy": {"modes": ["Aeolian", "Dorian"], "tempo_range": (60, 90), "density": "Sparse"},
    "nostalgia": {"modes": ["Mixolydian", "Dorian"], "tempo_range": (70, 100), "density": "Moderate"},
    "catharsis": {"modes": ["Aeolian", "Phrygian"], "tempo_range": (80, 130), "density": "Dense"},
    "dissociation": {"modes": ["Lydian", "Locrian"], "tempo_range": (60, 90), "density": "Sparse"},
    "determination": {"modes": ["Ionian", "Mixolydian"], "tempo_range": (100, 130), "density": "Full"},
    "surrender": {"modes": ["Aeolian", "Dorian"], "tempo_range": (50, 75), "density": "Sparse"},
}


# =================================================================
# IMAGERY → GENRE MAPPING
# =================================================================

IMAGERY_TO_GENRE_MAP = {
    # Maps imagery textures to genre suggestions
    "Sharp Edges": ["Industrial Pop", "Post-Punk", "Art Rock"],
    "Muffled": ["Lo-Fi Bedroom", "Confessional Acoustic", "Shoegaze"],
    "Open/Vast": ["Cinematic Neo-Soul", "Dream Pop", "Electronic"],
    "Claustrophobic": ["Industrial Pop", "Post-Punk", "Alternative"],
    "Hazy/Dreamy": ["Dream Pop", "Shoegaze", "Lo-Fi Bedroom"],
    "Crystalline": ["Cinematic Neo-Soul", "Chamber Pop", "Art Rock"],
    "Muddy/Thick": ["Lo-Fi Bedroom", "Alternative", "Indie Folk"],
    "Sparse/Empty": ["Confessional Acoustic", "Indie Folk", "Lo-Fi Bedroom"],
    "Chaotic": ["Industrial Pop", "Post-Punk", "Electronic"],
    "Flowing/Liquid": ["Dream Pop", "Shoegaze", "Cinematic Neo-Soul"],
    "Fractured": ["Post-Punk", "Art Rock", "Alternative"],
    "Warm/Enveloping": ["Chamber Pop", "Indie Folk", "R&B"],
    "Cold/Distant": ["Synthwave", "Electronic", "Shoegaze"],
    "Blinding Light": ["Cinematic Neo-Soul", "Euphoria", "Electronic"],
    "Deep Shadow": ["Lo-Fi Bedroom", "Shoegaze", "Alternative"],
}


# =================================================================
# NARRATIVE ARC → GROOVE FEEL MAPPING
# =================================================================

NARRATIVE_ARC_TO_GROOVE_MAP = {
    # Maps narrative arcs to groove feel suggestions
    "Climb-to-Climax": ["Straight/Driving", "Push-Pull", "Organic/Breathing"],
    "Slow Reveal": ["Laid Back", "Organic/Breathing", "Rubato/Free"],
    "Repetitive Despair": ["Mechanical", "Straight/Driving", "Syncopated"],
    "Static Reflection": ["Rubato/Free", "Laid Back", "Organic/Breathing"],
    "Sudden Shift": ["Straight/Driving", "Push-Pull", "Mechanical"],
    "Descent": ["Laid Back", "Rubato/Free", "Organic/Breathing"],
    "Rise and Fall": ["Push-Pull", "Straight/Driving", "Organic/Breathing"],
    "Spiral": ["Mechanical", "Syncopated", "Straight/Driving"],
}


# =================================================================
# TEXTURE → PRODUCTION MAPPING
# =================================================================

TEXTURE_PRODUCTION_MAP = {
    # Maps texture types to suggested production characteristics
    "Ethereal": {
        "reverb": "long",
        "delay": "ambient",
        "eq_character": "air",
        "stereo_width": "wide",
        "compression": "light",
    },
    "Intimate": {
        "reverb": "none/room",
        "delay": "none/subtle",
        "eq_character": "presence",
        "stereo_width": "narrow",
        "compression": "moderate",
    },
    "Massive": {
        "reverb": "medium-large",
        "delay": "rhythmic",
        "eq_character": "full",
        "stereo_width": "wide",
        "compression": "heavy",
    },
    "Skeletal": {
        "reverb": "none/room",
        "delay": "none",
        "eq_character": "thin",
        "stereo_width": "mono/narrow",
        "compression": "none",
    },
    "Lush": {
        "reverb": "medium-long",
        "delay": "modulated",
        "eq_character": "warm",
        "stereo_width": "wide",
        "compression": "glue",
    },
    "Harsh": {
        "reverb": "short/none",
        "delay": "none",
        "eq_character": "bright/cutting",
        "stereo_width": "variable",
        "compression": "aggressive",
    },
    "Murky": {
        "reverb": "dark",
        "delay": "degraded",
        "eq_character": "muddy",
        "stereo_width": "mono",
        "compression": "squashed",
    },
    "Crystalline": {
        "reverb": "short/bright",
        "delay": "clean",
        "eq_character": "bright/clear",
        "stereo_width": "precise",
        "compression": "light",
    },
}


# =================================================================
# DATA CLASSES: Song Intent Structure
# =================================================================

@dataclass
class SongRoot:
    """
    Phase 0: The Core Wound/Desire
    
    Deep interrogation to find what the song NEEDS to express.
    """
    core_event: str = ""           # The inciting moment/realization
    core_resistance: str = ""      # What's holding you back
    core_longing: str = ""         # What you ultimately want to feel
    core_stakes: str = ""          # What's at risk
    core_transformation: str = ""  # How you want to feel when done


@dataclass
class SongIntent:
    """
    Phase 1: Emotional & Intent
    
    Validated by Phase 0, guides all technical decisions.
    """
    mood_primary: str = ""                  # Primary emotion
    mood_secondary_tension: float = 0.5     # Tension level 0.0-1.0
    imagery_texture: str = ""               # Visual/tactile quality
    vulnerability_scale: str = "Medium"     # Low/Medium/High
    narrative_arc: str = ""                 # Structural emotion


@dataclass
class TechnicalConstraints:
    """
    Phase 2: Technical Constraints
    
    Implementation of intent into concrete musical decisions.
    """
    technical_genre: str = ""
    technical_tempo_range: Tuple[int, int] = (80, 120)
    technical_key: str = ""
    technical_mode: str = ""
    technical_groove_feel: str = ""
    technical_rule_to_break: str = ""
    rule_breaking_justification: str = ""


@dataclass
class SystemDirective:
    """What DAiW should generate."""
    output_target: str = ""           # What to generate
    output_feedback_loop: str = ""    # Which modules to iterate


@dataclass
class CompleteSongIntent:
    """
    Complete song intent combining all phases.
    
    This is the full specification for a song that DAiW
    uses to generate meaningful, emotionally-aligned output.
    """
    # Phase 0
    song_root: SongRoot = field(default_factory=SongRoot)
    
    # Phase 1
    song_intent: SongIntent = field(default_factory=SongIntent)
    
    # Phase 2
    technical_constraints: TechnicalConstraints = field(default_factory=TechnicalConstraints)
    
    # System
    system_directive: SystemDirective = field(default_factory=SystemDirective)
    
    # Meta
    title: str = ""
    created: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "created": self.created,
            "song_root": {
                "core_event": self.song_root.core_event,
                "core_resistance": self.song_root.core_resistance,
                "core_longing": self.song_root.core_longing,
                "core_stakes": self.song_root.core_stakes,
                "core_transformation": self.song_root.core_transformation,
            },
            "song_intent": {
                "mood_primary": self.song_intent.mood_primary,
                "mood_secondary_tension": self.song_intent.mood_secondary_tension,
                "imagery_texture": self.song_intent.imagery_texture,
                "vulnerability_scale": self.song_intent.vulnerability_scale,
                "narrative_arc": self.song_intent.narrative_arc,
            },
            "technical_constraints": {
                "technical_genre": self.technical_constraints.technical_genre,
                "technical_tempo_range": list(self.technical_constraints.technical_tempo_range),
                "technical_key": self.technical_constraints.technical_key,
                "technical_mode": self.technical_constraints.technical_mode,
                "technical_groove_feel": self.technical_constraints.technical_groove_feel,
                "technical_rule_to_break": self.technical_constraints.technical_rule_to_break,
                "rule_breaking_justification": self.technical_constraints.rule_breaking_justification,
            },
            "system_directive": {
                "output_target": self.system_directive.output_target,
                "output_feedback_loop": self.system_directive.output_feedback_loop,
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CompleteSongIntent":
        """Create from dictionary."""
        intent = cls()
        intent.title = data.get("title", "")
        intent.created = data.get("created", "")
        
        if "song_root" in data:
            root = data["song_root"]
            intent.song_root = SongRoot(
                core_event=root.get("core_event", ""),
                core_resistance=root.get("core_resistance", ""),
                core_longing=root.get("core_longing", ""),
                core_stakes=root.get("core_stakes", ""),
                core_transformation=root.get("core_transformation", ""),
            )
        
        if "song_intent" in data:
            si = data["song_intent"]
            intent.song_intent = SongIntent(
                mood_primary=si.get("mood_primary", ""),
                mood_secondary_tension=si.get("mood_secondary_tension", 0.5),
                imagery_texture=si.get("imagery_texture", ""),
                vulnerability_scale=si.get("vulnerability_scale", "Medium"),
                narrative_arc=si.get("narrative_arc", ""),
            )
        
        if "technical_constraints" in data:
            tc = data["technical_constraints"]
            tempo = tc.get("technical_tempo_range", [80, 120])
            intent.technical_constraints = TechnicalConstraints(
                technical_genre=tc.get("technical_genre", ""),
                technical_tempo_range=tuple(tempo) if isinstance(tempo, list) else tempo,
                technical_key=tc.get("technical_key", ""),
                technical_mode=tc.get("technical_mode", ""),
                technical_groove_feel=tc.get("technical_groove_feel", ""),
                technical_rule_to_break=tc.get("technical_rule_to_break", ""),
                rule_breaking_justification=tc.get("rule_breaking_justification", ""),
            )
        
        if "system_directive" in data:
            sd = data["system_directive"]
            intent.system_directive = SystemDirective(
                output_target=sd.get("output_target", ""),
                output_feedback_loop=sd.get("output_feedback_loop", ""),
            )
        
        return intent
    
    def save(self, path: str):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "CompleteSongIntent":
        """Load from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# =================================================================
# HELPER FUNCTIONS
# =================================================================

def suggest_rule_break(emotion: str) -> List[Dict]:
    """
    Suggest appropriate rules to break based on target emotion.
    
    Args:
        emotion: Target emotion (grief, defiance, etc.)
    
    Returns:
        List of rule-breaking suggestions with justifications
    """
    emotion_lower = emotion.lower()
    suggestions = []
    
    for rule_key, rule_data in RULE_BREAKING_EFFECTS.items():
        if any(e in emotion_lower for e in rule_data.get("example_emotions", [])):
            suggestions.append({
                "rule": rule_key,
                "description": rule_data["description"],
                "effect": rule_data["effect"],
                "use_when": rule_data["use_when"],
            })
    
    return suggestions


def get_rule_breaking_info(rule_key: str) -> Optional[Dict]:
    """Get detailed info about a rule-breaking option."""
    return RULE_BREAKING_EFFECTS.get(rule_key)


def validate_intent(intent: CompleteSongIntent) -> List[str]:
    """
    Validate a song intent for completeness and consistency.
    
    Returns list of issues found (empty = valid).
    """
    issues = []
    
    # Phase 0 checks
    if not intent.song_root.core_event:
        issues.append("Phase 0: Missing core_event - what happened?")
    if not intent.song_root.core_longing:
        issues.append("Phase 0: Missing core_longing - what do you want to feel?")
    
    # Phase 1 checks (use enhanced validation)
    phase1_issues = validate_phase1(intent.song_intent)
    issues.extend(phase1_issues)
    
    # Phase 2 checks (use enhanced validation)
    phase2_issues = validate_phase2(intent.technical_constraints)
    issues.extend(phase2_issues)
    
    # Consistency checks
    if intent.song_intent.vulnerability_scale == "High":
        if intent.song_intent.mood_secondary_tension < 0.3:
            issues.append("Consistency: High vulnerability usually implies some tension (tension is very low)")
    
    return issues


def list_all_rules() -> Dict[str, List[str]]:
    """Get all available rule-breaking options by category."""
    return {
        "Harmony": [e.value for e in HarmonyRuleBreak],
        "Rhythm": [e.value for e in RhythmRuleBreak],
        "Arrangement": [e.value for e in ArrangementRuleBreak],
        "Production": [e.value for e in ProductionRuleBreak],
        "Melody": [e.value for e in MelodyRuleBreak],
        "Texture": [e.value for e in TextureRuleBreak],
        "Temporal": [e.value for e in TemporalRuleBreak],
    }


def get_all_rule_breaking_options() -> List[str]:
    """Get flat list of all rule-breaking option values."""
    rules_dict = list_all_rules()
    all_rules = []
    for category_rules in rules_dict.values():
        all_rules.extend(category_rules)
    return all_rules


def get_rule_categories() -> Dict[str, List[str]]:
    """Get rules organized by category for selection."""
    return list_all_rules()


def get_affect_mapping(emotion: str) -> Optional[Dict]:
    """
    Get musical parameter suggestions for an emotional state.

    Args:
        emotion: The target emotion (grief, hope, etc.)

    Returns:
        Dict with modes, tempo_range, density suggestions or None if not found
    """
    return AFFECT_MODE_MAP.get(emotion.lower())


def get_texture_production(texture: str) -> Optional[Dict]:
    """
    Get production parameter suggestions for a texture type.

    Args:
        texture: The target texture (Ethereal, Intimate, etc.)

    Returns:
        Dict with reverb, delay, eq, stereo, compression suggestions or None
    """
    return TEXTURE_PRODUCTION_MAP.get(texture)


def suggest_full_palette(emotion: str) -> Dict:
    """
    Get a complete musical palette suggestion for an emotion.

    Combines affect mapping, rule suggestions, and texture recommendations.

    Args:
        emotion: Target emotion

    Returns:
        Dict with modes, tempo, density, suggested_rules, texture recommendations
    """
    result = {
        "emotion": emotion,
        "affect_mapping": get_affect_mapping(emotion),
        "suggested_rules": suggest_rule_break(emotion),
        "texture_options": [],
    }

    # Suggest textures based on emotion
    emotion_texture_map = {
        "grief": ["Murky", "Skeletal", "Ethereal"],
        "longing": ["Ethereal", "Intimate", "Lush"],
        "defiance": ["Harsh", "Massive", "Crystalline"],
        "hope": ["Ethereal", "Lush", "Crystalline"],
        "rage": ["Harsh", "Dense", "Massive"],
        "tenderness": ["Intimate", "Lush", "Ethereal"],
        "anxiety": ["Harsh", "Skeletal", "Murky"],
        "euphoria": ["Massive", "Lush", "Crystalline"],
        "melancholy": ["Murky", "Ethereal", "Intimate"],
        "nostalgia": ["Murky", "Intimate", "Lush"],
        "catharsis": ["Massive", "Dense", "Ethereal"],
        "dissociation": ["Ethereal", "Murky", "Skeletal"],
        "determination": ["Massive", "Crystalline", "Full"],
        "surrender": ["Ethereal", "Skeletal", "Intimate"],
    }

    textures = emotion_texture_map.get(emotion.lower(), [])
    for tex in textures:
        prod = get_texture_production(tex)
        if prod:
            result["texture_options"].append({
                "texture": tex,
                "production": prod,
            })

    return result


def load_schema_options() -> Dict[str, List[str]]:
    """
    Load enum options from song_intent_schema.yaml.
    
    Returns:
        Dict mapping option names to lists of valid values
    """
    schema_path = Path(__file__).parent.parent / "data" / "song_intent_schema.yaml"
    
    if not HAS_YAML:
        # Fallback to hardcoded values if PyYAML not available
        return _get_fallback_options()
    
    try:
        with open(schema_path, 'r') as f:
            data = yaml.safe_load(f)
        
        enums = data.get("enums", {})
        return {
            "mood_primary": enums.get("mood_primary_options", []),
            "imagery_texture": enums.get("imagery_texture_options", []),
            "vulnerability_scale": enums.get("vulnerability_scale_options", []),
            "narrative_arc": enums.get("narrative_arc_options", []),
            "core_stakes": enums.get("core_stakes_options", []),
            "genre": enums.get("genre_options", []),
            "groove_feel": enums.get("groove_feel_options", []),
        }
    except Exception as e:
        # Fallback to hardcoded values if YAML can't be loaded
        return _get_fallback_options()


def suggest_phase2_from_phase1(song_intent: SongIntent) -> Dict:
    """
    Suggest Phase 2 technical constraints based on Phase 1 emotional intent.
    
    Args:
        song_intent: Phase 1 SongIntent data
    
    Returns:
        Dict with suggested values for Phase 2 fields
    """
    suggestions = {
        "technical_genre": "",
        "technical_tempo_range": (80, 120),
        "technical_key": "F",
        "technical_mode": "major",
        "technical_groove_feel": "Organic/Breathing",
        "technical_rule_to_break": "",
    }
    
    # Get affect mapping for mood_primary
    mood_lower = song_intent.mood_primary.lower()
    affect_map = get_affect_mapping(mood_lower)
    
    if affect_map:
        # Suggest mode and tempo from affect mapping
        modes = affect_map.get("modes", [])
        if modes:
            suggestions["technical_mode"] = modes[0].lower()  # Use first suggested mode
        tempo_range = affect_map.get("tempo_range")
        if tempo_range:
            suggestions["technical_tempo_range"] = tempo_range
    
    # Suggest genre from imagery_texture
    imagery_lower = song_intent.imagery_texture.lower()
    for imagery_key, genres in IMAGERY_TO_GENRE_MAP.items():
        if imagery_key.lower() in imagery_lower or imagery_lower in imagery_key.lower():
            if genres:
                suggestions["technical_genre"] = genres[0]  # Use first suggested genre
            break
    
    # Suggest groove_feel from narrative_arc
    arc_lower = song_intent.narrative_arc.lower()
    for arc_key, groove_feels in NARRATIVE_ARC_TO_GROOVE_MAP.items():
        if arc_key.lower() in arc_lower or arc_lower in arc_key.lower():
            if groove_feels:
                suggestions["technical_groove_feel"] = groove_feels[0]  # Use first suggested
            break
    
    # Suggest rule to break from mood_primary
    rule_suggestions = suggest_rule_break(song_intent.mood_primary)
    if rule_suggestions:
        suggestions["technical_rule_to_break"] = rule_suggestions[0]["rule"]
    
    # Suggest key based on mode (simple mapping)
    mode_to_key = {
        "ionian": "C",
        "aeolian": "A",
        "dorian": "D",
        "phrygian": "E",
        "lydian": "F",
        "mixolydian": "G",
        "locrian": "B",
    }
    mode_lower = suggestions["technical_mode"].lower()
    if mode_lower in mode_to_key:
        suggestions["technical_key"] = mode_to_key[mode_lower]
    
    return suggestions


def _get_fallback_options() -> Dict[str, List[str]]:
    """Fallback options if YAML can't be loaded."""
    return {
        "mood_primary": ["Grief", "Joy", "Defiance", "Longing", "Rage", "Nostalgia", "Melancholy", "Euphoria", "Desperation", "Serenity", "Confusion", "Determination", "Bittersweet", "Triumphant Hope", "Dissociation", "Acceptance", "Liberation", "Nervousness"],
        "imagery_texture": ["Sharp Edges", "Muffled", "Open/Vast", "Claustrophobic", "Hazy/Dreamy", "Crystalline", "Muddy/Thick", "Sparse/Empty", "Chaotic", "Flowing/Liquid", "Fractured", "Warm/Enveloping", "Cold/Distant", "Blinding Light", "Deep Shadow"],
        "vulnerability_scale": ["Low", "Medium", "High"],
        "narrative_arc": ["Climb-to-Climax", "Slow Reveal", "Repetitive Despair", "Static Reflection", "Sudden Shift", "Descent", "Rise and Fall", "Spiral"],
        "core_stakes": ["Personal", "Relational", "Existential", "Survival", "Creative", "Moral"],
        "genre": ["Cinematic Neo-Soul", "Lo-Fi Bedroom", "Industrial Pop", "Synthwave", "Confessional Acoustic", "Art Rock", "Indie Folk", "Post-Punk", "Chamber Pop", "Electronic", "Hip-Hop", "R&B", "Alternative", "Shoegaze", "Dream Pop"],
        "groove_feel": ["Straight/Driving", "Laid Back", "Swung", "Syncopated", "Rubato/Free", "Mechanical", "Organic/Breathing", "Push-Pull"],
    }


def collect_phase1_interactive(song_root: Optional[SongRoot] = None) -> SongIntent:
    """
    Interactively collect Phase 1 (Emotional Intent) data from user.
    
    Args:
        song_root: Optional Phase 0 data to inform questions
    
    Returns:
        SongIntent with collected data
    """
    options = load_schema_options()
    
    print("\n" + "=" * 60)
    print("PHASE 1: EMOTIONAL & INTENT")
    print("=" * 60)
    print("\nValidated by Phase 0, guides all technical decisions.\n")
    
    # Mood Primary
    print("1. PRIMARY EMOTION")
    print("   What's the dominant emotion?")
    if options["mood_primary"]:
        print(f"   Options: {', '.join(options['mood_primary'][:10])}...")
        print("   (You can type your own or choose from the list)")
    mood_primary = input("   > ").strip()
    if not mood_primary:
        mood_primary = "[Primary emotion]"
    
    # Secondary Tension
    print("\n2. SECONDARY TENSION (0.0 - 1.0)")
    print("   Internal conflict level:")
    print("   0.0 = calm, resolved")
    print("   0.5 = balanced tension")
    print("   1.0 = anxious, high tension")
    while True:
        try:
            tension_input = input("   > ").strip()
            if not tension_input:
                mood_secondary_tension = 0.5
                break
            mood_secondary_tension = float(tension_input)
            if 0.0 <= mood_secondary_tension <= 1.0:
                break
            else:
                print("   Please enter a value between 0.0 and 1.0")
        except ValueError:
            print("   Please enter a number between 0.0 and 1.0")
    
    # Imagery Texture
    print("\n3. IMAGERY TEXTURE")
    print("   Visual/tactile quality - how should this song FEEL?")
    if options["imagery_texture"]:
        print("   Options:")
        for i, opt in enumerate(options["imagery_texture"], 1):
            print(f"   {i}. {opt}")
        print("   (You can type your own or choose from the list)")
    imagery_texture = input("   > ").strip()
    if not imagery_texture:
        imagery_texture = "[Visual/tactile quality]"
    
    # Vulnerability Scale
    print("\n4. VULNERABILITY SCALE")
    print("   How exposed should this song be?")
    if options["vulnerability_scale"]:
        for i, opt in enumerate(options["vulnerability_scale"], 1):
            desc = {
                "Low": "Guarded, protective",
                "Medium": "Honest but controlled",
                "High": "Raw, exposed"
            }.get(opt, "")
            print(f"   {i}. {opt} - {desc}")
    while True:
        vuln_input = input("   > ").strip()
        if not vuln_input:
            vulnerability_scale = "Medium"
            break
        vuln_lower = vuln_input.lower()
        if vuln_lower in ["low", "medium", "high"]:
            vulnerability_scale = vuln_input.capitalize()
            break
        elif vuln_input in options["vulnerability_scale"]:
            vulnerability_scale = vuln_input
            break
        else:
            print("   Please enter: Low, Medium, or High")
    
    # Narrative Arc
    print("\n5. NARRATIVE ARC")
    print("   How should the emotion evolve structurally?")
    if options["narrative_arc"]:
        print("   Options:")
        for i, opt in enumerate(options["narrative_arc"], 1):
            print(f"   {i}. {opt}")
        print("   (You can type your own or choose from the list)")
    narrative_arc = input("   > ").strip()
    if not narrative_arc:
        narrative_arc = "Climb-to-Climax"
    
    print("\n" + "=" * 60)
    print("Phase 1 Complete!")
    print("=" * 60 + "\n")
    
    return SongIntent(
        mood_primary=mood_primary,
        mood_secondary_tension=mood_secondary_tension,
        imagery_texture=imagery_texture,
        vulnerability_scale=vulnerability_scale,
        narrative_arc=narrative_arc,
    )


def validate_phase1(song_intent: SongIntent) -> List[str]:
    """
    Validate Phase 1 fields against schema options.
    
    Returns list of validation issues (empty = valid).
    """
    issues = []
    options = load_schema_options()
    
    # Validate mood_primary
    if not song_intent.mood_primary or song_intent.mood_primary.startswith("["):
        issues.append("Phase 1: mood_primary is required")
    elif options["mood_primary"] and song_intent.mood_primary not in options["mood_primary"]:
        # Warn but don't fail - allow custom emotions
        pass
    
    # Validate mood_secondary_tension
    if not (0.0 <= song_intent.mood_secondary_tension <= 1.0):
        issues.append("Phase 1: mood_secondary_tension must be between 0.0 and 1.0")
    
    # Validate imagery_texture
    if not song_intent.imagery_texture or song_intent.imagery_texture.startswith("["):
        issues.append("Phase 1: imagery_texture is required")
    
    # Validate vulnerability_scale
    if song_intent.vulnerability_scale not in options["vulnerability_scale"]:
        issues.append(f"Phase 1: vulnerability_scale must be one of {options['vulnerability_scale']}")
    
    # Validate narrative_arc
    if not song_intent.narrative_arc or song_intent.narrative_arc.startswith("["):
        issues.append("Phase 1: narrative_arc is required")
    
    return issues


def collect_phase2_interactive(song_intent: Optional[SongIntent] = None) -> TechnicalConstraints:
    """
    Interactively collect Phase 2 (Technical Constraints) data from user.
    
    Args:
        song_intent: Optional Phase 1 data to inform suggestions
    
    Returns:
        TechnicalConstraints with collected data
    """
    options = load_schema_options()
    
    print("\n" + "=" * 60)
    print("PHASE 2: TECHNICAL CONSTRAINTS")
    print("=" * 60)
    print("\nImplementation of intent into concrete musical decisions.\n")
    
    # Show smart suggestions if Phase 1 data provided
    suggestions = None
    if song_intent:
        suggestions = suggest_phase2_from_phase1(song_intent)
        print("💡 Smart suggestions based on Phase 1:")
        if suggestions.get("technical_genre"):
            print(f"   Genre: {suggestions['technical_genre']}")
        if suggestions.get("technical_tempo_range"):
            print(f"   Tempo: {suggestions['technical_tempo_range'][0]}-{suggestions['technical_tempo_range'][1]} BPM")
        if suggestions.get("technical_key") and suggestions.get("technical_mode"):
            print(f"   Key/Mode: {suggestions['technical_key']} {suggestions['technical_mode']}")
        if suggestions.get("technical_groove_feel"):
            print(f"   Groove Feel: {suggestions['technical_groove_feel']}")
        if suggestions.get("technical_rule_to_break"):
            rule_info = get_rule_breaking_info(suggestions['technical_rule_to_break'])
            if rule_info:
                print(f"   Rule to Break: {suggestions['technical_rule_to_break']}")
                print(f"      Effect: {rule_info.get('effect', '')}")
        print()
    
    # 1. Genre
    print("1. GENRE")
    print("   What genre/style fits this song?")
    if options["genre"]:
        print("   Options:")
        for i, opt in enumerate(options["genre"], 1):
            marker = " ← suggested" if suggestions and suggestions.get("technical_genre") == opt else ""
            print(f"   {i}. {opt}{marker}")
        print("   (You can type your own or choose from the list)")
    genre_input = input("   > ").strip()
    if not genre_input:
        technical_genre = suggestions.get("technical_genre", "") if suggestions else ""
    elif genre_input.isdigit() and options["genre"]:
        idx = int(genre_input) - 1
        if 0 <= idx < len(options["genre"]):
            technical_genre = options["genre"][idx]
        else:
            technical_genre = genre_input
    else:
        technical_genre = genre_input
    
    # 2. Tempo Range
    print("\n2. TEMPO RANGE (BPM)")
    print("   Enter tempo range as two numbers (e.g., 80 120)")
    if suggestions and suggestions.get("technical_tempo_range"):
        print(f"   Suggested: {suggestions['technical_tempo_range'][0]}-{suggestions['technical_tempo_range'][1]} BPM")
    while True:
        tempo_input = input("   > ").strip()
        if not tempo_input:
            if suggestions and suggestions.get("technical_tempo_range"):
                technical_tempo_range = suggestions["technical_tempo_range"]
                break
            else:
                technical_tempo_range = (80, 120)
                break
        parts = tempo_input.split()
        if len(parts) >= 2:
            try:
                low = int(parts[0])
                high = int(parts[1])
                if 40 <= low <= 240 and 40 <= high <= 240 and low <= high:
                    technical_tempo_range = (low, high)
                    break
                else:
                    print("   Please enter two numbers between 40-240 BPM, with first <= second")
            except ValueError:
                print("   Please enter two numbers (e.g., 80 120)")
        else:
            print("   Please enter two numbers separated by space (e.g., 80 120)")
    
    # 3. Key
    print("\n3. MUSICAL KEY")
    print("   What key? (e.g., C, F#, Bb, E minor)")
    if suggestions and suggestions.get("technical_key"):
        print(f"   Suggested: {suggestions['technical_key']}")
    key_input = input("   > ").strip()
    if not key_input:
        technical_key = suggestions.get("technical_key", "") if suggestions else ""
    else:
        technical_key = key_input
    
    # 4. Mode
    print("\n4. MODE")
    print("   Mode: major, minor, or modal (e.g., Dorian, Lydian)")
    mode_options = ["major", "minor", "Ionian", "Dorian", "Phrygian", "Lydian", "Mixolydian", "Aeolian", "Locrian"]
    print("   Common options: major, minor, Dorian, Lydian, Mixolydian, Aeolian")
    if suggestions and suggestions.get("technical_mode"):
        print(f"   Suggested: {suggestions['technical_mode']}")
    mode_input = input("   > ").strip()
    if not mode_input:
        technical_mode = suggestions.get("technical_mode", "major") if suggestions else "major"
    else:
        technical_mode = mode_input
    
    # 5. Groove Feel
    print("\n5. GROOVE FEEL")
    print("   What's the rhythmic feel?")
    if options["groove_feel"]:
        print("   Options:")
        for i, opt in enumerate(options["groove_feel"], 1):
            marker = " ← suggested" if suggestions and suggestions.get("technical_groove_feel") == opt else ""
            print(f"   {i}. {opt}{marker}")
    groove_input = input("   > ").strip()
    if not groove_input:
        technical_groove_feel = suggestions.get("technical_groove_feel", "") if suggestions else ""
    elif groove_input.isdigit() and options["groove_feel"]:
        idx = int(groove_input) - 1
        if 0 <= idx < len(options["groove_feel"]):
            technical_groove_feel = options["groove_feel"][idx]
        else:
            technical_groove_feel = groove_input
    else:
        technical_groove_feel = groove_input
    
    # 6. Rule to Break
    print("\n6. RULE TO BREAK")
    print("   What musical rule should be intentionally broken? (Optional)")
    print("   Type 'list' to see all categories, or 'skip' to skip")
    if suggestions and suggestions.get("technical_rule_to_break"):
        rule_info = get_rule_breaking_info(suggestions['technical_rule_to_break'])
        if rule_info:
            print(f"   Suggested: {suggestions['technical_rule_to_break']}")
            print(f"      {rule_info.get('description', '')}")
            print(f"      Effect: {rule_info.get('effect', '')}")
    
    rule_categories = get_rule_categories()
    technical_rule_to_break = ""
    
    while True:
        rule_input = input("   > ").strip()
        if not rule_input or rule_input.lower() == "skip":
            break
        elif rule_input.lower() == "list":
            print("\n   Rule Categories:")
            for cat_idx, (category, rules) in enumerate(rule_categories.items(), 1):
                print(f"\n   {cat_idx}. {category}:")
                for rule_idx, rule in enumerate(rules[:5], 1):  # Show first 5
                    rule_info = get_rule_breaking_info(rule)
                    desc = rule_info.get("description", "") if rule_info else ""
                    print(f"      {rule_idx}. {rule}")
                    if desc:
                        print(f"         {desc}")
                if len(rules) > 5:
                    print(f"      ... and {len(rules) - 5} more")
            print("\n   Enter rule name or 'skip' to skip:")
            continue
        elif rule_input in get_all_rule_breaking_options():
            technical_rule_to_break = rule_input
            rule_info = get_rule_breaking_info(rule_input)
            if rule_info:
                print(f"   ✓ Selected: {rule_info.get('description', rule_input)}")
            break
        else:
            print(f"   '{rule_input}' not found. Type 'list' to see options or 'skip' to skip")
    
    # 7. Rule Breaking Justification
    rule_breaking_justification = ""
    if technical_rule_to_break:
        print("\n7. RULE BREAKING JUSTIFICATION")
        print("   WHY break this rule? (Required)")
        print("   What emotional effect does breaking this rule achieve?")
        while True:
            justification_input = input("   > ").strip()
            if justification_input:
                rule_breaking_justification = justification_input
                break
            else:
                print("   Justification is required when a rule is selected. Please explain WHY.")
    
    print("\n" + "=" * 60)
    print("Phase 2 Complete!")
    print("=" * 60 + "\n")
    
    return TechnicalConstraints(
        technical_genre=technical_genre,
        technical_tempo_range=technical_tempo_range,
        technical_key=technical_key,
        technical_mode=technical_mode,
        technical_groove_feel=technical_groove_feel,
        technical_rule_to_break=technical_rule_to_break,
        rule_breaking_justification=rule_breaking_justification,
    )


def validate_phase2(technical_constraints: TechnicalConstraints) -> List[str]:
    """
    Validate Phase 2 fields against schema options and constraints.
    
    Returns list of validation issues (empty = valid).
    """
    issues = []
    options = load_schema_options()
    
    # Validate genre (warn if custom, but allow)
    if technical_constraints.technical_genre:
        if options["genre"] and technical_constraints.technical_genre not in options["genre"]:
            # Warn but don't fail - allow custom genres
            pass
    
    # Validate tempo_range
    tempo_range = technical_constraints.technical_tempo_range
    if not isinstance(tempo_range, tuple) or len(tempo_range) != 2:
        issues.append("Phase 2: technical_tempo_range must be a tuple of two integers")
    else:
        low, high = tempo_range
        if not isinstance(low, int) or not isinstance(high, int):
            issues.append("Phase 2: technical_tempo_range values must be integers")
        elif not (40 <= low <= 240) or not (40 <= high <= 240):
            issues.append("Phase 2: technical_tempo_range values must be between 40-240 BPM")
        elif low > high:
            issues.append("Phase 2: technical_tempo_range first value must be <= second value")
    
    # Validate key (lenient - just check not empty if mode is set)
    if technical_constraints.technical_mode and not technical_constraints.technical_key:
        # Key is optional, but warn if mode is set
        pass
    
    # Validate mode (lenient - allow custom)
    if technical_constraints.technical_mode:
        # Mode can be custom, just check it's not obviously invalid
        pass
    
    # Validate groove_feel
    if technical_constraints.technical_groove_feel:
        if options["groove_feel"] and technical_constraints.technical_groove_feel not in options["groove_feel"]:
            issues.append(f"Phase 2: technical_groove_feel must be one of {options['groove_feel']}")
    
    # Validate rule_to_break exists in available rules (if set)
    if technical_constraints.technical_rule_to_break:
        all_rules = get_all_rule_breaking_options()
        if technical_constraints.technical_rule_to_break not in all_rules:
            issues.append(f"Phase 2: technical_rule_to_break '{technical_constraints.technical_rule_to_break}' not found in available rules")
        
        # Require justification if rule is set
        if not technical_constraints.rule_breaking_justification:
            issues.append("Phase 2: Rule to break specified without justification - WHY break this rule?")
    
    return issues
