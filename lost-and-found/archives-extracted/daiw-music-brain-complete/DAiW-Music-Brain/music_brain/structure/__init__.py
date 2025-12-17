"""DAiW Structure Module - Data models and engines"""

from .models import (
    CoreWoundModel,
    IntentModel,
    ConstraintModel,
    RuleBreakModel,
    DirectiveModel,
    InstrumentPalette,
    FinalPayload,
    NarrativeArc,
    VulnerabilityScale,
    GrooveFeel,
    HarmonicComplexity,
    RuleToBreak,
    OutputTarget,
    FeedbackLoop,
    OutputFormat,
    PALETTE_C1_BASIC,
    PALETTE_C2_ADVANCED,
    create_example_payload,
)

from .tension import generate_tension_curve, choose_structure_type_for_mood
from .progression import parse_chord, parse_progression_string, analyze_progression
from .comprehensive_engine import (
    TherapySession,
    HarmonyPlan,
    AffectResult,
    AffectAnalyzer,
    render_plan_to_midi,
    select_kit_for_mood,
)

__all__ = [
    # Models
    "CoreWoundModel",
    "IntentModel",
    "ConstraintModel",
    "RuleBreakModel",
    "DirectiveModel",
    "InstrumentPalette",
    "FinalPayload",
    # Enums
    "NarrativeArc",
    "VulnerabilityScale",
    "GrooveFeel",
    "HarmonicComplexity",
    "RuleToBreak",
    "OutputTarget",
    "FeedbackLoop",
    "OutputFormat",
    # Presets
    "PALETTE_C1_BASIC",
    "PALETTE_C2_ADVANCED",
    # Functions
    "create_example_payload",
    "generate_tension_curve",
    "choose_structure_type_for_mood",
    "parse_chord",
    "parse_progression_string",
    "analyze_progression",
    # Engine
    "TherapySession",
    "HarmonyPlan",
    "AffectResult",
    "AffectAnalyzer",
    "render_plan_to_midi",
    "select_kit_for_mood",
]
