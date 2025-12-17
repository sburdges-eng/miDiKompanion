"""
DAiW Music Brain
================

Digital Audio intelligent Workstation - A CLI toolkit and Python library for
emotion-first music generation.

Core Philosophy: "Interrogate Before Generate"

Modules:
    structure   - Data models, interrogation, constraints, comprehensive engine
    modules     - Chord generation, harmony
    groove      - Humanization, groove templates, extraction
    session     - Intent processing, vernacular translation, teaching
    audio       - Audio analysis (requires numpy/librosa)
    daw         - DAW integration (Logic Pro, etc.)

Quick Start:
    from music_brain.structure.comprehensive_engine import TherapySession
    
    session = TherapySession()
    mood = session.process_core_input("I feel broken")
    session.set_scales(motivation=6, chaos=0.4)
    plan = session.generate_plan()

CLI Usage:
    daiw new              # Start emotional interrogation
    daiw constraints      # Set technical parameters
    daiw execute          # Generate MIDI output
    daiw diagnose "F-C-Am-Dm"  # Analyze progression
"""

__version__ = "0.1.0"
__author__ = "DAiW Project"

from .structure.models import (
    CoreWoundModel,
    IntentModel,
    ConstraintModel,
    RuleBreakModel,
    DirectiveModel,
    FinalPayload,
    NarrativeArc,
    VulnerabilityScale,
    GrooveFeel,
    HarmonicComplexity,
    RuleToBreak,
)

from .structure.comprehensive_engine import (
    TherapySession,
    HarmonyPlan,
    AffectAnalyzer,
    render_plan_to_midi,
)

__all__ = [
    # Models
    "CoreWoundModel",
    "IntentModel",
    "ConstraintModel",
    "RuleBreakModel",
    "DirectiveModel",
    "FinalPayload",
    # Enums
    "NarrativeArc",
    "VulnerabilityScale",
    "GrooveFeel",
    "HarmonicComplexity",
    "RuleToBreak",
    # Engine
    "TherapySession",
    "HarmonyPlan",
    "AffectAnalyzer",
    "render_plan_to_midi",
]
