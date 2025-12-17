"""
Penta Core - Music Theory Rule-Breaking Framework
==================================================

A comprehensive music theory library integrating DAiW and iDAW systems.
Provides voice leading, harmony, counterpoint rules with context filtering
and intentional rule-breaking for emotional effect.

Core Philosophy:
    "Interrogate Before Generate" - Understand the emotion, then translate to music
    "Every Rule-Break Needs Justification" - Breaking rules requires emotional reasoning
    "The wrong note played with conviction is the right note" - Beethoven

Modules:
    teachers    - Interactive teaching and demonstration
    rules       - Core rule definitions and filtering
    harmony     - Harmonic analysis and generation
    rhythm      - Rhythmic patterns and groove
    counterpoint - Species counterpoint rules
    analysis    - Music analysis tools
    utils       - Utility functions

Quick Start:
    from penta_core.teachers import RuleBreakingTeacher
    from penta_core.rules import VoiceLeadingRules, RuleSeverity
    
    # Get all strict voice leading rules
    strict_rules = VoiceLeadingRules.get_rules_by_severity(RuleSeverity.STRICT)
    
    # Interactive teaching
    teacher = RuleBreakingTeacher()
    teacher.demonstrate_rule_break("parallel_fifths")
"""

__version__ = "0.1.0"
__author__ = "Penta Core Project"
__license__ = "MIT"

# Core exports
from .rules.severity import RuleSeverity
from .rules.species import Species
from .rules.context import MusicalContext
from .rules.base import Rule, RuleViolation

# Rule collections
from .rules.voice_leading import VoiceLeadingRules
from .rules.harmony_rules import HarmonyRules
from .rules.counterpoint_rules import CounterpointRules
from .rules.rhythm_rules import RhythmRules

# Teachers
from .teachers.rule_breaking_teacher import RuleBreakingTeacher
from .teachers.counterpoint_teacher import CounterpointTeacher

__all__ = [
    # Version
    "__version__",
    # Enums
    "RuleSeverity",
    "Species", 
    "MusicalContext",
    # Base classes
    "Rule",
    "RuleViolation",
    # Rule collections
    "VoiceLeadingRules",
    "HarmonyRules",
    "CounterpointRules",
    "RhythmRules",
    # Teachers
    "RuleBreakingTeacher",
    "CounterpointTeacher",
]
