"""
Penta Core Rules Package
========================

Comprehensive music theory rules with context-dependent severity.
"""

from .severity import RuleSeverity
from .species import Species
from .context import MusicalContext, CONTEXT_GROUPS
from .base import Rule, RuleViolation, RuleBreakSuggestion
from .voice_leading import VoiceLeadingRules
from .harmony_rules import HarmonyRules
from .counterpoint_rules import CounterpointRules
from .rhythm_rules import RhythmRules

__all__ = [
    # Enums
    "RuleSeverity",
    "Species",
    "MusicalContext",
    "CONTEXT_GROUPS",
    # Base classes
    "Rule",
    "RuleViolation",
    "RuleBreakSuggestion",
    # Rule collections
    "VoiceLeadingRules",
    "HarmonyRules",
    "CounterpointRules",
    "RhythmRules",
]
