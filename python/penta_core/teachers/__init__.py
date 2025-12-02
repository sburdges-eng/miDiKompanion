"""
Music theory teachers and interactive learning modules.

This package provides comprehensive music theory education tools including:
- Rule-breaking teacher for learning through counterexamples
- Voice leading rulebook (classical, jazz, contemporary)
- Harmony rulebook (functional harmony, chord construction, progressions)
- Counterpoint rulebook (species counterpoint, Fux's principles)
"""

from .rule_breaking_teacher import RuleBreakingTeacher
from .voice_leading_rules import VoiceLeadingRules, RuleSeverity
from .harmony_rules import HarmonyRules, ChordQuality
from .counterpoint_rules import CounterpointRules, Species

__all__ = [
    "RuleBreakingTeacher",
    "VoiceLeadingRules",
    "HarmonyRules",
    "CounterpointRules",
    "RuleSeverity",
    "ChordQuality",
    "Species",
]
