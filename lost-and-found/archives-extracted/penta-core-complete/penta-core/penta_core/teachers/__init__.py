"""
Penta Core Teachers Package
===========================

Interactive teaching modules for music theory rule-breaking.
"""

from .rule_breaking_teacher import RuleBreakingTeacher, RuleBreakExample, Lesson
from .counterpoint_teacher import CounterpointTeacher

__all__ = [
    "RuleBreakingTeacher",
    "RuleBreakExample",
    "Lesson",
    "CounterpointTeacher",
]
