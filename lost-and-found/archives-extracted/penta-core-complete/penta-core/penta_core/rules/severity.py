"""
Rule Severity Levels
====================

Defines how strictly a rule should be followed across different contexts.
"""

from enum import Enum


class RuleSeverity(Enum):
    """
    How strictly a music theory rule should be followed.
    
    STRICT: Never break in this context (e.g., parallel 5ths in Bach chorale)
    MODERATE: Avoid unless intentional with justification
    FLEXIBLE: Context-dependent, often broken in modern music
    STYLISTIC: Genre-specific preference, not a "rule" per se
    ENCOURAGED: Breaking this rule is often desired (e.g., power chords)
    """
    STRICT = "strict"
    MODERATE = "moderate"
    FLEXIBLE = "flexible"
    STYLISTIC = "stylistic"
    ENCOURAGED = "encouraged"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def description(self) -> str:
        """Human-readable description of the severity level."""
        descriptions = {
            "strict": "Never break - fundamental to the style",
            "moderate": "Avoid unless intentionally justified",
            "flexible": "Context-dependent - use judgment",
            "stylistic": "Genre-specific preference",
            "encouraged": "Breaking often creates desired effect",
        }
        return descriptions.get(self.value, "")
