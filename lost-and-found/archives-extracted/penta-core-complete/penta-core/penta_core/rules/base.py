"""
Base Rule Classes
=================

Foundation classes for all music theory rules.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from .severity import RuleSeverity
from .context import MusicalContext


@dataclass
class Rule:
    """
    A music theory rule with context-dependent severity.
    
    Attributes:
        id: Unique identifier (e.g., "parallel_fifths")
        name: Human-readable name
        description: Full explanation of the rule
        reason: Why the rule exists
        severity: Default severity level
        contexts: Which contexts this rule applies to
        exceptions: Known exceptions to the rule
        severity_by_context: Override severity for specific contexts
        check_function: Optional function to check for violations
        examples: Famous examples of this rule being broken
    """
    id: str
    name: str
    description: str
    reason: str
    severity: RuleSeverity = RuleSeverity.MODERATE
    contexts: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    severity_by_context: Dict[str, RuleSeverity] = field(default_factory=dict)
    check_function: Optional[Callable] = None
    examples: List[Dict[str, str]] = field(default_factory=list)
    emotional_uses: List[str] = field(default_factory=list)
    
    def get_severity_for_context(self, context: str) -> RuleSeverity:
        """Get the severity level for a specific context."""
        if context in self.severity_by_context:
            return self.severity_by_context[context]
        return self.severity
    
    def applies_to_context(self, context: str) -> bool:
        """Check if this rule applies to a given context."""
        if not self.contexts:
            return True  # Applies to all if not specified
        return context in self.contexts or "all" in self.contexts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "reason": self.reason,
            "severity": self.severity.value,
            "contexts": self.contexts,
            "exceptions": self.exceptions,
            "emotional_uses": self.emotional_uses,
        }


@dataclass
class RuleViolation:
    """
    A detected violation of a music theory rule.
    
    Attributes:
        rule: The rule that was violated
        location: Where in the music (bar, beat, voice)
        description: Human-readable description of the violation
        severity: How severe this violation is in context
        suggestion: How to fix (if desired)
        is_intentional: Whether this appears to be intentional rule-breaking
        emotional_justification: Potential emotional reason for the violation
    """
    rule: Rule
    location: str
    description: str
    severity: RuleSeverity = RuleSeverity.MODERATE
    suggestion: str = ""
    is_intentional: bool = False
    emotional_justification: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule.id,
            "rule_name": self.rule.name,
            "location": self.location,
            "description": self.description,
            "severity": self.severity.value,
            "suggestion": self.suggestion,
            "is_intentional": self.is_intentional,
            "emotional_justification": self.emotional_justification,
        }


@dataclass
class RuleBreakSuggestion:
    """
    A suggestion to intentionally break a rule for emotional effect.
    
    From DAiW philosophy: "Every Rule-Break Needs Justification"
    """
    rule: Rule
    emotion: str
    justification: str
    implementation: str
    examples: List[str] = field(default_factory=list)
    intensity: float = 0.5  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule.id,
            "rule_name": self.rule.name,
            "emotion": self.emotion,
            "justification": self.justification,
            "implementation": self.implementation,
            "examples": self.examples,
            "intensity": self.intensity,
        }
