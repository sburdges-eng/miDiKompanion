"""
Voice Leading Rules
===================

Comprehensive voice leading rules from common practice period through modern contexts.
Integrates DAiW emotional rule-breaking framework.
"""

from typing import Dict, List, Optional, Any
from .base import Rule, RuleBreakSuggestion
from .severity import RuleSeverity
from .context import MusicalContext


class VoiceLeadingRules:
    """
    Collection of voice leading rules with context-dependent severity.
    
    Categories:
        - parallel_motion: Rules about parallel intervals
        - contrary_motion: Rules about contrary motion
        - voice_spacing: Rules about distance between voices
        - tendency_tones: Rules about resolution of tendency tones
        - voice_crossing: Rules about voices crossing
        - jazz: Jazz-specific voice leading conventions
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RULE DEFINITIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    RULES: Dict[str, Dict[str, Rule]] = {
        "parallel_motion": {
            "parallel_fifths": Rule(
                id="parallel_fifths",
                name="No Parallel Perfect Fifths",
                description="Two voices moving in parallel perfect fifths (P5→P5) are forbidden",
                reason="Destroys voice independence; two voices fuse into one perceived line",
                severity=RuleSeverity.STRICT,
                contexts=["classical", "baroque", "romantic", "educational"],
                severity_by_context={
                    "classical": RuleSeverity.STRICT,
                    "baroque": RuleSeverity.STRICT,
                    "jazz": RuleSeverity.FLEXIBLE,
                    "rock": RuleSeverity.ENCOURAGED,
                    "metal": RuleSeverity.ENCOURAGED,
                    "impressionist": RuleSeverity.FLEXIBLE,
                    "contemporary": RuleSeverity.FLEXIBLE,
                },
                exceptions=[
                    "Parallel fifths in octave doublings",
                    "Planing in impressionist music (Debussy)",
                    "Power chords in rock/metal",
                    "Intentional primitive/folk effect",
                ],
                examples=[
                    {"artist": "Beethoven", "piece": "Symphony No. 6 'Pastoral'", 
                     "detail": "Intentional parallel fifths for rustic effect"},
                    {"artist": "Debussy", "piece": "La Cathédrale engloutie",
                     "detail": "Planing entire triads for medieval atmosphere"},
                    {"artist": "Power Chords", "piece": "All rock/metal",
                     "detail": "Root + P5 moving in parallel"},
                ],
                emotional_uses=["power", "defiance", "primitive_energy", "folk_simplicity"],
            ),
            
            "parallel_octaves": Rule(
                id="parallel_octaves",
                name="No Parallel Octaves",
                description="Two voices moving in parallel octaves (P8→P8) are forbidden",
                reason="Reduces voice count; voices merge into unison",
                severity=RuleSeverity.STRICT,
                contexts=["classical", "baroque", "romantic"],
                severity_by_context={
                    "classical": RuleSeverity.STRICT,
                    "jazz": RuleSeverity.MODERATE,
                    "rock": RuleSeverity.FLEXIBLE,
                    "electronic": RuleSeverity.FLEXIBLE,
                },
                exceptions=[
                    "Intentional octave doubling for reinforcement",
                    "Orchestral doublings at the octave",
                ],
                emotional_uses=["reinforcement", "boldness", "strength"],
            ),
            
            "hidden_fifths": Rule(
                id="hidden_fifths",
                name="No Hidden (Direct) Fifths",
                description="Two voices approaching a perfect fifth by similar motion",
                reason="Creates the effect of parallel fifths even without consecutive P5s",
                severity=RuleSeverity.MODERATE,
                contexts=["classical", "baroque"],
                severity_by_context={
                    "classical": RuleSeverity.MODERATE,
                    "jazz": RuleSeverity.FLEXIBLE,
                    "rock": RuleSeverity.FLEXIBLE,
                },
                exceptions=[
                    "When upper voice moves by step",
                    "In inner voices (less audible)",
                ],
                emotional_uses=[],
            ),
            
            "hidden_octaves": Rule(
                id="hidden_octaves",
                name="No Hidden (Direct) Octaves",
                description="Two voices approaching an octave by similar motion",
                reason="Creates the effect of parallel octaves",
                severity=RuleSeverity.MODERATE,
                contexts=["classical", "baroque"],
                exceptions=["When soprano moves by step"],
                emotional_uses=[],
            ),
        },
        
        "tendency_tones": {
            "leading_tone_resolution": Rule(
                id="leading_tone_resolution",
                name="Leading Tone Must Resolve Up",
                description="The 7th scale degree (leading tone) must resolve up to tonic",
                reason="Leading tone has strong tendency toward tonic; leaving it unresolved creates tension",
                severity=RuleSeverity.STRICT,
                contexts=["classical", "baroque", "romantic"],
                severity_by_context={
                    "classical": RuleSeverity.STRICT,
                    "jazz": RuleSeverity.FLEXIBLE,
                    "rock": RuleSeverity.FLEXIBLE,
                    "impressionist": RuleSeverity.FLEXIBLE,
                },
                exceptions=[
                    "In inner voice when another voice has the resolution",
                    "In sequences",
                    "Intentional non-resolution for emotional effect",
                ],
                emotional_uses=["unresolved_longing", "tension", "anticipation"],
            ),
            
            "seventh_resolution": Rule(
                id="seventh_resolution",
                name="Chordal 7th Must Resolve Down",
                description="The 7th of a chord must resolve down by step",
                reason="The 7th is a dissonance that creates expectation of downward resolution",
                severity=RuleSeverity.MODERATE,
                contexts=["classical", "baroque"],
                severity_by_context={
                    "classical": RuleSeverity.MODERATE,
                    "jazz": RuleSeverity.FLEXIBLE,
                    "contemporary": RuleSeverity.FLEXIBLE,
                },
                exceptions=[
                    "Common-tone 7ths",
                    "Jazz chord extensions",
                ],
                emotional_uses=["sustained_tension", "jazz_color"],
            ),
            
            "tritone_resolution": Rule(
                id="tritone_resolution",
                name="Tritone Must Resolve",
                description="Augmented 4th expands outward to 6th; diminished 5th contracts inward to 3rd",
                reason="The tritone ('diabolus in musica') creates maximum tension requiring resolution",
                severity=RuleSeverity.STRICT,
                contexts=["classical", "baroque", "romantic"],
                severity_by_context={
                    "classical": RuleSeverity.STRICT,
                    "jazz": RuleSeverity.FLEXIBLE,
                    "metal": RuleSeverity.ENCOURAGED,  # Unresolved tritone = metal
                    "rock": RuleSeverity.FLEXIBLE,
                },
                examples=[
                    {"artist": "Black Sabbath", "piece": "Black Sabbath",
                     "detail": "Built entire riff on unresolved tritone (G to Db)"},
                ],
                emotional_uses=["evil", "darkness", "unease", "metal_heaviness"],
            ),
        },
        
        "voice_spacing": {
            "voice_crossing": Rule(
                id="voice_crossing",
                name="No Voice Crossing",
                description="Voices should not cross (soprano below alto, etc.)",
                reason="Confuses the listener's perception of separate melodic lines",
                severity=RuleSeverity.MODERATE,
                contexts=["classical", "baroque"],
                severity_by_context={
                    "classical": RuleSeverity.MODERATE,
                    "jazz": RuleSeverity.FLEXIBLE,
                    "contemporary": RuleSeverity.FLEXIBLE,
                },
                exceptions=[
                    "Brief crossings for melodic reasons",
                    "Keyboard music where hands cross",
                ],
                emotional_uses=["confusion", "complexity"],
            ),
            
            "spacing_limits": Rule(
                id="spacing_limits",
                name="Spacing Between Upper Voices",
                description="No more than an octave between adjacent upper voices (S-A, A-T)",
                reason="Maintains cohesive sound; prevents gaps in harmonic texture",
                severity=RuleSeverity.MODERATE,
                contexts=["classical"],
                exceptions=[
                    "Orchestral writing with different timbres",
                    "Intentional sparse texture",
                ],
                emotional_uses=["spaciousness", "isolation"],
            ),
            
            "bass_spacing": Rule(
                id="bass_spacing",
                name="Bass May Be Distant",
                description="Bass can be more than an octave from tenor",
                reason="Bass functions as harmonic foundation, not melodic voice",
                severity=RuleSeverity.FLEXIBLE,
                contexts=["all"],
                emotional_uses=[],
            ),
        },
        
        "doubling": {
            "no_double_leading_tone": Rule(
                id="no_double_leading_tone",
                name="Never Double Leading Tone",
                description="The leading tone (7th scale degree) should not be doubled",
                reason="Both voices would need to resolve up, creating parallel octaves",
                severity=RuleSeverity.STRICT,
                contexts=["classical", "baroque", "romantic"],
                severity_by_context={
                    "classical": RuleSeverity.STRICT,
                    "jazz": RuleSeverity.MODERATE,
                    "rock": RuleSeverity.FLEXIBLE,
                },
                exceptions=[
                    "Melody doubling at octave (modern usage)",
                ],
                emotional_uses=["reinforcement"],
            ),
            
            "no_double_seventh": Rule(
                id="no_double_seventh",
                name="Never Double Chordal 7th",
                description="The 7th of a chord should not be doubled",
                reason="Creates parallel octaves when both resolve down",
                severity=RuleSeverity.MODERATE,
                contexts=["classical"],
                emotional_uses=[],
            ),
        },
        
        "jazz": {
            "voice_leading_by_step": Rule(
                id="voice_leading_by_step",
                name="Smooth Voice Leading",
                description="Voices should move by step or small intervals",
                reason="Creates smooth, connected progressions; minimizes motion",
                severity=RuleSeverity.MODERATE,
                contexts=["jazz"],
                emotional_uses=["smoothness", "sophistication"],
            ),
            
            "guide_tone_lines": Rule(
                id="guide_tone_lines",
                name="Guide Tone Lines",
                description="3rds and 7ths form smooth melodic lines through changes",
                reason="These tones define chord quality; smooth lines create continuity",
                severity=RuleSeverity.STYLISTIC,
                contexts=["jazz"],
                emotional_uses=["flow", "coherence"],
            ),
            
            "drop_voicings": Rule(
                id="drop_voicings",
                name="Drop Voicings",
                description="Drop 2, Drop 3, Drop 2+4 voicings for guitar/piano",
                reason="Creates playable voicings with good spacing",
                severity=RuleSeverity.STYLISTIC,
                contexts=["jazz"],
                emotional_uses=[],
            ),
        },
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CLASS METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @classmethod
    def get_all_rules(cls) -> Dict[str, Dict[str, Rule]]:
        """Get all voice leading rules."""
        return cls.RULES
    
    @classmethod
    def get_rule(cls, rule_id: str) -> Optional[Rule]:
        """Get a specific rule by ID."""
        for category, rules in cls.RULES.items():
            if rule_id in rules:
                return rules[rule_id]
        return None
    
    @classmethod
    def get_rules_by_context(cls, context: str) -> Dict[str, Dict[str, Rule]]:
        """
        Get rules filtered by musical context.
        
        Args:
            context: Musical context (e.g., "classical", "jazz", "rock")
        
        Returns:
            Rules applicable to that context
        """
        filtered = {}
        for category, rules in cls.RULES.items():
            matching = {}
            for rule_id, rule in rules.items():
                if rule.applies_to_context(context):
                    # Include severity for this specific context
                    matching[rule_id] = {
                        "name": rule.name,
                        "description": rule.description,
                        "reason": rule.reason,
                        "severity": rule.get_severity_for_context(context),
                        "exceptions": rule.exceptions,
                    }
            if matching:
                filtered[category] = matching
        return filtered
    
    @classmethod
    def get_rules_by_severity(cls, severity: RuleSeverity) -> Dict[str, Dict[str, Rule]]:
        """
        Get rules filtered by severity level.
        
        Args:
            severity: The severity level to filter by
        
        Returns:
            Rules matching that severity
        """
        filtered = {}
        for category, rules in cls.RULES.items():
            matching = {}
            for rule_id, rule in rules.items():
                if rule.severity == severity:
                    matching[rule_id] = {
                        "name": rule.name,
                        "description": rule.description,
                        "reason": rule.reason,
                        "severity": rule.severity,
                    }
            if matching:
                filtered[category] = matching
        return filtered
    
    @classmethod
    def get_rule_break_suggestions(cls, emotion: str) -> List[RuleBreakSuggestion]:
        """
        Get suggestions for rules to break based on emotional intent.
        
        From DAiW: "Every Rule-Break Needs Justification"
        
        Args:
            emotion: Target emotion (e.g., "grief", "power", "anxiety")
        
        Returns:
            List of suggested rule breaks with justifications
        """
        suggestions = []
        
        for category, rules in cls.RULES.items():
            for rule_id, rule in rules.items():
                if emotion in rule.emotional_uses:
                    suggestions.append(RuleBreakSuggestion(
                        rule=rule,
                        emotion=emotion,
                        justification=f"Breaking '{rule.name}' can express {emotion}",
                        implementation=f"Intentionally violate: {rule.description}",
                        examples=[ex.get("detail", "") for ex in rule.examples],
                    ))
        
        return suggestions
