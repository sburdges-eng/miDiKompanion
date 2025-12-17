"""
Harmony Rules
=============

Comprehensive harmonic rules from common practice through modern contexts.
Integrates DAiW rule-breaking framework and iDAW emotional mapping.
"""

from typing import Dict, List, Optional, Any
from .base import Rule, RuleBreakSuggestion
from .severity import RuleSeverity


class HarmonyRules:
    """
    Collection of harmonic rules with emotional rule-breaking suggestions.
    
    Categories:
        - chord_progression: Rules about chord movement
        - resolution: Rules about harmonic resolution
        - modal: Rules about modes and modal interchange
        - dissonance: Rules about dissonance treatment
        - chromatic: Rules about chromatic harmony
    
    DAiW Integration:
        Each rule includes emotional_uses for intentional rule-breaking.
        "The audience doesn't hear 'borrowed from Dorian.' They hear 'that part made me cry.'"
    """
    
    RULES: Dict[str, Dict[str, Rule]] = {
        "chord_progression": {
            "functional_harmony": Rule(
                id="functional_harmony",
                name="Functional Harmonic Progression",
                description="Chords should progress according to functional relationships (T→PD→D→T)",
                reason="Creates sense of motion and goal-directed harmony",
                severity=RuleSeverity.MODERATE,
                contexts=["classical", "baroque", "romantic", "pop"],
                severity_by_context={
                    "classical": RuleSeverity.MODERATE,
                    "jazz": RuleSeverity.FLEXIBLE,
                    "rock": RuleSeverity.FLEXIBLE,
                    "impressionist": RuleSeverity.FLEXIBLE,
                },
                exceptions=[
                    "Plagal progressions (IV→I)",
                    "Modal harmony",
                    "Retrograde progressions for effect",
                ],
                emotional_uses=["subverting_expectations", "dream_state"],
            ),
            
            "root_motion_by_fifth": Rule(
                id="root_motion_by_fifth",
                name="Root Motion by Fifth Preferred",
                description="Root motion by 5th (or 4th) is strongest progression",
                reason="Creates strongest sense of harmonic motion",
                severity=RuleSeverity.STYLISTIC,
                contexts=["classical", "baroque"],
                severity_by_context={
                    "classical": RuleSeverity.STYLISTIC,
                    "jazz": RuleSeverity.FLEXIBLE,
                },
                examples=[
                    {"artist": "Coltrane", "piece": "Giant Steps",
                     "detail": "Root motion by major 3rds instead of 5ths"},
                ],
                emotional_uses=["rapid_key_changes", "harmonic_complexity"],
            ),
            
            "avoid_retrogression": Rule(
                id="avoid_retrogression",
                name="Avoid Retrogression",
                description="Don't move backwards in the functional cycle (D→PD, PD→T, etc.)",
                reason="Creates sense of harmonic backsliding",
                severity=RuleSeverity.MODERATE,
                contexts=["classical"],
                exceptions=[
                    "Deceptive cadences (V→vi)",
                    "Intentional harmonic surprise",
                ],
                emotional_uses=["surprise", "disappointment", "plot_twist"],
            ),
        },
        
        "resolution": {
            "dominant_to_tonic": Rule(
                id="dominant_to_tonic",
                name="Dominant Must Resolve to Tonic",
                description="V (or V7) should resolve to I",
                reason="Most fundamental harmonic expectation in tonal music",
                severity=RuleSeverity.STRICT,
                contexts=["classical", "baroque", "romantic"],
                severity_by_context={
                    "classical": RuleSeverity.STRICT,
                    "jazz": RuleSeverity.FLEXIBLE,
                    "rock": RuleSeverity.FLEXIBLE,
                    "impressionist": RuleSeverity.FLEXIBLE,
                },
                exceptions=[
                    "Deceptive cadence (V→vi)",
                    "Half cadence (ending on V)",
                    "Interrupted/avoided cadence",
                ],
                examples=[
                    {"artist": "Chopin", "piece": "Prelude in E Minor, Op. 28 No. 4",
                     "detail": "Enharmonic spelling creates ambiguous resolution"},
                    {"artist": "Radiohead", "piece": "Various",
                     "detail": "Songs that never establish clear tonic"},
                ],
                emotional_uses=["unresolved_longing", "grief", "openness", "ambiguity"],
            ),
            
            "tonic_resolution": Rule(
                id="tonic_resolution",
                name="End on Tonic",
                description="Pieces should resolve to tonic chord at the end",
                reason="Provides closure and completion",
                severity=RuleSeverity.MODERATE,
                contexts=["classical", "baroque", "pop"],
                severity_by_context={
                    "classical": RuleSeverity.MODERATE,
                    "jazz": RuleSeverity.FLEXIBLE,
                    "rock": RuleSeverity.FLEXIBLE,
                    "film": RuleSeverity.FLEXIBLE,
                },
                exceptions=[
                    "Fade outs",
                    "Intentional cliff-hanger endings",
                    "Transitional pieces",
                ],
                emotional_uses=["unfinished_business", "continuation", "grief"],
            ),
        },
        
        "modal": {
            "stay_in_key": Rule(
                id="stay_in_key",
                name="Stay Within Key",
                description="Use only diatonic chords from the established key",
                reason="Maintains tonal coherence",
                severity=RuleSeverity.FLEXIBLE,
                contexts=["classical"],
                severity_by_context={
                    "classical": RuleSeverity.MODERATE,
                    "romantic": RuleSeverity.FLEXIBLE,
                    "jazz": RuleSeverity.FLEXIBLE,
                    "rock": RuleSeverity.FLEXIBLE,
                    "pop": RuleSeverity.FLEXIBLE,
                },
                exceptions=[
                    "Secondary dominants",
                    "Modal interchange/borrowed chords",
                    "Modulation",
                ],
                emotional_uses=["predictability"],
            ),
            
            "modal_interchange": Rule(
                id="modal_interchange",
                name="Modal Interchange (Borrowed Chords)",
                description="Borrowing chords from parallel major/minor modes",
                reason="Adds color and emotional complexity",
                severity=RuleSeverity.ENCOURAGED,
                contexts=["jazz", "rock", "pop", "romantic", "film"],
                examples=[
                    {"artist": "Radiohead", "piece": "Creep",
                     "detail": "G-B-C-Cm (I-III-IV-iv) - B from Lydian, Cm from minor"},
                    {"artist": "Beatles", "piece": "Norwegian Wood",
                     "detail": "I-bVII-I progression - bVII from Mixolydian"},
                    {"artist": "Kelly Song", "piece": "DAiW Example",
                     "detail": "F-C-Dm-Bbm - Bbm borrowed from F minor"},
                ],
                emotional_uses=["bittersweet", "nostalgia", "longing", "hope_through_grief"],
            ),
            
            "no_mixing_modes": Rule(
                id="no_mixing_modes",
                name="Don't Mix Major and Minor Freely",
                description="Major and minor modes should not be freely mixed",
                reason="Classical rule for modal purity",
                severity=RuleSeverity.FLEXIBLE,
                contexts=["baroque"],
                severity_by_context={
                    "baroque": RuleSeverity.MODERATE,
                    "classical": RuleSeverity.FLEXIBLE,
                    "romantic": RuleSeverity.FLEXIBLE,
                    "jazz": RuleSeverity.ENCOURAGED,
                    "pop": RuleSeverity.ENCOURAGED,
                },
                emotional_uses=["happy_sad_ambiguity", "emotional_complexity"],
            ),
        },
        
        "dissonance": {
            "dissonance_preparation": Rule(
                id="dissonance_preparation",
                name="Prepare Dissonances",
                description="Dissonances should be prepared (present as consonance before becoming dissonant)",
                reason="Softens the impact of dissonance",
                severity=RuleSeverity.STRICT,
                contexts=["renaissance", "baroque"],
                severity_by_context={
                    "renaissance": RuleSeverity.STRICT,
                    "baroque": RuleSeverity.MODERATE,
                    "classical": RuleSeverity.MODERATE,
                    "jazz": RuleSeverity.FLEXIBLE,
                },
                exceptions=[
                    "Passing tones",
                    "Neighbor tones",
                    "Anticipations",
                ],
                emotional_uses=["shock", "impact"],
            ),
            
            "dissonance_resolution": Rule(
                id="dissonance_resolution",
                name="Resolve Dissonances",
                description="Dissonances (7ths, 2nds, tritones) must resolve by step to consonances",
                reason="Tension requires release",
                severity=RuleSeverity.STRICT,
                contexts=["classical", "baroque", "romantic"],
                severity_by_context={
                    "classical": RuleSeverity.STRICT,
                    "jazz": RuleSeverity.FLEXIBLE,
                    "rock": RuleSeverity.FLEXIBLE,
                    "metal": RuleSeverity.ENCOURAGED,  # Unresolved = heavy
                },
                examples=[
                    {"artist": "Thelonious Monk", "piece": "'Round Midnight",
                     "detail": "Deliberate unresolved semitone clusters"},
                ],
                emotional_uses=["tension", "anxiety", "unease", "darkness"],
            ),
            
            "unresolved_dissonance": Rule(
                id="unresolved_dissonance",
                name="Intentional Unresolved Dissonance",
                description="Leaving dissonances unresolved for effect",
                reason="Creates sustained tension and unease",
                severity=RuleSeverity.ENCOURAGED,
                contexts=["jazz", "metal", "contemporary", "film"],
                examples=[
                    {"artist": "Monk", "piece": "Various",
                     "detail": "Wrong notes that sound meaningfully wrong"},
                    {"artist": "Black Sabbath", "piece": "Black Sabbath",
                     "detail": "Unresolved tritone defines the genre"},
                ],
                emotional_uses=["anxiety", "evil", "unease", "wrongness"],
            ),
        },
        
        "chromatic": {
            "chromatic_movement": Rule(
                id="chromatic_movement",
                name="Chromatic Voice Movement",
                description="Chromatic notes should resolve in the direction of their alteration",
                reason="Raised notes tend up, lowered notes tend down",
                severity=RuleSeverity.MODERATE,
                contexts=["classical", "romantic"],
                exceptions=[
                    "Modal interchange",
                    "Jazz alterations",
                ],
                emotional_uses=["color", "sophistication"],
            ),
            
            "polytonality": Rule(
                id="polytonality",
                name="Polytonality (Multiple Keys)",
                description="Superimposing chords from different keys simultaneously",
                reason="Classical rule: maintain single tonal center",
                severity=RuleSeverity.FLEXIBLE,
                contexts=["twentieth_century", "contemporary", "film"],
                examples=[
                    {"artist": "Stravinsky", "piece": "The Rite of Spring",
                     "detail": "Eb7 + E major simultaneously - the 'Augurs of Spring' chord"},
                    {"artist": "Stravinsky", "piece": "Petrushka",
                     "detail": "C major + F# major - 'Petrushka Chord'"},
                ],
                emotional_uses=["chaos", "conflict", "dual_nature", "primal_energy"],
            ),
            
            "tritone_substitution": Rule(
                id="tritone_substitution",
                name="Tritone Substitution",
                description="Replace V7 with bII7 (a tritone away)",
                reason="Both share the same tritone; creates chromatic bass movement",
                severity=RuleSeverity.ENCOURAGED,
                contexts=["jazz"],
                examples=[
                    {"artist": "Jazz standard", "piece": "ii-V-I",
                     "detail": "Dm7-Db7-Cmaj7 instead of Dm7-G7-Cmaj7"},
                    {"artist": "Schubert", "piece": "String Quintet in C",
                     "detail": "Ends with Db7 instead of G7"},
                ],
                emotional_uses=["sophistication", "chromatic_color", "jazz_flavor"],
            ),
        },
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CLASS METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @classmethod
    def get_all_rules(cls) -> Dict[str, Dict[str, Rule]]:
        """Get all harmony rules."""
        return cls.RULES
    
    @classmethod
    def get_rule(cls, rule_id: str) -> Optional[Rule]:
        """Get a specific rule by ID."""
        for category, rules in cls.RULES.items():
            if rule_id in rules:
                return rules[rule_id]
        return None
    
    @classmethod
    def get_rules_by_context(cls, context: str) -> Dict[str, Dict[str, Any]]:
        """Get rules filtered by musical context."""
        filtered = {}
        for category, rules in cls.RULES.items():
            matching = {}
            for rule_id, rule in rules.items():
                if rule.applies_to_context(context):
                    matching[rule_id] = {
                        "name": rule.name,
                        "description": rule.description,
                        "reason": rule.reason,
                        "severity": rule.get_severity_for_context(context),
                    }
            if matching:
                filtered[category] = matching
        return filtered
    
    @classmethod
    def get_rules_by_severity(cls, severity: RuleSeverity) -> Dict[str, Dict[str, Any]]:
        """Get rules filtered by severity level."""
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
    def get_rule_break_for_emotion(cls, emotion: str) -> List[RuleBreakSuggestion]:
        """
        Get harmony rule-breaking suggestions for an emotion.
        
        DAiW Integration: Maps emotions to harmonic rule-breaks.
        
        Args:
            emotion: Target emotion (grief, power, anxiety, bittersweet, etc.)
        """
        emotion_map = {
            "grief": ["dominant_to_tonic", "tonic_resolution"],
            "bittersweet": ["modal_interchange", "no_mixing_modes"],
            "power": [],  # More voice leading than harmony
            "anxiety": ["dissonance_resolution", "unresolved_dissonance"],
            "chaos": ["polytonality", "functional_harmony"],
            "longing": ["dominant_to_tonic", "modal_interchange"],
            "nostalgia": ["modal_interchange"],
            "hope_through_grief": ["modal_interchange"],
        }
        
        suggestions = []
        rule_ids = emotion_map.get(emotion, [])
        
        for rule_id in rule_ids:
            rule = cls.get_rule(rule_id)
            if rule:
                suggestions.append(RuleBreakSuggestion(
                    rule=rule,
                    emotion=emotion,
                    justification=f"Breaking '{rule.name}' creates {emotion}",
                    implementation=f"Intentionally: {rule.description}",
                    examples=[ex.get("detail", "") for ex in rule.examples],
                ))
        
        return suggestions
