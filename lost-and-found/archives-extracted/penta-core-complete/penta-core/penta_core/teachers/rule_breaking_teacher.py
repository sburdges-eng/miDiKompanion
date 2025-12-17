"""
Rule Breaking Teacher
=====================

Interactive teaching system for intentional rule-breaking in music.
Integrates DAiW philosophy: "Every Rule-Break Needs Justification"

Usage:
    from penta_core.teachers import RuleBreakingTeacher
    
    teacher = RuleBreakingTeacher()
    teacher.demonstrate_rule_break("parallel_fifths")
    teacher.get_examples_for_emotion("grief")
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from ..rules.base import Rule, RuleBreakSuggestion
from ..rules.severity import RuleSeverity
from ..rules.voice_leading import VoiceLeadingRules
from ..rules.harmony_rules import HarmonyRules
from ..rules.rhythm_rules import RhythmRules
from ..rules.counterpoint_rules import CounterpointRules


@dataclass
class RuleBreakExample:
    """An example of intentional rule-breaking."""
    artist: str
    piece: str
    rule_broken: str
    notation_detail: str
    why_it_works: str
    emotional_effect: str
    context: str = ""


@dataclass
class Lesson:
    """A teaching lesson about a rule or concept."""
    title: str
    rule: Rule
    explanation: str
    examples: List[RuleBreakExample]
    exercise: str = ""
    key_insight: str = ""


class RuleBreakingTeacher:
    """
    Interactive teacher for intentional rule-breaking in music.
    
    Philosophy from DAiW:
        "The audience doesn't hear 'borrowed from Dorian.' They hear 'that part made me cry.'"
    
    Methods:
        demonstrate_rule_break(rule_id) - Show examples of a rule being broken
        get_examples_for_emotion(emotion) - Get rule-breaks that create an emotion
        explain_rule(rule_id) - Full explanation of a rule
        suggest_rule_breaks(emotion, context) - Get personalized suggestions
        create_lesson(topic) - Generate an interactive lesson
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MASTERPIECE EXAMPLES DATABASE
    # ═══════════════════════════════════════════════════════════════════════════
    
    MASTERPIECE_EXAMPLES: Dict[str, List[RuleBreakExample]] = {
        "parallel_fifths": [
            RuleBreakExample(
                artist="Beethoven",
                piece="Symphony No. 6 'Pastoral' (1808)",
                rule_broken="Parallel Perfect Fifths",
                notation_detail="Violins and violas moving in parallel P5 motion in the Storm movement",
                why_it_works="Creates rustic, folk-like quality; evokes peasant dances and village music",
                emotional_effect="Folk simplicity, primitive energy",
                context="When questioned, Beethoven asked 'Well, who has forbidden them?' Upon citation of Fux: 'Well, I allow them!'",
            ),
            RuleBreakExample(
                artist="Debussy",
                piece="La Cathédrale engloutie (1910)",
                rule_broken="Parallel Perfect Fifths",
                notation_detail="'Planing' - moving entire triads in parallel (C-E-G → D-F#-A → E-G#-B)",
                why_it_works="Deliberately evokes pre-tonal organum; creates impressionistic wash of color",
                emotional_effect="Medieval atmosphere, mystery, otherworldly",
                context="Systematic use throughout his body of work",
            ),
            RuleBreakExample(
                artist="Power Chords",
                piece="All rock/metal (1960s-present)",
                rule_broken="Parallel Perfect Fifths",
                notation_detail="Root + P5 moving in parallel (E5→A5→D5 = E-B→A-E→D-A)",
                why_it_works="Distortion creates additional harmonics; parallel motion creates massive sound",
                emotional_effect="Power, defiance, aggression",
                context="Adding 3rds creates 'mud' while P5s remain clear through distortion",
            ),
        ],
        
        "polytonality": [
            RuleBreakExample(
                artist="Stravinsky",
                piece="The Rite of Spring (1913) - 'Augurs of Spring'",
                rule_broken="Single Tonal Center",
                notation_detail="Eb7 (Eb-G-Bb-Db) + E major (E-G#-B) played simultaneously",
                why_it_works="Neither key 'wins' - creates permanent tension; represents primal chaos",
                emotional_effect="Primitive chaos, primal energy, violence",
                context="Minor 2nd clash between Eb and E, plus augmented unison between G and G#",
            ),
            RuleBreakExample(
                artist="Stravinsky",
                piece="Petrushka (1911) - 'Petrushka Chord'",
                rule_broken="Single Tonal Center",
                notation_detail="C major (C-E-G) + F# major (F#-A#-C#) simultaneously",
                why_it_works="Tritone relationship creates maximum tonal distance; represents dual nature",
                emotional_effect="Dual nature, puppet vs human, conflict",
                context="Piano splits: RH plays F# major, LH plays C major",
            ),
        ],
        
        "unresolved_dissonance": [
            RuleBreakExample(
                artist="Thelonious Monk",
                piece="'Round Midnight (various recordings)",
                rule_broken="Dissonance Must Resolve",
                notation_detail="Semitone clusters (Bb against B natural) that never resolve",
                why_it_works="'Wrong notes' make sense because they sound wrong in a meaningful way",
                emotional_effect="Commentary, irony, meaningful wrongness",
                context="'Monk voicings' = semitone at bottom + 3rd on top",
            ),
            RuleBreakExample(
                artist="Black Sabbath",
                piece="Black Sabbath (1970)",
                rule_broken="Tritone Must Resolve",
                notation_detail="Riff built on G-G(octave)-Db (augmented 4th / tritone)",
                why_it_works="The tension that begged for resolution becomes the genre's identity",
                emotional_effect="Evil, darkness, dread, heaviness",
                context="Heavy metal literally embraces the 'devil's interval'",
            ),
        ],
        
        "modal_interchange": [
            RuleBreakExample(
                artist="Radiohead",
                piece="Creep (1992)",
                rule_broken="Stay Within Key",
                notation_detail="G-B-C-Cm (I-III-IV-iv) - B from Lydian, Cm from parallel minor",
                why_it_works="Creates signature 'happy-to-sad' emotional shift",
                emotional_effect="Bittersweet, self-loathing, beautiful sadness",
                context="Pedal notes tie disparate chords together",
            ),
            RuleBreakExample(
                artist="The Beatles",
                piece="Norwegian Wood (1965)",
                rule_broken="Stay Within Key",
                notation_detail="I-bVII-I progression (E-D-E) - bVII borrowed from Mixolydian",
                why_it_works="bVII functions as dominant substitute with folk quality",
                emotional_effect="Folk, nostalgia, wistful",
                context="Helped popularize bVII in 1960s rock",
            ),
            RuleBreakExample(
                artist="Kelly Song",
                piece="DAiW Example",
                rule_broken="Stay Within Key",
                notation_detail="F-C-Dm-Bbm - Bbm borrowed from F minor (iv chord)",
                why_it_works="First 3 chords sound like love; Bbm is THE REVEAL (grief speaking)",
                emotional_effect="Hope through grief, bittersweet darkness",
                context="From DAiW: 'Bbm makes hope feel earned'",
            ),
        ],
        
        "metric_ambiguity": [
            RuleBreakExample(
                artist="Stravinsky",
                piece="The Rite of Spring - 'Sacrificial Dance'",
                rule_broken="Consistent Meter",
                notation_detail="Alternating 2/16, 3/16, 5/16, 2/8, 3/8, 5/8, 7/8 etc.",
                why_it_works="Impossible to predict accent pattern; evokes primitive ritual chaos",
                emotional_effect="Primitive chaos, violence, unpredictability",
                context="'Additive rhythms' - figures never repeat the same way",
            ),
            RuleBreakExample(
                artist="Radiohead",
                piece="Pyramid Song (2001)",
                rule_broken="Clear Meter",
                notation_detail="Fans notate as 12/8, 6/8, 4/4, or 3/4+5/4 alternating",
                why_it_works="Piano chords have no discernible pulse until drums enter",
                emotional_effect="Floating, unsettled, dreaming",
                context="Ambiguity remains central to the piece's quality",
            ),
        ],
        
        "non_resolution": [
            RuleBreakExample(
                artist="Chopin",
                piece="Prelude in E Minor, Op. 28 No. 4",
                rule_broken="Resolve to Tonic",
                notation_detail="Chord spelled as dominant 4/2 with Bb in bass, 'resolves' up to B natural across silence",
                why_it_works="Silence embodies 'negative action' - intention NOT to resolve",
                emotional_effect="Grief, yearning, unfinished business",
                context="Enharmonically ambiguous: Neapolitan or German Aug 6th?",
            ),
            RuleBreakExample(
                artist="Radiohead",
                piece="Various songs",
                rule_broken="Establish Clear Tonic",
                notation_detail="'Double-tonic complex' - Am-F-C-G creates ambiguity between Am and C",
                why_it_works="Neither key fully asserts itself; proportion varies throughout",
                emotional_effect="Floating, unresolved, emotional ambiguity",
                context="Analysis term: songs that never establish clear tonic",
            ),
        ],
        
        "coltrane_changes": [
            RuleBreakExample(
                artist="John Coltrane",
                piece="Giant Steps (1959)",
                rule_broken="Traditional ii-V-I Cycle",
                notation_detail="Roots move by major 3rds: B→G→Eb (26 chords in 16 bars)",
                why_it_works="Each V7 resolves to I, but I immediately becomes passing to next key",
                emotional_effect="Harmonic complexity, 'sheets of sound', virtuosity",
                context="Scales shift every 2 beats; creates dense cascading arpeggios",
            ),
        ],
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EMOTION TO RULE-BREAK MAPPING (from DAiW)
    # ═══════════════════════════════════════════════════════════════════════════
    
    EMOTION_RULE_MAP: Dict[str, List[str]] = {
        "grief": ["non_resolution", "modal_interchange", "tempo_fluctuation"],
        "bittersweet": ["modal_interchange"],
        "power": ["parallel_fifths"],
        "defiance": ["parallel_fifths"],
        "anxiety": ["metric_ambiguity", "unresolved_dissonance", "constant_displacement"],
        "chaos": ["polytonality", "metric_ambiguity"],
        "nostalgia": ["modal_interchange"],
        "longing": ["non_resolution", "modal_interchange"],
        "vulnerability": ["tempo_fluctuation", "pitch_imperfection"],
        "evil": ["unresolved_dissonance"],  # Tritone
        "primitive_energy": ["parallel_fifths", "metric_ambiguity"],
        "floating": ["metric_ambiguity", "non_resolution"],
        "hope_through_grief": ["modal_interchange"],
    }
    
    def __init__(self):
        """Initialize the teacher with all rule collections."""
        self.voice_leading = VoiceLeadingRules
        self.harmony = HarmonyRules
        self.rhythm = RhythmRules
        self.counterpoint = CounterpointRules
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def demonstrate_rule_break(self, rule_id: str) -> Dict[str, Any]:
        """
        Get examples demonstrating a specific rule being broken.
        
        Args:
            rule_id: The rule identifier (e.g., "parallel_fifths")
        
        Returns:
            Dictionary with rule info and masterpiece examples
        """
        # Find the rule
        rule = self._find_rule(rule_id)
        examples = self.MASTERPIECE_EXAMPLES.get(rule_id, [])
        
        return {
            "rule_id": rule_id,
            "rule": rule.to_dict() if rule else None,
            "examples": [
                {
                    "artist": ex.artist,
                    "piece": ex.piece,
                    "notation_detail": ex.notation_detail,
                    "why_it_works": ex.why_it_works,
                    "emotional_effect": ex.emotional_effect,
                    "context": ex.context,
                }
                for ex in examples
            ],
            "key_insight": self._get_key_insight(rule_id),
        }
    
    def get_examples_for_emotion(self, emotion: str) -> List[Dict[str, Any]]:
        """
        Get rule-breaking examples that create a specific emotion.
        
        Args:
            emotion: Target emotion (grief, power, anxiety, etc.)
        
        Returns:
            List of examples with justifications
        """
        rule_ids = self.EMOTION_RULE_MAP.get(emotion.lower(), [])
        results = []
        
        for rule_id in rule_ids:
            examples = self.MASTERPIECE_EXAMPLES.get(rule_id, [])
            rule = self._find_rule(rule_id)
            
            for ex in examples:
                results.append({
                    "rule_broken": rule_id,
                    "rule_name": rule.name if rule else rule_id,
                    "artist": ex.artist,
                    "piece": ex.piece,
                    "how": ex.notation_detail,
                    "why": ex.why_it_works,
                    "emotional_effect": ex.emotional_effect,
                })
        
        return results
    
    def explain_rule(self, rule_id: str) -> Dict[str, Any]:
        """
        Get full explanation of a rule.
        
        Args:
            rule_id: The rule identifier
        
        Returns:
            Comprehensive rule explanation
        """
        rule = self._find_rule(rule_id)
        if not rule:
            return {"error": f"Rule '{rule_id}' not found"}
        
        return {
            "id": rule.id,
            "name": rule.name,
            "description": rule.description,
            "reason": rule.reason,
            "default_severity": rule.severity.value,
            "contexts": rule.contexts,
            "severity_by_context": {
                ctx: sev.value for ctx, sev in rule.severity_by_context.items()
            },
            "exceptions": rule.exceptions,
            "emotional_uses": rule.emotional_uses,
            "examples": [
                {"artist": ex.get("artist"), "piece": ex.get("piece"), "detail": ex.get("detail")}
                for ex in rule.examples
            ],
        }
    
    def suggest_rule_breaks(
        self, 
        emotion: str, 
        context: str = "contemporary"
    ) -> List[RuleBreakSuggestion]:
        """
        Get personalized rule-breaking suggestions.
        
        DAiW Integration: "Every Rule-Break Needs Justification"
        
        Args:
            emotion: Target emotion
            context: Musical context
        
        Returns:
            List of suggestions with justifications
        """
        suggestions = []
        
        # Get from harmony rules
        suggestions.extend(self.harmony.get_rule_break_for_emotion(emotion))
        
        # Get from voice leading rules
        suggestions.extend(self.voice_leading.get_rule_break_suggestions(emotion))
        
        # Get from rhythm rules
        suggestions.extend(self.rhythm.get_rule_break_for_emotion(emotion))
        
        return suggestions
    
    def create_lesson(self, topic: str) -> Lesson:
        """
        Generate an interactive lesson on a topic.
        
        Args:
            topic: Lesson topic (e.g., "parallel_fifths", "modal_interchange")
        
        Returns:
            Lesson object with content and exercises
        """
        rule = self._find_rule(topic)
        examples = [
            RuleBreakExample(**{
                "artist": ex.artist,
                "piece": ex.piece,
                "rule_broken": topic,
                "notation_detail": ex.notation_detail,
                "why_it_works": ex.why_it_works,
                "emotional_effect": ex.emotional_effect,
            })
            for ex in self.MASTERPIECE_EXAMPLES.get(topic, [])
        ]
        
        return Lesson(
            title=f"Rule-Breaking: {rule.name if rule else topic}",
            rule=rule,
            explanation=rule.description if rule else "",
            examples=examples,
            exercise=self._get_exercise(topic),
            key_insight=self._get_key_insight(topic),
        )
    
    def list_all_emotions(self) -> List[str]:
        """Get list of all emotions with rule-break mappings."""
        return list(self.EMOTION_RULE_MAP.keys())
    
    def list_all_rule_breaks(self) -> List[str]:
        """Get list of all documented rule-breaks."""
        return list(self.MASTERPIECE_EXAMPLES.keys())
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRIVATE METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _find_rule(self, rule_id: str) -> Optional[Rule]:
        """Find a rule by ID across all collections."""
        rule = self.voice_leading.get_rule(rule_id)
        if rule:
            return rule
        
        rule = self.harmony.get_rule(rule_id)
        if rule:
            return rule
        
        rule = self.rhythm.get_rule(rule_id)
        if rule:
            return rule
        
        rule = self.counterpoint.get_rule(rule_id)
        return rule
    
    def _get_key_insight(self, topic: str) -> str:
        """Get the key insight for a topic."""
        insights = {
            "parallel_fifths": "As Beethoven said: 'Well, who has forbidden them?' - Rules exist for a reason, but the ultimate authority is your ear.",
            "modal_interchange": "The audience doesn't hear 'borrowed from Dorian.' They hear 'that part made me cry.'",
            "unresolved_dissonance": "The wrong note played with conviction is the right note.",
            "metric_ambiguity": "The grid is just a suggestion. The pocket is where life happens.",
            "non_resolution": "Sometimes the most powerful gesture is the intention NOT to resolve.",
            "polytonality": "When neither key 'wins,' the conflict itself becomes the expression.",
            "coltrane_changes": "Breaking the ii-V-I cycle created one of jazz's most challenging standards.",
        }
        return insights.get(topic, "Every rule-break needs justification.")
    
    def _get_exercise(self, topic: str) -> str:
        """Get a practice exercise for a topic."""
        exercises = {
            "parallel_fifths": "Write an 8-bar phrase using only power chords, then write the same progression with full triads. Compare the emotional effect.",
            "modal_interchange": "Take a I-V-vi-IV progression and substitute the iv chord for IV. Notice the shift from hope to bittersweet.",
            "unresolved_dissonance": "Play a V7 chord and don't resolve it. Sit with the tension. When does it become expressive rather than 'wrong'?",
            "metric_ambiguity": "Play a simple pattern but shift where you feel 'beat one.' How does the emotional quality change?",
            "non_resolution": "End a piece on the V chord instead of I. What story does this tell?",
        }
        return exercises.get(topic, "Experiment with intentionally breaking this rule. Document when it works and why.")
