"""
Rhythm Rules
============

Rhythmic rules and groove conventions from classical through modern.
Integrates DAiW groove philosophy: "The grid is just a suggestion. The pocket is where life happens."
"""

from typing import Dict, List, Optional, Any
from .base import Rule, RuleBreakSuggestion
from .severity import RuleSeverity


class RhythmRules:
    """
    Collection of rhythmic rules with groove and feel conventions.
    
    Categories:
        - meter: Rules about metric organization
        - groove: Rules about rhythmic feel and pocket
        - syncopation: Rules about off-beat emphasis
        - tempo: Rules about tempo and rubato
    
    DAiW Philosophy:
        "Feel isn't random—it's systematic deviation from perfection."
    """
    
    RULES: Dict[str, Dict[str, Rule]] = {
        "meter": {
            "consistent_meter": Rule(
                id="consistent_meter",
                name="Maintain Consistent Meter",
                description="Music should maintain a consistent time signature",
                reason="Allows listeners to entrain to the pulse",
                severity=RuleSeverity.MODERATE,
                contexts=["classical", "baroque", "pop"],
                severity_by_context={
                    "classical": RuleSeverity.MODERATE,
                    "twentieth_century": RuleSeverity.FLEXIBLE,
                    "jazz": RuleSeverity.FLEXIBLE,
                    "progressive_rock": RuleSeverity.FLEXIBLE,
                },
                exceptions=[
                    "Metric modulation",
                    "Asymmetrical meters in 20th century",
                    "Progressive rock",
                ],
                examples=[
                    {"artist": "Stravinsky", "piece": "The Rite of Spring - Sacrificial Dance",
                     "detail": "Constant meter changes: 2/16, 3/16, 5/16, 2/8, 3/8, 5/8, 7/8..."},
                    {"artist": "Radiohead", "piece": "Pyramid Song",
                     "detail": "Ambiguous meter that resists classification"},
                ],
                emotional_uses=["chaos", "primitive_energy", "disorientation", "unease"],
            ),
            
            "downbeat_emphasis": Rule(
                id="downbeat_emphasis",
                name="Emphasize Downbeats",
                description="Strong beats should be emphasized with accents or harmonic changes",
                reason="Creates metric clarity",
                severity=RuleSeverity.MODERATE,
                contexts=["classical"],
                severity_by_context={
                    "classical": RuleSeverity.MODERATE,
                    "jazz": RuleSeverity.FLEXIBLE,
                    "funk": RuleSeverity.FLEXIBLE,  # "The One"
                },
                exceptions=[
                    "Syncopation",
                    "Jazz anticipations",
                ],
                emotional_uses=["floating_feeling", "groove_complexity"],
            ),
            
            "metric_ambiguity": Rule(
                id="metric_ambiguity",
                name="Metric Ambiguity",
                description="Deliberately obscuring the meter for effect",
                reason="Creates unsettled, floating quality",
                severity=RuleSeverity.ENCOURAGED,
                contexts=["impressionist", "jazz", "contemporary", "lofi"],
                examples=[
                    {"artist": "Radiohead", "piece": "Pyramid Song",
                     "detail": "Fans notate it as 12/8, 6/8, 4/4, or 3/4+5/4 alternating"},
                ],
                emotional_uses=["floating", "dream_state", "ambiguity"],
            ),
        },
        
        "groove": {
            "quantized_rhythm": Rule(
                id="quantized_rhythm",
                name="Play on the Grid",
                description="Notes should align with the rhythmic grid",
                reason="Creates precision and clarity",
                severity=RuleSeverity.FLEXIBLE,
                contexts=["electronic", "pop"],
                severity_by_context={
                    "electronic": RuleSeverity.MODERATE,
                    "jazz": RuleSeverity.FLEXIBLE,
                    "funk": RuleSeverity.FLEXIBLE,
                    "lofi": RuleSeverity.FLEXIBLE,
                },
                exceptions=[
                    "Intentional humanization",
                    "Groove pocket",
                    "Rubato",
                ],
                emotional_uses=["mechanical", "robotic", "precision"],
            ),
            
            "pocket_timing": Rule(
                id="pocket_timing",
                name="Play in the Pocket",
                description="Notes deviate systematically from the grid for feel",
                reason="Creates groove and human feel",
                severity=RuleSeverity.ENCOURAGED,
                contexts=["funk", "jazz", "hip_hop", "lofi", "soul"],
                examples=[
                    {"artist": "J Dilla", "piece": "Various",
                     "detail": "Heavy swing (62%), drums behind beat by 15-20ms"},
                    {"artist": "Questlove", "piece": "Various",
                     "detail": "Funk pocket with hi-hat pulling ahead"},
                ],
                emotional_uses=["groove", "humanity", "feel", "laid_back"],
            ),
            
            "constant_displacement": Rule(
                id="constant_displacement",
                name="Constant Displacement",
                description="Systematically displacing notes from expected positions",
                reason="Creates off-kilter, anxious feel",
                severity=RuleSeverity.STYLISTIC,
                contexts=["jazz", "contemporary"],
                emotional_uses=["anxiety", "unease", "anticipation"],
            ),
            
            "swing": Rule(
                id="swing",
                name="Swing Feel",
                description="Offbeat eighth notes played late (between straight and triplet)",
                reason="Creates bounce and swing feel",
                severity=RuleSeverity.STYLISTIC,
                contexts=["jazz", "funk", "hip_hop"],
                exceptions=[
                    "Straight eighth contexts",
                    "Latin jazz (often straight)",
                ],
                emotional_uses=["bounce", "energy", "life"],
            ),
        },
        
        "syncopation": {
            "avoid_syncopation": Rule(
                id="avoid_syncopation",
                name="Avoid Excessive Syncopation",
                description="Don't place too many accents on weak beats",
                reason="Classical preference for metric clarity",
                severity=RuleSeverity.FLEXIBLE,
                contexts=["baroque"],
                severity_by_context={
                    "baroque": RuleSeverity.MODERATE,
                    "classical": RuleSeverity.FLEXIBLE,
                    "jazz": RuleSeverity.ENCOURAGED,
                    "funk": RuleSeverity.ENCOURAGED,
                },
                emotional_uses=["tension", "forward_motion", "groove"],
            ),
            
            "polyrhythm": Rule(
                id="polyrhythm",
                name="Polyrhythm",
                description="Multiple conflicting rhythmic patterns simultaneously",
                reason="Creates rhythmic complexity and tension",
                severity=RuleSeverity.FLEXIBLE,
                contexts=["twentieth_century", "jazz", "african", "contemporary"],
                examples=[
                    {"artist": "Stravinsky", "piece": "Rite of Spring",
                     "detail": "Additive rhythms creating polyrhythmic layers"},
                    {"artist": "Coltrane", "piece": "Sheets of Sound",
                     "detail": "Notes grouped in 5s and 7s against the beat"},
                ],
                emotional_uses=["complexity", "tension", "african_influence"],
            ),
        },
        
        "tempo": {
            "steady_tempo": Rule(
                id="steady_tempo",
                name="Maintain Steady Tempo",
                description="Tempo should remain constant throughout a section",
                reason="Allows ensemble coordination and listener entrainment",
                severity=RuleSeverity.MODERATE,
                contexts=["classical", "pop", "electronic"],
                severity_by_context={
                    "classical": RuleSeverity.MODERATE,
                    "romantic": RuleSeverity.FLEXIBLE,  # More rubato
                    "jazz": RuleSeverity.FLEXIBLE,
                    "lofi": RuleSeverity.FLEXIBLE,
                },
                exceptions=[
                    "Rubato",
                    "Accelerando/ritardando",
                    "Tempo changes between sections",
                ],
                emotional_uses=["breathing", "organic_feel", "vulnerability"],
            ),
            
            "tempo_fluctuation": Rule(
                id="tempo_fluctuation",
                name="Tempo Fluctuation / Rubato",
                description="Intentional speeding up and slowing down for expression",
                reason="Creates organic, breathing quality",
                severity=RuleSeverity.ENCOURAGED,
                contexts=["romantic", "jazz", "lofi"],
                examples=[
                    {"artist": "Chopin", "piece": "Nocturnes",
                     "detail": "Expressive rubato throughout"},
                    {"artist": "Lo-Fi Producers", "piece": "Various",
                     "detail": "Slight tempo drift for tape machine effect"},
                ],
                emotional_uses=["intimacy", "vulnerability", "breathing", "humanity"],
            ),
        },
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GENRE POCKET MAPS (from DAiW)
    # ═══════════════════════════════════════════════════════════════════════════
    
    GENRE_POCKETS = {
        "funk": {
            "swing": 0.58,  # 58%
            "kick_offset_ms": 15,
            "snare_offset_ms": -8,
            "hihat_offset_ms": -10,
            "character": "Laid back groove, 'The One' emphasized",
        },
        "boom_bap": {
            "swing": 0.54,
            "kick_offset_ms": 12,
            "snare_offset_ms": -5,
            "hihat_offset_ms": -15,
            "character": "Hip-hop pocket, slightly behind",
        },
        "dilla": {
            "swing": 0.62,
            "kick_offset_ms": 20,
            "snare_offset_ms": -12,
            "hihat_offset_ms": -18,
            "character": "Heavy swing, very behind the beat",
        },
        "straight": {
            "swing": 0.50,
            "kick_offset_ms": 0,
            "snare_offset_ms": 0,
            "hihat_offset_ms": 0,
            "character": "Quantized, add humanize as needed",
        },
        "trap": {
            "swing": 0.51,
            "kick_offset_ms": 3,
            "snare_offset_ms": -2,
            "hihat_offset_ms": -5,
            "character": "Minimal swing, tight hihat rolls",
        },
        "lofi_grief": {
            "swing": 0.52,
            "kick_offset_ms": 8,
            "snare_offset_ms": -3,
            "hihat_offset_ms": -10,
            "character": "Minimal swing, intimate, bedroom feel",
        },
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CLASS METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @classmethod
    def get_all_rules(cls) -> Dict[str, Dict[str, Rule]]:
        """Get all rhythm rules."""
        return cls.RULES
    
    @classmethod
    def get_rule(cls, rule_id: str) -> Optional[Rule]:
        """Get a specific rule by ID."""
        for category, rules in cls.RULES.items():
            if rule_id in rules:
                return rules[rule_id]
        return None
    
    @classmethod
    def get_genre_pocket(cls, genre: str) -> Optional[Dict[str, Any]]:
        """Get groove pocket settings for a genre."""
        return cls.GENRE_POCKETS.get(genre)
    
    @classmethod
    def get_rule_break_for_emotion(cls, emotion: str) -> List[RuleBreakSuggestion]:
        """Get rhythm rule-breaking suggestions for an emotion."""
        emotion_map = {
            "anxiety": ["consistent_meter", "constant_displacement"],
            "chaos": ["consistent_meter", "polyrhythm"],
            "floating": ["metric_ambiguity", "downbeat_emphasis"],
            "intimacy": ["tempo_fluctuation", "pocket_timing"],
            "vulnerability": ["tempo_fluctuation"],
            "groove": ["quantized_rhythm", "pocket_timing"],
            "primitive_energy": ["consistent_meter"],
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
