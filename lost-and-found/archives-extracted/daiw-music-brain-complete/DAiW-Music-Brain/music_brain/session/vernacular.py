"""
Vernacular Translation Engine for DAiW
=======================================

Translates casual music descriptions to technical parameters.
Enables understanding of "make it fat and laid back" or "needs more glue"
and converting to actionable DAiW generation parameters.

Usage:
    from vernacular import VernacularTranslator
    
    translator = VernacularTranslator()
    params = translator.translate("fat laid back swung")
    # Returns: {'eq.low_mid': '+3dB', 'saturation': 'light', 
    #           'groove.pocket': 'behind', 'groove.swing': 0.62, ...}
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class VernacularCategory(Enum):
    """Categories of vernacular terms."""
    RHYTHMIC = "rhythmic_onomatopoeia"
    TIMBRE = "timbre_texture"
    GROOVE = "groove_feel"
    MIX = "mix_production"
    MEME_THEORY = "meme_theory"
    RULE_BREAK = "rule_breaking"


class RuleBreakCode(Enum):
    """Rule-breaking codes for intent schema."""
    HARMONY_ParallelMotion = "HARMONY_ParallelMotion"
    HARMONY_ModalInterchange = "HARMONY_ModalInterchange"
    HARMONY_UnresolvedDissonance = "HARMONY_UnresolvedDissonance"
    HARMONY_TritoneSubstitution = "HARMONY_TritoneSubstitution"
    HARMONY_Polytonality = "HARMONY_Polytonality"
    RHYTHM_MeterAmbiguity = "RHYTHM_MeterAmbiguity"
    RHYTHM_ConstantDisplacement = "RHYTHM_ConstantDisplacement"
    RHYTHM_TempoFluctuation = "RHYTHM_TempoFluctuation"
    STRUCTURE_NonResolution = "STRUCTURE_NonResolution"
    PRODUCTION_BuriedVocals = "PRODUCTION_BuriedVocals"
    PRODUCTION_PitchImperfection = "PRODUCTION_PitchImperfection"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VernacularMatch:
    """Result of matching a vernacular term."""
    term: str
    category: VernacularCategory
    meaning: str
    daiw_params: Dict
    confidence: float = 1.0


@dataclass
class TranslationResult:
    """Complete result of translating vernacular input."""
    original_input: str
    matched_terms: List[VernacularMatch]
    combined_params: Dict
    unmatched_words: List[str]
    suggested_rule_breaks: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# VERNACULAR DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

VERNACULAR_DB = {
    "rhythmic_onomatopoeia": {
        "boots and cats": {"meaning": "Basic 4/4 beat", "daiw_params": {"groove.pattern": "four_on_floor"}},
        "untz": {"meaning": "Techno/EDM kick pattern", "daiw_params": {"groove.pattern": "four_on_floor", "tempo_range": [120, 140]}},
        "boom bap": {"meaning": "Hip-hop groove", "daiw_params": {"groove.pattern": "boom_bap"}},
        "chugga": {"meaning": "Palm-muted power chords", "daiw_params": {"groove.feel": "chug", "muting": True}},
        "brr": {"meaning": "Trap hi-hat rolls", "daiw_params": {"hihat.subdivision": 32, "pattern": "trap_roll"}},
        "skrrt": {"meaning": "Record scratch effect", "daiw_params": {"fx.type": "scratch"}},
    },
    
    "timbre_texture": {
        "fat": {"meaning": "Full low-mid frequencies", "daiw_params": {"eq.low_mid": "+3dB", "saturation": "light"}},
        "phat": {"meaning": "Full low-mid frequencies", "daiw_params": {"eq.low_mid": "+3dB", "saturation": "light"}},
        "thin": {"meaning": "Lacking low frequencies", "daiw_params": {"eq.low": "-6dB"}},
        "muddy": {"meaning": "Cluttered low-mids", "daiw_params": {"eq.problem": "mud", "target": "200-500Hz"}},
        "crispy": {"meaning": "Pleasant high-frequency presence", "daiw_params": {"eq.presence": "+2dB", "dist": "light_saturation"}},
        "crunchy": {"meaning": "Pleasant high-frequency presence", "daiw_params": {"eq.presence": "+2dB", "dist": "light_saturation"}},
        "warm": {"meaning": "Analog character", "daiw_params": {"character": "analog", "warmth": 0.7}},
        "bright": {"meaning": "Emphasized highs", "daiw_params": {"eq.high": "+3dB"}},
        "dark": {"meaning": "Subdued highs", "daiw_params": {"eq.high": "-4dB", "character": "dark"}},
        "punchy": {"meaning": "Strong transient attack", "daiw_params": {"comp.attack": "fast", "punch": "high"}},
        "scooped": {"meaning": "Cut mids (metal tone)", "daiw_params": {"eq.mid": "-6dB"}},
        "honky": {"meaning": "Problem nasal midrange", "daiw_params": {"eq.problem": "honk", "target": "800-1200Hz"}},
        "boxy": {"meaning": "Cardboard midrange", "daiw_params": {"eq.problem": "box", "target": "300-600Hz"}},
        "glassy": {"meaning": "Crystalline highs", "daiw_params": {"eq.high_shelf": "+2dB", "dist": "none"}},
        "airy": {"meaning": "Air frequencies", "daiw_params": {"eq.air": "+3dB", "target": "12kHz+"}},
        "lo-fi": {"meaning": "Degraded, vintage", "daiw_params": {"character": "lofi", "degradation": 0.6}},
    },
    
    "groove_feel": {
        "laid back": {"meaning": "Behind the beat", "daiw_params": {"groove.pocket": "behind", "offset_ms": 15}},
        "pushing": {"meaning": "Ahead of beat", "daiw_params": {"groove.pocket": "ahead", "offset_ms": -10}},
        "on top": {"meaning": "Ahead of beat", "daiw_params": {"groove.pocket": "ahead", "offset_ms": -10}},
        "in the pocket": {"meaning": "Perfect lock", "daiw_params": {"groove.pocket": "locked"}},
        "swung": {"meaning": "Triplet timing", "daiw_params": {"groove.swing": 0.62}},
        "straight": {"meaning": "No swing", "daiw_params": {"groove.swing": 0.5}},
        "tight": {"meaning": "Precise timing", "daiw_params": {"humanize": 0.1}},
        "loose": {"meaning": "Relaxed timing", "daiw_params": {"humanize": 0.6}},
        "drunk": {"meaning": "Very loose", "daiw_params": {"humanize": 0.8, "groove.pocket": "behind"}},
        "driving": {"meaning": "Forward momentum", "daiw_params": {"groove.pocket": "ahead", "energy": "high"}},
        "breathing": {"meaning": "Tempo fluctuation", "daiw_params": {"tempo.rubato": True}},
    },
    
    "mix_production": {
        "glue": {"meaning": "Bus compression cohesion", "daiw_params": {"mix.bus_comp": True, "mix.glue": 0.7}},
        "separated": {"meaning": "Clear instrument distinction", "daiw_params": {"mix.separation": "high"}},
        "wall of sound": {"meaning": "Dense layers", "daiw_params": {"arrangement.density": "high", "mix.width": "wide"}},
        "intimate": {"meaning": "Close, dry", "daiw_params": {"mix.reverb": "short", "mix.distance": "close"}},
        "spacious": {"meaning": "Wide reverb", "daiw_params": {"mix.reverb": "long", "mix.width": "wide"}},
        "in your face": {"meaning": "Aggressive, forward", "daiw_params": {"mix.distance": "close", "comp.ratio": "high"}},
        "bedroom": {"meaning": "Lo-fi, intimate", "daiw_params": {"character": "lofi", "mix.distance": "close"}},
    },
}

MEME_PROGRESSIONS = {
    "creep": {
        "progression": "I-III-IV-iv",
        "formal_name": "Modal Interchange",
        "emotions": ["bittersweet", "melancholy"],
        "daiw_params": {"rule_break": "HARMONY_ModalInterchange"},
    },
    "mario cadence": {
        "progression": "bVI-bVII-I",
        "formal_name": "Double Plagal Cadence",
        "emotions": ["triumphant", "heroic"],
        "daiw_params": {"progression.type": "mario_cadence"},
    },
    "pachelbel": {
        "progression": "I-V-vi-iii-IV-I-IV-V",
        "formal_name": "Pachelbel Canon",
        "emotions": ["nostalgic", "wedding"],
        "daiw_params": {"progression.type": "pachelbel"},
    },
    "axis": {
        "progression": "I-V-vi-IV",
        "formal_name": "Axis Progression",
        "emotions": ["universal", "pop"],
        "daiw_params": {"progression.type": "axis"},
    },
    "andalusian": {
        "progression": "i-bVII-bVI-V",
        "formal_name": "Phrygian Descent",
        "emotions": ["spanish", "dramatic"],
        "daiw_params": {"progression.type": "andalusian"},
    },
}

EMOTION_TO_RULE_BREAK = {
    "bittersweet": [RuleBreakCode.HARMONY_ModalInterchange],
    "nostalgia": [RuleBreakCode.HARMONY_ModalInterchange],
    "longing": [RuleBreakCode.STRUCTURE_NonResolution, RuleBreakCode.HARMONY_ModalInterchange],
    "grief": [RuleBreakCode.STRUCTURE_NonResolution, RuleBreakCode.PRODUCTION_BuriedVocals],
    "power": [RuleBreakCode.HARMONY_ParallelMotion],
    "defiance": [RuleBreakCode.HARMONY_ParallelMotion],
    "anxiety": [RuleBreakCode.RHYTHM_ConstantDisplacement, RuleBreakCode.RHYTHM_MeterAmbiguity],
    "chaos": [RuleBreakCode.HARMONY_Polytonality, RuleBreakCode.RHYTHM_MeterAmbiguity],
    "vulnerability": [RuleBreakCode.PRODUCTION_PitchImperfection, RuleBreakCode.RHYTHM_TempoFluctuation],
    "intimacy": [RuleBreakCode.RHYTHM_TempoFluctuation, RuleBreakCode.PRODUCTION_PitchImperfection],
    "dissociation": [RuleBreakCode.PRODUCTION_BuriedVocals],
    "unfinished": [RuleBreakCode.STRUCTURE_NonResolution],
    "tension": [RuleBreakCode.HARMONY_UnresolvedDissonance],
}


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSLATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class VernacularTranslator:
    """
    Translates casual music descriptions to DAiW parameters.
    
    Philosophy: Musicians describe sound in vernacular, not technical terms.
    This class bridges that gap.
    """
    
    def __init__(self, custom_db: Optional[Dict] = None):
        """Initialize with optional custom database."""
        self.db = custom_db or VERNACULAR_DB
        self.meme_progressions = MEME_PROGRESSIONS
        self.emotion_rules = EMOTION_TO_RULE_BREAK
        
        # Build reverse lookup
        self._build_lookup()
    
    def _build_lookup(self) -> None:
        """Build flat lookup dictionary for matching."""
        self._lookup: Dict[str, Tuple[VernacularCategory, Dict]] = {}
        
        for category_name, terms in self.db.items():
            category = VernacularCategory(category_name)
            for term, data in terms.items():
                self._lookup[term.lower()] = (category, data)
        
        # Add meme progressions
        for term, data in self.meme_progressions.items():
            self._lookup[term.lower()] = (VernacularCategory.MEME_THEORY, data)
    
    def translate(self, input_text: str) -> TranslationResult:
        """
        Translate vernacular description to DAiW parameters.
        
        Args:
            input_text: Natural language description
        
        Returns:
            TranslationResult with matched terms and combined parameters
        """
        input_lower = input_text.lower()
        matched = []
        combined_params = {}
        matched_spans = set()
        
        # Try to match multi-word phrases first (longer = higher priority)
        sorted_terms = sorted(self._lookup.keys(), key=len, reverse=True)
        
        for term in sorted_terms:
            if term in input_lower:
                # Check if this span was already matched
                start = input_lower.find(term)
                end = start + len(term)
                span = (start, end)
                
                # Skip if overlapping with existing match
                overlaps = any(
                    not (end <= s or start >= e)
                    for s, e in matched_spans
                )
                if overlaps:
                    continue
                
                matched_spans.add(span)
                
                category, data = self._lookup[term]
                match = VernacularMatch(
                    term=term,
                    category=category,
                    meaning=data.get("meaning", ""),
                    daiw_params=data.get("daiw_params", {})
                )
                matched.append(match)
                
                # Merge parameters
                combined_params.update(data.get("daiw_params", {}))
        
        # Find unmatched words
        words = input_lower.split()
        matched_words = set()
        for start, end in matched_spans:
            for word in input_lower[start:end].split():
                matched_words.add(word)
        
        unmatched = [w for w in words if w not in matched_words and len(w) > 2]
        
        # Suggest rule breaks based on detected emotions
        suggested_breaks = []
        for term, data in self.meme_progressions.items():
            if term in input_lower:
                for emotion in data.get("emotions", []):
                    if emotion in self.emotion_rules:
                        for rule in self.emotion_rules[emotion]:
                            if rule.value not in suggested_breaks:
                                suggested_breaks.append(rule.value)
        
        return TranslationResult(
            original_input=input_text,
            matched_terms=matched,
            combined_params=combined_params,
            unmatched_words=unmatched,
            suggested_rule_breaks=suggested_breaks
        )
    
    def get_rule_breaks_for_emotion(self, emotion: str) -> List[str]:
        """Get suggested rule breaks for an emotion."""
        emotion_lower = emotion.lower()
        if emotion_lower in self.emotion_rules:
            return [r.value for r in self.emotion_rules[emotion_lower]]
        return []
    
    def explain_term(self, term: str) -> Optional[Dict]:
        """Explain what a vernacular term means."""
        term_lower = term.lower()
        if term_lower in self._lookup:
            category, data = self._lookup[term_lower]
            return {
                "term": term,
                "category": category.value,
                "meaning": data.get("meaning", ""),
                "parameters": data.get("daiw_params", {})
            }
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def translate_vernacular(text: str) -> Dict[str, Any]:
    """Quick function to translate vernacular text."""
    translator = VernacularTranslator()
    result = translator.translate(text)
    return result.combined_params


def explain_vernacular(term: str) -> Optional[Dict]:
    """Quick function to explain a vernacular term."""
    translator = VernacularTranslator()
    return translator.explain_term(term)
