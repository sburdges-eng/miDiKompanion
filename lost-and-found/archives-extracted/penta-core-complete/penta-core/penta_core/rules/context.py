"""
Musical Context Definitions
===========================

Different musical styles and periods have different rules.
"""

from enum import Enum
from typing import List


class MusicalContext(Enum):
    """
    Musical style contexts that affect which rules apply.
    
    Rules that are strict in one context may be flexible or even
    encouraged in another.
    """
    # Historical periods
    RENAISSANCE = "renaissance"
    BAROQUE = "baroque"
    CLASSICAL = "classical"
    ROMANTIC = "romantic"
    IMPRESSIONIST = "impressionist"
    TWENTIETH_CENTURY = "twentieth_century"
    CONTEMPORARY = "contemporary"
    
    # Genre contexts
    JAZZ = "jazz"
    ROCK = "rock"
    METAL = "metal"
    POP = "pop"
    ELECTRONIC = "electronic"
    HIP_HOP = "hip_hop"
    LOFI = "lofi"
    
    # Functional contexts
    FILM = "film"
    GAME = "game"
    LITURGICAL = "liturgical"
    EDUCATIONAL = "educational"
    
    def __str__(self) -> str:
        return self.value.replace("_", " ").title()
    
    @property
    def is_classical(self) -> bool:
        """Is this a 'classical' (CPP-influenced) context?"""
        return self in [
            MusicalContext.RENAISSANCE,
            MusicalContext.BAROQUE,
            MusicalContext.CLASSICAL,
            MusicalContext.ROMANTIC,
            MusicalContext.LITURGICAL,
            MusicalContext.EDUCATIONAL,
        ]
    
    @property
    def allows_parallel_fifths(self) -> bool:
        """Does this context allow parallel perfect fifths?"""
        return self in [
            MusicalContext.ROCK,
            MusicalContext.METAL,
            MusicalContext.POP,
            MusicalContext.ELECTRONIC,
            MusicalContext.IMPRESSIONIST,
            MusicalContext.CONTEMPORARY,
            MusicalContext.LOFI,
            MusicalContext.HIP_HOP,
        ]
    
    @property
    def typical_rules(self) -> List[str]:
        """Key rules that apply in this context."""
        if self.is_classical:
            return [
                "no_parallel_fifths",
                "no_parallel_octaves",
                "resolve_dissonance",
                "resolve_leading_tone",
                "no_voice_crossing",
            ]
        elif self == MusicalContext.JAZZ:
            return [
                "tritone_substitution_allowed",
                "extended_chords",
                "chromatic_approach",
                "modal_interchange",
            ]
        elif self in [MusicalContext.ROCK, MusicalContext.METAL]:
            return [
                "parallel_fifths_encouraged",
                "power_chords",
                "modal_mixture",
            ]
        elif self == MusicalContext.LOFI:
            return [
                "imperfection_intentional",
                "buried_elements",
                "tempo_fluctuation",
                "pitch_imperfection",
            ]
        return []


# Context groups for filtering
CONTEXT_GROUPS = {
    "common_practice": [
        MusicalContext.BAROQUE,
        MusicalContext.CLASSICAL,
        MusicalContext.ROMANTIC,
    ],
    "modern": [
        MusicalContext.JAZZ,
        MusicalContext.ROCK,
        MusicalContext.POP,
        MusicalContext.ELECTRONIC,
        MusicalContext.CONTEMPORARY,
    ],
    "strict": [
        MusicalContext.RENAISSANCE,
        MusicalContext.BAROQUE,
        MusicalContext.LITURGICAL,
        MusicalContext.EDUCATIONAL,
    ],
    "flexible": [
        MusicalContext.JAZZ,
        MusicalContext.ROCK,
        MusicalContext.METAL,
        MusicalContext.ELECTRONIC,
        MusicalContext.LOFI,
        MusicalContext.CONTEMPORARY,
    ],
}
