"""
Species Counterpoint Definitions
================================

The five species of counterpoint as codified by Fux (1725).
"""

from enum import Enum


class Species(Enum):
    """
    Species counterpoint categories from Fux's Gradus ad Parnassum.
    
    Each species introduces new rhythmic relationships and techniques.
    """
    FIRST = 1    # Note against note (1:1)
    SECOND = 2   # Two notes against one (2:1)
    THIRD = 3    # Four notes against one (4:1)
    FOURTH = 4   # Syncopation and suspensions
    FIFTH = 5    # Florid/free counterpoint (mixed)
    
    def __str__(self) -> str:
        return f"Species {self.value}"
    
    @property
    def ratio(self) -> str:
        """The note ratio for this species."""
        ratios = {1: "1:1", 2: "2:1", 3: "4:1", 4: "syncopated", 5: "mixed/free"}
        return ratios.get(self.value, "unknown")
    
    @property
    def description(self) -> str:
        """Human-readable description."""
        descriptions = {
            1: "Note against note - one counterpoint note per cantus firmus note",
            2: "Two notes against one - introduces passing tones",
            3: "Four notes against one - more elaborate melodic motion",
            4: "Syncopation - suspensions and their resolutions",
            5: "Florid/free - combines all previous species",
        }
        return descriptions.get(self.value, "")
    
    @property
    def allowed_intervals(self) -> list:
        """Intervals typically allowed on strong beats for this species."""
        # Perfect consonances and imperfect consonances
        if self.value == 1:
            return ["P1", "P5", "P8", "m3", "M3", "m6", "M6"]
        elif self.value == 2:
            # Strong beats: consonances; weak beats: passing tones allowed
            return ["P1", "P5", "P8", "m3", "M3", "m6", "M6", "P4"]  # P4 on weak only
        elif self.value == 3:
            return ["P1", "P5", "P8", "m3", "M3", "m6", "M6", "P4", "m2", "M2"]
        elif self.value == 4:
            # Suspensions resolve down by step
            return ["P1", "P5", "P8", "m3", "M3", "m6", "M6", "7-6", "4-3", "9-8"]
        else:  # Fifth species
            return ["all"]  # Free combination
