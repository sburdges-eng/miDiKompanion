"""
Counterpoint Teacher
====================

Interactive teaching for species counterpoint.
"""

from typing import Dict, List, Optional, Any
from ..rules.counterpoint_rules import CounterpointRules
from ..rules.species import Species
from ..rules.base import Rule


class CounterpointTeacher:
    """
    Teacher for species counterpoint with progressive lessons.
    
    Methods:
        get_species_overview(species) - Overview of a species
        get_rules_for_species(species) - All rules for a species
        check_counterpoint(cantus, counterpoint, species) - Check for violations
        generate_exercise(species, difficulty) - Generate practice exercises
    """
    
    SPECIES_DESCRIPTIONS = {
        Species.FIRST: {
            "name": "First Species (1:1)",
            "description": "Note against note - one counterpoint note per cantus firmus note",
            "key_concepts": [
                "Only consonances allowed (P1, P5, P8, m3, M3, m6, M6)",
                "Begin and end on perfect consonance",
                "Contrary motion preferred",
                "No parallel or hidden fifths/octaves",
            ],
            "difficulty": "Beginner",
            "focus": "Pure consonance and voice independence",
        },
        Species.SECOND: {
            "name": "Second Species (2:1)",
            "description": "Two notes against one - introduces passing tones",
            "key_concepts": [
                "Strong beats must be consonant",
                "Weak beats can have passing dissonances",
                "Dissonances approached by step",
                "Introduces melodic elaboration",
            ],
            "difficulty": "Beginner-Intermediate",
            "focus": "Introducing dissonance as passing motion",
        },
        Species.THIRD: {
            "name": "Third Species (4:1)",
            "description": "Four notes against one - elaborate melodic motion",
            "key_concepts": [
                "First of each four-note group consonant",
                "Cambiata figure allowed",
                "More freedom for melodic shapes",
                "Greater rhythmic variety",
            ],
            "difficulty": "Intermediate",
            "focus": "Elaborate melodic writing",
        },
        Species.FOURTH: {
            "name": "Fourth Species (Syncopation)",
            "description": "Suspensions and their resolutions",
            "key_concepts": [
                "Tied notes create suspensions",
                "Prepare-suspend-resolve pattern",
                "Standard suspensions: 7-6, 4-3, 9-8",
                "Resolution always down by step",
            ],
            "difficulty": "Intermediate-Advanced",
            "focus": "Controlled dissonance through suspension",
        },
        Species.FIFTH: {
            "name": "Fifth Species (Florid)",
            "description": "Free counterpoint combining all techniques",
            "key_concepts": [
                "Combines techniques from all species",
                "Rhythmic variety within metric clarity",
                "Most expressive species",
                "Foundation for free composition",
            ],
            "difficulty": "Advanced",
            "focus": "Integration and musical expression",
        },
    }
    
    def __init__(self):
        self.rules = CounterpointRules
    
    def get_species_overview(self, species: Species) -> Dict[str, Any]:
        """Get overview information for a species."""
        info = self.SPECIES_DESCRIPTIONS.get(species, {})
        rules = self.rules.get_all_rules_for_species(species)
        
        return {
            **info,
            "ratio": species.ratio,
            "allowed_intervals": species.allowed_intervals,
            "rule_count": len(rules),
            "rules_summary": [
                {"id": r.id, "name": r.name, "severity": r.severity.value}
                for r in rules.values()
            ],
        }
    
    def get_rules_for_species(self, species: Species) -> Dict[str, Rule]:
        """Get all rules for a specific species."""
        return self.rules.get_all_rules_for_species(species)
    
    def get_progressive_curriculum(self) -> List[Dict[str, Any]]:
        """Get a progressive learning curriculum through all species."""
        curriculum = []
        
        for species in Species:
            info = self.SPECIES_DESCRIPTIONS.get(species, {})
            rules = self.rules.get_species_rules(species)
            
            curriculum.append({
                "species": species.value,
                "name": info.get("name", f"Species {species.value}"),
                "difficulty": info.get("difficulty", "Unknown"),
                "focus": info.get("focus", ""),
                "key_concepts": info.get("key_concepts", []),
                "new_rules_introduced": len(rules),
                "prerequisites": [s.value for s in Species if s.value < species.value],
            })
        
        return curriculum
    
    def generate_exercise(
        self, 
        species: Species, 
        length: int = 8,
        mode: str = "dorian"
    ) -> Dict[str, Any]:
        """
        Generate a counterpoint exercise.
        
        Args:
            species: Which species to practice
            length: Number of notes in cantus firmus
            mode: Modal basis (dorian, phrygian, etc.)
        
        Returns:
            Exercise with cantus firmus and instructions
        """
        # Example cantus firmi for different modes
        cantus_examples = {
            "dorian": [62, 64, 65, 67, 69, 67, 65, 64, 62],  # D dorian
            "phrygian": [64, 62, 64, 65, 67, 65, 64, 62, 64],  # E phrygian
            "mixolydian": [67, 69, 71, 72, 71, 69, 67, 65, 67],  # G mixolydian
        }
        
        cantus = cantus_examples.get(mode, cantus_examples["dorian"])[:length]
        
        rules = self.rules.get_all_rules_for_species(species)
        
        return {
            "species": species.value,
            "species_name": self.SPECIES_DESCRIPTIONS[species]["name"],
            "mode": mode,
            "cantus_firmus": cantus,
            "cantus_note_names": self._midi_to_names(cantus),
            "instructions": self._get_exercise_instructions(species),
            "rules_to_follow": [
                {"name": r.name, "description": r.description}
                for r in rules.values()
            ],
            "tips": self._get_exercise_tips(species),
        }
    
    def _midi_to_names(self, midi_notes: List[int]) -> List[str]:
        """Convert MIDI note numbers to note names."""
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return [f"{note_names[n % 12]}{n // 12 - 1}" for n in midi_notes]
    
    def _get_exercise_instructions(self, species: Species) -> str:
        """Get instructions for an exercise."""
        instructions = {
            Species.FIRST: "Write one note against each cantus firmus note. Use only consonances. Begin and end on a perfect consonance (P1, P5, or P8).",
            Species.SECOND: "Write two notes against each cantus firmus note. Strong beats must be consonant; weak beats may be passing tones.",
            Species.THIRD: "Write four notes against each cantus firmus note. First note of each group must be consonant.",
            Species.FOURTH: "Write syncopated half notes (tied across the bar). Create suspensions that resolve down by step.",
            Species.FIFTH: "Combine techniques from all species. Create a musical, flowing line with rhythmic variety.",
        }
        return instructions.get(species, "Complete the counterpoint exercise.")
    
    def _get_exercise_tips(self, species: Species) -> List[str]:
        """Get tips for completing an exercise."""
        tips = {
            Species.FIRST: [
                "Start by identifying all possible consonances above/below each note",
                "Prioritize contrary motion",
                "Check for parallel fifths and octaves",
                "Aim for a single climax point in your line",
            ],
            Species.SECOND: [
                "Plan your strong beats first (they must be consonant)",
                "Use step motion to connect strong beats",
                "Passing tones should fill in leaps of a third",
                "Avoid accented dissonances",
            ],
            Species.THIRD: [
                "Think in terms of four-note groups",
                "The cambiata is your friend for difficult spots",
                "Maintain metric clarity despite faster notes",
                "Use the faster rhythm to create melodic interest",
            ],
            Species.FOURTH: [
                "The suspension is: prepare → suspend → resolve",
                "7-6 and 4-3 are the most common suspensions",
                "The resolution is always down by step",
                "Don't suspend into a unison",
            ],
            Species.FIFTH: [
                "Plan your overall shape before writing details",
                "Use faster notes for decoration, not constantly",
                "Save your most elaborate writing for climactic moments",
                "The line should feel natural to sing",
            ],
        }
        return tips.get(species, [])
