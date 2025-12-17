"""
Counterpoint Rules
==================

Species counterpoint rules from Fux through modern applications.
"""

from typing import Dict, List, Optional, Any
from .base import Rule
from .severity import RuleSeverity
from .species import Species


class CounterpointRules:
    """
    Collection of counterpoint rules organized by species.
    
    Based on Fux's Gradus ad Parnassum (1725) with modern interpretations.
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GENERAL COUNTERPOINT RULES (All Species)
    # ═══════════════════════════════════════════════════════════════════════════
    
    GENERAL_RULES: Dict[str, Rule] = {
        "melodic_motion": Rule(
            id="melodic_motion",
            name="Prefer Stepwise Motion",
            description="Melodies should move primarily by step with occasional leaps",
            reason="Creates smooth, singable lines",
            severity=RuleSeverity.MODERATE,
            contexts=["all"],
            exceptions=["Triadic outlines", "Expressive leaps"],
            emotional_uses=["angular_melody", "dramatic_leaps"],
        ),
        
        "leap_recovery": Rule(
            id="leap_recovery",
            name="Recover Leaps by Step",
            description="Large leaps should be followed by stepwise motion in the opposite direction",
            reason="Balances melodic tension",
            severity=RuleSeverity.MODERATE,
            contexts=["all"],
            exceptions=["Consecutive thirds (broken chord)", "Triadic outlines"],
            emotional_uses=["relentless_drive", "mounting_tension"],
        ),
        
        "avoid_augmented_intervals": Rule(
            id="avoid_augmented_intervals",
            name="Avoid Augmented Melodic Intervals",
            description="Don't leap by augmented intervals (aug 2nd, aug 4th)",
            reason="Augmented intervals are difficult to sing in tune",
            severity=RuleSeverity.STRICT,
            contexts=["renaissance", "baroque"],
            severity_by_context={
                "renaissance": RuleSeverity.STRICT,
                "twentieth_century": RuleSeverity.FLEXIBLE,
                "jazz": RuleSeverity.FLEXIBLE,
            },
            emotional_uses=["exotic_color", "tension"],
        ),
        
        "contrary_motion_preferred": Rule(
            id="contrary_motion_preferred",
            name="Prefer Contrary Motion",
            description="Voices should move in opposite directions when possible",
            reason="Maintains voice independence",
            severity=RuleSeverity.MODERATE,
            contexts=["all"],
            emotional_uses=[],
        ),
        
        "climax_placement": Rule(
            id="climax_placement",
            name="Single Climax Point",
            description="Each melodic line should have one clear high point",
            reason="Creates shape and direction",
            severity=RuleSeverity.MODERATE,
            contexts=["all"],
            exceptions=["Long melodies may have secondary peaks"],
            emotional_uses=["multiple_climaxes", "relentless_intensity"],
        ),
        
        "variety_of_intervals": Rule(
            id="variety_of_intervals",
            name="Variety of Vertical Intervals",
            description="Use a variety of consonant intervals, not just one type",
            reason="Creates interest and color",
            severity=RuleSeverity.MODERATE,
            contexts=["all"],
            emotional_uses=[],
        ),
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SPECIES-SPECIFIC RULES
    # ═══════════════════════════════════════════════════════════════════════════
    
    SPECIES_RULES: Dict[Species, Dict[str, Rule]] = {
        Species.FIRST: {
            "consonance_only": Rule(
                id="first_consonance_only",
                name="Consonances Only (First Species)",
                description="Only consonant intervals allowed: P1, P5, P8, m3, M3, m6, M6",
                reason="First species establishes pure consonance",
                severity=RuleSeverity.STRICT,
                contexts=["educational", "renaissance"],
                emotional_uses=[],
            ),
            
            "begin_end_perfect": Rule(
                id="first_begin_end_perfect",
                name="Begin and End on Perfect Consonance",
                description="Start and end on P1, P5, or P8",
                reason="Creates stability at structural points",
                severity=RuleSeverity.STRICT,
                contexts=["educational", "renaissance"],
                emotional_uses=["ambiguous_ending"],
            ),
            
            "no_unison_except_ends": Rule(
                id="first_no_unison_mid",
                name="No Unison in Middle",
                description="Unisons (P1) only at beginning or end",
                reason="Unison eliminates independence of voices",
                severity=RuleSeverity.MODERATE,
                contexts=["educational"],
                emotional_uses=["convergence", "unity"],
            ),
        },
        
        Species.SECOND: {
            "strong_beat_consonance": Rule(
                id="second_strong_consonance",
                name="Consonance on Strong Beats",
                description="Strong beats (1, 3) must be consonant",
                reason="Maintains harmonic stability",
                severity=RuleSeverity.STRICT,
                contexts=["educational"],
                emotional_uses=[],
            ),
            
            "passing_tones_allowed": Rule(
                id="second_passing_tones",
                name="Passing Tones on Weak Beats",
                description="Dissonances allowed on weak beats as passing tones",
                reason="Creates melodic fluidity",
                severity=RuleSeverity.ENCOURAGED,
                contexts=["educational"],
                emotional_uses=[],
            ),
            
            "approach_by_step": Rule(
                id="second_approach_by_step",
                name="Approach Dissonance by Step",
                description="Dissonant weak beats must be approached by step",
                reason="Smooths the dissonance",
                severity=RuleSeverity.STRICT,
                contexts=["educational"],
                emotional_uses=["jarring_leap_to_dissonance"],
            ),
        },
        
        Species.THIRD: {
            "first_note_consonant": Rule(
                id="third_first_consonant",
                name="First Note of Each Group Consonant",
                description="First of each four-note group must be consonant",
                reason="Anchors each beat",
                severity=RuleSeverity.STRICT,
                contexts=["educational"],
                emotional_uses=[],
            ),
            
            "cambiata_allowed": Rule(
                id="third_cambiata",
                name="Cambiata Figure Allowed",
                description="Special dissonant pattern: step down, skip down third, step up",
                reason="Traditional melodic ornament",
                severity=RuleSeverity.ENCOURAGED,
                contexts=["educational", "renaissance"],
                emotional_uses=[],
            ),
            
            "nota_cambiata": Rule(
                id="third_nota_cambiata",
                name="Nota Cambiata Pattern",
                description="Consonance-dissonance-consonance-consonance pattern",
                reason="Establishes melodic figure vocabulary",
                severity=RuleSeverity.STYLISTIC,
                contexts=["renaissance"],
                emotional_uses=[],
            ),
        },
        
        Species.FOURTH: {
            "suspension_preparation": Rule(
                id="fourth_suspension_prep",
                name="Prepare Suspensions",
                description="Suspended note must be consonant on previous weak beat",
                reason="Establishes the note as consonance before it becomes dissonance",
                severity=RuleSeverity.STRICT,
                contexts=["educational", "baroque"],
                emotional_uses=["unprepared_shock"],
            ),
            
            "suspension_resolution": Rule(
                id="fourth_suspension_resolve",
                name="Resolve Suspensions Down by Step",
                description="Suspended dissonance resolves down by step to consonance",
                reason="Creates satisfying tension-release",
                severity=RuleSeverity.STRICT,
                contexts=["educational", "baroque"],
                examples=[
                    {"artist": "Bach", "piece": "Various",
                     "detail": "7-6, 4-3, 9-8 suspensions throughout"},
                ],
                emotional_uses=["unresolved_suspension", "upward_resolution"],
            ),
            
            "standard_suspensions": Rule(
                id="fourth_standard_suspensions",
                name="Standard Suspension Types",
                description="Use 7-6, 4-3, and 9-8 suspensions",
                reason="These are the consonant resolutions",
                severity=RuleSeverity.MODERATE,
                contexts=["educational"],
                emotional_uses=[],
            ),
        },
        
        Species.FIFTH: {
            "combine_all_species": Rule(
                id="fifth_combine_species",
                name="Combine All Species",
                description="Florid counterpoint combines techniques from all species",
                reason="Creates rich, varied melodic writing",
                severity=RuleSeverity.STYLISTIC,
                contexts=["educational", "renaissance", "baroque"],
                emotional_uses=[],
            ),
            
            "maintain_pulse": Rule(
                id="fifth_maintain_pulse",
                name="Maintain Metric Clarity",
                description="Despite rhythmic variety, keep metric structure clear",
                reason="Prevents rhythmic chaos",
                severity=RuleSeverity.MODERATE,
                contexts=["educational"],
                emotional_uses=["metric_ambiguity", "polyrhythm"],
            ),
        },
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CLASS METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @classmethod
    def get_general_rules(cls) -> Dict[str, Rule]:
        """Get rules that apply to all species."""
        return cls.GENERAL_RULES
    
    @classmethod
    def get_species_rules(cls, species: Species) -> Dict[str, Rule]:
        """Get rules specific to a particular species."""
        return cls.SPECIES_RULES.get(species, {})
    
    @classmethod
    def get_all_rules_for_species(cls, species: Species) -> Dict[str, Rule]:
        """Get both general and species-specific rules."""
        rules = dict(cls.GENERAL_RULES)
        rules.update(cls.SPECIES_RULES.get(species, {}))
        return rules
    
    @classmethod
    def get_rule(cls, rule_id: str) -> Optional[Rule]:
        """Get a specific rule by ID."""
        if rule_id in cls.GENERAL_RULES:
            return cls.GENERAL_RULES[rule_id]
        
        for species, rules in cls.SPECIES_RULES.items():
            if rule_id in rules:
                return rules[rule_id]
        
        return None
    
    @classmethod
    def get_all_rules(cls) -> Dict[str, Dict[str, Rule]]:
        """Get all counterpoint rules organized by category."""
        return {
            "general": cls.GENERAL_RULES,
            "first_species": cls.SPECIES_RULES.get(Species.FIRST, {}),
            "second_species": cls.SPECIES_RULES.get(Species.SECOND, {}),
            "third_species": cls.SPECIES_RULES.get(Species.THIRD, {}),
            "fourth_species": cls.SPECIES_RULES.get(Species.FOURTH, {}),
            "fifth_species": cls.SPECIES_RULES.get(Species.FIFTH, {}),
        }
