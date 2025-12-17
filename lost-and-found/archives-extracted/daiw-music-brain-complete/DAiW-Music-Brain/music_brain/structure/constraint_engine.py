"""
DAiW - Constraint Engine

Executes Phase 2: Technical & Groovy Constraints.
Translates emotional intent into technical parameters and proposes rule-breaking.

Based on: rule_breaking_masterpieces.md database
"""

from typing import Tuple, Dict, List, Optional
from random import choice
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from .models import (
    IntentModel, ConstraintModel, RuleBreakModel,
    GrooveFeel, RuleToBreak, VulnerabilityScale, NarrativeArc
)

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# MAPPING DATABASES
# ═══════════════════════════════════════════════════════════════════════════════

# Mood → Genre suggestions
MOOD_TO_GENRE = {
    "grief": ["Lo-Fi Bedroom Emo", "Ambient/Post-Rock", "Slowcore"],
    "anger": ["Industrial", "Punk", "Aggressive Electronic"],
    "defiance": ["Alt-Rock", "Industrial Pop", "Anthemic"],
    "nostalgia": ["Indie Folk", "Dream Pop", "Lo-Fi"],
    "fear": ["Dark Ambient", "Industrial", "Experimental"],
    "awe": ["Post-Rock", "Ambient", "Neo-Classical"],
    "longing": ["Shoegaze", "Dream Pop", "Chamber Pop"],
}

# Mood → Tempo suggestions
MOOD_TO_TEMPO = {
    "grief": (60, 85),
    "anger": (120, 160),
    "defiance": (100, 140),
    "nostalgia": (70, 95),
    "fear": (80, 120),
    "awe": (65, 90),
    "longing": (70, 90),
}

# Mood → Key suggestions
MOOD_TO_KEY = {
    "grief": ["F", "Am", "Dm", "Gm"],
    "anger": ["Em", "Bm", "D", "G"],
    "defiance": ["E", "A", "D", "G"],
    "nostalgia": ["C", "G", "D", "F"],
    "fear": ["Dm", "Am", "Em", "Bm"],
    "awe": ["C", "G", "F", "Bb"],
    "longing": ["F", "C", "Am", "Dm"],
}

# Mood + Arc → Rule Break suggestions
RULE_BREAK_SUGGESTIONS: Dict[str, Dict[str, List[RuleToBreak]]] = {
    "grief": {
        "Climb-to-Climax": [RuleToBreak.HARMONY_ModalInterchange, RuleToBreak.STRUCTURE_NonResolution],
        "Slow Reveal": [RuleToBreak.PRODUCTION_BuriedVocals, RuleToBreak.HARMONY_AvoidTonicResolution],
        "Repetitive Despair": [RuleToBreak.RHYTHM_ConstantDisplacement, RuleToBreak.ARRANGEMENT_SparseClimax],
        "Sudden Shift": [RuleToBreak.HARMONY_ModalInterchange, RuleToBreak.ARRANGEMENT_InstrumentDrop],
        "Static Reflection": [RuleToBreak.STRUCTURE_NonResolution, RuleToBreak.PRODUCTION_LoFiArtifacts],
    },
    "anger": {
        "Climb-to-Climax": [RuleToBreak.HARMONY_ParallelMotion, RuleToBreak.RHYTHM_ConstantDisplacement],
        "default": [RuleToBreak.HARMONY_ParallelMotion, RuleToBreak.STRUCTURE_AsymmetricForm],
    },
    "defiance": {
        "default": [RuleToBreak.HARMONY_ParallelMotion, RuleToBreak.STRUCTURE_AntiDrop],
    },
    "fear": {
        "default": [RuleToBreak.HARMONY_UnresolvedDissonance, RuleToBreak.RHYTHM_MeterAmbiguity],
    },
    "nostalgia": {
        "default": [RuleToBreak.HARMONY_ModalInterchange, RuleToBreak.PRODUCTION_LoFiArtifacts],
    },
    "awe": {
        "default": [RuleToBreak.STRUCTURE_NonResolution, RuleToBreak.HARMONY_Polytonality],
    },
    "longing": {
        "default": [RuleToBreak.HARMONY_AvoidTonicResolution, RuleToBreak.STRUCTURE_NonResolution],
    },
}

# Rule break justifications
RULE_BREAK_JUSTIFICATIONS = {
    RuleToBreak.HARMONY_ModalInterchange: "Borrowed chords create bittersweet ambiguity - hope that feels earned, not given",
    RuleToBreak.HARMONY_ParallelMotion: "Parallel fifths create raw, defiant power - unity in defiance",
    RuleToBreak.HARMONY_UnresolvedDissonance: "Unresolved tension mirrors the emotional state - no easy answers",
    RuleToBreak.HARMONY_AvoidTonicResolution: "Avoiding home creates perpetual yearning - the ache of incompleteness",
    RuleToBreak.HARMONY_Polytonality: "Multiple keys create disorientation - fractured reality",
    RuleToBreak.RHYTHM_ConstantDisplacement: "Off-kilter timing creates unease - ground shifting underfoot",
    RuleToBreak.RHYTHM_MeterAmbiguity: "Unclear pulse creates floating anxiety - losing grip on time",
    RuleToBreak.STRUCTURE_NonResolution: "No resolution mirrors emotional truth - some things don't resolve",
    RuleToBreak.STRUCTURE_AsymmetricForm: "Lopsided structure mirrors chaos - refusing neat packaging",
    RuleToBreak.STRUCTURE_AntiDrop: "Quiet instead of explosion - devastation through restraint",
    RuleToBreak.PRODUCTION_BuriedVocals: "Half-heard lyrics create intimacy - overheard confession",
    RuleToBreak.PRODUCTION_LoFiArtifacts: "Imperfection signals authenticity - rawness as honesty",
    RuleToBreak.PRODUCTION_PitchImperfection: "Slightly off pitch creates vulnerability - human, not machine",
    RuleToBreak.ARRANGEMENT_SparseClimax: "Less is more at peak - restraint as devastation",
    RuleToBreak.ARRANGEMENT_InstrumentDrop: "Sudden absence creates presence - what's missing screams",
}


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRAINT PROPOSAL
# ═══════════════════════════════════════════════════════════════════════════════

def _propose_technical_constraints(intent: IntentModel) -> ConstraintModel:
    """Generate technical constraint proposals based on intent."""
    mood = intent.mood_primary.lower().split()[0] if intent.mood_primary else "neutral"
    
    # Get suggestions or defaults
    genres = MOOD_TO_GENRE.get(mood, ["Alternative", "Indie", "Electronic"])
    tempo_range = MOOD_TO_TEMPO.get(mood, (100, 120))
    keys = MOOD_TO_KEY.get(mood, ["C", "Am", "F", "G"])
    
    # Groove feel based on vulnerability
    groove_map = {
        VulnerabilityScale.LOW: GrooveFeel.STRAIGHT,
        VulnerabilityScale.MEDIUM: GrooveFeel.SYNCOPATED,
        VulnerabilityScale.HIGH: GrooveFeel.FLOATING,
    }
    groove = groove_map.get(intent.vulnerability_scale, GrooveFeel.STRAIGHT)
    
    return ConstraintModel(
        technical_genre=choice(genres),
        technical_tempo_range=tempo_range,
        technical_groove_feel=groove,
        technical_key=choice(keys),
        technical_mode="minor" if mood in ["grief", "fear", "anger"] else "major"
    )


def _propose_rule_break(intent: IntentModel) -> RuleBreakModel:
    """Propose a rule to break based on emotional intent."""
    mood = intent.mood_primary.lower().split()[0] if intent.mood_primary else "neutral"
    arc = intent.narrative_arc.value if intent.narrative_arc else "default"
    
    # Get suggestions
    mood_rules = RULE_BREAK_SUGGESTIONS.get(mood, {})
    rule_options = mood_rules.get(arc, mood_rules.get("default", [RuleToBreak.NONE]))
    
    if not rule_options:
        rule_options = [RuleToBreak.NONE]
    
    selected_rule = choice(rule_options)
    justification = RULE_BREAK_JUSTIFICATIONS.get(selected_rule, "")
    
    # Intensity based on vulnerability
    intensity_map = {
        VulnerabilityScale.LOW: 0.3,
        VulnerabilityScale.MEDIUM: 0.5,
        VulnerabilityScale.HIGH: 0.8,
    }
    intensity = intensity_map.get(intent.vulnerability_scale, 0.5)
    
    return RuleBreakModel(
        technical_rule_to_break=selected_rule,
        rule_breaking_justification=justification,
        rule_breaking_intensity=intensity
    )


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE REVIEW
# ═══════════════════════════════════════════════════════════════════════════════

def _review_and_finalize_constraints(
    constraint: ConstraintModel,
    rule_break: RuleBreakModel
) -> Tuple[ConstraintModel, RuleBreakModel]:
    """Interactive review and modification of constraints."""
    
    # Display current proposals
    table = Table(title="Proposed Technical Constraints")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Genre", constraint.technical_genre)
    table.add_row("Tempo", f"{constraint.technical_tempo_range[0]}-{constraint.technical_tempo_range[1]} BPM")
    table.add_row("Key", constraint.technical_key)
    table.add_row("Mode", constraint.technical_mode)
    table.add_row("Groove Feel", constraint.technical_groove_feel.value)
    table.add_row("Rule to Break", rule_break.technical_rule_to_break.value)
    table.add_row("Justification", rule_break.rule_breaking_justification[:60] + "...")
    
    console.print(table)
    
    # Ask for approval
    if Confirm.ask("\n[cyan]Accept these constraints?[/cyan]", default=True):
        return constraint, rule_break
    
    # Allow modifications
    console.print("\n[yellow]Let's adjust...[/yellow]")
    
    # Tempo
    new_tempo = Prompt.ask(
        f"[green]Tempo range (current: {constraint.technical_tempo_range})[/green]",
        default=f"{constraint.technical_tempo_range[0]}-{constraint.technical_tempo_range[1]}"
    )
    try:
        parts = new_tempo.replace(" ", "").split("-")
        tempo_range = (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        tempo_range = constraint.technical_tempo_range
    
    # Key
    new_key = Prompt.ask(
        f"[green]Key (current: {constraint.technical_key})[/green]",
        default=constraint.technical_key
    )
    
    # Return modified constraints
    return ConstraintModel(
        technical_genre=constraint.technical_genre,
        technical_tempo_range=tempo_range,
        technical_groove_feel=constraint.technical_groove_feel,
        technical_key=new_key,
        technical_mode=constraint.technical_mode
    ), rule_break


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def propose_constraints(intent: IntentModel) -> Tuple[ConstraintModel, RuleBreakModel]:
    """
    Execute Phase 2: Technical Constraints.
    Proposes and finalizes technical parameters and rule-breaking.
    """
    console.print(Panel.fit(
        "[bold yellow]PHASE 2: TECHNICAL & GROOVY CONSTRAINTS[/bold yellow]\n"
        "[dim]Translating emotion into sonic parameters.[/dim]",
        border_style="yellow"
    ))
    
    # Generate proposals
    constraint = _propose_technical_constraints(intent)
    rule_break = _propose_rule_break(intent)
    
    # Interactive review
    final_constraint, final_rule_break = _review_and_finalize_constraints(
        constraint, rule_break
    )
    
    console.print("\n[green]✓ Constraints finalized[/green]")
    
    return final_constraint, final_rule_break
