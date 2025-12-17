"""
DAiW - Interrogation Engine

Executes Phase 0 (Core Wound) and Phase 1 (Intent) of the user journey.
Collects deep psychological and emotional data via conversational prompts.

This is where the "therapist" lives.

References:
    - Batshit Mode (Phase F): Avoidance/contradiction detection for re-interrogation
    - User Journey Step 1: Force honesty and define root cause
"""

from typing import Tuple, Dict, Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from .models import (
    CoreWoundModel, IntentModel, VulnerabilityScale,
    NarrativeArc, HarmonicComplexity
)

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0: THE CORE WOUND
# ═══════════════════════════════════════════════════════════════════════════════

THERAPIST_PROMPTS = {
    "core_event": {
        "question": "What is the actual story you are trying to process or celebrate?",
        "subtext": "Focus on the specific moment you want the song to freeze in time.",
        "example": "e.g., 'Finding my friend after she overdosed', 'The moment I quit my job'",
    },
    "core_resistance": {
        "question": "What part of you does NOT want this song written?",
        "subtext": "This helps us find the most vulnerable truth to put into the music.",
        "example": "e.g., 'Afraid of being judged', 'Don't want to relive it'",
    },
    "core_longing": {
        "question": "What do you wish you could feel right now?",
        "subtext": "The emotional destination, even if it feels unreachable.",
        "example": "e.g., 'Peace', 'Permission to be angry', 'Freedom from guilt'",
    },
    "core_stakes": {
        "question": "What happens if this song never gets written?",
        "subtext": "Why does this matter enough to create?",
        "example": "e.g., 'The grief stays stuck', 'I never confront this'",
    },
    "core_transformation": {
        "question": "How should you feel when this song is finished?",
        "subtext": "Not 'happy' - what specific emotional shift?",
        "example": "e.g., 'Lighter', 'Like I finally said the unsaid'",
    },
}


def _ask_therapist_question(key: str) -> str:
    """Ask a single therapist question with formatting."""
    prompt_data = THERAPIST_PROMPTS.get(key, {})
    
    console.print(f"\n[bold cyan]{prompt_data.get('question', 'Tell me more...')}[/bold cyan]")
    console.print(f"[dim]{prompt_data.get('subtext', '')}[/dim]")
    console.print(f"[dim italic]{prompt_data.get('example', '')}[/dim italic]")
    
    return Prompt.ask("[green]>[/green]")


def run_phase_0() -> CoreWoundModel:
    """
    Execute Phase 0: Core Wound interrogation.
    Returns a populated CoreWoundModel.
    """
    console.print(Panel.fit(
        "[bold red]PHASE 0: THE CORE WOUND[/bold red]\n"
        "[dim]Let's find what needs to be said.[/dim]",
        border_style="red"
    ))
    
    core_event = _ask_therapist_question("core_event")
    core_resistance = _ask_therapist_question("core_resistance")
    core_longing = _ask_therapist_question("core_longing")
    core_stakes = _ask_therapist_question("core_stakes")
    core_transformation = _ask_therapist_question("core_transformation")
    
    # Optional: Name the entity (narrative therapy)
    console.print("\n[cyan]Give your emotion/experience a name - a character you can externalize.[/cyan]")
    console.print("[dim]e.g., 'The Grey Weight', 'The Burning', 'The Silence'[/dim]")
    entity_name = Prompt.ask("[green]Name (or skip)[/green]", default="")
    
    return CoreWoundModel(
        core_event=core_event,
        core_resistance=core_resistance,
        core_longing=core_longing,
        core_stakes=core_stakes,
        core_transformation=core_transformation,
        narrative_entity_name=entity_name
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: EMOTIONAL INTENT
# ═══════════════════════════════════════════════════════════════════════════════

MOOD_SUGGESTIONS = {
    "grief": ["Grief and Longing", "Quiet Devastation", "Heavy Emptiness"],
    "anger": ["Defiance", "Burning Rage", "Righteous Fury"],
    "fear": ["Creeping Dread", "Panic", "Frozen Terror"],
    "joy": ["Liberation", "Triumph", "Unexpected Lightness"],
    "love": ["Aching Tenderness", "Fierce Protection", "Bittersweet Memory"],
    "confusion": ["Lost", "Spinning", "Fractured"],
}


def run_phase_1(wound: CoreWoundModel) -> IntentModel:
    """
    Execute Phase 1: Emotional Intent.
    Uses the CoreWoundModel to propose and refine intent.
    """
    console.print(Panel.fit(
        "[bold yellow]PHASE 1: EMOTIONAL INTENT[/bold yellow]\n"
        "[dim]Translating your wound into musical direction.[/dim]",
        border_style="yellow"
    ))
    
    # Analyze wound for mood suggestions
    wound_text = f"{wound.core_event} {wound.core_resistance} {wound.core_longing}"
    suggested_moods = _analyze_for_mood(wound_text)
    
    # Display suggestions
    console.print("\n[cyan]Based on what you shared, consider these emotional directions:[/cyan]")
    for i, mood in enumerate(suggested_moods, 1):
        console.print(f"  {i}. {mood}")
    console.print(f"  {len(suggested_moods) + 1}. [dim]Something else...[/dim]")
    
    choice = Prompt.ask("[green]Select or type your own[/green]")
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(suggested_moods):
            mood_primary = suggested_moods[idx]
        else:
            mood_primary = Prompt.ask("[green]Describe the primary emotion[/green]")
    except ValueError:
        mood_primary = choice
    
    # Tension level
    console.print("\n[cyan]How much internal conflict/tension?[/cyan]")
    console.print("[dim]0 = Resolved, peaceful | 10 = Maximum unresolved tension[/dim]")
    tension = Prompt.ask("[green]Tension (0-10)[/green]", default="5")
    try:
        tension_float = float(tension) / 10.0
    except ValueError:
        tension_float = 0.5
    
    # Narrative arc
    console.print("\n[cyan]What shape should the emotional journey take?[/cyan]")
    arcs = list(NarrativeArc)
    for i, arc in enumerate(arcs, 1):
        console.print(f"  {i}. {arc.value}")
    
    arc_choice = Prompt.ask("[green]Select arc[/green]", default="1")
    try:
        narrative_arc = arcs[int(arc_choice) - 1]
    except (ValueError, IndexError):
        narrative_arc = NarrativeArc.CLIMB_TO_CLIMAX
    
    # Vulnerability
    console.print("\n[cyan]How raw/exposed should the content feel?[/cyan]")
    console.print("  1. Low (Protected, metaphorical)")
    console.print("  2. Medium (Honest but guarded)")
    console.print("  3. High (Brutally direct)")
    
    vuln_choice = Prompt.ask("[green]Vulnerability[/green]", default="2")
    vuln_map = {"1": VulnerabilityScale.LOW, "2": VulnerabilityScale.MEDIUM, "3": VulnerabilityScale.HIGH}
    vulnerability = vuln_map.get(vuln_choice, VulnerabilityScale.MEDIUM)
    
    # Imagery
    console.print("\n[cyan]Close your eyes. What do you see/feel/smell?[/cyan]")
    console.print("[dim]Quick sensory impressions that capture the mood.[/dim]")
    imagery = Prompt.ask("[green]Imagery[/green]", default="")
    
    return IntentModel(
        mood_primary=mood_primary,
        mood_secondary_tension=tension_float,
        narrative_arc=narrative_arc,
        harmonic_complexity=HarmonicComplexity.MODERATE,
        vulnerability_scale=vulnerability,
        imagery_texture=imagery
    )


def _analyze_for_mood(text: str) -> List[str]:
    """Analyze text and return mood suggestions."""
    text_lower = text.lower()
    
    suggestions = []
    
    # Check for keywords
    if any(w in text_lower for w in ["death", "died", "lost", "gone", "miss"]):
        suggestions.extend(MOOD_SUGGESTIONS["grief"])
    if any(w in text_lower for w in ["angry", "hate", "furious", "rage"]):
        suggestions.extend(MOOD_SUGGESTIONS["anger"])
    if any(w in text_lower for w in ["scared", "afraid", "terror", "panic"]):
        suggestions.extend(MOOD_SUGGESTIONS["fear"])
    if any(w in text_lower for w in ["love", "heart", "tender", "care"]):
        suggestions.extend(MOOD_SUGGESTIONS["love"])
    if any(w in text_lower for w in ["confused", "lost", "why", "spinning"]):
        suggestions.extend(MOOD_SUGGESTIONS["confusion"])
    
    # Default suggestions if nothing matched
    if not suggestions:
        suggestions = ["Melancholy", "Nostalgia", "Quiet Intensity"]
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    
    return unique[:5]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_emotional_interrogation() -> Tuple[CoreWoundModel, IntentModel]:
    """
    Run the complete emotional interrogation (Phase 0 + Phase 1).
    Returns both models for use in Phase 2.
    """
    wound = run_phase_0()
    intent = run_phase_1(wound)
    
    # Summary
    console.print(Panel.fit(
        f"[bold green]INTERROGATION COMPLETE[/bold green]\n\n"
        f"[cyan]Wound:[/cyan] {wound.core_event[:50]}...\n"
        f"[cyan]Primary Mood:[/cyan] {intent.mood_primary}\n"
        f"[cyan]Arc:[/cyan] {intent.narrative_arc.value}\n"
        f"[cyan]Vulnerability:[/cyan] {intent.vulnerability_scale.value}",
        border_style="green"
    ))
    
    return wound, intent


if __name__ == "__main__":
    wound, intent = run_emotional_interrogation()
    print(f"\nWound: {wound}")
    print(f"\nIntent: {intent}")
