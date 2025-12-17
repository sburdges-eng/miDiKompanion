#!/usr/bin/env python3
"""
DAiW CLI - Digital Audio intelligent Workstation
================================================

Main command-line interface for the DAiW music generation system.

Commands:
    daiw new              - Start new song session (Phase 0-1 interrogation)
    daiw constraints      - Set technical constraints (Phase 2)
    daiw directive        - Set output directive (Phase 3)
    daiw execute          - Generate output
    daiw diagnose         - Analyze a chord progression
    daiw apply            - Apply groove template to MIDI
    daiw extract          - Extract groove from MIDI
    daiw teach            - Interactive teaching mode
    daiw intent suggest   - Get rule-break suggestions for emotion
    daiw status           - Show current session

Philosophy: "Interrogate Before Generate"
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json
import uuid

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Install rich and typer: pip install typer[all] rich")

# Initialize
app = typer.Typer(
    name="daiw",
    help="DAiW - Digital Audio intelligent Workstation",
    add_completion=False
)
console = Console() if HAS_RICH else None

# Session storage
CONFIG_DIR = Path.home() / ".daiw"
CURRENT_SESSION: Dict[str, Any] = {}


def ensure_config_dir():
    """Ensure config directory exists."""
    CONFIG_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

@app.command()
def new(
    title: str = typer.Option("", "--title", "-t", help="Song title"),
    skip_intro: bool = typer.Option(False, "--skip-intro", help="Skip intro prompt"),
):
    """
    🎭 Start a new song session.
    
    Begins the emotional interrogation process (Phase 0 & 1).
    """
    console.print(Panel.fit(
        "[bold cyan]DAiW - Digital Audio intelligent Workstation[/bold cyan]\n"
        "[dim]Interrogate Before Generate[/dim]",
        border_style="cyan"
    ))
    
    session_id = str(uuid.uuid4())[:8]
    
    if not skip_intro:
        proceed = _intro_consent()
        if not proceed:
            console.print("[yellow]Session cancelled. Use 'daiw new' when ready.[/yellow]")
            raise typer.Exit()
    
    # Run interrogation
    from music_brain.structure.interrogation_engine import run_emotional_interrogation
    
    wound, intent = run_emotional_interrogation()
    
    # Store in session
    CURRENT_SESSION["session_id"] = session_id
    CURRENT_SESSION["title"] = title or f"session_{session_id}"
    CURRENT_SESSION["wound"] = wound
    CURRENT_SESSION["intent"] = intent
    CURRENT_SESSION["phase"] = "intent_complete"
    CURRENT_SESSION["created_at"] = datetime.now().isoformat()
    
    # Save session
    _save_session()
    
    console.print("\n[green]✓ Phase 0 & 1 complete. Run 'daiw constraints' to continue.[/green]")


@app.command()
def constraints(
    genre: Optional[str] = typer.Option(None, "--genre", "-g", help="Override genre"),
    tempo: Optional[str] = typer.Option(None, "--tempo", "-t", help="Tempo range"),
    key: Optional[str] = typer.Option(None, "--key", "-k", help="Key center"),
):
    """
    ⚙️ Set technical constraints (Phase 2).
    
    Defines genre, tempo, groove, key, and rule-to-break.
    """
    _load_session()
    
    if "intent" not in CURRENT_SESSION:
        console.print("[red]No active session. Run 'daiw new' first.[/red]")
        raise typer.Exit(1)
    
    console.print(Panel.fit(
        "[bold yellow]Phase 2: Technical Constraints[/bold yellow]",
        border_style="yellow"
    ))
    
    from music_brain.structure.constraint_engine import propose_constraints
    
    intent = CURRENT_SESSION.get("intent")
    constraint_model, rule_break = propose_constraints(intent)
    
    CURRENT_SESSION["constraints"] = constraint_model
    CURRENT_SESSION["rule_break"] = rule_break
    CURRENT_SESSION["phase"] = "constraints_complete"
    
    _save_session()
    
    console.print("\n[green]✓ Phase 2 complete. Run 'daiw execute' to generate.[/green]")


@app.command()
def execute(
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    format: str = typer.Option("midi", "--format", "-f", help="Output format"),
):
    """
    🚀 Execute generation and produce output.
    """
    _load_session()
    
    if "constraints" not in CURRENT_SESSION:
        console.print("[red]No constraints set. Run 'daiw constraints' first.[/red]")
        raise typer.Exit(1)
    
    console.print(Panel.fit(
        "[bold green]Generating...[/bold green]",
        border_style="green"
    ))
    
    from music_brain.modules.chord import generate_progression, export_to_midi
    
    # Build payload
    payload = _build_payload()
    
    # Generate
    chords = generate_progression(payload)
    
    # Export
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    title = CURRENT_SESSION.get("title", "output")
    midi_path = output_path / f"{title}.mid"
    
    tempo = payload.get("technical_tempo_range", (100, 120))
    avg_tempo = (tempo[0] + tempo[1]) // 2
    
    export_to_midi(chords, str(midi_path), tempo=avg_tempo)
    
    console.print(f"\n[green]✓ Generated: {midi_path}[/green]")
    console.print(f"[dim]Chords: {len(chords)} | Tempo: {avg_tempo} BPM[/dim]")


@app.command()
def diagnose(
    progression: str = typer.Argument(..., help="Chord progression (e.g., 'F-C-Am-Dm')"),
    key: Optional[str] = typer.Option(None, "--key", "-k", help="Known key"),
):
    """
    🔍 Analyze a chord progression.
    
    Example: daiw diagnose "F-C-Am-Dm"
    """
    from music_brain.structure.progression import analyze_progression
    
    result = analyze_progression(progression)
    
    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        raise typer.Exit(1)
    
    table = Table(title="Chord Analysis")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Input", progression)
    table.add_row("Estimated Key", result.get("estimated_key", "Unknown"))
    table.add_row("Roman Numerals", result.get("progression_string", ""))
    table.add_row("Chord Count", str(result.get("num_chords", 0)))
    
    console.print(table)


@app.command()
def apply(
    input_file: str = typer.Argument(..., help="Input MIDI file"),
    output_file: str = typer.Argument(..., help="Output MIDI file"),
    genre: str = typer.Option("funk", "--genre", "-g", help="Groove template"),
    intensity: float = typer.Option(1.0, "--intensity", "-i", help="Application intensity"),
):
    """
    🎛️ Apply groove template to MIDI.
    
    Example: daiw apply drums.mid drums_grooved.mid --genre boom_bap
    """
    from music_brain.groove.engine import GrooveApplicator
    
    applicator = GrooveApplicator()
    template = applicator.get_preset(genre)
    
    if not template:
        console.print(f"[red]Unknown genre: {genre}[/red]")
        console.print("[dim]Available: funk, boom_bap, dilla, lofi, straight, driving[/dim]")
        raise typer.Exit(1)
    
    result = applicator.apply_groove(input_file, output_file, template, intensity)
    console.print(f"[green]✓ Applied {genre} groove to {output_file}[/green]")


@app.command()
def extract(
    midi_file: str = typer.Argument(..., help="MIDI file to analyze"),
):
    """
    📊 Extract groove characteristics from MIDI.
    """
    from music_brain.groove.engine import extract_groove
    
    result = extract_groove(midi_file)
    
    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        raise typer.Exit(1)
    
    table = Table(title="Groove Analysis")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    for key, value in result.items():
        table.add_row(key, str(value))
    
    console.print(table)


# ═══════════════════════════════════════════════════════════════════════════════
# INTENT SUBCOMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

intent_app = typer.Typer(help="Intent and rule-breaking commands")
app.add_typer(intent_app, name="intent")


@intent_app.command("suggest")
def intent_suggest(
    emotion: str = typer.Argument(..., help="Emotion to get suggestions for"),
):
    """
    💡 Get rule-break suggestions for an emotion.
    
    Example: daiw intent suggest grief
    """
    from music_brain.session.vernacular import VernacularTranslator
    
    translator = VernacularTranslator()
    suggestions = translator.get_rule_breaks_for_emotion(emotion)
    
    if not suggestions:
        console.print(f"[yellow]No specific suggestions for '{emotion}'[/yellow]")
        console.print("[dim]Try: grief, anger, anxiety, nostalgia, defiance, longing[/dim]")
        return
    
    console.print(f"\n[cyan]Suggested rule-breaks for '{emotion}':[/cyan]")
    for rule in suggestions:
        console.print(f"  • {rule}")


@intent_app.command("list")
def intent_list():
    """
    📋 List available emotions and rule-breaks.
    """
    from music_brain.session.vernacular import EMOTION_TO_RULE_BREAK
    
    table = Table(title="Emotion → Rule-Break Mapping")
    table.add_column("Emotion", style="cyan")
    table.add_column("Suggested Rule-Breaks", style="white")
    
    for emotion, rules in EMOTION_TO_RULE_BREAK.items():
        rule_names = ", ".join(r.value for r in rules)
        table.add_row(emotion, rule_names)
    
    console.print(table)


# ═══════════════════════════════════════════════════════════════════════════════
# TEACH COMMAND
# ═══════════════════════════════════════════════════════════════════════════════

@app.command()
def teach(
    topic: str = typer.Argument("rulebreaking", help="Topic to teach"),
):
    """
    📚 Interactive teaching mode.
    
    Topics: rulebreaking, harmony, rhythm, production
    """
    console.print(Panel.fit(
        f"[bold magenta]Teaching: {topic}[/bold magenta]",
        border_style="magenta"
    ))
    
    # Import the teaching module
    try:
        from music_brain.session.teaching import RuleBreakingTeacher
        teacher = RuleBreakingTeacher()
        teacher.start_lesson(topic)
    except ImportError:
        console.print("[yellow]Teaching module not fully installed.[/yellow]")
        console.print("[dim]Basic rule-breaking info:[/dim]")
        _show_basic_teaching(topic)


def _show_basic_teaching(topic: str):
    """Show basic teaching content."""
    content = {
        "rulebreaking": """
Rule-breaking in music is about intentional violation of conventions for emotional effect.

Key categories:
• HARMONY - Parallel fifths, modal interchange, unresolved dissonance
• RHYTHM - Metric ambiguity, constant displacement, tempo fluctuation  
• STRUCTURE - Non-resolution, asymmetric form, anti-drop
• PRODUCTION - Buried vocals, lo-fi artifacts, pitch imperfection

Famous example: Radiohead's "Creep" uses I-III-IV-iv (modal interchange)
The borrowed iv chord creates the signature bittersweet feel.
        """,
        "harmony": """
Harmonic rule-breaking:

1. PARALLEL FIFTHS - Moving voices in parallel (power chords!)
   - Beethoven did it intentionally in Symphony 6
   - Every power chord progression breaks this rule

2. MODAL INTERCHANGE - Borrowing from parallel mode
   - Example: Using iv instead of IV in a major key
   - Creates "happy-sad" ambiguity

3. UNRESOLVED DISSONANCE - Letting tension hang
   - Thelonious Monk made this his signature
        """,
    }
    
    console.print(content.get(topic, "Topic not found."))


# ═══════════════════════════════════════════════════════════════════════════════
# STATUS COMMAND
# ═══════════════════════════════════════════════════════════════════════════════

@app.command()
def status():
    """
    📊 Show current session status.
    """
    _load_session()
    
    if not CURRENT_SESSION:
        console.print("[yellow]No active session. Run 'daiw new' to start.[/yellow]")
        return
    
    table = Table(title="Current Session")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Session ID", CURRENT_SESSION.get("session_id", "N/A"))
    table.add_row("Title", CURRENT_SESSION.get("title", "Untitled"))
    table.add_row("Phase", CURRENT_SESSION.get("phase", "Not started"))
    table.add_row("Created", CURRENT_SESSION.get("created_at", "Unknown"))
    
    if "intent" in CURRENT_SESSION:
        intent = CURRENT_SESSION["intent"]
        table.add_row("Mood", getattr(intent, "mood_primary", "N/A"))
    
    if "constraints" in CURRENT_SESSION:
        constraints = CURRENT_SESSION["constraints"]
        table.add_row("Key", getattr(constraints, "technical_key", "N/A"))
        table.add_row("Genre", getattr(constraints, "technical_genre", "N/A"))
    
    console.print(table)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _intro_consent() -> bool:
    """Show intro and get consent."""
    console.print("\n[cyan]This process will ask you personal questions.[/cyan]")
    console.print("[dim]The goal is to find the emotional truth for your song.[/dim]")
    return Confirm.ask("\n[green]Ready to begin?[/green]", default=True)


def _save_session():
    """Save current session to disk."""
    ensure_config_dir()
    session_file = CONFIG_DIR / "current_session.json"
    
    # Serialize dataclasses
    serializable = {}
    for key, value in CURRENT_SESSION.items():
        if hasattr(value, "__dict__"):
            serializable[key] = {
                "_type": type(value).__name__,
                **{k: v.value if hasattr(v, "value") else v 
                   for k, v in value.__dict__.items()}
            }
        else:
            serializable[key] = value
    
    with open(session_file, "w") as f:
        json.dump(serializable, f, indent=2, default=str)


def _load_session():
    """Load session from disk."""
    global CURRENT_SESSION
    session_file = CONFIG_DIR / "current_session.json"
    
    if session_file.exists():
        with open(session_file) as f:
            data = json.load(f)
            # For now, just load the raw data
            # Full deserialization would reconstruct dataclasses
            CURRENT_SESSION = data


def _build_payload() -> Dict[str, Any]:
    """Build generation payload from session."""
    payload = {
        "session_id": CURRENT_SESSION.get("session_id", ""),
    }
    
    # Add wound fields
    if "wound" in CURRENT_SESSION:
        wound = CURRENT_SESSION["wound"]
        if isinstance(wound, dict):
            payload.update(wound)
        else:
            payload["core_event"] = getattr(wound, "core_event", "")
    
    # Add intent fields
    if "intent" in CURRENT_SESSION:
        intent = CURRENT_SESSION["intent"]
        if isinstance(intent, dict):
            payload["mood_primary"] = intent.get("mood_primary", "")
            payload["narrative_arc"] = intent.get("narrative_arc", "Climb-to-Climax")
            payload["mood_secondary_tension"] = intent.get("mood_secondary_tension", 0.5)
        else:
            payload["mood_primary"] = getattr(intent, "mood_primary", "")
            payload["narrative_arc"] = getattr(intent, "narrative_arc", "Climb-to-Climax")
    
    # Add constraint fields
    if "constraints" in CURRENT_SESSION:
        constraints = CURRENT_SESSION["constraints"]
        if isinstance(constraints, dict):
            payload["technical_key"] = constraints.get("technical_key", "C")
            payload["technical_mode"] = constraints.get("technical_mode", "major")
            payload["technical_tempo_range"] = constraints.get("technical_tempo_range", (100, 120))
            payload["technical_genre"] = constraints.get("technical_genre", "")
        else:
            payload["technical_key"] = getattr(constraints, "technical_key", "C")
            payload["technical_mode"] = getattr(constraints, "technical_mode", "major")
            payload["technical_tempo_range"] = getattr(constraints, "technical_tempo_range", (100, 120))
    
    # Add rule break
    if "rule_break" in CURRENT_SESSION:
        rule_break = CURRENT_SESSION["rule_break"]
        if isinstance(rule_break, dict):
            payload["technical_rule_to_break"] = rule_break.get("technical_rule_to_break", "NONE")
        else:
            payload["technical_rule_to_break"] = getattr(
                rule_break, "technical_rule_to_break", "NONE"
            )
    
    return payload


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    ensure_config_dir()
    app()


if __name__ == "__main__":
    main()
