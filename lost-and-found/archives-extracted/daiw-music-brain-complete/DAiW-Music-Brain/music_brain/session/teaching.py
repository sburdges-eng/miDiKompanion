"""
Rule-Breaking Teacher - Interactive music theory lessons.

Teaches intentional rule-breaking through examples and exercises,
connecting theory concepts to emotional outcomes.
"""

from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
import json
from pathlib import Path


console = Console()


class RuleBreakingTeacher:
    """
    Interactive teacher for intentional rule-breaking in music.
    """
    
    def __init__(self):
        self.current_topic = None
        self.progress = {}
        self.database = self._get_default_database()
    
    def start_lesson(self, topic: str = None):
        """Start an interactive lesson."""
        console.print(Panel.fit(
            "[bold cyan]RULE-BREAKING MASTERCLASS[/bold cyan]\n"
            "[dim]Learn to break rules with purpose[/dim]",
            border_style="cyan"
        ))
        
        if not topic:
            topic = self._select_topic()
        
        self.current_topic = topic
        self._teach_topic(topic)
    
    def _select_topic(self) -> str:
        """Let user select a topic."""
        topics = [
            ("harmony", "Harmonic Rule-Breaking", "Parallel fifths, modal interchange, polytonality"),
            ("rhythm", "Rhythmic Rule-Breaking", "Metric ambiguity, polyrhythm, displacement"),
            ("production", "Production Rule-Breaking", "Buried vocals, lo-fi artifacts, pitch imperfection"),
            ("arrangement", "Arrangement Rule-Breaking", "Sparse climax, anti-drop, structural asymmetry"),
        ]
        
        console.print("\n[cyan]Available Topics:[/cyan]")
        for i, (key, name, desc) in enumerate(topics, 1):
            console.print(f"  {i}. [bold]{name}[/bold]")
            console.print(f"     [dim]{desc}[/dim]")
        
        choice = Prompt.ask("\n[green]Select topic[/green]", choices=["1", "2", "3", "4"])
        return topics[int(choice) - 1][0]
    
    def _teach_topic(self, topic: str):
        """Teach a specific topic."""
        topic_data = self.database.get(topic, {})
        
        if not topic_data:
            console.print(f"[yellow]Topic '{topic}' not found[/yellow]")
            return
        
        # Display overview
        console.print(f"\n[bold magenta]{topic_data.get('title', topic.upper())}[/bold magenta]")
        console.print(f"[dim]{topic_data.get('overview', '')}[/dim]")
        
        # Show the rule being broken
        console.print(f"\n[red]THE RULE:[/red]")
        console.print(f"  {topic_data.get('rule', 'N/A')}")
        
        # Show examples
        examples = topic_data.get('examples', [])
        if examples:
            console.print(f"\n[cyan]FAMOUS EXAMPLES:[/cyan]")
            for ex in examples[:3]:
                console.print(f"\n  [bold]{ex.get('artist', 'Unknown')} - {ex.get('piece', 'Unknown')}[/bold]")
                console.print(f"  [dim]What they broke:[/dim] {ex.get('what_broken', '')}")
                console.print(f"  [dim]Why it works:[/dim] {ex.get('why_works', '')}")
        
        # Interactive quiz
        if Confirm.ask("\n[cyan]Try a quick exercise?[/cyan]", default=True):
            self._run_exercise(topic)
    
    def _run_exercise(self, topic: str):
        """Run an interactive exercise."""
        exercises = {
            "harmony": self._harmony_exercise,
            "rhythm": self._rhythm_exercise,
            "production": self._production_exercise,
            "arrangement": self._arrangement_exercise,
        }
        
        exercise_func = exercises.get(topic, self._generic_exercise)
        exercise_func()
    
    def _harmony_exercise(self):
        """Harmony-specific exercise."""
        console.print("\n[bold cyan]EXERCISE: Identify the Rule-Break[/bold cyan]")
        console.print("\nListen to this progression: [bold]G - B - C - Cm[/bold]")
        console.print("[dim](Radiohead's 'Creep' progression)[/dim]")
        
        console.print("\n[cyan]Which rule is being broken?[/cyan]")
        console.print("  1. Parallel fifths")
        console.print("  2. Modal interchange (borrowed chords)")
        console.print("  3. Unresolved dissonance")
        
        answer = Prompt.ask("[green]Your answer[/green]", choices=["1", "2", "3"])
        
        if answer == "2":
            console.print("\n[green]✓ Correct![/green]")
            console.print("[dim]The B major (III) is borrowed from G Lydian, and Cm (iv) is borrowed from G minor.[/dim]")
            console.print("[dim]This creates the signature 'happy-to-sad' emotional shift.[/dim]")
        else:
            console.print("\n[yellow]Not quite.[/yellow]")
            console.print("[dim]The answer is modal interchange - mixing parallel major and minor modes.[/dim]")
    
    def _rhythm_exercise(self):
        """Rhythm-specific exercise."""
        console.print("\n[bold cyan]EXERCISE: Metric Ambiguity[/bold cyan]")
        console.print("\nRadiohead's 'Pyramid Song' has been notated as:")
        console.print("  - 12/8")
        console.print("  - 6/8")
        console.print("  - 4/4")
        console.print("  - 3/4 + 5/4 alternating")
        
        console.print("\n[cyan]What makes this ambiguity effective emotionally?[/cyan]")
        response = Prompt.ask("[green]Your thoughts[/green]")
        
        console.print("\n[cyan]Key insight:[/cyan]")
        console.print("[dim]The piano chords have no discernible pulse until drums enter.[/dim]")
        console.print("[dim]This ambiguity creates an unsettled, floating quality that matches the song's themes.[/dim]")
    
    def _production_exercise(self):
        """Production-specific exercise."""
        console.print("\n[bold cyan]EXERCISE: Lo-Fi as Authenticity[/bold cyan]")
        console.print("\nElliott Smith's recordings often feature:")
        console.print("  - Audible room noise")
        console.print("  - Imperfect pitch")
        console.print("  - Buried vocals")
        
        console.print("\n[cyan]Why might these 'flaws' serve the emotional content?[/cyan]")
        response = Prompt.ask("[green]Your thoughts[/green]")
        
        console.print("\n[cyan]Key insight:[/cyan]")
        console.print("[dim]Lo-fi artifacts create intimacy - like overhearing a private moment.[/dim]")
        console.print("[dim]Technical perfection can create emotional distance; imperfection signals vulnerability.[/dim]")
    
    def _arrangement_exercise(self):
        """Arrangement-specific exercise."""
        console.print("\n[bold cyan]EXERCISE: The Anti-Drop[/bold cyan]")
        console.print("\nSome songs build tension then... go quiet instead of dropping.")
        
        console.print("\n[cyan]Name a song that does this, or imagine one:[/cyan]")
        response = Prompt.ask("[green]Your example[/green]")
        
        console.print("\n[cyan]Key insight:[/cyan]")
        console.print("[dim]Subverting the expected drop can be more powerful than the drop itself.[/dim]")
        console.print("[dim]The absence of resolution can feel like a punch to the gut.[/dim]")
    
    def _generic_exercise(self):
        """Generic exercise fallback."""
        console.print("\n[yellow]Exercise not available for this topic yet.[/yellow]")
    
    def _get_default_database(self) -> Dict[str, Any]:
        """Return default database."""
        return {
            "harmony": {
                "title": "HARMONIC RULE-BREAKING",
                "overview": "Breaking voice-leading and harmonic function rules for emotional effect.",
                "rule": "Parallel fifths destroy voice independence; dissonances must resolve.",
                "examples": [
                    {
                        "artist": "Beethoven",
                        "piece": "Symphony No. 6 'Pastoral'",
                        "what_broken": "Parallel fifths in the Storm movement",
                        "why_works": "Creates rustic, folk-like quality",
                    },
                    {
                        "artist": "Radiohead",
                        "piece": "Creep",
                        "what_broken": "Modal interchange (I-III-IV-iv)",
                        "why_works": "Happy-to-sad emotional ambiguity",
                    },
                    {
                        "artist": "Thelonious Monk",
                        "piece": "'Round Midnight",
                        "what_broken": "Unresolved dissonances, semitone clusters",
                        "why_works": "Wrong notes that sound meaningfully wrong",
                    },
                ],
            },
            "rhythm": {
                "title": "RHYTHMIC RULE-BREAKING",
                "overview": "Breaking metric and rhythmic conventions for disorientation or groove.",
                "rule": "Music should maintain consistent meter with predictable strong beats.",
                "examples": [
                    {
                        "artist": "Stravinsky",
                        "piece": "The Rite of Spring - Sacrificial Dance",
                        "what_broken": "Constant meter changes (2/16, 3/16, 5/16...)",
                        "why_works": "Impossible to predict accent pattern; evokes primal chaos",
                    },
                    {
                        "artist": "Radiohead",
                        "piece": "Pyramid Song",
                        "what_broken": "Ambiguous meter that resists classification",
                        "why_works": "Creates floating, unsettled quality",
                    },
                ],
            },
            "production": {
                "title": "PRODUCTION RULE-BREAKING",
                "overview": "Breaking technical standards for emotional authenticity.",
                "rule": "Vocals should be clear and prominent; recordings should be clean.",
                "examples": [
                    {
                        "artist": "Bon Iver",
                        "piece": "For Emma, Forever Ago",
                        "what_broken": "Lo-fi recording, buried vocals, pitch imperfection",
                        "why_works": "Creates intimacy and vulnerability",
                    },
                    {
                        "artist": "Elliott Smith",
                        "piece": "Various",
                        "what_broken": "Room noise, imperfect takes",
                        "why_works": "Feels like overhearing a private moment",
                    },
                ],
            },
            "arrangement": {
                "title": "ARRANGEMENT RULE-BREAKING",
                "overview": "Breaking structural expectations for emotional impact.",
                "rule": "Climaxes should be the biggest, loudest moment; songs should resolve.",
                "examples": [
                    {
                        "artist": "Billie Eilish",
                        "piece": "bury a friend",
                        "what_broken": "Anti-drop - tension builds then goes quiet",
                        "why_works": "Subverts expectations; absence is more powerful than presence",
                    },
                    {
                        "artist": "Radiohead",
                        "piece": "Various",
                        "what_broken": "Songs that never establish clear tonic",
                        "why_works": "Creates floating, unresolved emotional state",
                    },
                ],
            },
        }
