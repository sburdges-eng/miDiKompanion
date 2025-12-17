# DAiW Music Brain

**Digital Audio intelligent Workstation** - Emotion-first music generation system.

> "Interrogate Before Generate" - The tool shouldn't finish art for people. It should make them braver.

## Philosophy

DAiW is built on three core principles:

1. **"Interrogate Before Generate"** - Understand the emotional truth before creating music
2. **"Imperfection is Intentional"** - Lo-fi aesthetic treats flaws as authenticity
3. **"Every Rule-Break Needs Justification"** - Breaking music theory rules requires emotional reasoning

## Installation

```bash
# Clone the repository
git clone https://github.com/seanburdgeseng/DAiW-Music-Brain.git
cd DAiW-Music-Brain

# Install in development mode
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

## Quick Start

### CLI Usage

```bash
# Start a new song session (emotional interrogation)
daiw new

# Set technical constraints
daiw constraints

# Generate output
daiw execute

# Analyze a chord progression
daiw diagnose "F-C-Am-Dm"

# Get rule-break suggestions for an emotion
daiw intent suggest grief

# Apply groove template to MIDI
daiw apply drums.mid drums_grooved.mid --genre boom_bap

# Interactive teaching mode
daiw teach rulebreaking
```

### Python API

```python
from music_brain import TherapySession, render_plan_to_midi

# Create a therapy session
session = TherapySession()

# Process emotional input
mood = session.process_core_input("I feel broken and lost")
print(f"Detected: {mood}")  # "grief"

# Set motivation and chaos scales
session.set_scales(motivation=7, chaos=0.4)

# Generate a plan
plan = session.generate_plan()
print(f"Plan: {plan.length_bars} bars @ {plan.tempo_bpm} BPM in {plan.mode}")

# Render to MIDI
render_plan_to_midi(plan, "output.mid", vulnerability=0.6)
```

## The Three-Phase Intent Schema

### Phase 0: Core Wound/Desire
- `core_event` - What happened?
- `core_resistance` - What holds you back from saying it?
- `core_longing` - What do you want to feel?
- `core_stakes` - What's at risk?
- `core_transformation` - How should you feel when done?

### Phase 1: Emotional Intent
- `mood_primary` - Dominant emotion
- `mood_secondary_tension` - Internal conflict (0.0-1.0)
- `vulnerability_scale` - Low/Medium/High
- `narrative_arc` - Climb-to-Climax, Slow Reveal, Repetitive Despair, etc.

### Phase 2: Technical Constraints
- `technical_genre`, `technical_key`, `technical_mode`
- `technical_rule_to_break` - Intentional rule violation
- `rule_breaking_justification` - WHY break this rule

## Rule-Breaking Categories

| Category | Examples | Use When |
|----------|----------|----------|
| HARMONY_ModalInterchange | Borrowing chords from parallel mode | Bittersweet, nostalgia |
| HARMONY_ParallelMotion | Power chords, parallel fifths | Power, defiance |
| HARMONY_UnresolvedDissonance | Tension that doesn't resolve | Anxiety, unease |
| STRUCTURE_NonResolution | No return to tonic | Grief, longing |
| PRODUCTION_BuriedVocals | Half-heard lyrics | Intimacy, confession |
| RHYTHM_ConstantDisplacement | Off-kilter timing | Anxiety, chaos |

## Project Structure

```
DAiW-Music-Brain/
├── music_brain/
│   ├── __init__.py
│   ├── cli.py                    # Main CLI interface
│   ├── structure/
│   │   ├── models.py             # Frozen dataclasses (Schema of Truth)
│   │   ├── comprehensive_engine.py # THE BRAIN
│   │   ├── interrogation_engine.py # Phase 0-1 therapist
│   │   ├── constraint_engine.py  # Phase 2 constraints
│   │   ├── tension.py            # Tension curve generation
│   │   └── progression.py        # Chord parser
│   ├── modules/
│   │   └── chord.py              # Chord generation
│   ├── groove/
│   │   └── engine.py             # Humanization, groove templates
│   └── session/
│       ├── vernacular.py         # Casual music description translation
│       └── teaching.py           # Interactive lessons
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Vernacular Translation

The system understands casual music descriptions:

| They say | We interpret |
|----------|--------------|
| "fat" | Full low-mids, light saturation |
| "laid back" | Behind-beat timing |
| "crispy" | Pleasant high-frequency presence |
| "boots and cats" | 4/4 kick-hat pattern |
| "boom bap" | Hip-hop groove |
| "lo-fi" | Degraded, vintage, imperfect |

## The Kelly Song

The canonical test case: A song about finding a friend after suicide.

- **Key:** F major
- **Progression:** F - C - Am - Dm (with Bbm borrowed chord)
- **Tempo:** 82 BPM
- **Style:** Lo-fi bedroom emo
- **Rule Break:** HARMONY_ModalInterchange (Bbm makes hope feel earned)

## Meta Principles

> "The audience doesn't hear 'borrowed from Dorian.' They hear 'that part made me cry.'"

> "The wrong note played with conviction is the right note."

> "The grid is just a suggestion. The pocket is where life happens."

## License

MIT

---

*"Well, who has forbidden them?" — Beethoven, when questioned about parallel fifths*
