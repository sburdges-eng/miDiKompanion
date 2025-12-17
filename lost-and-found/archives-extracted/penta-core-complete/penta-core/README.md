# Penta Core

**Music Theory Rule-Breaking Framework** - Comprehensive voice leading, harmony, counterpoint, and rhythm rules with context-dependent severity and intentional rule-breaking for emotional effect.

Integrates **DAiW** (Digital Audio intelligent Workstation) and **iDAW** (intelligent Digital Audio Workspace) philosophies.

## Philosophy

> "Interrogate Before Generate" - Understand the emotion first, then translate to music

> "Every Rule-Break Needs Justification" - Breaking rules requires emotional reasoning

> "The wrong note played with conviction is the right note" - Beethoven (paraphrased)

> "The grid is just a suggestion. The pocket is where life happens." - DAiW

## Features

- **Context-Dependent Rules**: Same rule can be STRICT in classical, FLEXIBLE in jazz, ENCOURAGED in rock
- **Severity Levels**: STRICT, MODERATE, FLEXIBLE, STYLISTIC, ENCOURAGED
- **Emotional Rule-Breaking**: Maps emotions to suggested rule-breaks
- **Species Counterpoint**: Full Fux-based counterpoint rules
- **Groove Templates**: Genre pocket maps from DAiW (funk, boom-bap, dilla, etc.)
- **Masterpiece Examples**: Learn from Beethoven, Debussy, Stravinsky, Coltrane, Radiohead, and more

## Quick Start

```python
from penta_core.teachers import RuleBreakingTeacher
from penta_core.rules import VoiceLeadingRules, RuleSeverity

# Initialize teacher
teacher = RuleBreakingTeacher()

# Get rule-breaking examples for an emotion
grief_examples = teacher.get_examples_for_emotion("grief")

# Filter rules by context
jazz_rules = VoiceLeadingRules.get_rules_by_context("jazz")

# Filter rules by severity
strict_rules = VoiceLeadingRules.get_rules_by_severity(RuleSeverity.STRICT)

# Get demonstration of a rule being broken
demo = teacher.demonstrate_rule_break("parallel_fifths")
```

## Installation

```bash
git clone https://github.com/seanburdgeseng/penta-core.git
cd penta-core
pip install -e .
```

## Rule Categories

### Voice Leading
- Parallel fifths/octaves
- Hidden fifths/octaves
- Leading tone resolution
- Voice crossing
- Spacing limits

### Harmony
- Functional progressions
- Modal interchange (borrowed chords)
- Dissonance resolution
- Polytonality
- Tritone substitution

### Counterpoint (Species)
- First Species (1:1)
- Second Species (2:1)
- Third Species (4:1)
- Fourth Species (suspensions)
- Fifth Species (florid)

### Rhythm
- Meter and metric ambiguity
- Groove and pocket timing
- Syncopation
- Tempo fluctuation/rubato

## Emotional Rule-Breaking Map

| Emotion | Suggested Rule-Breaks |
|---------|----------------------|
| Grief | Non-resolution, Modal interchange, Tempo fluctuation |
| Bittersweet | Modal interchange |
| Power | Parallel fifths |
| Anxiety | Metric ambiguity, Unresolved dissonance |
| Chaos | Polytonality, Metric ambiguity |
| Nostalgia | Modal interchange |
| Longing | Non-resolution |
| Vulnerability | Tempo fluctuation |

## Famous Examples

### Parallel Fifths
- **Beethoven** - Symphony No. 6 "Pastoral" (folk quality)
- **Debussy** - La Cathédrale engloutie (medieval atmosphere)
- **Power Chords** - All rock/metal (massive, unified sound)

### Modal Interchange
- **Radiohead** - "Creep" (G-B-C-Cm = bittersweet)
- **Beatles** - "Norwegian Wood" (I-♭VII-I = folk nostalgia)
- **Kelly Song** - F-C-Dm-B♭m (hope through grief)

### Unresolved Dissonance
- **Monk** - Semitone clusters (meaningful wrongness)
- **Black Sabbath** - Tritone riff (metal identity)

## Integration with DAiW

Penta Core provides the theoretical foundation for DAiW's emotional rule-breaking system:

```python
# DAiW Integration Example
from penta_core.rules import HarmonyRules

# Get rule-break suggestions for Kelly song emotion
suggestions = HarmonyRules.get_rule_break_for_emotion("hope_through_grief")
# Returns: Modal interchange (♭VI chord makes hope feel earned)
```

## Groove Pockets (from DAiW)

```python
from penta_core.rules import RhythmRules

# Get genre pocket template
funk = RhythmRules.get_genre_pocket("funk")
# {'swing': 0.58, 'kick_offset_ms': 15, 'snare_offset_ms': -8, ...}

dilla = RhythmRules.get_genre_pocket("dilla")
# {'swing': 0.62, 'kick_offset_ms': 20, 'snare_offset_ms': -12, ...}
```

## Project Structure

```
penta-core/
├── penta_core/
│   ├── __init__.py
│   ├── rules/
│   │   ├── severity.py      # RuleSeverity enum
│   │   ├── species.py       # Species counterpoint enum
│   │   ├── context.py       # MusicalContext enum
│   │   ├── base.py          # Rule, RuleViolation classes
│   │   ├── voice_leading.py # Voice leading rules
│   │   ├── harmony_rules.py # Harmony rules
│   │   ├── counterpoint_rules.py # Species counterpoint
│   │   └── rhythm_rules.py  # Rhythm and groove rules
│   └── teachers/
│       ├── rule_breaking_teacher.py  # Main teacher
│       └── counterpoint_teacher.py   # Counterpoint lessons
├── tests/
├── demo.py
├── pyproject.toml
└── README.md
```

## License

MIT

---

*"Well, who has forbidden them?" — Beethoven, when questioned about parallel fifths*
