# Kelly Phase 2 - Transitions

## Energy Curves

```python
CURVES = {
    "build": lambda t: t ** 1.5,
    "drop": lambda t: 1 - (1-t)**2,
    "breakdown": lambda t: 1 - t,
    "sustain": lambda t: 0.7,
    "swell": lambda t: 0.5 + 0.3*sin(t*pi),
    "impact": lambda t: 1.0 if t > 0.9 else t*0.5,
    "spike": lambda t: exp(-10*(t-0.5)**2),
    "plateau": lambda t: min(1, t*3) if t < 0.7 else 1 - (t-0.7)*3
}

def energy_curve(curve_type: str, points: int = 8) -> list:
    f = CURVES.get(curve_type, lambda t: t)
    return [f(i/(points-1)) for i in range(points)]
```

## Tension Curves

```python
TENSION_SHAPES = {
    "linear_build": lambda t: t,
    "linear_release": lambda t: 1 - t,
    "spike": lambda t: 4*t*(1-t),
    "plateau": lambda t: min(1, t*2) if t < 0.8 else 1-(t-0.8)*5,
    "sawtooth": lambda t: (t*4) % 1,
    "wave": lambda t: 0.5 + 0.5*sin(2*pi*t),
    "climax": lambda t: t**2,
    "delayed_resolution": lambda t: min(1, t*1.5) if t < 0.9 else (1-t)*10
}

def tension_curve(shape: str, points: int, base: float = 0.2, peak: float = 0.8) -> list:
    f = TENSION_SHAPES.get(shape, lambda t: t)
    return [base + (peak-base)*f(i/(points-1)) for i in range(points)]
```

## Crossfade Parameters

```python
def crossfade_params(from_e: float, to_e: float, bars: int) -> dict:
    diff = abs(to_e - from_e)
    return {
        "duration": min(bars, max(1, int(diff * 4))),
        "curve": "exponential" if to_e > from_e else "logarithmic",
        "from": from_e,
        "to": to_e
    }
```

## Section Transition Map

| From → To | Energy Curve | Tension Shape | Duration |
|-----------|--------------|---------------|----------|
| Verse → Chorus | build | linear_build | 2-4 bars |
| Chorus → Verse | breakdown | linear_release | 1-2 bars |
| Verse → Bridge | swell | wave | 2 bars |
| Bridge → Chorus | build | climax | 4 bars |
| Intro → Verse | build | delayed_resolution | 4 bars |
| Chorus → Outro | breakdown | linear_release | 4 bars |

## Emotion Transition Planning

```python
def section_emotion_transition(from_section: str, to_section: str, emotion: EmotionVector) -> dict:
    transitions = {
        ("verse", "chorus"): {"arousal_delta": 0.2, "valence_delta": 0.1},
        ("chorus", "verse"): {"arousal_delta": -0.15, "valence_delta": 0},
        ("verse", "bridge"): {"arousal_delta": 0, "valence_delta": -0.2},
        ("bridge", "chorus"): {"arousal_delta": 0.3, "valence_delta": 0.2},
    }
    t = transitions.get((from_section, to_section), {"arousal_delta": 0, "valence_delta": 0})
    return EmotionVector(
        emotion.valence + t["valence_delta"],
        min(1, max(0, emotion.arousal + t["arousal_delta"])),
        emotion.dominance
    )
```
