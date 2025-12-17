# Kelly Phase 2 - Core Formulas

## 1. VAD Emotion Vector

```python
@dataclass
class EmotionVector:
    valence: float    # -1 to +1
    arousal: float    # 0 to 1
    dominance: float  # 0 to 1

def emotion_distance(e1, e2):
    return sqrt((e1.valence-e2.valence)**2 + (e1.arousal-e2.arousal)**2 + (e1.dominance-e2.dominance)**2)
```

## 2. Emotion → Music Mapping

| Parameter | Formula |
|-----------|---------|
| Tempo | `60 + 120 * arousal` |
| Velocity | `60 + 67 * dominance` |
| Mode | `"major" if valence > 0 else "minor"` |
| Dissonance | `0.2 + abs(valence)*0.3 + (1-dominance)*0.3` |
| Legato | `0.7 - arousal * 0.4` |
| Register | `60 + valence*12 + arousal*6` |

```python
def map_emotion_to_music(e: EmotionVector) -> dict:
    return {
        "tempo": int(60 + 120 * e.arousal),
        "velocity": int(60 + 67 * e.dominance),
        "mode": "major" if e.valence > 0.3 else "minor" if e.valence < -0.3 else "dorian",
        "dissonance": 0.2 + abs(e.valence)*0.3 + (1-e.dominance)*0.3,
        "legato": 0.7 - e.arousal * 0.4,
        "register": 60 + int(e.valence*12) + int(e.arousal*6)
    }
```

## 3. Emotion → Mode Map

```python
EMOTION_MODE = {
    "joy": "lydian", "euphoria": "lydian", "hope": "ionian",
    "grief": "aeolian", "sadness": "dorian", "despair": "phrygian",
    "anger": "phrygian", "fear": "locrian", "anxiety": "locrian",
    "longing": "dorian", "nostalgia": "mixolydian", "defiance": "mixolydian"
}
```

## 4. Trajectory Planning

```python
def plan_trajectory(start, end, bars, curve="linear"):
    points = []
    for i in range(bars):
        t = i / (bars - 1) if bars > 1 else 0
        if curve == "sigmoid": t = 1 / (1 + exp(-10*(t-0.5)))
        elif curve == "exp": t = t ** 2
        elif curve == "log": t = sqrt(t)
        points.append(EmotionVector(
            start.valence + t*(end.valence - start.valence),
            start.arousal + t*(end.arousal - start.arousal),
            start.dominance + t*(end.dominance - start.dominance)
        ))
    return points
```

## 5. Resonance Reward

```python
def resonance_reward(bio_prev, bio_new, emotion, coherence):
    d_hrv = bio_new["hrv"] - bio_prev["hrv"]
    d_eda = bio_prev["eda"] - bio_new["eda"]
    return 0.3*d_hrv + 0.2*d_eda + 0.3*emotion.valence + 0.2*coherence
```

## 6. Aesthetic Reward

```python
def aesthetic_reward(emotion_match, coherence, novelty, feedback=0):
    return 0.4*emotion_match + 0.3*coherence + 0.2*novelty + 0.1*feedback
```

## 7. Coherence Score

```python
def coherence_score(intended, generated, tension_curve):
    mode_score = 1.0 if (generated["mode"]=="major") == (intended.valence>0) else 0.5
    tempo_expected = 60 + 120*intended.arousal
    tempo_score = max(0, 1 - abs(generated["tempo"]-tempo_expected)/60)
    return (mode_score + tempo_score) / 2
```
