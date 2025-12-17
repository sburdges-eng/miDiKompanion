# Kelly Phase 2 - Biometric Integration

## EEG Band → Emotion

| Band | Hz | Meaning | Mapping |
|------|-----|---------|---------|
| α Alpha | 8-12 | Relaxed | ↑Valence, ↓Arousal |
| β Beta | 13-30 | Focus | ↑Arousal |
| θ Theta | 4-7 | Drowsy | ↓Dominance |
| γ Gamma | 30-50 | Engaged | ↑Dominance |

```python
def eeg_to_emotion(bands: dict) -> EmotionVector:
    a, b, t, g = bands["alpha"], bands["beta"], bands["theta"], bands["gamma"]
    return EmotionVector(
        valence = clamp((a - b) * 0.5 + 0.5 * (1 - t), -1, 1),
        arousal = clamp(b / (a + 0.001) / 2, 0, 1),
        dominance = clamp(g / (t + 0.001) / 2, 0, 1)
    )
```

## Biometric Signals

| Signal | Source | Emotional Correlation |
|--------|--------|----------------------|
| HR | Smartwatch | Arousal (↑HR = ↑arousal) |
| HRV | Smartwatch | Calm (↑HRV = ↑valence) |
| EDA/GSR | Skin sensor | Stress (↑EDA = ↓valence) |
| Temp | Body sensor | Comfort |

```python
def bio_to_emotion(bio: dict) -> EmotionVector:
    hr_norm = (bio["hr"] - 60) / 60  # 60-120 BPM range
    hrv_norm = bio["hrv"]  # Already 0-1
    eda_norm = bio["eda"]  # Already 0-1
    
    return EmotionVector(
        valence = hrv_norm - eda_norm * 0.5,
        arousal = hr_norm,
        dominance = 0.5 + hrv_norm * 0.3
    )
```

## Resonance Calculation

```python
def compute_resonance(prev: dict, new: dict, emotion: EmotionVector, coherence: float) -> tuple:
    d_hrv = new["hrv"] - prev["hrv"]  # Positive = calming
    d_eda = prev["eda"] - new["eda"]  # Positive = relaxing
    
    reward = 0.3*d_hrv + 0.2*d_eda + 0.3*emotion.valence + 0.2*coherence
    resonance = (1 + reward) / 2
    
    return round(reward, 3), round(resonance, 3)
```

## Circadian Integration

```python
def circadian_modifier(hour: int) -> dict:
    """Adjust parameters based on time of day."""
    if 6 <= hour < 10:    # Morning
        return {"tempo_mod": 0.9, "brightness_mod": 1.1}
    elif 10 <= hour < 14:  # Midday
        return {"tempo_mod": 1.0, "brightness_mod": 1.0}
    elif 14 <= hour < 18:  # Afternoon
        return {"tempo_mod": 1.05, "brightness_mod": 0.95}
    elif 18 <= hour < 22:  # Evening
        return {"tempo_mod": 0.95, "brightness_mod": 0.85}
    else:                  # Night
        return {"tempo_mod": 0.85, "brightness_mod": 0.7}
```

## Fusion Model (Simplified)

```python
def fuse_inputs(eeg: dict, bio: dict, context: dict) -> EmotionVector:
    e1 = eeg_to_emotion(eeg) if eeg else None
    e2 = bio_to_emotion(bio) if bio else None
    
    if e1 and e2:
        return EmotionVector(
            (e1.valence + e2.valence) / 2,
            (e1.arousal + e2.arousal) / 2,
            (e1.dominance + e2.dominance) / 2
        )
    return e1 or e2 or EmotionVector(0, 0.5, 0.5)
```
