# Kelly Phase 2 - Mix Parameters

## Emotion → Mix Mapping

```python
@dataclass
class MixParams:
    low_shelf_db: float
    high_shelf_db: float
    compression_ratio: float
    attack_ms: float
    release_ms: float
    reverb_amount: float
    reverb_decay: float
    stereo_width: float
    saturation: float

def emotion_to_mix(e: EmotionVector) -> MixParams:
    return MixParams(
        low_shelf_db = -3 + 6*(1-e.valence)*e.dominance,
        high_shelf_db = -2 + 4*(e.valence + e.arousal)/2,
        compression_ratio = 2.0 + 4.0*e.arousal,
        attack_ms = 30 - 20*e.arousal,
        release_ms = 100 + 200*(1-e.arousal),
        reverb_amount = 0.3 + 0.4*(1-e.arousal) + 0.2*(1-e.dominance),
        reverb_decay = 1.0 + 2.0*(1-e.arousal),
        stereo_width = 0.5 + 0.3*e.valence + 0.2*(1-e.arousal),
        saturation = 0.1 + 0.4*e.arousal*(1-e.valence)
    )
```

## Quick Reference

| Emotion State | Low | High | Comp | Reverb | Width | Sat |
|---------------|-----|------|------|--------|-------|-----|
| Joy (high V) | -3dB | +2dB | 2:1 | 0.3 | 0.8 | 0.1 |
| Grief (low V) | +3dB | -2dB | 2:1 | 0.7 | 0.3 | 0.3 |
| Anger (low V, high A) | +3dB | +2dB | 6:1 | 0.3 | 0.5 | 0.5 |
| Peace (high V, low A) | -3dB | +2dB | 2:1 | 0.9 | 0.8 | 0.1 |

## Production Rule-Breaks by Emotion

```python
PRODUCTION_RULES = {
    "grief": ["excessive_mud", "pitch_imperfection", "room_noise"],
    "anger": ["distortion", "clipping_peaks", "mono_collapse"],
    "dissociation": ["buried_vocals", "lo_fi_degradation"],
    "intimacy": ["room_noise", "pitch_imperfection"],
    "anxiety": ["mono_collapse", "excessive_compression"]
}
```
