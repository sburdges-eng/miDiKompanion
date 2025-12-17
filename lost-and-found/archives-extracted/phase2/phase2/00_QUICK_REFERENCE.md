# Kelly Phase 2 - Quick Reference

## Core Equations

| Formula | Code |
|---------|------|
| Tempo | `60 + 120 * arousal` |
| Velocity | `60 + 67 * dominance` |
| Mode | `major if v>0 else minor` |
| Swing | `base + 0.1*(1-arousal)` |
| Reverb | `0.3 + 0.4*(1-arousal)` |
| Resonance | `0.3*Δhrv + 0.2*Δeda + 0.3*v + 0.2*c` |
| Reward | `0.4*E + 0.3*C + 0.2*N + 0.1*F` |

## VAD Ranges

| Dimension | Range | Low | High |
|-----------|-------|-----|------|
| Valence | -1 to +1 | Negative | Positive |
| Arousal | 0 to 1 | Calm | Excited |
| Dominance | 0 to 1 | Submissive | Dominant |

## Emotion → Mode

| Valence | Arousal | Mode |
|---------|---------|------|
| High | High | Lydian |
| High | Low | Ionian |
| Low | High | Phrygian |
| Low | Low | Aeolian |
| Neutral | Any | Dorian |

## Genre Swing

| Genre | Swing | Offset (ms) |
|-------|-------|-------------|
| Jazz | 0.67 | +10 |
| Hip-hop | 0.55 | +15 |
| Funk | 0.52 | -5 |
| Lo-fi | 0.62 | +20 |
| Rock | 0.50 | -3 |
| EDM | 0.50 | 0 |

## EEG Bands

| Band | Hz | → Emotion |
|------|-----|-----------|
| Alpha | 8-12 | +V, -A |
| Beta | 13-30 | +A |
| Theta | 4-7 | -D |
| Gamma | 30-50 | +D |

## Tension Levels

| Level | Value | Use |
|-------|-------|-----|
| Comatose | 0 | Drone |
| Minimal | 1 | Ambient |
| Low | 2 | Verse |
| Moderate | 4 | Pre-chorus |
| High | 6 | Chorus |
| Extreme | 8 | Climax |

## Rule Breaks by Emotion

| Emotion | Harmony | Rhythm | Production |
|---------|---------|--------|------------|
| Grief | AvoidResolution | TempoFluctuation | ExcessiveMud |
| Anger | ParallelMotion | ConstantDisplacement | Distortion |
| Fear | UnresolvedDissonance | DroppedBeats | MonoCollapse |
| Longing | ModalInterchange | TempoFluctuation | PitchImperfection |
| Dissociation | Polytonality | MetricModulation | BuriedVocals |

## Agent Outputs

| Agent | Key Params |
|-------|------------|
| Harmony | mode, extensions, progression |
| Rhythm | density, swing, syncopation |
| Melody | range, leap_prob, rest_prob |
| Dynamics | velocity, range, accent |

## OSC Routes

```
/kelly/emotion/valence   float -1..1
/kelly/emotion/arousal   float 0..1
/kelly/music/tempo       int 60..180
/kelly/music/mode        string
/kelly/reward/coherence  float 0..1
```

## MIDI CC

| CC | Param | Source |
|----|-------|--------|
| 1 | Mod | arousal |
| 7 | Vol | dominance |
| 74 | Bright | valence |
| 91 | Reverb | 1-arousal |
