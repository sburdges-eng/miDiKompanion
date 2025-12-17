# Kelly Phase 2 - Humanization Engine

## 1. Timing Humanization

```python
GENRE_OFFSETS_MS = {
    "hiphop": 15, "jazz": 10, "lofi": 20,
    "rock": -3, "funk": -5, "edm": 0
}

def humanize_timing(note_time, arousal, genre, ppq=480):
    max_dev = 10 if arousal > 0.7 else 25
    offset = GENRE_OFFSETS_MS.get(genre, 0)
    ms_per_tick = 60000 / (120 * ppq)
    deviation = int(gauss(offset, max_dev) / ms_per_tick)
    return note_time + deviation
```

## 2. Velocity Humanization

```python
def humanize_velocity(base_vel, beat_pos, emotion):
    # Accent pattern
    accent = 1.15 if beat_pos < 0.1 or 0.5 <= beat_pos < 0.6 else 1.0
    variation = gauss(0, 5 + 10*(1-emotion.dominance))
    return clamp(int(base_vel * accent * (1 + (emotion.arousal-0.5)*0.2) + variation), 1, 127)
```

## 3. Ghost Notes

```python
GHOST_PROB = {"funk": 0.4, "hiphop": 0.3, "jazz": 0.25, "neo_soul": 0.35, "rock": 0.1}

def generate_ghosts(notes, emotion, genre):
    prob = GHOST_PROB.get(genre, 0.15) * (1.3 if emotion.arousal < 0.4 else 1.0)
    ghosts = []
    for i, n in enumerate(notes[:-1]):
        gap = notes[i+1].time - n.time
        if gap > 240 and random() < prob:
            ghosts.append(Note(n.pitch, n.time + gap//2 + randint(-20,20), gap//4, randint(20,45)))
    return ghosts
```

## 4. Pitch Drift (Vocals/Leads)

```python
def pitch_drift(pitch, emotion, phrase_pos):
    if emotion.valence < -0.3:
        center, range_ = -8, 15  # Sad: flat
    elif emotion.arousal > 0.7:
        center, range_ = 5, 10   # Excited: sharp
    else:
        center, range_ = 0, 8
    end_drift = -5 * phrase_pos if phrase_pos > 0.7 else 0
    return clamp(int(gauss(center + end_drift, range_/3)), -50, 50)  # cents
```

## 5. Vibrato

```python
def vibrato(duration_ticks, emotion, ppq=480):
    rate = 5 + emotion.arousal * 2  # Hz
    depth = (20 + emotion.arousal*30) if emotion.valence < -0.3 else (10 + emotion.arousal*20)
    samples_per_cycle = int(ppq * 60 / (120 * rate))
    curve = []
    for t in range(0, duration_ticks, ppq//8):
        fade = min(1.0, t / (duration_ticks * 0.25))
        bend = int(depth * fade * sin(2*pi*t/samples_per_cycle))
        curve.append((t, bend))
    return curve
```

## 6. Swing Application

```python
GENRE_SWING = {"jazz": 0.67, "hiphop": 0.55, "funk": 0.52, "lofi": 0.62, "rock": 0.5}

def apply_swing(note_time, swing_amount, ppq=480):
    beat_pos = (note_time % ppq) / ppq
    if 0.4 < beat_pos < 0.6:  # Off-beat
        shift = int((swing_amount - 0.5) * ppq * 0.5)
        return note_time + shift
    return note_time
```
