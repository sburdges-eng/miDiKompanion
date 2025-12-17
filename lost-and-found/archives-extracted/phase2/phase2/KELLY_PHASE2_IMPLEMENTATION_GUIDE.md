# KELLY PHASE 2 - Implementation Guide
## Extracted from Suno Architecture Analysis + OMEGA CEFE

**Date:** December 16, 2025  
**Status:** Ready for Implementation  
**Target:** Kelly MIDI Companion Enhancement

---

## 1. EMOTION VECTOR SYSTEM

### 1.1 VAD Continuous Space
Kelly's 216-node thesaurus maps to continuous VAD vectors:

```python
# Each emotion node should include VAD coordinates
@dataclass
class EmotionVector:
    valence: float    # -1.0 (negative) to +1.0 (positive)
    arousal: float    # 0.0 (calm) to 1.0 (excited)
    dominance: float  # 0.0 (submissive) to 1.0 (dominant)
```

### 1.2 Emotion Trajectory Planning
Music unfolds over time - plan emotional arcs:

```python
def plan_emotion_trajectory(
    start_emotion: EmotionVector,
    end_emotion: EmotionVector,
    duration_bars: int,
    curve_type: str = "linear"
) -> List[EmotionVector]:
    """
    Generate emotion trajectory over time.
    
    curve_type: "linear", "exponential", "logarithmic", "sigmoid"
    """
    points = []
    for i in range(duration_bars):
        t = i / (duration_bars - 1) if duration_bars > 1 else 0
        
        if curve_type == "sigmoid":
            t = 1 / (1 + math.exp(-10 * (t - 0.5)))
        elif curve_type == "exponential":
            t = t ** 2
        elif curve_type == "logarithmic":
            t = math.sqrt(t)
        
        points.append(EmotionVector(
            valence=start_emotion.valence + t * (end_emotion.valence - start_emotion.valence),
            arousal=start_emotion.arousal + t * (end_emotion.arousal - start_emotion.arousal),
            dominance=start_emotion.dominance + t * (end_emotion.dominance - start_emotion.dominance)
        ))
    return points
```

### 1.3 Emotion Distance Calculation

```python
def emotion_distance(e1: EmotionVector, e2: EmotionVector) -> float:
    """Euclidean distance in VAD space."""
    return math.sqrt(
        (e1.valence - e2.valence) ** 2 +
        (e1.arousal - e2.arousal) ** 2 +
        (e1.dominance - e2.dominance) ** 2
    )
```

---

## 2. EMOTION-TO-MUSIC MAPPING

### 2.1 Core Parameter Mapping

| VAD Dimension | Music Parameter | Formula |
|---------------|-----------------|---------|
| Valence | Mode | `"major" if valence > 0 else "minor"` |
| Arousal | Tempo | `tempo = 60 + 120 * arousal` |
| Dominance | Dynamics | `velocity = 60 + 67 * dominance` |
| Valence + Arousal | Instrumentation | See matrix below |

### 2.2 Complete Mapping Function

```python
def map_emotion_to_music(emotion: EmotionVector) -> MusicParams:
    """Convert emotion vector to musical parameters."""
    
    # Mode selection
    if emotion.valence > 0.3:
        mode = "major"
    elif emotion.valence < -0.3:
        mode = "minor"
    else:
        mode = "dorian"  # Ambiguous emotions
    
    # Tempo (BPM)
    tempo = int(60 + 120 * emotion.arousal)
    
    # Velocity/Dynamics
    base_velocity = int(60 + 67 * emotion.dominance)
    
    # Harmonic rhythm (chords per bar)
    if emotion.arousal > 0.7:
        harmonic_rhythm = 2  # Fast changes
    elif emotion.arousal < 0.3:
        harmonic_rhythm = 0.5  # Slow changes
    else:
        harmonic_rhythm = 1
    
    # Dissonance tolerance
    dissonance = 0.2 + abs(emotion.valence) * 0.3 + (1 - emotion.dominance) * 0.3
    
    # Register (MIDI note range center)
    register_center = 60 + int(emotion.valence * 12) + int(emotion.arousal * 6)
    
    # Articulation
    legato = 0.7 - emotion.arousal * 0.4  # Higher arousal = more staccato
    
    return MusicParams(
        mode=mode,
        tempo=tempo,
        velocity=base_velocity,
        harmonic_rhythm=harmonic_rhythm,
        dissonance_tolerance=dissonance,
        register_center=register_center,
        legato=legato
    )
```

### 2.3 Emotion-to-Mode Extended Mapping

```python
EMOTION_MODE_MAP = {
    # High valence
    "joy": ["ionian", "lydian"],
    "euphoria": ["lydian", "ionian"],
    "hope": ["ionian", "mixolydian"],
    "contentment": ["ionian", "dorian"],
    
    # Low valence
    "grief": ["aeolian", "phrygian"],
    "sadness": ["aeolian", "dorian"],
    "despair": ["phrygian", "locrian"],
    "melancholy": ["dorian", "aeolian"],
    
    # Mixed/Complex
    "longing": ["dorian", "mixolydian"],
    "nostalgia": ["mixolydian", "dorian"],
    "anxiety": ["locrian", "phrygian"],
    "anger": ["phrygian", "aeolian"],
    "defiance": ["mixolydian", "dorian"],
    "dissociation": ["lydian", "locrian"],  # Floating quality
}
```

---

## 3. HUMANIZATION ENGINE

### 3.1 Timing Humanization

```python
def humanize_timing(
    note_time: int,
    emotion: EmotionVector,
    genre: str,
    ppq: int = 480
) -> int:
    """
    Apply timing deviations based on emotion and genre.
    
    Returns adjusted note time in ticks.
    """
    # Base deviation in ms
    if emotion.arousal > 0.7:
        # High arousal: tighter timing
        max_deviation_ms = 10
    else:
        # Low arousal: looser, more human
        max_deviation_ms = 25
    
    # Genre-specific offset
    genre_offsets = {
        "hiphop": 15,    # Laid back
        "jazz": 10,      # Behind the beat
        "rock": -3,      # Pushed forward
        "edm": 0,        # Grid-tight
        "lofi": 20,      # Very laid back
        "funk": -5,      # Slightly ahead
    }
    base_offset = genre_offsets.get(genre, 0)
    
    # Convert ms to ticks (assuming 120 BPM as reference)
    ms_per_tick = 60000 / (120 * ppq)
    deviation_ticks = int((random.gauss(base_offset, max_deviation_ms)) / ms_per_tick)
    
    return note_time + deviation_ticks
```

### 3.2 Velocity Humanization

```python
def humanize_velocity(
    base_velocity: int,
    beat_position: float,
    emotion: EmotionVector
) -> int:
    """
    Apply velocity variations based on beat position and emotion.
    
    beat_position: 0.0-1.0 within the bar
    """
    # Accent pattern (emphasize beats 1 and 3 in 4/4)
    accent = 1.0
    if beat_position < 0.1 or (0.5 <= beat_position < 0.6):
        accent = 1.15  # Downbeat accent
    elif (0.25 <= beat_position < 0.35) or (0.75 <= beat_position < 0.85):
        accent = 1.05  # Backbeat accent
    
    # Random human variation
    variation = random.gauss(0, 5 + 10 * (1 - emotion.dominance))
    
    # Emotional dynamics
    emotional_mod = 1.0 + (emotion.arousal - 0.5) * 0.2
    
    result = int(base_velocity * accent * emotional_mod + variation)
    return max(1, min(127, result))
```

### 3.3 Ghost Notes

```python
def generate_ghost_notes(
    main_notes: List[Note],
    emotion: EmotionVector,
    genre: str
) -> List[Note]:
    """Generate ghost notes between main notes for groove."""
    
    ghost_probability = {
        "funk": 0.4,
        "hiphop": 0.3,
        "jazz": 0.25,
        "neo_soul": 0.35,
        "rock": 0.1,
        "edm": 0.05,
    }.get(genre, 0.15)
    
    # Emotion affects ghost density
    if emotion.arousal < 0.4:
        ghost_probability *= 1.3  # More ghosts when laid back
    
    ghost_velocity_range = (20, 45)
    ghosts = []
    
    for i, note in enumerate(main_notes[:-1]):
        next_note = main_notes[i + 1]
        gap = next_note.time - note.time
        
        if gap > 240 and random.random() < ghost_probability:  # 8th note gap minimum
            ghost_time = note.time + gap // 2 + random.randint(-20, 20)
            ghost_vel = random.randint(*ghost_velocity_range)
            ghosts.append(Note(
                pitch=note.pitch,
                time=ghost_time,
                duration=gap // 4,
                velocity=ghost_vel
            ))
    
    return ghosts
```

---

## 4. MICRO-EXPRESSIVE MODULATION

### 4.1 Pitch Drift (Vocal/Lead Lines)

```python
def apply_pitch_drift(
    pitch: int,
    emotion: EmotionVector,
    position_in_phrase: float  # 0.0 = start, 1.0 = end
) -> Tuple[int, int]:
    """
    Apply subtle pitch drift for emotional authenticity.
    
    Returns (pitch, pitch_bend) where pitch_bend is in cents.
    """
    # Drift range in cents
    if emotion.valence < -0.3:
        # Sad: slight flat tendency
        drift_center = -8
        drift_range = 15
    elif emotion.arousal > 0.7:
        # Excited: sharper tendency
        drift_center = 5
        drift_range = 10
    else:
        drift_center = 0
        drift_range = 8
    
    # Phrases often drift slightly flat at the end
    end_drift = -5 * position_in_phrase if position_in_phrase > 0.7 else 0
    
    pitch_bend = int(random.gauss(drift_center + end_drift, drift_range / 3))
    pitch_bend = max(-50, min(50, pitch_bend))  # Clamp to ±50 cents
    
    return pitch, pitch_bend
```

### 4.2 Vibrato Generation

```python
def generate_vibrato(
    duration_ticks: int,
    emotion: EmotionVector,
    ppq: int = 480
) -> List[Tuple[int, int]]:
    """
    Generate vibrato curve as list of (time, pitch_bend) pairs.
    
    Returns pitch bend values in cents.
    """
    # Vibrato rate: 5-7 Hz typical
    rate_hz = 5 + emotion.arousal * 2
    
    # Vibrato depth in cents
    if emotion.valence < -0.3:
        depth = 20 + emotion.arousal * 30  # More expressive when sad
    else:
        depth = 10 + emotion.arousal * 20
    
    # Convert to ticks
    samples_per_cycle = int(ppq * 60 / (120 * rate_hz))
    
    vibrato = []
    t = 0
    while t < duration_ticks:
        # Fade in vibrato over first 25% of note
        fade_in = min(1.0, t / (duration_ticks * 0.25))
        
        # Sine wave vibrato
        phase = 2 * math.pi * t / samples_per_cycle
        bend = int(depth * fade_in * math.sin(phase))
        
        vibrato.append((t, bend))
        t += ppq // 8  # Sample every 32nd note
    
    return vibrato
```

---

## 5. ADAPTIVE MIX PARAMETERS

### 5.1 Emotion-to-Mix Mapping

```python
@dataclass
class MixParams:
    # EQ
    low_shelf_db: float      # Bass boost/cut
    high_shelf_db: float     # Brightness
    mid_presence: float      # 0-1, presence amount
    
    # Dynamics
    compression_ratio: float
    attack_ms: float
    release_ms: float
    
    # Space
    reverb_amount: float     # 0-1
    reverb_decay: float      # seconds
    stereo_width: float      # 0-1
    
    # Character
    saturation: float        # 0-1
    
def emotion_to_mix(emotion: EmotionVector) -> MixParams:
    """Map emotion to mix parameters."""
    
    # Low end: more bass for power/grief
    low_shelf = -3 + 6 * (1 - emotion.valence) * emotion.dominance
    
    # High end: brighter for positive/energetic
    high_shelf = -2 + 4 * (emotion.valence + emotion.arousal) / 2
    
    # Presence: more for dominant emotions
    presence = 0.3 + 0.4 * emotion.dominance
    
    # Compression: tighter for high arousal
    comp_ratio = 2.0 + 4.0 * emotion.arousal
    attack = 30 - 20 * emotion.arousal  # Faster attack when energetic
    release = 100 + 200 * (1 - emotion.arousal)
    
    # Reverb: more space for low arousal, intimate emotions
    reverb = 0.3 + 0.4 * (1 - emotion.arousal) + 0.2 * (1 - emotion.dominance)
    decay = 1.0 + 2.0 * (1 - emotion.arousal)
    
    # Stereo: wider for positive, expansive emotions
    width = 0.5 + 0.3 * emotion.valence + 0.2 * (1 - emotion.arousal)
    
    # Saturation: more for anger, intensity
    saturation = 0.1 + 0.4 * emotion.arousal * (1 - emotion.valence)
    
    return MixParams(
        low_shelf_db=low_shelf,
        high_shelf_db=high_shelf,
        mid_presence=presence,
        compression_ratio=comp_ratio,
        attack_ms=attack,
        release_ms=release,
        reverb_amount=reverb,
        reverb_decay=decay,
        stereo_width=width,
        saturation=saturation
    )
```

---

## 6. COHERENCE & REWARD SYSTEM

### 6.1 Musical Coherence Score

```python
def calculate_coherence(
    intended_emotion: EmotionVector,
    generated_params: MusicParams,
    progression_tension: List[float]
) -> float:
    """
    Calculate how well the generated music matches intended emotion.
    
    Returns 0.0-1.0 coherence score.
    """
    scores = []
    
    # Mode coherence
    expected_mode = "major" if intended_emotion.valence > 0 else "minor"
    mode_score = 1.0 if generated_params.mode == expected_mode else 0.5
    scores.append(mode_score)
    
    # Tempo coherence
    expected_tempo = 60 + 120 * intended_emotion.arousal
    tempo_diff = abs(generated_params.tempo - expected_tempo)
    tempo_score = max(0, 1 - tempo_diff / 60)
    scores.append(tempo_score)
    
    # Tension arc coherence
    if intended_emotion.arousal > 0.6:
        # High arousal should have building tension
        tension_trend = sum(progression_tension[i+1] - progression_tension[i] 
                          for i in range(len(progression_tension)-1))
        tension_score = 0.5 + 0.5 * (tension_trend / len(progression_tension))
    else:
        # Low arousal should have stable/releasing tension
        tension_variance = sum((t - 0.5)**2 for t in progression_tension) / len(progression_tension)
        tension_score = 1 - tension_variance
    scores.append(tension_score)
    
    return sum(scores) / len(scores)
```

### 6.2 Aesthetic Reward Function

```python
def aesthetic_reward(
    emotion_match: float,      # How well music matches intended emotion
    novelty: float,            # KL divergence from training distribution
    coherence: float,          # Internal musical coherence
    human_feedback: float = 0  # Optional user rating 0-1
) -> float:
    """
    Calculate aesthetic reward for reinforcement learning.
    
    R = 0.4*emotion_match + 0.3*coherence + 0.2*novelty + 0.1*feedback
    """
    weights = {
        'emotion': 0.4,
        'coherence': 0.3,
        'novelty': 0.2,
        'feedback': 0.1
    }
    
    reward = (
        weights['emotion'] * emotion_match +
        weights['coherence'] * coherence +
        weights['novelty'] * novelty +
        weights['feedback'] * human_feedback
    )
    
    return reward
```

---

## 7. RESONANCE ENGINE (From OMEGA)

### 7.1 Biometric Integration Formulas

```python
def compute_resonance(
    bio_prev: dict,
    bio_new: dict,
    emotion: EmotionVector,
    coherence: float
) -> Tuple[float, float]:
    """
    Calculate resonance reward from biometric feedback.
    
    Returns (reward, resonance_score)
    """
    # Heart rate variability change (positive = calming)
    delta_hrv = bio_new.get("hrv", 0.5) - bio_prev.get("hrv", 0.5)
    
    # Skin conductance change (negative = calming)
    delta_eda = bio_prev.get("eda", 0.5) - bio_new.get("eda", 0.5)
    
    # Weights
    w = [0.3, 0.2, 0.3, 0.2]
    
    reward = (
        w[0] * delta_hrv +
        w[1] * delta_eda +
        w[2] * emotion.valence +
        w[3] * coherence
    )
    
    resonance = (1 + reward) / 2  # Normalize to 0-1
    
    return round(reward, 3), round(resonance, 3)
```

### 7.2 EEG-to-Emotion Mapping (Optional Future)

```python
def eeg_to_emotion(eeg_bands: dict) -> EmotionVector:
    """
    Convert EEG frequency bands to emotion vector.
    
    eeg_bands: {alpha, beta, theta, gamma} each 0-1 normalized
    """
    alpha = eeg_bands.get("alpha", 0.5)
    beta = eeg_bands.get("beta", 0.5)
    theta = eeg_bands.get("theta", 0.5)
    gamma = eeg_bands.get("gamma", 0.5)
    
    # Alpha indicates relaxation (positive valence, low arousal)
    # Beta indicates focus/tension (high arousal)
    # Theta indicates drowsiness (low dominance)
    # Gamma indicates engagement (high dominance)
    
    valence = (alpha - beta) * 0.5 + 0.5 * (1 - theta)
    arousal = beta / (alpha + 0.001)
    arousal = min(1.0, arousal / 2)  # Normalize
    dominance = gamma / (theta + 0.001)
    dominance = min(1.0, dominance / 2)
    
    return EmotionVector(
        valence=max(-1, min(1, valence)),
        arousal=max(0, min(1, arousal)),
        dominance=max(0, min(1, dominance))
    )
```

---

## 8. MULTI-AGENT COORDINATION

### 8.1 Agent Message Protocol

```python
@dataclass
class AgentMessage:
    """Standard message format for agent communication."""
    timestamp: str
    agent_id: str
    emotion_state: dict
    action_state: dict
    coherence: float = 1.0

class MusicAgent:
    """Base agent for music generation subsystems."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.state = {}
        self.coherence = 1.0
    
    def update(self, emotion: EmotionVector, context: dict) -> None:
        """Update agent state based on emotion and context."""
        raise NotImplementedError
    
    def message(self) -> AgentMessage:
        return AgentMessage(
            timestamp=datetime.utcnow().isoformat(),
            agent_id=self.agent_id,
            emotion_state=self.state.get("emotion", {}),
            action_state=self.state.get("action", {}),
            coherence=self.coherence
        )
```

### 8.2 Harmony Agent

```python
class HarmonyAgent(MusicAgent):
    """Generates chord progressions based on emotion."""
    
    def update(self, emotion: EmotionVector, context: dict) -> None:
        mode = "major" if emotion.valence > 0 else "minor"
        
        # Tension level affects chord complexity
        if abs(emotion.valence) > 0.6:
            extensions = "7th"
        elif emotion.arousal > 0.7:
            extensions = "9th"
        else:
            extensions = "triad"
        
        # Progression type
        if emotion.valence > 0.3:
            progression = "I-V-vi-IV"
        elif emotion.valence < -0.3:
            progression = "i-bVI-bVII-i"
        else:
            progression = "i-iv-V-i"
        
        self.state = {
            "emotion": {"v": emotion.valence, "a": emotion.arousal},
            "action": {
                "mode": mode,
                "extensions": extensions,
                "progression": progression
            }
        }
```

### 8.3 Rhythm Agent

```python
class RhythmAgent(MusicAgent):
    """Generates rhythmic patterns based on emotion."""
    
    def update(self, emotion: EmotionVector, context: dict) -> None:
        # Density based on arousal
        density = 0.3 + 0.7 * emotion.arousal
        
        # Swing based on genre and emotion
        genre = context.get("genre", "pop")
        base_swing = {"jazz": 0.67, "hiphop": 0.55, "rock": 0.5, "funk": 0.52}.get(genre, 0.5)
        swing = base_swing + 0.1 * (1 - emotion.arousal)
        
        # Syncopation
        syncopation = 0.2 + 0.4 * emotion.arousal + 0.2 * (1 - emotion.dominance)
        
        self.state = {
            "emotion": {"v": emotion.valence, "a": emotion.arousal},
            "action": {
                "density": density,
                "swing": swing,
                "syncopation": syncopation
            }
        }
```

---

## 9. TRANSITION FORMULAS

### 9.1 Energy Curve Generation

```python
def generate_energy_curve(
    transition_type: str,
    num_points: int = 8
) -> List[float]:
    """Generate energy curve for section transitions."""
    
    curves = {
        "build": lambda t: t ** 1.5,
        "drop": lambda t: 1 - (1 - t) ** 2,
        "breakdown": lambda t: 1 - t,
        "sustain": lambda t: 0.7,
        "swell": lambda t: 0.5 + 0.3 * math.sin(t * math.pi),
        "impact": lambda t: 1.0 if t > 0.9 else t * 0.5,
    }
    
    func = curves.get(transition_type, lambda t: t)
    
    return [func(i / (num_points - 1)) for i in range(num_points)]
```

### 9.2 Crossfade Parameters

```python
def calculate_crossfade(
    from_energy: float,
    to_energy: float,
    transition_bars: int
) -> dict:
    """Calculate crossfade parameters for smooth transition."""
    
    energy_diff = abs(to_energy - from_energy)
    
    # Longer crossfade for bigger energy changes
    crossfade_bars = min(transition_bars, max(1, int(energy_diff * 4)))
    
    # Curve type
    if to_energy > from_energy:
        curve = "exponential"  # Building energy
    else:
        curve = "logarithmic"  # Releasing energy
    
    return {
        "duration_bars": crossfade_bars,
        "curve": curve,
        "from_level": from_energy,
        "to_level": to_energy
    }
```

---

## 10. DATA STRUCTURES

### 10.1 Complete Song Intent

```python
@dataclass
class SongIntent:
    # Phase 0: Core Wound
    wound_description: str
    wound_intensity: float
    wound_temporal: str  # "acute", "chronic", "deep"
    
    # Phase 1: Emotional Intent
    primary_emotion: str
    secondary_emotions: List[str]
    emotion_vector: EmotionVector
    vulnerability_level: float
    narrative_arc: str
    
    # Phase 2: Technical
    genre: str
    tempo_range: Tuple[int, int]
    key: str
    mode: str
    rule_breaks: List[str]
    rule_break_justifications: Dict[str, str]
    
    # Generated
    music_params: Optional[MusicParams] = None
    mix_params: Optional[MixParams] = None
```

### 10.2 Section Specification

```python
@dataclass  
class SectionSpec:
    name: str                    # "verse", "chorus", "bridge"
    duration_bars: int
    emotion_start: EmotionVector
    emotion_end: EmotionVector
    energy_curve: List[float]
    tension_curve: List[float]
    instruments_active: List[str]
    rule_breaks_active: List[str]
```

---

## 11. OUTPUT FORMATS

### 11.1 MIDI Export

```python
def export_to_midi(
    notes: List[Note],
    params: MusicParams,
    output_path: str
) -> None:
    """Export notes to MIDI file with humanization."""
    from mido import MidiFile, MidiTrack, Message
    
    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Set tempo
    tempo_us = int(60_000_000 / params.tempo)
    track.append(Message('set_tempo', tempo=tempo_us, time=0))
    
    # Sort notes by time
    notes = sorted(notes, key=lambda n: n.time)
    
    current_time = 0
    for note in notes:
        # Note on
        delta = note.time - current_time
        track.append(Message('note_on', 
                            note=note.pitch, 
                            velocity=note.velocity, 
                            time=delta))
        
        # Note off
        track.append(Message('note_off',
                            note=note.pitch,
                            velocity=0,
                            time=note.duration))
        
        current_time = note.time + note.duration
    
    mid.save(output_path)
```

### 11.2 OSC Output (Real-time)

```python
def send_osc_params(
    emotion: EmotionVector,
    params: MusicParams,
    osc_client
) -> None:
    """Send parameters via OSC for live visualization."""
    
    osc_client.send_message("/kelly/emotion/valence", emotion.valence)
    osc_client.send_message("/kelly/emotion/arousal", emotion.arousal)
    osc_client.send_message("/kelly/emotion/dominance", emotion.dominance)
    
    osc_client.send_message("/kelly/music/tempo", params.tempo)
    osc_client.send_message("/kelly/music/mode", params.mode)
    osc_client.send_message("/kelly/music/velocity", params.velocity)
```

---

## 12. IMPLEMENTATION PRIORITIES

### Phase 2A (Immediate)
1. Add VAD coordinates to all 216 emotion nodes
2. Implement `map_emotion_to_music()` in intent processor
3. Add humanization to MIDI generator
4. Implement emotion trajectory planning

### Phase 2B (Next)
1. Multi-agent architecture (Harmony, Rhythm, Melody agents)
2. Coherence scoring system
3. Real-time OSC output
4. Transition engine integration

### Phase 2C (Future)
1. Biometric input layer
2. Resonance reward loop
3. Memory persistence (user profiles)
4. Live visualization dashboard

---

## QUICK REFERENCE

### Core Formulas

| Formula | Use |
|---------|-----|
| `tempo = 60 + 120 * arousal` | Base tempo from arousal |
| `velocity = 60 + 67 * dominance` | Base dynamics |
| `mode = major if valence > 0` | Mode selection |
| `R = 0.4E + 0.3C + 0.2N + 0.1F` | Aesthetic reward |
| `distance = sqrt(Σ(v1-v2)²)` | Emotion distance |

### Key Mappings

| Emotion | Mode | Tempo Range | Character |
|---------|------|-------------|-----------|
| Joy | Lydian/Ionian | 100-130 | Bright, open |
| Grief | Aeolian/Phrygian | 50-70 | Heavy, sparse |
| Anger | Phrygian | 120-160 | Driving, dissonant |
| Fear | Locrian | 80-110 | Unstable, tense |
| Hope | Ionian/Mixolydian | 90-120 | Rising, resolved |
| Longing | Dorian | 70-90 | Suspended, yearning |
