"""
Kelly Phase 2 - Core Implementation
Drop-in module for emotion-to-music generation
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import math
import random

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EmotionVector:
    valence: float    # -1 to +1
    arousal: float    # 0 to 1
    dominance: float  # 0 to 1
    
    def distance(self, other: 'EmotionVector') -> float:
        return math.sqrt(
            (self.valence - other.valence)**2 +
            (self.arousal - other.arousal)**2 +
            (self.dominance - other.dominance)**2
        )

@dataclass
class MusicParams:
    tempo: int
    mode: str
    velocity: int
    dissonance: float
    legato: float
    register: int

@dataclass
class MixParams:
    low_shelf_db: float
    high_shelf_db: float
    compression_ratio: float
    reverb_amount: float
    stereo_width: float
    saturation: float

# =============================================================================
# CORE MAPPING FUNCTIONS
# =============================================================================

EMOTION_MODE = {
    "joy": "lydian", "euphoria": "lydian", "hope": "ionian",
    "grief": "aeolian", "sadness": "dorian", "despair": "phrygian",
    "anger": "phrygian", "fear": "locrian", "anxiety": "locrian",
    "longing": "dorian", "nostalgia": "mixolydian", "defiance": "mixolydian"
}

def emotion_to_music(e: EmotionVector) -> MusicParams:
    """Convert emotion vector to musical parameters."""
    if e.valence > 0.3:
        mode = "major"
    elif e.valence < -0.3:
        mode = "minor"
    else:
        mode = "dorian"
    
    return MusicParams(
        tempo=int(60 + 120 * e.arousal),
        mode=mode,
        velocity=int(60 + 67 * e.dominance),
        dissonance=0.2 + abs(e.valence)*0.3 + (1-e.dominance)*0.3,
        legato=0.7 - e.arousal * 0.4,
        register=60 + int(e.valence*12) + int(e.arousal*6)
    )

def emotion_to_mix(e: EmotionVector) -> MixParams:
    """Convert emotion vector to mix parameters."""
    return MixParams(
        low_shelf_db=-3 + 6*(1-e.valence)*e.dominance,
        high_shelf_db=-2 + 4*(e.valence + e.arousal)/2,
        compression_ratio=2.0 + 4.0*e.arousal,
        reverb_amount=0.3 + 0.4*(1-e.arousal) + 0.2*(1-e.dominance),
        stereo_width=0.5 + 0.3*e.valence + 0.2*(1-e.arousal),
        saturation=0.1 + 0.4*e.arousal*(1-e.valence)
    )

# =============================================================================
# TRAJECTORY PLANNING
# =============================================================================

def plan_trajectory(
    start: EmotionVector,
    end: EmotionVector,
    bars: int,
    curve: str = "linear"
) -> List[EmotionVector]:
    """Generate emotion trajectory over time."""
    points = []
    for i in range(bars):
        t = i / (bars - 1) if bars > 1 else 0
        
        if curve == "sigmoid":
            t = 1 / (1 + math.exp(-10 * (t - 0.5)))
        elif curve == "exp":
            t = t ** 2
        elif curve == "log":
            t = math.sqrt(t)
        
        points.append(EmotionVector(
            start.valence + t * (end.valence - start.valence),
            start.arousal + t * (end.arousal - start.arousal),
            start.dominance + t * (end.dominance - start.dominance)
        ))
    return points

# =============================================================================
# HUMANIZATION
# =============================================================================

GENRE_SWING = {"jazz": 0.67, "hiphop": 0.55, "funk": 0.52, "lofi": 0.62, "rock": 0.5, "edm": 0.5}
GENRE_OFFSET = {"hiphop": 15, "jazz": 10, "lofi": 20, "rock": -3, "funk": -5, "edm": 0}

def humanize_timing(note_time: int, arousal: float, genre: str, ppq: int = 480) -> int:
    """Apply timing humanization."""
    max_dev = 10 if arousal > 0.7 else 25
    offset = GENRE_OFFSET.get(genre, 0)
    ms_per_tick = 60000 / (120 * ppq)
    deviation = int(random.gauss(offset, max_dev) / ms_per_tick)
    return note_time + deviation

def humanize_velocity(base_vel: int, beat_pos: float, emotion: EmotionVector) -> int:
    """Apply velocity humanization."""
    accent = 1.15 if beat_pos < 0.1 or 0.5 <= beat_pos < 0.6 else 1.0
    variation = random.gauss(0, 5 + 10 * (1 - emotion.dominance))
    result = int(base_vel * accent * (1 + (emotion.arousal - 0.5) * 0.2) + variation)
    return max(1, min(127, result))

def apply_swing(note_time: int, genre: str, ppq: int = 480) -> int:
    """Apply swing to note timing."""
    swing = GENRE_SWING.get(genre, 0.5)
    beat_pos = (note_time % ppq) / ppq
    if 0.4 < beat_pos < 0.6:
        shift = int((swing - 0.5) * ppq * 0.5)
        return note_time + shift
    return note_time

# =============================================================================
# COHERENCE & REWARD
# =============================================================================

def coherence_score(intended: EmotionVector, params: MusicParams) -> float:
    """Calculate coherence between intention and output."""
    mode_score = 1.0 if (params.mode == "major") == (intended.valence > 0) else 0.5
    tempo_expected = 60 + 120 * intended.arousal
    tempo_score = max(0, 1 - abs(params.tempo - tempo_expected) / 60)
    return (mode_score + tempo_score) / 2

def aesthetic_reward(emotion_match: float, coherence: float, novelty: float, feedback: float = 0) -> float:
    """Calculate aesthetic reward."""
    return 0.4*emotion_match + 0.3*coherence + 0.2*novelty + 0.1*feedback

def resonance_reward(bio_prev: dict, bio_new: dict, emotion: EmotionVector, coherence: float) -> Tuple[float, float]:
    """Calculate resonance from biometric feedback."""
    d_hrv = bio_new.get("hrv", 0.5) - bio_prev.get("hrv", 0.5)
    d_eda = bio_prev.get("eda", 0.5) - bio_new.get("eda", 0.5)
    reward = 0.3*d_hrv + 0.2*d_eda + 0.3*emotion.valence + 0.2*coherence
    resonance = (1 + reward) / 2
    return round(reward, 3), round(resonance, 3)

# =============================================================================
# EEG / BIOMETRIC
# =============================================================================

def eeg_to_emotion(bands: dict) -> EmotionVector:
    """Convert EEG bands to emotion vector."""
    a = bands.get("alpha", 0.5)
    b = bands.get("beta", 0.5)
    t = bands.get("theta", 0.5)
    g = bands.get("gamma", 0.5)
    
    return EmotionVector(
        valence=max(-1, min(1, (a - b) * 0.5 + 0.5 * (1 - t))),
        arousal=max(0, min(1, b / (a + 0.001) / 2)),
        dominance=max(0, min(1, g / (t + 0.001) / 2))
    )

def bio_to_emotion(bio: dict) -> EmotionVector:
    """Convert biometric signals to emotion vector."""
    hr_norm = (bio.get("hr", 75) - 60) / 60
    hrv = bio.get("hrv", 0.5)
    eda = bio.get("eda", 0.5)
    
    return EmotionVector(
        valence=hrv - eda * 0.5,
        arousal=max(0, min(1, hr_norm)),
        dominance=0.5 + hrv * 0.3
    )

# =============================================================================
# ENERGY CURVES
# =============================================================================

def energy_curve(curve_type: str, points: int = 8) -> List[float]:
    """Generate energy curve for transitions."""
    curves = {
        "build": lambda t: t ** 1.5,
        "drop": lambda t: 1 - (1-t)**2,
        "breakdown": lambda t: 1 - t,
        "sustain": lambda t: 0.7,
        "swell": lambda t: 0.5 + 0.3 * math.sin(t * math.pi),
        "impact": lambda t: 1.0 if t > 0.9 else t * 0.5,
    }
    f = curves.get(curve_type, lambda t: t)
    return [f(i / (points - 1)) for i in range(points)]

# =============================================================================
# MIDI CC MAPPING
# =============================================================================

def emotion_to_cc(e: EmotionVector) -> Dict[int, int]:
    """Convert emotion to MIDI CC values."""
    return {
        1: int(e.arousal * 127),       # Mod wheel
        7: int(e.dominance * 127),     # Volume
        74: int((e.valence + 1) * 63.5), # Brightness
        91: int((1 - e.arousal) * 127)  # Reverb
    }
