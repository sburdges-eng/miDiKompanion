"""
FX Engine - Emotion-Based Audio Effects System

Maps emotions to audio effects and embeds FX data in MIDI tracks.
Displays FX chains like DAW channel strips.

FX Categories:
- Time-based: Reverb, Delay, Echo
- Modulation: Chorus, Flanger, Phaser, Tremolo
- Dynamics: Compression, Saturation, Distortion
- Filter: Low-pass, High-pass, Band-pass, EQ
- Special: Bitcrusher, Tape, Vinyl, Shimmer
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum, auto
import json
import mido


# =============================================================================
# FX DEFINITIONS
# =============================================================================

class FXCategory(Enum):
    TIME = auto()
    MODULATION = auto()
    DYNAMICS = auto()
    FILTER = auto()
    SPECIAL = auto()


class FXType(Enum):
    # Time-based
    REVERB_HALL = "reverb_hall"
    REVERB_ROOM = "reverb_room"
    REVERB_PLATE = "reverb_plate"
    REVERB_SPRING = "reverb_spring"
    REVERB_SHIMMER = "reverb_shimmer"
    DELAY_QUARTER = "delay_quarter"
    DELAY_EIGHTH = "delay_eighth"
    DELAY_DOTTED = "delay_dotted"
    DELAY_PINGPONG = "delay_pingpong"
    DELAY_TAPE = "delay_tape"
    DELAY_SLAPBACK = "delay_slapback"
    
    # Modulation
    CHORUS_LIGHT = "chorus_light"
    CHORUS_DEEP = "chorus_deep"
    FLANGER = "flanger"
    PHASER = "phaser"
    TREMOLO = "tremolo"
    VIBRATO = "vibrato"
    ROTARY = "rotary"
    
    # Dynamics
    COMP_GENTLE = "comp_gentle"
    COMP_PUNCHY = "comp_punchy"
    COMP_SQUASH = "comp_squash"
    SATURATION_WARM = "saturation_warm"
    SATURATION_GRIT = "saturation_grit"
    DISTORTION_LIGHT = "distortion_light"
    DISTORTION_HEAVY = "distortion_heavy"
    FUZZ = "fuzz"
    
    # Filter
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    EQ_WARMTH = "eq_warmth"
    EQ_AIR = "eq_air"
    EQ_SCOOP = "eq_scoop"
    EQ_PRESENCE = "eq_presence"
    
    # Special
    BITCRUSHER = "bitcrusher"
    TAPE_SATURATION = "tape_saturation"
    VINYL_CRACKLE = "vinyl_crackle"
    LOFI_DEGRADE = "lofi_degrade"
    SHIMMER = "shimmer"
    GRANULAR = "granular"
    FREEZE = "freeze"
    NONE = "none"


@dataclass
class FXParameter:
    """Single FX parameter with range and default."""
    name: str
    value: float  # 0.0 - 1.0 normalized
    min_val: float = 0.0
    max_val: float = 1.0
    display_name: str = ""
    unit: str = ""
    midi_cc: Optional[int] = None
    
    def get_display_value(self) -> str:
        actual = self.min_val + (self.value * (self.max_val - self.min_val))
        if self.unit == "ms":
            return f"{actual:.0f}ms"
        elif self.unit == "Hz":
            return f"{actual:.0f}Hz"
        elif self.unit == "dB":
            return f"{actual:+.1f}dB"
        elif self.unit == "%":
            return f"{actual:.0f}%"
        else:
            return f"{actual:.2f}"


@dataclass
class FXInstance:
    """An instance of an effect with parameters."""
    fx_type: FXType
    enabled: bool = True
    wet_dry: float = 0.5  # 0.0 = dry, 1.0 = wet
    parameters: Dict[str, FXParameter] = field(default_factory=dict)
    
    # DAW display info
    display_name: str = ""
    category: FXCategory = FXCategory.TIME
    color: str = "#666666"
    
    def to_dict(self) -> Dict:
        return {
            "fx_type": self.fx_type.value,
            "enabled": self.enabled,
            "wet_dry": self.wet_dry,
            "display_name": self.display_name,
            "category": self.category.name,
            "color": self.color,
            "parameters": {
                k: {"value": v.value, "display": v.get_display_value()}
                for k, v in self.parameters.items()
            }
        }


@dataclass
class FXChain:
    """Complete FX chain for a track."""
    name: str = "FX Chain"
    effects: List[FXInstance] = field(default_factory=list)
    bypass_all: bool = False
    
    def add_effect(self, fx: FXInstance) -> None:
        self.effects.append(fx)
    
    def remove_effect(self, index: int) -> None:
        if 0 <= index < len(self.effects):
            self.effects.pop(index)
    
    def move_effect(self, from_idx: int, to_idx: int) -> None:
        if 0 <= from_idx < len(self.effects) and 0 <= to_idx < len(self.effects):
            fx = self.effects.pop(from_idx)
            self.effects.insert(to_idx, fx)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "bypass_all": self.bypass_all,
            "effects": [fx.to_dict() for fx in self.effects]
        }
    
    def to_display_string(self, compact: bool = False) -> str:
        """Format for DAW-style display."""
        if not self.effects:
            return "[No FX]"
        
        if compact:
            return " → ".join(
                f"{'⊘' if not fx.enabled else '●'}{fx.display_name}"
                for fx in self.effects
            )
        
        lines = [f"╔══ {self.name} {'[BYPASSED]' if self.bypass_all else ''} ══╗"]
        for i, fx in enumerate(self.effects):
            status = "○" if not fx.enabled else "●"
            wet = f"{fx.wet_dry*100:.0f}%"
            lines.append(f"║ {i+1}. {status} {fx.display_name:<15} [{wet:>4}] ║")
            for param_name, param in fx.parameters.items():
                lines.append(f"║    ├─ {param_name}: {param.get_display_value():<8} ║")
        lines.append("╚" + "═" * 36 + "╝")
        return "\n".join(lines)


# =============================================================================
# FX FACTORY - Create pre-configured effects
# =============================================================================

class FXFactory:
    """Factory to create pre-configured FX instances."""
    
    @staticmethod
    def create(fx_type: FXType, wet_dry: float = 0.5) -> FXInstance:
        """Create an FX instance with default parameters."""
        
        configs = {
            # REVERBS
            FXType.REVERB_HALL: {
                "display_name": "Hall Reverb",
                "category": FXCategory.TIME,
                "color": "#4A90D9",
                "params": {
                    "decay": FXParameter("decay", 0.7, 0.5, 10.0, "Decay Time", "s"),
                    "predelay": FXParameter("predelay", 0.3, 0, 100, "Pre-Delay", "ms"),
                    "damping": FXParameter("damping", 0.5, 0, 100, "Damping", "%"),
                    "size": FXParameter("size", 0.8, 0, 100, "Size", "%"),
                }
            },
            FXType.REVERB_ROOM: {
                "display_name": "Room Reverb",
                "category": FXCategory.TIME,
                "color": "#5A9FE9",
                "params": {
                    "decay": FXParameter("decay", 0.3, 0.1, 2.0, "Decay Time", "s"),
                    "predelay": FXParameter("predelay", 0.1, 0, 50, "Pre-Delay", "ms"),
                    "damping": FXParameter("damping", 0.6, 0, 100, "Damping", "%"),
                }
            },
            FXType.REVERB_PLATE: {
                "display_name": "Plate Reverb",
                "category": FXCategory.TIME,
                "color": "#6AAFFA",
                "params": {
                    "decay": FXParameter("decay", 0.5, 0.3, 5.0, "Decay Time", "s"),
                    "brightness": FXParameter("brightness", 0.7, 0, 100, "Brightness", "%"),
                }
            },
            FXType.REVERB_SHIMMER: {
                "display_name": "Shimmer Verb",
                "category": FXCategory.SPECIAL,
                "color": "#9B59B6",
                "params": {
                    "decay": FXParameter("decay", 0.8, 1.0, 15.0, "Decay Time", "s"),
                    "pitch_shift": FXParameter("pitch_shift", 0.5, -12, 12, "Pitch Shift", "st"),
                    "shimmer_amt": FXParameter("shimmer_amt", 0.6, 0, 100, "Shimmer", "%"),
                }
            },
            
            # DELAYS
            FXType.DELAY_QUARTER: {
                "display_name": "1/4 Delay",
                "category": FXCategory.TIME,
                "color": "#27AE60",
                "params": {
                    "feedback": FXParameter("feedback", 0.4, 0, 95, "Feedback", "%"),
                    "filter": FXParameter("filter", 0.5, 500, 15000, "Filter", "Hz"),
                }
            },
            FXType.DELAY_EIGHTH: {
                "display_name": "1/8 Delay",
                "category": FXCategory.TIME,
                "color": "#2ECC71",
                "params": {
                    "feedback": FXParameter("feedback", 0.35, 0, 95, "Feedback", "%"),
                    "filter": FXParameter("filter", 0.6, 500, 15000, "Filter", "Hz"),
                }
            },
            FXType.DELAY_DOTTED: {
                "display_name": "Dotted Delay",
                "category": FXCategory.TIME,
                "color": "#58D68D",
                "params": {
                    "feedback": FXParameter("feedback", 0.45, 0, 95, "Feedback", "%"),
                    "filter": FXParameter("filter", 0.55, 500, 15000, "Filter", "Hz"),
                }
            },
            FXType.DELAY_PINGPONG: {
                "display_name": "Ping Pong",
                "category": FXCategory.TIME,
                "color": "#1ABC9C",
                "params": {
                    "feedback": FXParameter("feedback", 0.5, 0, 95, "Feedback", "%"),
                    "spread": FXParameter("spread", 0.8, 0, 100, "Stereo Spread", "%"),
                }
            },
            FXType.DELAY_TAPE: {
                "display_name": "Tape Delay",
                "category": FXCategory.TIME,
                "color": "#E67E22",
                "params": {
                    "feedback": FXParameter("feedback", 0.55, 0, 95, "Feedback", "%"),
                    "wow_flutter": FXParameter("wow_flutter", 0.3, 0, 100, "Wow/Flutter", "%"),
                    "saturation": FXParameter("saturation", 0.4, 0, 100, "Saturation", "%"),
                }
            },
            FXType.DELAY_SLAPBACK: {
                "display_name": "Slapback",
                "category": FXCategory.TIME,
                "color": "#3498DB",
                "params": {
                    "time": FXParameter("time", 0.5, 50, 150, "Time", "ms"),
                    "feedback": FXParameter("feedback", 0.1, 0, 30, "Feedback", "%"),
                }
            },
            
            # MODULATION
            FXType.CHORUS_LIGHT: {
                "display_name": "Light Chorus",
                "category": FXCategory.MODULATION,
                "color": "#E74C3C",
                "params": {
                    "rate": FXParameter("rate", 0.3, 0.1, 5.0, "Rate", "Hz"),
                    "depth": FXParameter("depth", 0.3, 0, 100, "Depth", "%"),
                    "voices": FXParameter("voices", 0.5, 2, 4, "Voices", ""),
                }
            },
            FXType.CHORUS_DEEP: {
                "display_name": "Deep Chorus",
                "category": FXCategory.MODULATION,
                "color": "#C0392B",
                "params": {
                    "rate": FXParameter("rate", 0.5, 0.1, 5.0, "Rate", "Hz"),
                    "depth": FXParameter("depth", 0.7, 0, 100, "Depth", "%"),
                    "voices": FXParameter("voices", 0.8, 2, 6, "Voices", ""),
                }
            },
            FXType.FLANGER: {
                "display_name": "Flanger",
                "category": FXCategory.MODULATION,
                "color": "#9B59B6",
                "params": {
                    "rate": FXParameter("rate", 0.4, 0.05, 2.0, "Rate", "Hz"),
                    "depth": FXParameter("depth", 0.6, 0, 100, "Depth", "%"),
                    "feedback": FXParameter("feedback", 0.5, -100, 100, "Feedback", "%"),
                }
            },
            FXType.PHASER: {
                "display_name": "Phaser",
                "category": FXCategory.MODULATION,
                "color": "#8E44AD",
                "params": {
                    "rate": FXParameter("rate", 0.3, 0.05, 4.0, "Rate", "Hz"),
                    "depth": FXParameter("depth", 0.5, 0, 100, "Depth", "%"),
                    "stages": FXParameter("stages", 0.5, 2, 12, "Stages", ""),
                }
            },
            FXType.TREMOLO: {
                "display_name": "Tremolo",
                "category": FXCategory.MODULATION,
                "color": "#D35400",
                "params": {
                    "rate": FXParameter("rate", 0.5, 1.0, 12.0, "Rate", "Hz"),
                    "depth": FXParameter("depth", 0.6, 0, 100, "Depth", "%"),
                    "shape": FXParameter("shape", 0.5, 0, 100, "Shape", "%"),
                }
            },
            FXType.VIBRATO: {
                "display_name": "Vibrato",
                "category": FXCategory.MODULATION,
                "color": "#E67E22",
                "params": {
                    "rate": FXParameter("rate", 0.5, 2.0, 8.0, "Rate", "Hz"),
                    "depth": FXParameter("depth", 0.4, 0, 100, "Depth", "%"),
                }
            },
            
            # DYNAMICS
            FXType.COMP_GENTLE: {
                "display_name": "Gentle Comp",
                "category": FXCategory.DYNAMICS,
                "color": "#F39C12",
                "params": {
                    "threshold": FXParameter("threshold", 0.4, -40, 0, "Threshold", "dB"),
                    "ratio": FXParameter("ratio", 0.2, 1, 8, "Ratio", ":1"),
                    "attack": FXParameter("attack", 0.3, 1, 100, "Attack", "ms"),
                    "release": FXParameter("release", 0.5, 50, 500, "Release", "ms"),
                }
            },
            FXType.COMP_PUNCHY: {
                "display_name": "Punchy Comp",
                "category": FXCategory.DYNAMICS,
                "color": "#E74C3C",
                "params": {
                    "threshold": FXParameter("threshold", 0.5, -40, 0, "Threshold", "dB"),
                    "ratio": FXParameter("ratio", 0.5, 1, 20, "Ratio", ":1"),
                    "attack": FXParameter("attack", 0.2, 0.1, 50, "Attack", "ms"),
                    "release": FXParameter("release", 0.4, 30, 300, "Release", "ms"),
                }
            },
            FXType.SATURATION_WARM: {
                "display_name": "Warm Sat",
                "category": FXCategory.DYNAMICS,
                "color": "#E67E22",
                "params": {
                    "drive": FXParameter("drive", 0.3, 0, 100, "Drive", "%"),
                    "tone": FXParameter("tone", 0.4, 0, 100, "Tone", "%"),
                }
            },
            FXType.SATURATION_GRIT: {
                "display_name": "Grit",
                "category": FXCategory.DYNAMICS,
                "color": "#D35400",
                "params": {
                    "drive": FXParameter("drive", 0.6, 0, 100, "Drive", "%"),
                    "tone": FXParameter("tone", 0.6, 0, 100, "Tone", "%"),
                }
            },
            FXType.DISTORTION_LIGHT: {
                "display_name": "Light Dist",
                "category": FXCategory.DYNAMICS,
                "color": "#C0392B",
                "params": {
                    "gain": FXParameter("gain", 0.4, 0, 100, "Gain", "%"),
                    "tone": FXParameter("tone", 0.5, 0, 100, "Tone", "%"),
                }
            },
            FXType.DISTORTION_HEAVY: {
                "display_name": "Heavy Dist",
                "category": FXCategory.DYNAMICS,
                "color": "#922B21",
                "params": {
                    "gain": FXParameter("gain", 0.8, 0, 100, "Gain", "%"),
                    "tone": FXParameter("tone", 0.6, 0, 100, "Tone", "%"),
                }
            },
            FXType.FUZZ: {
                "display_name": "Fuzz",
                "category": FXCategory.DYNAMICS,
                "color": "#641E16",
                "params": {
                    "fuzz": FXParameter("fuzz", 0.7, 0, 100, "Fuzz", "%"),
                    "tone": FXParameter("tone", 0.5, 0, 100, "Tone", "%"),
                }
            },
            
            # FILTER
            FXType.LOWPASS: {
                "display_name": "Low Pass",
                "category": FXCategory.FILTER,
                "color": "#3498DB",
                "params": {
                    "cutoff": FXParameter("cutoff", 0.5, 100, 20000, "Cutoff", "Hz"),
                    "resonance": FXParameter("resonance", 0.3, 0, 100, "Resonance", "%"),
                }
            },
            FXType.HIGHPASS: {
                "display_name": "High Pass",
                "category": FXCategory.FILTER,
                "color": "#2980B9",
                "params": {
                    "cutoff": FXParameter("cutoff", 0.2, 20, 5000, "Cutoff", "Hz"),
                    "resonance": FXParameter("resonance", 0.3, 0, 100, "Resonance", "%"),
                }
            },
            FXType.EQ_WARMTH: {
                "display_name": "EQ Warmth",
                "category": FXCategory.FILTER,
                "color": "#E67E22",
                "params": {
                    "low_boost": FXParameter("low_boost", 0.4, 0, 12, "Low Boost", "dB"),
                    "high_cut": FXParameter("high_cut", 0.3, 5000, 20000, "High Cut", "Hz"),
                }
            },
            FXType.EQ_AIR: {
                "display_name": "EQ Air",
                "category": FXCategory.FILTER,
                "color": "#85C1E9",
                "params": {
                    "high_shelf": FXParameter("high_shelf", 0.5, 0, 8, "High Shelf", "dB"),
                    "freq": FXParameter("freq", 0.7, 8000, 16000, "Frequency", "Hz"),
                }
            },
            
            # SPECIAL
            FXType.BITCRUSHER: {
                "display_name": "Bitcrusher",
                "category": FXCategory.SPECIAL,
                "color": "#9B59B6",
                "params": {
                    "bits": FXParameter("bits", 0.5, 4, 16, "Bit Depth", ""),
                    "sample_rate": FXParameter("sample_rate", 0.5, 1000, 44100, "Sample Rate", "Hz"),
                }
            },
            FXType.TAPE_SATURATION: {
                "display_name": "Tape",
                "category": FXCategory.SPECIAL,
                "color": "#8B4513",
                "params": {
                    "saturation": FXParameter("saturation", 0.4, 0, 100, "Saturation", "%"),
                    "hiss": FXParameter("hiss", 0.2, 0, 100, "Hiss", "%"),
                    "wow_flutter": FXParameter("wow_flutter", 0.2, 0, 100, "Wow/Flutter", "%"),
                }
            },
            FXType.VINYL_CRACKLE: {
                "display_name": "Vinyl",
                "category": FXCategory.SPECIAL,
                "color": "#2C3E50",
                "params": {
                    "crackle": FXParameter("crackle", 0.3, 0, 100, "Crackle", "%"),
                    "dust": FXParameter("dust", 0.2, 0, 100, "Dust", "%"),
                    "warp": FXParameter("warp", 0.1, 0, 100, "Warp", "%"),
                }
            },
            FXType.LOFI_DEGRADE: {
                "display_name": "Lo-Fi",
                "category": FXCategory.SPECIAL,
                "color": "#7F8C8D",
                "params": {
                    "noise": FXParameter("noise", 0.3, 0, 100, "Noise", "%"),
                    "flutter": FXParameter("flutter", 0.25, 0, 100, "Flutter", "%"),
                    "filter": FXParameter("filter", 0.4, 500, 8000, "LP Filter", "Hz"),
                }
            },
            FXType.GRANULAR: {
                "display_name": "Granular",
                "category": FXCategory.SPECIAL,
                "color": "#1ABC9C",
                "params": {
                    "grain_size": FXParameter("grain_size", 0.5, 10, 500, "Grain Size", "ms"),
                    "density": FXParameter("density", 0.6, 0, 100, "Density", "%"),
                    "pitch_rand": FXParameter("pitch_rand", 0.3, 0, 100, "Pitch Rand", "%"),
                }
            },
            FXType.NONE: {
                "display_name": "Bypass",
                "category": FXCategory.SPECIAL,
                "color": "#95A5A6",
                "params": {}
            },
        }
        
        config = configs.get(fx_type, configs[FXType.NONE])
        
        return FXInstance(
            fx_type=fx_type,
            enabled=True,
            wet_dry=wet_dry,
            display_name=config["display_name"],
            category=config["category"],
            color=config["color"],
            parameters=config["params"],
        )


# =============================================================================
# EMOTION → FX MAPPING
# =============================================================================

EMOTION_FX_PRESETS: Dict[str, Dict] = {
    "grief": {
        "description": "Spacious, enveloping, washed out",
        "chain": [
            (FXType.REVERB_HALL, 0.6),
            (FXType.DELAY_QUARTER, 0.3),
            (FXType.EQ_WARMTH, 0.4),
            (FXType.COMP_GENTLE, 0.5),
        ],
        "characteristics": [
            "Long reverb tails = lingering sadness",
            "Warmth = comfort in sorrow",
            "Gentle compression = vulnerability",
        ]
    },
    "hope": {
        "description": "Bright, open, ascending",
        "chain": [
            (FXType.REVERB_PLATE, 0.4),
            (FXType.DELAY_DOTTED, 0.35),
            (FXType.CHORUS_LIGHT, 0.25),
            (FXType.EQ_AIR, 0.3),
        ],
        "characteristics": [
            "Plate reverb = clarity with space",
            "Dotted delays = forward momentum",
            "Light chorus = shimmer/life",
            "Air EQ = brightness/optimism",
        ]
    },
    "rage": {
        "description": "Aggressive, compressed, distorted",
        "chain": [
            (FXType.DISTORTION_HEAVY, 0.6),
            (FXType.COMP_PUNCHY, 0.7),
            (FXType.REVERB_ROOM, 0.2),
            (FXType.EQ_PRESENCE, 0.5),
        ],
        "characteristics": [
            "Distortion = aggression",
            "Heavy compression = power/loudness",
            "Short room = tight/controlled",
            "Presence = cutting through",
        ]
    },
    "fear": {
        "description": "Unsettling, disorienting, tense",
        "chain": [
            (FXType.REVERB_SHIMMER, 0.4),
            (FXType.PHASER, 0.3),
            (FXType.TREMOLO, 0.4),
            (FXType.HIGHPASS, 0.2),
        ],
        "characteristics": [
            "Shimmer = otherworldly unease",
            "Phaser = disorientation",
            "Tremolo = anxiety/instability",
            "High-pass = thin/exposed",
        ]
    },
    "joy": {
        "description": "Bright, bouncy, energetic",
        "chain": [
            (FXType.REVERB_ROOM, 0.3),
            (FXType.DELAY_SLAPBACK, 0.25),
            (FXType.SATURATION_WARM, 0.3),
            (FXType.CHORUS_LIGHT, 0.2),
        ],
        "characteristics": [
            "Room reverb = intimate space",
            "Slapback = energy/bounce",
            "Warm saturation = fullness",
            "Light chorus = life/movement",
        ]
    },
    "nostalgia": {
        "description": "Warm, degraded, memory-like",
        "chain": [
            (FXType.TAPE_SATURATION, 0.5),
            (FXType.VINYL_CRACKLE, 0.25),
            (FXType.REVERB_SPRING, 0.35),
            (FXType.LOWPASS, 0.3),
        ],
        "characteristics": [
            "Tape = warmth of old recordings",
            "Vinyl = imperfect memory",
            "Spring reverb = vintage character",
            "Low-pass = distance from present",
        ]
    },
    "melancholy": {
        "description": "Subdued, introspective, gentle weight",
        "chain": [
            (FXType.REVERB_HALL, 0.5),
            (FXType.DELAY_TAPE, 0.35),
            (FXType.SATURATION_WARM, 0.25),
            (FXType.LOWPASS, 0.2),
        ],
        "characteristics": [
            "Hall reverb = vast emptiness",
            "Tape delay = degraded echoes",
            "Warmth = soft edges",
            "Low-pass = muted/withdrawn",
        ]
    },
    "tension": {
        "description": "Building, unsettled, anticipatory",
        "chain": [
            (FXType.REVERB_ROOM, 0.2),
            (FXType.FLANGER, 0.3),
            (FXType.COMP_SQUASH, 0.5),
            (FXType.BITCRUSHER, 0.15),
        ],
        "characteristics": [
            "Tight room = claustrophobic",
            "Flanger = movement/unease",
            "Squash compression = pressure",
            "Light bitcrush = digital anxiety",
        ]
    },
    "peace": {
        "description": "Spacious, clear, serene",
        "chain": [
            (FXType.REVERB_HALL, 0.45),
            (FXType.DELAY_QUARTER, 0.2),
            (FXType.CHORUS_LIGHT, 0.15),
            (FXType.EQ_AIR, 0.2),
        ],
        "characteristics": [
            "Hall = open space",
            "Gentle delays = calm reflection",
            "Light chorus = subtle shimmer",
            "Air = clarity/openness",
        ]
    },
    "lofi_bedroom": {
        "description": "Intimate, imperfect, tape-degraded",
        "chain": [
            (FXType.LOFI_DEGRADE, 0.5),
            (FXType.TAPE_SATURATION, 0.4),
            (FXType.REVERB_ROOM, 0.3),
            (FXType.COMP_GENTLE, 0.4),
        ],
        "characteristics": [
            "Lo-fi processing = bedroom aesthetic",
            "Tape saturation = warmth",
            "Room = intimate space",
            "Gentle comp = cohesion",
        ]
    },
    "ethereal": {
        "description": "Dreamy, washed, otherworldly",
        "chain": [
            (FXType.REVERB_SHIMMER, 0.7),
            (FXType.DELAY_PINGPONG, 0.4),
            (FXType.CHORUS_DEEP, 0.35),
            (FXType.GRANULAR, 0.25),
        ],
        "characteristics": [
            "Shimmer = transcendent quality",
            "Ping pong = spatial movement",
            "Deep chorus = thick texture",
            "Granular = particle dissolution",
        ]
    },
    "defiance": {
        "description": "Bold, cutting, unapologetic",
        "chain": [
            (FXType.SATURATION_GRIT, 0.5),
            (FXType.COMP_PUNCHY, 0.6),
            (FXType.DELAY_SLAPBACK, 0.3),
            (FXType.EQ_PRESENCE, 0.45),
        ],
        "characteristics": [
            "Grit = edge/attitude",
            "Punchy compression = in-your-face",
            "Slapback = energy",
            "Presence = cutting through",
        ]
    },
}


class EmotionFXEngine:
    """
    Maps emotions to FX chains and manages FX state.
    
    Usage:
        engine = EmotionFXEngine()
        chain = engine.get_chain_for_emotion("grief")
        print(chain.to_display_string())
    """
    
    def __init__(self):
        self.factory = FXFactory()
        self.presets = EMOTION_FX_PRESETS
        self.custom_chains: Dict[str, FXChain] = {}
    
    def get_chain_for_emotion(
        self,
        emotion: str,
        intensity: float = 0.5
    ) -> FXChain:
        """Get FX chain for an emotion with intensity scaling."""
        emotion_lower = emotion.lower()
        
        if emotion_lower not in self.presets:
            # Try partial match
            for key in self.presets:
                if key in emotion_lower or emotion_lower in key:
                    emotion_lower = key
                    break
            else:
                emotion_lower = "melancholy"  # default
        
        preset = self.presets[emotion_lower]
        chain = FXChain(name=f"{emotion.title()} FX")
        
        for fx_type, base_wet in preset["chain"]:
            scaled_wet = base_wet * (0.5 + intensity * 0.5)
            scaled_wet = min(1.0, scaled_wet)
            
            fx = self.factory.create(fx_type, wet_dry=scaled_wet)
            chain.add_effect(fx)
        
        return chain
    
    def get_emotion_presets(self) -> List[str]:
        """Get list of available emotion presets."""
        return list(self.presets.keys())
    
    def get_preset_info(self, emotion: str) -> Optional[Dict]:
        """Get info about an emotion preset."""
        return self.presets.get(emotion.lower())
    
    def create_custom_chain(self, name: str, fx_types: List[Tuple[FXType, float]]) -> FXChain:
        """Create a custom FX chain."""
        chain = FXChain(name=name)
        for fx_type, wet in fx_types:
            chain.add_effect(self.factory.create(fx_type, wet_dry=wet))
        self.custom_chains[name] = chain
        return chain
    
    def blend_chains(
        self,
        chain_a: FXChain,
        chain_b: FXChain,
        blend: float = 0.5
    ) -> FXChain:
        """Blend two FX chains (for transitioning between emotions)."""
        result = FXChain(name=f"Blend ({blend:.0%})")
        
        # Combine unique effects from both chains
        seen_types = set()
        
        for fx in chain_a.effects:
            if fx.fx_type not in seen_types:
                new_fx = self.factory.create(fx.fx_type, fx.wet_dry * (1 - blend))
                result.add_effect(new_fx)
                seen_types.add(fx.fx_type)
        
        for fx in chain_b.effects:
            if fx.fx_type not in seen_types:
                new_fx = self.factory.create(fx.fx_type, fx.wet_dry * blend)
                result.add_effect(new_fx)
        
        return result


# =============================================================================
# MIDI CC MAPPING FOR FX AUTOMATION
# =============================================================================

MIDI_CC_FX_MAP = {
    # Standard CC numbers for FX control
    1: "mod_wheel",          # Often mapped to filter/vibrato
    7: "volume",
    10: "pan",
    11: "expression",
    64: "sustain",
    
    # Custom FX CCs (can be remapped in DAW)
    70: "fx_wet_dry",        # Master FX wet/dry
    71: "reverb_size",
    72: "reverb_decay",
    73: "delay_feedback",
    74: "filter_cutoff",
    75: "filter_resonance",
    76: "chorus_depth",
    77: "distortion_gain",
    78: "compression_threshold",
    79: "tremolo_rate",
    
    # FX bypass/enable (use 0-63 = off, 64-127 = on)
    80: "fx_slot_1_bypass",
    81: "fx_slot_2_bypass",
    82: "fx_slot_3_bypass",
    83: "fx_slot_4_bypass",
    
    # Additional automation
    91: "reverb_send",
    92: "tremolo_depth",
    93: "chorus_send",
    94: "delay_send",
}


def generate_fx_cc_automation(
    chain: FXChain,
    channel: int = 0
) -> List[mido.Message]:
    """Generate MIDI CC messages to initialize FX settings."""
    messages = []
    
    # Master wet/dry
    avg_wet = sum(fx.wet_dry for fx in chain.effects) / max(len(chain.effects), 1)
    messages.append(mido.Message('control_change', channel=channel, control=70, value=int(avg_wet * 127)))
    
    # Per-effect parameters
    for i, fx in enumerate(chain.effects[:4]):  # Max 4 slots for CC control
        bypass_cc = 80 + i
        bypass_val = 0 if not fx.enabled else 127
        messages.append(mido.Message('control_change', channel=channel, control=bypass_cc, value=bypass_val))
        
        # Effect-specific CCs
        if fx.fx_type in [FXType.REVERB_HALL, FXType.REVERB_ROOM, FXType.REVERB_PLATE]:
            if "size" in fx.parameters:
                messages.append(mido.Message('control_change', channel=channel, control=71, 
                                            value=int(fx.parameters["size"].value * 127)))
            if "decay" in fx.parameters:
                messages.append(mido.Message('control_change', channel=channel, control=72, 
                                            value=int(fx.parameters["decay"].value * 127)))
        
        elif fx.fx_type in [FXType.DELAY_QUARTER, FXType.DELAY_EIGHTH, FXType.DELAY_TAPE]:
            if "feedback" in fx.parameters:
                messages.append(mido.Message('control_change', channel=channel, control=73, 
                                            value=int(fx.parameters["feedback"].value * 127)))
        
        elif fx.fx_type in [FXType.LOWPASS, FXType.HIGHPASS]:
            if "cutoff" in fx.parameters:
                messages.append(mido.Message('control_change', channel=channel, control=74, 
                                            value=int(fx.parameters["cutoff"].value * 127)))
            if "resonance" in fx.parameters:
                messages.append(mido.Message('control_change', channel=channel, control=75, 
                                            value=int(fx.parameters["resonance"].value * 127)))
    
    return messages


# =============================================================================
# MIDI TRACK FX METADATA
# =============================================================================

@dataclass
class TrackFXState:
    """Complete FX state for a MIDI track (for display/storage)."""
    track_name: str
    channel: int
    fx_chain: FXChain
    
    # Display state
    show_expanded: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "track_name": self.track_name,
            "channel": self.channel,
            "fx_chain": self.fx_chain.to_dict(),
            "show_expanded": self.show_expanded,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrackFXState':
        chain = FXChain(name=data["fx_chain"]["name"])
        for fx_data in data["fx_chain"]["effects"]:
            fx_type = FXType(fx_data["fx_type"])
            fx = FXFactory.create(fx_type, fx_data["wet_dry"])
            fx.enabled = fx_data["enabled"]
            chain.add_effect(fx)
        
        return cls(
            track_name=data["track_name"],
            channel=data["channel"],
            fx_chain=chain,
            show_expanded=data.get("show_expanded", True),
        )
    
    def to_display_string(self) -> str:
        """DAW-style track display with FX."""
        header = f"┌─ {self.track_name} [Ch.{self.channel}] ─┐"
        chain_str = self.fx_chain.to_display_string(compact=False)
        return f"{header}\n{chain_str}"


def embed_fx_in_midi(
    midi_file: mido.MidiFile,
    track_fx_states: List[TrackFXState]
) -> mido.MidiFile:
    """
    Embed FX metadata and CC automation into a MIDI file.
    
    FX data is stored as:
    1. MIDI CC messages at track start
    2. Text meta-event with JSON FX chain data
    """
    for track_state in track_fx_states:
        if track_state.channel < len(midi_file.tracks):
            track = midi_file.tracks[track_state.channel]
            
            # Add FX chain as text metadata
            fx_json = json.dumps(track_state.to_dict())
            meta_msg = mido.MetaMessage('text', text=f"KELLY_FX:{fx_json}")
            track.insert(0, meta_msg)
            
            # Add CC automation
            cc_messages = generate_fx_cc_automation(track_state.fx_chain, track_state.channel)
            for i, msg in enumerate(cc_messages):
                track.insert(1 + i, msg)
    
    return midi_file


def extract_fx_from_midi(midi_file: mido.MidiFile) -> List[TrackFXState]:
    """Extract FX metadata from MIDI file."""
    states = []
    
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == 'text' and msg.text.startswith('KELLY_FX:'):
                try:
                    fx_json = msg.text[9:]  # Remove "KELLY_FX:" prefix
                    data = json.loads(fx_json)
                    states.append(TrackFXState.from_dict(data))
                except (json.JSONDecodeError, KeyError):
                    pass
    
    return states


# =============================================================================
# DAW-STYLE FX DISPLAY
# =============================================================================

def render_channel_strip(
    track_name: str,
    channel: int,
    fx_chain: FXChain,
    volume: float = 0.8,
    pan: float = 0.5,
    muted: bool = False,
    soloed: bool = False,
) -> str:
    """Render a DAW-style channel strip with FX."""
    
    # Meter visualization
    meter_height = int(volume * 10)
    meter = "█" * meter_height + "░" * (10 - meter_height)
    
    # Pan visualization
    pan_pos = int(pan * 10)
    pan_display = "◄" + "─" * pan_pos + "●" + "─" * (10 - pan_pos) + "►"
    
    # Track status
    status = ""
    if muted:
        status += "[M] "
    if soloed:
        status += "[S] "
    
    lines = [
        "╔══════════════════════════╗",
        f"║ {track_name:<22} ║",
        f"║ Ch.{channel:<2} {status:<16} ║",
        "╠══════════════════════════╣",
        f"║ VOL │{meter}│ ║",
        f"║ PAN │{pan_display}│ ║",
        "╠══════ FX CHAIN ══════════╣",
    ]
    
    for i, fx in enumerate(fx_chain.effects):
        status_icon = "●" if fx.enabled else "○"
        wet_pct = f"{fx.wet_dry*100:3.0f}%"
        lines.append(f"║ {i+1}.{status_icon} {fx.display_name:<12} {wet_pct} ║")
    
    if not fx_chain.effects:
        lines.append("║     [No FX]              ║")
    
    lines.append("╚══════════════════════════╝")
    
    return "\n".join(lines)


def render_mixer_view(tracks: List[TrackFXState]) -> str:
    """Render full mixer view with all tracks."""
    strips = []
    for track_state in tracks:
        strip = render_channel_strip(
            track_state.track_name,
            track_state.channel,
            track_state.fx_chain,
        )
        strips.append(strip.split("\n"))
    
    # Combine horizontally
    if not strips:
        return "[No Tracks]"
    
    max_lines = max(len(s) for s in strips)
    result_lines = []
    
    for line_idx in range(max_lines):
        row_parts = []
        for strip in strips:
            if line_idx < len(strip):
                row_parts.append(strip[line_idx])
            else:
                row_parts.append(" " * 28)
        result_lines.append("  ".join(row_parts))
    
    return "\n".join(result_lines)


# =============================================================================
# MAIN API
# =============================================================================

def get_fx_for_emotion(emotion: str, intensity: float = 0.5) -> FXChain:
    """Quick API to get FX chain for emotion."""
    engine = EmotionFXEngine()
    return engine.get_chain_for_emotion(emotion, intensity)


def create_fx_from_list(fx_list: List[str], wet_dry: float = 0.5) -> FXChain:
    """Create FX chain from list of effect names."""
    chain = FXChain(name="Custom Chain")
    
    for fx_name in fx_list:
        # Try to match FXType
        fx_name_clean = fx_name.lower().replace(" ", "_").replace("-", "_")
        
        for fx_type in FXType:
            if fx_type.value == fx_name_clean or fx_name_clean in fx_type.value:
                chain.add_effect(FXFactory.create(fx_type, wet_dry))
                break
    
    return chain


def list_all_fx() -> Dict[str, List[str]]:
    """List all available FX by category."""
    result = {}
    for fx_type in FXType:
        if fx_type == FXType.NONE:
            continue
        fx = FXFactory.create(fx_type)
        category = fx.category.name
        if category not in result:
            result[category] = []
        result[category].append(fx_type.value)
    return result


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("KELLY FX ENGINE - Emotion-Based Audio Effects")
    print("=" * 60)
    
    engine = EmotionFXEngine()
    
    # Demo: Get FX for different emotions
    emotions = ["grief", "hope", "rage", "nostalgia"]
    
    for emotion in emotions:
        print(f"\n{'='*40}")
        print(f"EMOTION: {emotion.upper()}")
        print(f"{'='*40}")
        
        preset = engine.get_preset_info(emotion)
        if preset:
            print(f"Description: {preset['description']}")
            print("Characteristics:")
            for char in preset['characteristics']:
                print(f"  • {char}")
        
        chain = engine.get_chain_for_emotion(emotion, intensity=0.7)
        print(f"\n{chain.to_display_string()}")
    
    # Demo: Channel strip display
    print("\n" + "=" * 60)
    print("CHANNEL STRIP DISPLAY")
    print("=" * 60)
    
    grief_chain = engine.get_chain_for_emotion("grief", 0.8)
    print(render_channel_strip("Lead Vocal", 0, grief_chain, volume=0.75, pan=0.5))
    
    # Demo: Mixer view
    print("\n" + "=" * 60)
    print("MIXER VIEW")
    print("=" * 60)
    
    tracks = [
        TrackFXState("Piano", 0, engine.get_chain_for_emotion("grief", 0.7)),
        TrackFXState("Strings", 1, engine.get_chain_for_emotion("melancholy", 0.6)),
        TrackFXState("Bass", 2, engine.get_chain_for_emotion("peace", 0.4)),
    ]
    
    print(render_mixer_view(tracks))
    
    # Demo: List all FX
    print("\n" + "=" * 60)
    print("ALL AVAILABLE FX")
    print("=" * 60)
    
    all_fx = list_all_fx()
    for category, fx_list in all_fx.items():
        print(f"\n{category}:")
        for fx in fx_list:
            print(f"  • {fx}")
