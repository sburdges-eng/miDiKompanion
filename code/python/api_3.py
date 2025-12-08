"""
Music Brain API Server
Provides endpoints for emotional music generation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
from pathlib import Path

app = FastAPI(title="Music Brain API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load emotion thesaurus
EMOTION_THESAURUS_PATH = Path(__file__).parent.parent / "emotion_thesaurus"

class EmotionalIntent(BaseModel):
    core_wound: Optional[str] = None
    core_desire: Optional[str] = None
    emotional_intent: Optional[str] = None  # Legacy field
    technical: Optional[Dict[str, Any]] = None
    # New format: base_emotion, intensity, specific_emotion
    base_emotion: Optional[str] = None
    intensity: Optional[str] = None
    specific_emotion: Optional[str] = None

    class Config:
        # Allow extra fields for backward compatibility
        extra = "allow"

class GenerateRequest(BaseModel):
    intent: EmotionalIntent
    output_format: str = "midi"

class InterrogateRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    return {
        "service": "Music Brain API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/generate", "/interrogate", "/emotions"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/generate")
async def generate_music(request: GenerateRequest):
    """Generate music from emotional intent"""
    try:
        from music_brain.session.generator import SongGenerator
        from music_brain.emotion_mapper import map_emotion_to_music
        import tempfile
        import os
        import base64
        from pathlib import Path

        # Extract parameters from intent
        intent = request.intent
        technical = intent.technical or {}

        # Initialize variables
        mode_name = None
        progression = None
        dynamics = None

        # Check if using new format (base_emotion, intensity, specific_emotion)
        if intent.base_emotion and intent.intensity:
            # Use emotion mapper
            from music_brain.emotion_mapper import mode_to_major_minor

            music_config = map_emotion_to_music(
                base=intent.base_emotion,
                intensity=intent.intensity,
                sub=intent.specific_emotion
            )

            key = music_config["key"]
            mode_name = music_config["mode"]  # Keep original mode name for response
            mode = mode_to_major_minor(mode_name)  # Convert to major/minor for SongGenerator
            tempo = music_config["tempo"]
            progression = music_config["progression"]
            dynamics = music_config["dynamics"]

            # Override with technical params if provided
            if technical.get("key"):
                key_str = str(technical["key"]).strip()
                if " " in key_str:
                    parts = key_str.split(" ", 1)
                    key = parts[0]
                    mode_override = parts[1].lower()
                    # Convert mode override to major/minor if needed
                    if mode_override not in ["major", "minor"]:
                        mode = mode_to_major_minor(mode_override)
                        mode_name = mode_override  # Update mode_name if override provided
                    else:
                        mode = mode_override
                        mode_name = mode_override
                else:
                    key = key_str
            if technical.get("bpm"):
                tempo = float(technical["bpm"])

            genre = technical.get("genre")
            mood = intent.specific_emotion or intent.base_emotion

        else:
            # Legacy format: parse from technical.key
            key = "C"
            mode = "major"
            if technical.get("key"):
                key_str = str(technical["key"]).strip()
                if " " in key_str:
                    parts = key_str.split(" ", 1)
                    key = parts[0]
                    mode = parts[1].lower() if len(parts) > 1 else "major"
                else:
                    key = key_str

            # Extract other parameters
            tempo = technical.get("bpm")
            genre = technical.get("genre")
            mood = intent.emotional_intent or None

        # Generate song
        generator = SongGenerator()
        song = generator.generate(
            key=key,
            mode=mode,
            mood=mood,
            genre=genre,
            tempo=float(tempo) if tempo else None
        )

        # Generate MIDI file
        midi_path = None
        midi_data_base64 = None
        try:
            # Create temporary directory for MIDI files
            temp_dir = Path(tempfile.gettempdir()) / "music_brain_midi"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique filename
            import time
            timestamp = int(time.time() * 1000)
            safe_title = "".join(c for c in song.title if c.isalnum() or c in (' ', '-', '_')).strip()[:30]
            midi_filename = f"{safe_title}_{timestamp}.mid"
            midi_path = str(temp_dir / midi_filename)

            # Export to MIDI
            song.export_to_midi(midi_path)

            # Read MIDI file and encode as base64
            if os.path.exists(midi_path):
                with open(midi_path, "rb") as f:
                    midi_bytes = f.read()
                    midi_data_base64 = base64.b64encode(midi_bytes).decode("utf-8")

        except Exception as midi_error:
            # Log error but don't fail the request
            import logging
            logging.warning(f"MIDI generation failed: {midi_error}")
            midi_path = None

        # Convert to dict for JSON response using the song's to_dict method
        result = {
            "success": True,
            "intent": intent.dict(),
            "song": song.to_dict(),
            "midi_path": midi_path,
            "midi_data": midi_data_base64,
            "message": f"Generated {len(song.sections)} sections, {song.total_bars} bars at {song.tempo_bpm} BPM"
        }

        # Add music config if using new format
        if intent.base_emotion and intent.intensity:
            result["music_config"] = {
                "key": key,
                "mode": mode_name or mode,  # Return original mode name if available
                "mode_simple": mode,  # Also include major/minor version
                "tempo": tempo,
                "progression": progression,
                "dynamics": dynamics,
            }

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Session storage (in-memory, could be replaced with Redis in production)
_interrogation_sessions: Dict[str, Any] = {}


@app.post("/interrogate")
async def interrogate(request: InterrogateRequest):
    """Conversational music creation"""
    try:
        from music_brain.interrogator import (
            InterrogationSession,
            process_interrogation_message,
        )

        # Get or create session
        session_id = request.session_id
        if not session_id or session_id not in _interrogation_sessions:
            session = InterrogationSession(session_id=session_id)
            _interrogation_sessions[session.session_id] = session
            session_id = session.session_id
        else:
            session = _interrogation_sessions[session_id]

        # Process message
        result = process_interrogation_message(session, request.message)

        return {
            "success": True,
            "ready": result["ready"],
            "session_id": session_id,
            "question": result.get("question"),
            "intent": result.get("intent"),
            "profile": result.get("profile"),
            "confidence": result.get("confidence", 0.0),
            "message": result.get("message"),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/emotions")
async def get_emotions():
    """Get the full 6x6x6 emotion thesaurus"""
    try:
        emotions = {}

        for emotion_file in EMOTION_THESAURUS_PATH.glob("*.json"):
            if emotion_file.stem not in ["metadata", "blends"]:
                with open(emotion_file) as f:
                    emotions[emotion_file.stem] = json.load(f)

        blends_path = EMOTION_THESAURUS_PATH / "blends.json"
        if blends_path.exists():
            with open(blends_path) as f:
                emotions["blends"] = json.load(f)

        return {
            "success": True,
            "emotions": emotions,
            "total_nodes": 216
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/emotions/{base_emotion}")
async def get_emotion_category(base_emotion: str):
    """Get specific emotion category"""
    emotion_file = EMOTION_THESAURUS_PATH / f"{base_emotion}.json"

    if not emotion_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Emotion '{base_emotion}' not found"
        )

    try:
        with open(emotion_file) as f:
            data = json.load(f)

        return {
            "success": True,
            "emotion": base_emotion,
            "data": data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# Groove API Endpoints
# =====================

class GrooveRequest(BaseModel):
    genre: str = "pop"
    tempo: float = 120.0
    swing: float = 0.0
    humanize: float = 0.5

@app.post("/groove/generate")
async def generate_groove(request: GrooveRequest):
    """Generate a groove pattern for the specified genre"""
    try:
        from music_brain.groove import GrooveTemplate

        template = GrooveTemplate(
            genre=request.genre,
            tempo=request.tempo,
            swing=request.swing
        )

        return {
            "success": True,
            "genre": request.genre,
            "tempo": request.tempo,
            "pattern": template.to_dict() if hasattr(template, 'to_dict') else {"name": request.genre},
            "message": f"Generated {request.genre} groove at {request.tempo} BPM"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/groove/genres")
async def get_groove_genres():
    """Get available groove genres"""
    genres = [
        "pop", "rock", "funk", "soul", "jazz", "hip_hop",
        "electronic", "lo_fi", "indie", "folk", "country",
        "metal", "punk", "reggae", "latin", "blues"
    ]
    return {"success": True, "genres": genres}


# =====================
# Harmony API Endpoints
# =====================

class HarmonyRequest(BaseModel):
    key: str = "C"
    mode: str = "major"
    mood: Optional[str] = None
    bars: int = 4

@app.post("/harmony/suggest")
async def suggest_harmony(request: HarmonyRequest):
    """Suggest chord progressions based on key, mode, and mood"""
    try:
        from music_brain.session.generator import SongGenerator

        generator = SongGenerator()
        result = generator.suggest_progression(
            mood=request.mood or "neutral",
            key=request.key,
            mode=request.mode,
            bars=request.bars
        )

        return {
            "success": True,
            "suggestion": result,
            "message": f"Suggested {request.bars}-bar progression in {request.key} {request.mode}"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/harmony/progressions")
async def get_progressions():
    """Get common chord progressions database"""
    try:
        progressions_path = Path(__file__).parent / "data" / "common_progressions.json"
        if progressions_path.exists():
            with open(progressions_path) as f:
                data = json.load(f)
            return {"success": True, "progressions": data}

        # Return basic progressions if file doesn't exist
        return {
            "success": True,
            "progressions": {
                "pop": ["I", "V", "vi", "IV"],
                "sad": ["vi", "IV", "I", "V"],
                "jazz": ["ii", "V", "I"],
                "blues": ["I", "I", "I", "I", "IV", "IV", "I", "I", "V", "IV", "I", "V"],
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# Rule Breaking Endpoints
# =====================

@app.get("/rules/breaking")
async def get_rule_breaking_options():
    """Get available rule-breaking techniques"""
    try:
        from music_brain.session.intent_schema import RULE_BREAKING_EFFECTS

        if 'RULE_BREAKING_EFFECTS' in dir():
            return {"success": True, "rules": RULE_BREAKING_EFFECTS}

        # Return default rule-breaking options
        return {
            "success": True,
            "rules": {
                "harmony": {
                    "avoid_tonic_resolution": "Creates unresolved yearning",
                    "parallel_fifths": "Adds raw, primitive power",
                    "borrowed_chords": "Adds color and surprise"
                },
                "rhythm": {
                    "constant_displacement": "Creates anxiety, restlessness",
                    "tempo_drift": "Adds organic, human feel",
                    "polyrhythm": "Creates complexity and tension"
                },
                "arrangement": {
                    "buried_vocals": "Creates dissociation effect",
                    "sudden_silence": "Creates shock and space",
                    "wrong_instrument": "Creates unexpected emotion"
                },
                "production": {
                    "pitch_imperfection": "Adds emotional honesty",
                    "room_noise": "Creates intimacy",
                    "distortion_on_clean": "Adds edge and rawness"
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RuleBreakRequest(BaseModel):
    emotion: str
    intensity: str = "moderate"

@app.post("/rules/suggest")
async def suggest_rule_break(request: RuleBreakRequest):
    """Suggest rule-breaking techniques for an emotion"""
    emotion_rules = {
        "grief": ["avoid_tonic_resolution", "buried_vocals", "pitch_imperfection"],
        "anger": ["parallel_fifths", "distortion_on_clean", "constant_displacement"],
        "joy": ["sudden_silence", "tempo_drift", "borrowed_chords"],
        "fear": ["constant_displacement", "sudden_silence", "room_noise"],
        "love": ["pitch_imperfection", "borrowed_chords", "room_noise"],
    }

    suggestions = emotion_rules.get(request.emotion.lower(), ["borrowed_chords"])

    return {
        "success": True,
        "emotion": request.emotion,
        "suggestions": suggestions,
        "message": f"Suggested {len(suggestions)} rule-breaking techniques for {request.emotion}"
    }


# =====================
# Learning Module Endpoints
# =====================

@app.get("/learning/instruments")
async def get_learning_instruments():
    """Get available instruments for learning"""
    try:
        from music_brain.learning import INSTRUMENTS, get_beginner_instruments

        beginner = get_beginner_instruments()
        return {
            "success": True,
            "total": len(INSTRUMENTS),
            "beginner_friendly": [i.name for i in beginner],
            "all_instruments": [{"name": i.name, "family": i.family.value, "difficulty": i.difficulty.value} for i in INSTRUMENTS]
        }
    except ImportError:
        return {
            "success": True,
            "instruments": ["piano", "guitar", "drums", "bass", "voice"],
            "message": "Learning module not fully loaded"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/learning/curriculum/{instrument}")
async def get_curriculum(instrument: str):
    """Get learning curriculum for an instrument"""
    try:
        from music_brain.learning import CurriculumBuilder, get_instrument

        inst = get_instrument(instrument)
        if not inst:
            raise HTTPException(status_code=404, detail=f"Instrument '{instrument}' not found")

        builder = CurriculumBuilder()
        curriculum = builder.build_curriculum(inst)

        return {
            "success": True,
            "instrument": instrument,
            "curriculum": curriculum.to_dict() if hasattr(curriculum, 'to_dict') else {"phases": []}
        }
    except ImportError:
        return {
            "success": True,
            "instrument": instrument,
            "curriculum": {
                "phases": [
                    {"name": "Basics", "weeks": 4},
                    {"name": "Fundamentals", "weeks": 8},
                    {"name": "Intermediate", "weeks": 12},
                    {"name": "Advanced", "weeks": 16}
                ]
            },
            "message": "Learning module not fully loaded"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# Voice Synthesis Endpoints
# =====================

class VoiceProfile(BaseModel):
    name: str = "Natural"
    pitch: float = 0
    formant: float = 0
    breathiness: float = 20
    vibrato: float = 30
    warmth: float = 50

class VoiceSynthRequest(BaseModel):
    text: str
    profile: VoiceProfile = VoiceProfile()
    emotion: Optional[str] = None

@app.post("/voice/synthesize")
async def synthesize_voice(request: VoiceSynthRequest):
    """Synthesize vocal audio from text with voice profile"""
    try:
        # Voice synthesis placeholder - would integrate with actual TTS/vocal synth
        # For now, return a simulated response
        import hashlib
        import base64

        # Create a unique identifier for this synthesis
        text_hash = hashlib.md5(request.text.encode()).hexdigest()[:8]

        # In production, this would call actual voice synthesis:
        # - ElevenLabs API
        # - Coqui TTS
        # - VITS model
        # - Custom vocal synthesis engine

        # Simulate processing based on voice profile
        voice_params = {
            "pitch_shift": request.profile.pitch,
            "formant_shift": request.profile.formant,
            "breathiness": request.profile.breathiness / 100.0,
            "vibrato_depth": request.profile.vibrato / 100.0,
            "warmth_filter": request.profile.warmth / 100.0,
        }

        # Apply emotion modifiers if specified
        emotion_mods = {}
        if request.emotion:
            emotion_effects = {
                "grief": {"pitch_shift": -1, "breathiness": 0.5, "vibrato_depth": 0.15},
                "anger": {"pitch_shift": 2, "breathiness": 0.1, "vibrato_depth": 0.05},
                "joy": {"pitch_shift": 3, "breathiness": 0.2, "vibrato_depth": 0.4},
                "longing": {"pitch_shift": 0, "breathiness": 0.35, "vibrato_depth": 0.25},
                "peace": {"pitch_shift": -1, "breathiness": 0.3, "vibrato_depth": 0.2},
            }
            emotion_mods = emotion_effects.get(request.emotion, {})

        return {
            "success": True,
            "text": request.text,
            "voice_profile": request.profile.name,
            "emotion": request.emotion,
            "parameters_applied": {**voice_params, **emotion_mods},
            "audio_url": None,  # Would be actual audio URL in production
            "message": f"Voice synthesis request processed for '{request.profile.name}' profile",
            "note": "Voice synthesis backend integration pending - this is a placeholder response"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voice/profiles")
async def get_voice_profiles():
    """Get available voice profiles"""
    return {
        "success": True,
        "profiles": {
            "natural": {
                "name": "Natural",
                "description": "Balanced, authentic vocal character",
                "pitch": 0, "formant": 0, "breathiness": 20, "vibrato": 30, "warmth": 50
            },
            "intimate": {
                "name": "Intimate",
                "description": "Close, soft, personal feel",
                "pitch": -2, "formant": -1, "breathiness": 40, "vibrato": 15, "warmth": 70
            },
            "powerful": {
                "name": "Powerful",
                "description": "Strong, bold, commanding",
                "pitch": 2, "formant": 1, "breathiness": 10, "vibrato": 40, "warmth": 40
            },
            "ethereal": {
                "name": "Ethereal",
                "description": "Airy, floating, dreamlike",
                "pitch": 5, "formant": 3, "breathiness": 50, "vibrato": 60, "warmth": 60
            },
            "raspy": {
                "name": "Raspy",
                "description": "Textured, gravelly, raw",
                "pitch": -3, "formant": -2, "breathiness": 60, "vibrato": 20, "warmth": 30
            },
            "robotic": {
                "name": "Robotic",
                "description": "Mechanical, processed, synthetic",
                "pitch": 0, "formant": 0, "breathiness": 0, "vibrato": 0, "warmth": 20
            }
        }
    }


@app.get("/voice/emotions")
async def get_voice_emotions():
    """Get emotional styles for voice synthesis"""
    return {
        "success": True,
        "emotions": {
            "grief": {
                "description": "Fragile, breaking voice with catch in throat",
                "adjustments": {"breathiness": 50, "vibrato": 15, "warmth": 60, "pitch": -1}
            },
            "anger": {
                "description": "Tight, controlled tension with sharp edges",
                "adjustments": {"breathiness": 10, "vibrato": 5, "warmth": 20, "pitch": 2}
            },
            "joy": {
                "description": "Bright, lifted tone with natural energy",
                "adjustments": {"breathiness": 20, "vibrato": 40, "warmth": 70, "pitch": 3}
            },
            "longing": {
                "description": "Distant, reaching quality with ache",
                "adjustments": {"breathiness": 35, "vibrato": 25, "warmth": 55, "pitch": 0}
            },
            "peace": {
                "description": "Settled, grounded with gentle flow",
                "adjustments": {"breathiness": 30, "vibrato": 20, "warmth": 65, "pitch": -1}
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
