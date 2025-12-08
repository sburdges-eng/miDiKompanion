"""
Music Brain API Server
Provides endpoints for emotional music generation
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import base64
import tempfile
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
    emotional_intent: Optional[str] = None
    technical: Optional[Dict[str, Any]] = None

# Request models removed - using Dict[str, Any] directly in endpoint

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
async def generate_music(request: Request):
    """Generate music from emotional intent"""
    try:
        # Parse JSON body manually
        body = await request.json()

        from music_brain.emotion_mapper import map_emotion_to_music
        from music_brain.harmony import HarmonyGenerator, generate_midi_from_harmony

        # Check if using new format (base_emotion, intensity, specific_emotion)
        if "base_emotion" in body and "intensity" in body:
            # New format: use emotion mapper
            base_emotion = body["base_emotion"]
            intensity = body["intensity"]
            specific_emotion = body.get("specific_emotion")

            # Map emotion to musical parameters
            music_params = map_emotion_to_music(
                base_emotion=base_emotion,
                intensity=intensity,
                specific_emotion=specific_emotion
            )

            key = music_params["key"]
            mode = music_params["mode"]
            tempo = music_params["tempo"]
            progression = music_params["progression"]

            # Generate harmony
            harmony_gen = HarmonyGenerator()
            harmony = harmony_gen.generate_basic_progression(
                key=key,
                mode=mode,
                pattern=progression
            )

            # Generate MIDI file
            output_path = Path(tempfile.gettempdir()) / "output.mid"
            generate_midi_from_harmony(harmony, str(output_path), tempo_bpm=tempo)

            # Read MIDI file and encode as base64
            with open(output_path, "rb") as f:
                midi_data = base64.b64encode(f.read()).decode("utf-8")

            return {
                "success": True,
                "file_path": str(output_path),
                "midi_data": midi_data,
                "parameters": {
                    "key": key,
                    "mode": mode,
                    "tempo": tempo,
                    "progression": progression,
                    "base_emotion": base_emotion,
                    "intensity": intensity,
                    "specific_emotion": specific_emotion,
                },
                "message": f"Generated MIDI: {key} {mode}, {tempo} BPM, {progression}"
            }

        # Legacy format: use SongGenerator
        elif "intent" in body:
            from music_brain.session.generator import SongGenerator

            # Extract parameters from intent
            intent_data = body["intent"]
            technical = intent_data.get("technical") or {}

            # Parse key and mode from technical.key (e.g., "F major" or "C minor")
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
            mood = intent_data.get("emotional_intent") or None

            # Generate song
            generator = SongGenerator()
            song = generator.generate(
                key=key,
                mode=mode,
                mood=mood,
                genre=genre,
                tempo=float(tempo) if tempo else None
            )

            # Convert to dict for JSON response using the song's to_dict method
            result = {
                "success": True,
                "intent": intent_data,
                "song": song.to_dict(),
                "message": f"Generated {len(song.sections)} sections, {song.total_bars} bars at {song.tempo_bpm} BPM"
            }

            return result
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'intent' or 'base_emotion' and 'intensity' must be provided"
            )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interrogate")
async def interrogate(request: InterrogateRequest):
    """Conversational music creation"""
    try:
        # TODO: Integrate with interrogator module
        return {
            "success": True,
            "response": {
                "message": "Interrogation endpoint ready - integration pending",
                "questions": []
            }
        }

    except Exception as e:
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
