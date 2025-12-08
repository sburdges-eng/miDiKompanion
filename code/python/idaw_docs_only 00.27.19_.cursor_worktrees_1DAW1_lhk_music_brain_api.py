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
    emotional_intent: str
    technical: Optional[Dict[str, Any]] = None
    
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
        # TODO: Integrate with actual music generation modules
        return {
            "success": True,
            "intent": request.intent.dict(),
            "result": {
                "message": "Generation endpoint ready - integration pending",
                "midi_data": None
            }
        }
        
    except Exception as e:
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
