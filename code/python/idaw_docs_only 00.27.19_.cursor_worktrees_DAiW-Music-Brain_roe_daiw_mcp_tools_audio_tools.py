"""
Audio Analysis Tools - MCP tools for audio file analysis.

Provides 4 tools:
- detect_bpm
- detect_key
- analyze_audio_feel
- extract_chords
"""

from typing import Any, Dict, List
import json
from pathlib import Path

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Import DAiW modules
try:
    from music_brain.audio.feel import analyze_feel, AudioFeatures
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Try to import audio cataloger for BPM/key detection
try:
    from tools.audio_cataloger.audio_cataloger import detect_bpm_key
    AUDIO_CATALOGER_AVAILABLE = True
except ImportError:
    AUDIO_CATALOGER_AVAILABLE = False


def register_tools(server: Server) -> None:
    """Register all audio analysis tools with the MCP server."""
    if not MCP_AVAILABLE:
        return
    
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available audio analysis tools."""
        return [
            Tool(
                name="detect_bpm",
                description="Detect tempo (BPM) from an audio file.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "audio_file": {
                            "type": "string",
                            "description": "Path to audio file (WAV, MP3, AIFF, etc.)"
                        }
                    },
                    "required": ["audio_file"]
                }
            ),
            Tool(
                name="detect_key",
                description="Detect musical key and mode from an audio file.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "audio_file": {
                            "type": "string",
                            "description": "Path to audio file"
                        }
                    },
                    "required": ["audio_file"]
                }
            ),
            Tool(
                name="analyze_audio_feel",
                description="Analyze groove feel and energy characteristics from audio.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "audio_file": {
                            "type": "string",
                            "description": "Path to audio file"
                        }
                    },
                    "required": ["audio_file"]
                }
            ),
            Tool(
                name="extract_chords",
                description="Extract chord progression from audio file.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "audio_file": {
                            "type": "string",
                            "description": "Path to audio file"
                        }
                    },
                    "required": ["audio_file"]
                }
            ),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls."""
        try:
            audio_file = arguments.get("audio_file", "")
            
            if not Path(audio_file).exists():
                return [TextContent(type="text", text=f"Error: Audio file not found: {audio_file}")]
            
            if name == "detect_bpm":
                if AUDIO_CATALOGER_AVAILABLE:
                    try:
                        bpm, _ = detect_bpm_key(audio_file)
                        result = {
                            "audio_file": audio_file,
                            "bpm": float(bpm) if bpm else None,
                            "confidence": "high" if bpm else "low"
                        }
                    except Exception as e:
                        result = {
                            "audio_file": audio_file,
                            "error": str(e),
                            "note": "BPM detection requires librosa. Install with: pip install librosa"
                        }
                else:
                    result = {
                        "audio_file": audio_file,
                        "error": "Audio cataloger not available",
                        "note": "Install librosa and ensure audio_cataloger is accessible"
                    }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "detect_key":
                if AUDIO_CATALOGER_AVAILABLE:
                    try:
                        _, key_info = detect_bpm_key(audio_file)
                        if key_info:
                            key, mode = key_info.split() if " " in key_info else (key_info, "major")
                            result = {
                                "audio_file": audio_file,
                                "key": key,
                                "mode": mode,
                                "confidence": "high"
                            }
                        else:
                            result = {
                                "audio_file": audio_file,
                                "key": None,
                                "mode": None,
                                "confidence": "low"
                            }
                    except Exception as e:
                        result = {
                            "audio_file": audio_file,
                            "error": str(e),
                            "note": "Key detection requires librosa. Install with: pip install librosa"
                        }
                else:
                    result = {
                        "audio_file": audio_file,
                        "error": "Audio cataloger not available",
                        "note": "Install librosa and ensure audio_cataloger is accessible"
                    }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "analyze_audio_feel":
                if AUDIO_AVAILABLE:
                    try:
                        features = analyze_feel(audio_file)
                        result = {
                            "audio_file": audio_file,
                            "tempo_bpm": features.tempo_bpm,
                            "energy_curve": features.energy_curve[:10] if len(features.energy_curve) > 10 else features.energy_curve,
                            "dynamic_range_db": features.dynamic_range_db,
                            "rms_mean": features.rms_mean
                        }
                    except Exception as e:
                        result = {
                            "audio_file": audio_file,
                            "error": str(e),
                            "note": "Audio analysis requires librosa. Install with: pip install librosa"
                        }
                else:
                    result = {
                        "audio_file": audio_file,
                        "error": "Audio analysis module not available",
                        "note": "Install librosa and ensure music_brain.audio is accessible"
                    }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "extract_chords":
                result = {
                    "audio_file": audio_file,
                    "status": "not_implemented",
                    "note": "Chord extraction from audio requires advanced DSP. This feature is planned for Phase 2."
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
        
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

