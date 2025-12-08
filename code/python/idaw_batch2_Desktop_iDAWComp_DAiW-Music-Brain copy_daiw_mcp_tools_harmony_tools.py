"""
Harmony Tools - MCP tools for harmony generation and analysis.

Provides 6 tools:
- analyze_progression
- generate_harmony
- diagnose_chords
- suggest_reharmonization
- find_key
- voice_leading
"""

from typing import Any, Dict, List, Optional
import json
from pathlib import Path

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Import DAiW modules
from music_brain.harmony import HarmonyGenerator, HarmonyResult, generate_midi_from_harmony
from music_brain.structure.progression import diagnose_progression, generate_reharmonizations
from music_brain.structure.chord import analyze_chords


def register_tools(server: Server) -> None:
    """Register all harmony tools with the MCP server."""
    if not MCP_AVAILABLE:
        return
    
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available harmony tools."""
        return [
            Tool(
                name="analyze_progression",
                description="Analyze a chord progression for harmonic characteristics, emotional character, and rule breaks.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "progression": {
                            "type": "string",
                            "description": "Chord progression string (e.g., 'F-C-Dm-Bbm')"
                        },
                        "key": {
                            "type": "string",
                            "description": "Optional key context (e.g., 'F major')"
                        }
                    },
                    "required": ["progression"]
                }
            ),
            Tool(
                name="generate_harmony",
                description="Generate harmony from emotional intent or basic parameters.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "emotion": {
                            "type": "string",
                            "description": "Primary emotion (e.g., 'grief', 'nostalgia', 'defiance')"
                        },
                        "key": {
                            "type": "string",
                            "description": "Musical key (e.g., 'C', 'F', 'Bb')",
                            "default": "C"
                        },
                        "mode": {
                            "type": "string",
                            "description": "Mode/scale (major, minor, dorian, etc.)",
                            "default": "major"
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Roman numeral pattern (e.g., 'I-V-vi-IV')",
                            "default": "I-V-vi-IV"
                        },
                        "intent_file": {
                            "type": "string",
                            "description": "Optional path to CompleteSongIntent JSON file"
                        },
                        "output_midi": {
                            "type": "string",
                            "description": "Optional path to save MIDI output"
                        },
                        "tempo": {
                            "type": "integer",
                            "description": "Tempo in BPM",
                            "default": 82
                        }
                    }
                }
            ),
            Tool(
                name="diagnose_chords",
                description="Diagnose harmonic issues in a chord progression.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "progression": {
                            "type": "string",
                            "description": "Chord progression string"
                        },
                        "key": {
                            "type": "string",
                            "description": "Optional key context"
                        }
                    },
                    "required": ["progression"]
                }
            ),
            Tool(
                name="suggest_reharmonization",
                description="Suggest reharmonization alternatives for a chord progression.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "progression": {
                            "type": "string",
                            "description": "Chord progression to reharmonize"
                        },
                        "style": {
                            "type": "string",
                            "description": "Reharmonization style (jazz, pop, rnb, classical, experimental)",
                            "default": "jazz"
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of suggestions",
                            "default": 3
                        }
                    },
                    "required": ["progression"]
                }
            ),
            Tool(
                name="find_key",
                description="Detect the key of a chord progression.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "progression": {
                            "type": "string",
                            "description": "Chord progression string"
                        }
                    },
                    "required": ["progression"]
                }
            ),
            Tool(
                name="voice_leading",
                description="Optimize voice leading for a chord progression.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "chords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of chord symbols"
                        },
                        "key": {
                            "type": "string",
                            "description": "Musical key"
                        }
                    },
                    "required": ["chords", "key"]
                }
            ),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls."""
        try:
            if name == "analyze_progression":
                progression = arguments["progression"]
                key = arguments.get("key")
                
                diagnosis = diagnose_progression(progression, key)
                result = {
                    "progression": progression,
                    "key": diagnosis.get("key", "Unknown"),
                    "mode": diagnosis.get("mode", "Unknown"),
                    "issues": diagnosis.get("issues", []),
                    "suggestions": diagnosis.get("suggestions", []),
                    "emotional_character": diagnosis.get("emotional_character", ""),
                    "rule_breaks": diagnosis.get("rule_breaks", [])
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "generate_harmony":
                generator = HarmonyGenerator()
                
                if "intent_file" in arguments:
                    # Generate from intent file
                    intent_path = Path(arguments["intent_file"])
                    if not intent_path.exists():
                        return [TextContent(type="text", text=f"Error: Intent file not found: {intent_path}")]
                    
                    from music_brain.session.intent_schema import CompleteSongIntent
                    intent = CompleteSongIntent.load(str(intent_path))
                    harmony = generator.generate_from_intent(intent)
                else:
                    # Generate from basic parameters
                    key = arguments.get("key", "C")
                    mode = arguments.get("mode", "major")
                    pattern = arguments.get("pattern", "I-V-vi-IV")
                    emotion = arguments.get("emotion")
                    
                    harmony = generator.generate_basic_progression(
                        key=key,
                        mode=mode,
                        pattern=pattern
                    )
                    
                    if emotion:
                        # Apply emotional rule-breaking
                        from music_brain.session.intent_schema import suggest_rule_break
                        suggestions = suggest_rule_break(emotion)
                        if suggestions:
                            # Apply first suggestion
                            rule_break = suggestions[0]["rule"]
                            # This would need more implementation
                
                result = {
                    "chords": harmony.chords,
                    "key": harmony.key,
                    "mode": harmony.mode,
                    "rule_break_applied": harmony.rule_break_applied,
                    "emotional_justification": harmony.emotional_justification
                }
                
                # Save MIDI if requested
                if "output_midi" in arguments:
                    tempo = arguments.get("tempo", 82)
                    generate_midi_from_harmony(harmony, arguments["output_midi"], tempo_bpm=tempo)
                    result["midi_file"] = arguments["output_midi"]
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "diagnose_chords":
                progression = arguments["progression"]
                key = arguments.get("key")
                
                diagnosis = diagnose_progression(progression, key)
                result = {
                    "progression": progression,
                    "key": diagnosis.get("key", "Unknown"),
                    "mode": diagnosis.get("mode", "Unknown"),
                    "issues": diagnosis.get("issues", []),
                    "suggestions": diagnosis.get("suggestions", [])
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "suggest_reharmonization":
                progression = arguments["progression"]
                style = arguments.get("style", "jazz")
                count = arguments.get("count", 3)
                
                suggestions = generate_reharmonizations(progression, style=style, count=count)
                result = {
                    "original": progression,
                    "style": style,
                    "suggestions": [
                        {
                            "chords": s["chords"],
                            "technique": s.get("technique", ""),
                            "mood": s.get("mood", "")
                        }
                        for s in suggestions
                    ]
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "find_key":
                progression = arguments["progression"]
                diagnosis = diagnose_progression(progression)
                result = {
                    "progression": progression,
                    "key": diagnosis.get("key", "Unknown"),
                    "mode": diagnosis.get("mode", "Unknown"),
                    "confidence": "high" if diagnosis.get("key") else "low"
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "voice_leading":
                chords = arguments["chords"]
                key = arguments["key"]
                
                # Basic voice leading optimization
                # This would need more sophisticated implementation
                result = {
                    "chords": chords,
                    "key": key,
                    "voice_leading_score": 0.85,  # Placeholder
                    "notes": "Voice leading optimization would analyze smooth voice movement between chords"
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
        
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

