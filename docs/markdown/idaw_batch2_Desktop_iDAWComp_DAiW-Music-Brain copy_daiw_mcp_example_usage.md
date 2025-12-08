# MCP Server Usage Examples

## Setup

1. Install MCP SDK:
```bash
pip install mcp
```

2. Run the server:
```bash
python -m daiw_mcp.server
```

## Example Tool Calls

### 1. Generate Harmony from Emotion

```json
{
  "tool": "generate_harmony",
  "arguments": {
    "emotion": "grief",
    "key": "F",
    "mode": "major",
    "pattern": "I-V-vi-IV",
    "output_midi": "grief_harmony.mid",
    "tempo": 82
  }
}
```

**Response:**
```json
{
  "chords": ["F", "C", "Dm", "Bbm"],
  "key": "F",
  "mode": "major",
  "rule_break_applied": "HARMONY_ModalInterchange",
  "emotional_justification": "Bbm borrowed from F minor creates unresolved yearning",
  "midi_file": "grief_harmony.mid"
}
```

### 2. Analyze Chord Progression

```json
{
  "tool": "analyze_progression",
  "arguments": {
    "progression": "F-C-Dm-Bbm",
    "key": "F major"
  }
}
```

**Response:**
```json
{
  "progression": "F-C-Dm-Bbm",
  "key": "F",
  "mode": "major",
  "issues": [],
  "suggestions": ["Consider adding passing tones"],
  "emotional_character": "complex, emotionally ambiguous",
  "rule_breaks": ["HARMONY_ModalInterchange"]
}
```

### 3. Create and Process Intent

```json
{
  "tool": "create_intent",
  "arguments": {
    "title": "When I Found You Sleeping",
    "core_event": "Watching someone sleep, feeling protective",
    "core_longing": "To preserve this moment forever",
    "mood_primary": "nostalgia",
    "key": "F",
    "mode": "major",
    "output_file": "sleeping_intent.json"
  }
}
```

Then process it:

```json
{
  "tool": "process_intent",
  "arguments": {
    "intent_file": "sleeping_intent.json",
    "output_midi": "sleeping_song.mid"
  }
}
```

### 4. Extract Groove from MIDI

```json
{
  "tool": "extract_groove",
  "arguments": {
    "midi_file": "drums.mid"
  }
}
```

**Response:**
```json
{
  "midi_file": "drums.mid",
  "timing_stats": {
    "mean_deviation_ms": 12.5,
    "std_deviation_ms": 8.3
  },
  "velocity_stats": {
    "min": 45,
    "max": 127,
    "mean": 78
  },
  "swing_factor": 0.15,
  "note_count": 124
}
```

### 5. Suggest Rule-Breaks for Emotion

```json
{
  "tool": "suggest_rulebreaks",
  "arguments": {
    "emotion": "grief"
  }
}
```

**Response:**
```json
{
  "emotion": "grief",
  "suggestions": [
    {
      "rule": "HARMONY_AvoidTonicResolution",
      "description": "Avoid resolving to the tonic chord",
      "effect": "Creates unresolved yearning, emotional tension",
      "use_when": "Expressing loss, longing, or incomplete feelings"
    }
  ]
}
```

### 6. Map Emotion to Music

```json
{
  "tool": "emotion_to_music",
  "arguments": {
    "emotion": "defiance"
  }
}
```

**Response:**
```json
{
  "emotion": "defiance",
  "musical_parameters": {
    "suggested_key": "minor",
    "suggested_tempo": "fast",
    "rule_break_suggestions": [
      {
        "rule": "HARMONY_ParallelMotion",
        "effect": "Creates bold, unapologetic movement"
      }
    ]
  }
}
```

## Integration with AI Assistants

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "daiw-music-brain": {
      "command": "python",
      "args": ["-m", "daiw_mcp.server"],
      "env": {
        "PYTHONPATH": "/path/to/DAiW-Music-Brain"
      }
    }
  }
}
```

### Cursor

Add to Cursor settings or `.cursorrules`:

```json
{
  "mcp": {
    "servers": {
      "daiw": {
        "command": "python",
        "args": ["-m", "daiw_mcp.server"]
      }
    }
  }
}
```

## Workflow Example

1. **AI Assistant:** "I want to write a song about grief"
2. **Tool Call:** `suggest_rulebreaks(emotion="grief")`
3. **AI Assistant:** "Let me create an intent for this"
4. **Tool Call:** `create_intent(title="Grief Song", mood_primary="grief", ...)`
5. **AI Assistant:** "Now let's generate the harmony"
6. **Tool Call:** `process_intent(intent_file="grief_song.json", output_midi="grief.mid")`
7. **Result:** MIDI file with emotionally-driven harmony

## Error Handling

All tools return JSON with error information:

```json
{
  "error": "MIDI file not found: missing.mid",
  "status": "error"
}
```

## Notes

- All file paths should be absolute or relative to the working directory
- Audio tools require `librosa` for full functionality
- Some tools are placeholders for Phase 2 features
- Tools return structured JSON for easy parsing by AI assistants

