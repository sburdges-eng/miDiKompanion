# DAiW MCP Server

Model Context Protocol (MCP) server for DAiW-Music-Brain. This allows AI assistants to use DAiW's music analysis and generation capabilities.

## Installation

```bash
# Install MCP SDK
pip install mcp

# Or install with DAiW
pip install -e ".[mcp]"
```

## Usage

### Running the Server

```bash
# Run MCP server
python -m daiw_mcp.server

# Or use the entry point
daiw-mcp-server
```

### Connecting an AI Assistant

Configure your AI assistant (Claude Desktop, Cursor, etc.) to use this MCP server:

```json
{
  "mcpServers": {
    "daiw-music-brain": {
      "command": "python",
      "args": ["-m", "daiw_mcp.server"]
    }
  }
}
```

## Available Tools

### Harmony Tools (6)
- `analyze_progression` - Analyze chord progression
- `generate_harmony` - Generate harmony from intent or parameters
- `diagnose_chords` - Diagnose harmonic issues
- `suggest_reharmonization` - Suggest chord substitutions
- `find_key` - Detect key from progression
- `voice_leading` - Optimize voice leading

### Groove Tools (5)
- `extract_groove` - Extract groove from MIDI
- `apply_groove` - Apply genre groove template
- `analyze_pocket` - Analyze timing pocket
- `humanize_midi` - Add human feel
- `quantize_smart` - Smart quantization

### Intent Tools (4)
- `create_intent` - Create song intent template
- `process_intent` - Process intent → music
- `validate_intent` - Validate intent schema
- `suggest_rulebreaks` - Suggest emotional rule-breaks

### Audio Analysis Tools (4)
- `detect_bpm` - Detect tempo from audio
- `detect_key` - Detect key from audio
- `analyze_audio_feel` - Analyze groove feel
- `extract_chords` - Extract chords from audio (planned)

### Teaching Tools (3)
- `explain_rulebreak` - Explain rule-breaking technique
- `get_progression_info` - Get progression details
- `emotion_to_music` - Map emotion to musical parameters

## Example Usage

### Generate Harmony

```python
# AI assistant can call:
generate_harmony(
    emotion="grief",
    key="F",
    mode="major",
    pattern="I-V-vi-IV"
)
```

### Process Intent

```python
# AI assistant can call:
process_intent(
    intent_file="song_intent.json",
    output_midi="output.mid"
)
```

### Analyze Progression

```python
# AI assistant can call:
analyze_progression(
    progression="F-C-Dm-Bbm",
    key="F major"
)
```

## Architecture

```
daiw_mcp/
├── __init__.py
├── server.py              # Main MCP server
├── tools/
│   ├── __init__.py
│   ├── harmony_tools.py   # 6 harmony tools
│   ├── groove_tools.py    # 5 groove tools
│   ├── intent_tools.py    # 4 intent tools
│   ├── audio_tools.py     # 4 audio tools
│   └── teaching_tools.py   # 3 teaching tools
└── README.md
```

## Development

### Adding a New Tool

1. Add tool definition to appropriate `tools/*.py` file
2. Implement handler in `call_tool()` function
3. Register with server in `register_tools()`

### Testing

```bash
# Test MCP server
python -m pytest tests/test_mcp_tools.py

# Test individual tools
python -c "from daiw_mcp.tools import harmony_tools; print('OK')"
```

## Dependencies

- `mcp` - Model Context Protocol SDK
- `music_brain` - DAiW core modules

## Notes

- All tools return JSON responses
- File paths should be absolute or relative to working directory
- Audio tools require `librosa` for full functionality
- Some tools are placeholders for Phase 2 features

