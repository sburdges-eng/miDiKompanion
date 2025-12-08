# DAiW MCP Bridge

The **daiw_mcp** package exposes the Music Brain capabilities through the
[Model Context Protocol](https://github.com/modelcontextprotocol). Any MCP-aware
copilot can call DAiW tools without bespoke integrations.

## Installation

```bash
# Install DAiW with MCP extras
pip install -e .[mcp]
```

## Running the server

```bash
# Default stdio transport (works with MCP-compatible CLIs)
python -m daiw_mcp.server

# SSE transport for browsers / external clients
python -m daiw_mcp.server sse
```

## Tools

| Tool Name | Description |
|-----------|-------------|
| `daiw.analyze_chords` | Analyze chords (and sections) from MIDI |
| `daiw.analyze_progression` | Diagnose chord progression text |
| `daiw.diagnose_chords` | Surface issues/suggestions for a progression |
| `daiw.generate_harmony` | Generate harmony from intent/basic params |
| `daiw.suggest_reharmonization` | Alternative reharmonizations |
| `daiw.extract_groove` | Extract groove profile from MIDI |
| `daiw.apply_groove` | Apply genre groove template |
| `daiw.humanize_midi` | Drunken Drummer humanization |
| `daiw.intent.create_template` | Emit CompleteSongIntent template |
| `daiw.intent.process` | Process intent → musical elements |
| `daiw.intent.validate` | Schema validation |
| `daiw.intent.suggest_rulebreaks` | Emotion-driven rule-break hints |
| `daiw.therapy.session` | Emotion → harmonic plan snapshot |

All MIDI-producing tools return a `midi_base64` payload plus display metadata.

## Customization

* Use `daiw_mcp.server.build_server(name="custom-name")` to adjust the server name.
* Additional tools can be added by implementing a new module in `daiw_mcp/tools`
  and exposing a `register_tools(server)` function.

## Testing

Run the smoke tests with:

```bash
pytest tests/test_mcp_tools.py -k "not therapy"
```

Tests exercise JSON-returning logic without requiring MCP runtime, keeping CI light.

