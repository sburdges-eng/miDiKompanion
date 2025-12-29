# MCP Web Parser - Setup Guide

Complete setup guide for parallel web parsing with 4 GPT Codex instances.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r mcp_web_parser/requirements.txt
```

Or install individually:
```bash
pip install requests beautifulsoup4 markdownify
```

### 2. Configure MCP Server

Add to your MCP client configuration (Cursor, Claude Desktop, etc.):

**For Cursor:**
Edit `~/.cursor/mcp.json` or your MCP config file:

```json
{
  "mcpServers": {
    "web-parser": {
      "command": "python",
      "args": ["-m", "mcp_web_parser.server"]
    }
  }
}
```

**For Claude Desktop:**
Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or equivalent:

```json
{
  "mcpServers": {
    "web-parser": {
      "command": "python",
      "args": ["-m", "mcp_web_parser.server"]
    }
  }
}
```

### 3. Test the Server

```bash
# Test CLI
python -m mcp_web_parser.cli parse https://example.com

# Check statistics
python -m mcp_web_parser.cli stats
```

### 4. Generate Commands for Parallel Instances

Create a file `urls.txt` with one URL per line:

```
https://example.com/page1
https://example.com/page2
https://example.com/page3
...
```

Generate commands:
```bash
# Generate for all 4 instances
python -m mcp_web_parser.generate_commands urls.txt --format markdown

# Generate for specific instance
python -m mcp_web_parser.generate_commands urls.txt --instance 1
```

## Usage with GPT Codex

### Step 1: Prepare URL List

Create a text file with URLs, one per line.

### Step 2: Generate Commands

```bash
python -m mcp_web_parser.generate_commands urls.txt --format markdown > commands.md
```

### Step 3: Launch 4 GPT Codex Instances

1. Open 4 separate GPT Codex sessions
2. Copy the prompt from `GPT_CODEX_PARALLEL_PROMPT.md` into each
3. Replace `{INSTANCE_ID}` with 1, 2, 3, and 4 respectively
4. Provide the generated commands to each instance

### Step 4: Monitor Progress

In any instance, use:
```json
{
  "tool": "web_get_statistics",
  "arguments": {}
}
```

## Data Storage

All data is stored in `~/.mcp_web_parser/`:

```
~/.mcp_web_parser/
├── parsed/
│   ├── {url_hash}.json    # Full parsed data
│   ├── {url_hash}.md      # Markdown version
│   └── ...
├── downloads/
│   ├── {domain}/
│   │   └── files...
│   └── ...
└── metadata.json           # Master metadata
```

## CLI Commands

### Parse URLs
```bash
python -m mcp_web_parser.cli parse https://example.com/page1 https://example.com/page2
```

### Download Files
```bash
python -m mcp_web_parser.cli download https://example.com/file.pdf
```

### Show Statistics
```bash
python -m mcp_web_parser.cli stats
```

### List Parsed Pages
```bash
python -m mcp_web_parser.cli list --limit 50
```

## MCP Tools Reference

See `GPT_CODEX_PARALLEL_PROMPT.md` for complete tool documentation.

## Troubleshooting

### "Dependencies not available"
Install required packages:
```bash
pip install requests beautifulsoup4 markdownify
```

### "Rate limiting errors"
The system automatically handles rate limiting (1 second per domain). If you see persistent errors, the target site may be blocking requests.

### "Parallel conflicts"
Each instance processes different URLs, so there should be no conflicts. Verify instance IDs are correctly assigned (1-4).

## Next Steps

1. Review `GPT_CODEX_PARALLEL_PROMPT.md` for detailed usage
2. Test with a small URL list first
3. Scale up to larger datasets
4. Use parsed data for training preparation

