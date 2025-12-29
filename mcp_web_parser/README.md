# MCP Web Parser

Parallel web parsing and training data collection system for GPT Codex instances.

## Features

- **Parallel Processing**: Optimized for 4 parallel GPT Codex instances
- **Web Parsing**: Extract structured content (text, markdown, metadata) from URLs
- **Preview Mode**: Quick URL assessment without full parsing
- **File Downloads**: Download files for training data collection
- **Metadata Tracking**: Comprehensive metadata for all parsed content
- **Rate Limiting**: Automatic rate limiting (1 second per domain)

## Installation

```bash
pip install requests beautifulsoup4 markdownify
```

## MCP Server Setup

Add to your MCP client configuration:

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

## Available Tools

See `GPT_CODEX_PARALLEL_PROMPT.md` for complete tool documentation and usage examples.

## Data Storage

All data is stored in `~/.mcp_web_parser/`:

- `parsed/` - Parsed web pages (JSON + Markdown)
- `downloads/` - Downloaded files
- `metadata.json` - Master metadata file

## Usage

See `GPT_CODEX_PARALLEL_PROMPT.md` for detailed usage instructions and parallel processing setup.

