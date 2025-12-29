# GPT Codex Parallel Web Parsing Prompt

## Overview

This prompt is designed for **4 parallel GPT Codex instances** to work together on automated web parsing, preview, and download tasks for training data collection. Each instance handles a subset of URLs, enabling efficient parallel processing.

---

## System Architecture

### Parallel Execution Model

```
GPT Codex Instance 1 → URLs [0, 4, 8, 12, ...]   (25% of workload)
GPT Codex Instance 2 → URLs [1, 5, 9, 13, ...]   (25% of workload)
GPT Codex Instance 3 → URLs [2, 6, 10, 14, ...] (25% of workload)
GPT Codex Instance 4 → URLs [3, 7, 11, 15, ...] (25% of workload)
```

### Data Flow

1. **Command Queue**: Shared list of URLs to process
2. **Parallel Parsing**: Each instance processes its assigned URLs
3. **Metadata Collection**: All results aggregated in `~/.mcp_web_parser/metadata.json`
4. **Training Data**: Parsed content saved as JSON + Markdown in `~/.mcp_web_parser/parsed/`

---

## GPT Codex Prompt Template

Copy this prompt into each of your 4 GPT Codex instances, modifying the `INSTANCE_ID`:

```markdown
# Web Parser Agent - Instance {INSTANCE_ID} of 4

You are a specialized web parsing agent working in a parallel processing system with 3 other identical instances.

## Your Role

- **Instance ID**: {INSTANCE_ID} (1-4)
- **Workload**: Process URLs at indices {INSTANCE_ID}, {INSTANCE_ID}+4, {INSTANCE_ID}+8, etc.
- **Goal**: Parse websites, extract content, and prepare training data for Llama-adjacent model training

## Security Features

**All downloads are automatically validated for security:**
- ✅ File extension whitelist (only safe extensions allowed)
- ✅ Content-Type validation
- ✅ File size limits (100 MB max)
- ✅ Content scanning for malicious patterns
- ✅ Quarantine system for suspicious files
- ✅ URL validation (only http/https allowed)

**See `mcp_web_parser/SECURITY.md` for complete security documentation.**

## Available MCP Tools

You have access to the following MCP tools via the `mcp_web_parser` server:

1. **web_parse_url** - Parse a single URL
   - Input: `{"url": "https://example.com", "save": true}`
   - Output: Parsed content (text, markdown, metadata) saved to disk

2. **web_parse_batch** - Parse multiple URLs in parallel
   - Input: `{"urls": ["url1", "url2", ...], "save": true}`
   - Output: List of parsed pages

3. **web_preview_url** - Quick preview without full parsing
   - Input: `{"url": "https://example.com"}`
   - Output: Title, description, basic metadata

4. **web_download_file** - Download a file
   - Input: `{"url": "https://example.com/file.pdf"}`
   - Output: Downloaded file path

5. **web_download_batch** - Download multiple files in parallel
   - Input: `{"urls": ["url1", "url2", ...]}`
   - Output: List of downloaded files

6. **web_get_statistics** - Get parsing/download statistics
   - Input: `{}`
   - Output: Statistics about collected data

7. **web_list_parsed** - List all parsed pages
   - Input: `{"limit": 100}`
   - Output: List of parsed pages with metadata

8. **web_get_security_info** - Get security configuration and status
   - Input: `{}`
   - Output: Security settings, validation status, quarantine directory

## Workflow

### Step 1: Receive URL List
You will receive a list of URLs to process. Your instance should process:
- URL at index: {INSTANCE_ID} - 1 (0-indexed: {INSTANCE_ID - 1})
- Then every 4th URL after that: {INSTANCE_ID + 3}, {INSTANCE_ID + 7}, etc.

### Step 2: Preview URLs (Optional)
Before full parsing, you may want to preview URLs to filter out irrelevant content:
```
Use: web_preview_url for each URL in your subset
```

### Step 3: Parse URLs
Parse your assigned URLs:
```
Use: web_parse_batch with your subset of URLs
```

### Step 4: Extract Download Links (If Needed)
If pages contain downloadable resources (PDFs, documents, etc.):
```
1. Parse the page
2. Extract links from metadata
3. Filter for downloadable file types (.pdf, .doc, .txt, etc.)
4. Use: web_download_batch to download files
```

### Step 5: Report Progress
After processing, report:
- Number of URLs processed
- Number successfully parsed
- Number of files downloaded (if any)
- Any errors encountered

## Example Commands

### Parse a single URL
```json
{
  "tool": "web_parse_url",
  "arguments": {
    "url": "https://example.com/article",
    "save": true
  }
}
```

### Parse multiple URLs in parallel
```json
{
  "tool": "web_parse_batch",
  "arguments": {
    "urls": [
      "https://example.com/page1",
      "https://example.com/page2",
      "https://example.com/page3"
    ],
    "save": true
  }
}
```

### Preview before parsing
```json
{
  "tool": "web_preview_url",
  "arguments": {
    "url": "https://example.com/article"
  }
}
```

### Download files
```json
{
  "tool": "web_download_batch",
  "arguments": {
    "urls": [
      "https://example.com/document.pdf",
      "https://example.com/data.txt"
    ]
  }
}
```

## Best Practices

1. **Respect Rate Limits**: The system handles rate limiting automatically (1 second between requests per domain)

2. **Error Handling**: If a URL fails to parse, log the error and continue with the next URL

3. **Content Quality**: Focus on high-quality content sources:
   - Educational websites
   - Documentation sites
   - Technical blogs
   - Research papers (if accessible)

4. **Metadata Preservation**: Always use `"save": true` to preserve parsed content for training

5. **Parallel Efficiency**: Process your assigned URLs in batches using `web_parse_batch` rather than one-by-one

6. **Progress Tracking**: Periodically check statistics with `web_get_statistics` to monitor overall progress

7. **Security Awareness**: 
   - All downloads are automatically validated
   - Suspicious files are quarantined (check `~/.mcp_web_parser/quarantine/`)
   - Review security warnings in logs
   - Use `web_get_security_info` to check security status

## Output Format

After processing your assigned URLs, provide a summary:

```markdown
## Processing Summary - Instance {INSTANCE_ID}

- **URLs Assigned**: X
- **Successfully Parsed**: Y
- **Failed**: Z
- **Files Downloaded**: N
- **Total Content Extracted**: ~X MB

### Errors (if any):
- URL: https://example.com/broken → Error: Connection timeout
```

## Coordination Notes

- You are working in parallel with 3 other instances
- Each instance processes a different subset of URLs
- All results are saved to the same shared directory: `~/.mcp_web_parser/`
- Check `web_get_statistics` to see overall progress across all instances
```

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install requests beautifulsoup4 markdownify
```

### 2. Configure MCP Server

Add to your MCP client configuration (e.g., Cursor, Claude Desktop):

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

### 3. Launch 4 GPT Codex Instances

1. Open 4 separate GPT Codex sessions
2. Copy the prompt template above into each
3. Replace `{INSTANCE_ID}` with 1, 2, 3, and 4 respectively
4. Provide the URL list to all instances

### 4. Distribute URLs

Provide the same URL list to all 4 instances. Each will automatically process its assigned subset:

- **Instance 1**: URLs at indices 0, 4, 8, 12, ...
- **Instance 2**: URLs at indices 1, 5, 9, 13, ...
- **Instance 3**: URLs at indices 2, 6, 10, 14, ...
- **Instance 4**: URLs at indices 3, 7, 11, 15, ...

---

## Example Workflow

### Input: URL List
```python
urls = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page3",
    "https://example.com/page4",
    "https://example.com/page5",
    "https://example.com/page6",
    "https://example.com/page7",
    "https://example.com/page8",
]
```

### Instance 1 Processing
- Processes: page1, page5
- Uses: `web_parse_batch({"urls": ["page1", "page5"]})`

### Instance 2 Processing
- Processes: page2, page6
- Uses: `web_parse_batch({"urls": ["page2", "page6"]})`

### Instance 3 Processing
- Processes: page3, page7
- Uses: `web_parse_batch({"urls": ["page3", "page7"]})`

### Instance 4 Processing
- Processes: page4, page8
- Uses: `web_parse_batch({"urls": ["page4", "page8"]})`

---

## Output Structure

All parsed content is saved to `~/.mcp_web_parser/`:

```
~/.mcp_web_parser/
├── parsed/
│   ├── {url_hash}.json    # Full parsed data (JSON)
│   ├── {url_hash}.md      # Markdown version
│   └── ...
├── downloads/
│   ├── {domain}/
│   │   ├── file1.pdf
│   │   └── file2.txt
│   └── ...
└── metadata.json           # Master metadata file
```

### JSON Format
```json
{
  "url": "https://example.com/page",
  "title": "Page Title",
  "content": "Extracted text content...",
  "markdown": "# Page Title\n\nMarkdown content...",
  "metadata": {
    "domain": "example.com",
    "content_length": 5000,
    "links": ["url1", "url2"],
    "images": ["img1", "img2"]
  },
  "timestamp": "2024-01-01T12:00:00",
  "url_hash": "abc123..."
}
```

---

## Advanced: Command Generation

For ongoing command creation, you can use this pattern:

```python
# Generate commands for all 4 instances
def generate_commands(urls, instance_id):
    """Generate MCP tool calls for a specific instance."""
    assigned_urls = [urls[i] for i in range(instance_id - 1, len(urls), 4)]
    
    commands = []
    for url in assigned_urls:
        commands.append({
            "tool": "web_parse_url",
            "arguments": {"url": url, "save": True}
        })
    
    return commands
```

---

## Monitoring Progress

Use `web_get_statistics` across instances to monitor overall progress:

```json
{
  "tool": "web_get_statistics",
  "arguments": {}
}
```

Response:
```json
{
  "statistics": {
    "total_parsed": 150,
    "total_downloaded": 25,
    "last_updated": "2024-01-01T12:00:00"
  }
}
```

---

## Troubleshooting

### Issue: Dependencies Not Available
**Solution**: Install required packages:
```bash
pip install requests beautifulsoup4 markdownify
```

### Issue: Rate Limiting Errors
**Solution**: The system automatically handles rate limiting (1 second per domain). If you see errors, the target site may be blocking requests.

### Issue: Parallel Conflicts
**Solution**: Each instance processes different URLs, so there should be no conflicts. If you see file conflicts, check that instance IDs are correctly assigned (1-4).

### Issue: Missing Content
**Solution**: Some sites use JavaScript rendering. Consider using a headless browser (Selenium/Playwright) for those sites (future enhancement).

---

## Future Enhancements

1. **JavaScript Rendering**: Add Selenium/Playwright support for dynamic content
2. **Content Filtering**: AI-powered content quality assessment
3. **Deduplication**: Automatic detection of duplicate content
4. **Format Conversion**: Convert parsed content to training formats (JSONL, Parquet, etc.)
5. **Incremental Updates**: Track and update only changed content

---

## Training Data Preparation

After parsing, you can prepare data for Llama training:

```python
import json
from pathlib import Path

def prepare_training_data(parsed_dir):
    """Convert parsed pages to training format."""
    training_data = []
    
    for json_file in Path(parsed_dir).glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            training_data.append({
                "text": data["content"],
                "metadata": {
                    "url": data["url"],
                    "title": data["title"],
                    "domain": data["metadata"]["domain"]
                }
            })
    
    return training_data
```

---

## Summary

This system enables:
- ✅ **4 parallel GPT Codex instances** working simultaneously
- ✅ **Automated web parsing** with structured content extraction
- ✅ **Preview functionality** for quick URL assessment
- ✅ **Parallel downloads** for training data collection
- ✅ **Metadata tracking** for all parsed content
- ✅ **Training-ready output** in JSON + Markdown formats

Each instance processes 25% of the workload, enabling 4x faster processing while maintaining organized, structured output for future model training.

