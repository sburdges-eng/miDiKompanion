# Security Features - MCP Web Parser

## Overview

The MCP Web Parser includes comprehensive security features to prevent malicious files and code from being downloaded or executed.

## Security Layers

### 1. URL Validation

- **Scheme Validation**: Only `http://` and `https://` URLs are allowed
- **Domain Validation**: Blocks localhost and suspicious domains
- **Format Validation**: Ensures URLs are properly formatted

### 2. File Extension Whitelist

**Allowed Extensions:**
- Text: `.txt`, `.md`, `.markdown`, `.rst`, `.org`
- Documents: `.pdf`, `.epub` (read-only, no macros)
- Data: `.json`, `.jsonl`, `.csv`, `.tsv`, `.xml`
- Archives: `.zip`, `.tar`, `.tar.gz` (will be extracted and validated)
- Code: `.py`, `.js`, `.ts`, `.cpp`, `.c`, `.h`, etc. (scanned carefully)
- Web: `.html`, `.htm`, `.css`

**Blocked Extensions:**
- Executables: `.exe`, `.bat`, `.cmd`, `.com`, `.scr`, `.vbs`, `.jar`
- Libraries: `.dll`, `.so`, `.dylib`, `.app`
- Installers: `.deb`, `.rpm`, `.msi`, `.pkg`

### 3. Content-Type Validation

Validates HTTP `Content-Type` header against allowed MIME types:
- `text/*` (text files)
- `application/json`, `application/xml`
- `application/pdf`
- `application/zip`, `application/x-tar`, `application/gzip`

### 4. File Size Limits

- **Maximum File Size**: 100 MB per file
- **Maximum Content-Length**: 200 MB (from HTTP header)
- Prevents resource exhaustion attacks

### 5. Content Scanning

Scans downloaded files for suspicious patterns:

**Executable Signatures:**
- Windows PE (`MZ\0\0`)
- Linux ELF (`\x7fELF`)
- ZIP archives (could contain executables)

**Code Injection Patterns:**
- `eval()`, `exec()`, `system()` calls
- `shell_exec`, `base64_decode`
- Network commands: `curl`, `wget`, `nc -l`

**Suspicious URLs:**
- IP addresses in content
- Long base64-encoded strings

### 6. File Type Detection

Uses magic number detection (python-magic) to verify actual file type matches Content-Type header. Prevents file type spoofing.

### 7. Quarantine System

Files that fail validation are moved to a quarantine directory:
- Location: `~/.mcp_web_parser/quarantine/`
- Logged with reason for quarantine
- Can be reviewed manually

## Security Configuration

### Default Settings

```python
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB
```

### Customization

You can modify security settings in `mcp_web_parser/security.py`:

- `SAFE_EXTENSIONS`: Add/remove allowed file extensions
- `DANGEROUS_EXTENSIONS`: Add extensions to block
- `MAX_FILE_SIZE`: Adjust maximum file size
- `SUSPICIOUS_PATTERNS`: Add custom pattern detection

## Usage

### Automatic Validation

Security validation is **enabled by default** for all downloads:

```python
# Validation is automatic
downloader.download_file("https://example.com/file.pdf")
```

### Disable Validation (Not Recommended)

```python
# Only disable if you trust the source
downloader.download_file("https://example.com/file.pdf", validate=False)
```

### Check Security Status

```json
{
  "tool": "web_get_security_info",
  "arguments": {}
}
```

## Quarantine Directory

Quarantined files are stored in:
```
~/.mcp_web_parser/quarantine/
├── file_name_hash.ext
└── quarantine.log
```

Review `quarantine.log` to see why files were quarantined.

## Best Practices

1. **Always validate downloads** (default behavior)
2. **Review quarantined files** periodically
3. **Use HTTPS** when possible
4. **Verify file types** match expectations
5. **Monitor security warnings** in logs

## Limitations

### Current Limitations

1. **No virus scanning**: Relies on pattern matching, not actual virus detection
2. **No sandboxing**: Files are saved to disk (but in isolated directory)
3. **No code execution prevention**: Code files are saved but not executed
4. **Limited archive validation**: ZIP/TAR files are allowed but not fully scanned

### Recommended Enhancements

1. **Integrate ClamAV** or similar for virus scanning
2. **Add sandboxing** for code file execution
3. **Deep archive scanning** for nested files
4. **Content-based filtering** using ML models
5. **Reputation checking** for domains/URLs

## Security Warnings

The system logs warnings for suspicious content but doesn't block downloads. Review warnings in logs:

```
[INFO] Security warning for https://example.com/file.py: Suspicious pattern detected: eval
```

## Reporting Security Issues

If you discover a security vulnerability:

1. **Do not** create a public issue
2. Review the code in `mcp_web_parser/security.py`
3. Report privately if needed

## Dependencies

### Required
- `requests`: HTTP client
- `beautifulsoup4`: HTML parsing

### Optional (Enhanced Security)
- `python-magic` or `python-magic-bin`: File type detection
  ```bash
  pip install python-magic-bin  # Windows
  pip install python-magic      # Linux/macOS
  ```

## Example: Secure Download Workflow

```python
from mcp_web_parser.server import DownloadManager
from mcp_web_parser.security import validate_url

# 1. Validate URL first
is_valid, error = validate_url("https://example.com/file.pdf")
if not is_valid:
    print(f"Invalid URL: {error}")
    return

# 2. Download with automatic validation
downloader = DownloadManager()
path = downloader.download_file("https://example.com/file.pdf")

if path:
    print(f"Downloaded safely: {path}")
else:
    print("Download failed or was quarantined")
```

## Security Checklist

Before using in production:

- [ ] Review and customize `SAFE_EXTENSIONS` for your use case
- [ ] Set appropriate `MAX_FILE_SIZE` limits
- [ ] Configure quarantine directory location
- [ ] Set up monitoring for security warnings
- [ ] Review quarantined files regularly
- [ ] Consider adding virus scanning for production use
- [ ] Test with known malicious files (in isolated environment)

