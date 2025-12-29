# Security Features Summary

## Quick Answer: How We Prevent Malicious Files

The MCP Web Parser includes **7 layers of security** to prevent malicious files and code:

1. ✅ **URL Validation** - Only http/https, blocks localhost
2. ✅ **File Extension Whitelist** - Only safe extensions allowed (no .exe, .bat, etc.)
3. ✅ **Content-Type Validation** - Verifies HTTP headers match allowed types
4. ✅ **File Size Limits** - 100 MB max per file
5. ✅ **Content Scanning** - Detects executable signatures and suspicious patterns
6. ✅ **File Type Detection** - Uses magic numbers to verify actual file type
7. ✅ **Quarantine System** - Suspicious files moved to isolated directory

## What Gets Blocked

### Automatically Blocked
- Executables: `.exe`, `.bat`, `.cmd`, `.com`, `.scr`, `.vbs`, `.jar`
- Libraries: `.dll`, `.so`, `.dylib`, `.app`
- Installers: `.deb`, `.rpm`, `.msi`, `.pkg`
- Files over 100 MB
- Files with executable signatures (PE, ELF)
- Files with suspicious code patterns (`eval()`, `exec()`, etc.)

### Allowed (But Scanned)
- Code files: `.py`, `.js`, `.ts`, `.cpp`, etc. (scanned for malicious patterns)
- Scripts: `.sh`, `.bash` (scanned carefully)
- Documents: `.pdf`, `.epub` (read-only)
- Data: `.json`, `.csv`, `.xml`

## Quarantine System

Files that fail validation are **not deleted** - they're moved to:
```
~/.mcp_web_parser/quarantine/
```

You can review them manually. Check `quarantine.log` for reasons.

## Security Status

Check security configuration:
```json
{
  "tool": "web_get_security_info",
  "arguments": {}
}
```

## Default Behavior

**Security validation is ON by default** for all downloads. You don't need to do anything - it's automatic.

## Customization

Edit `mcp_web_parser/security.py` to:
- Add/remove allowed file extensions
- Adjust file size limits
- Add custom suspicious patterns

## Limitations

- **No virus scanning**: Uses pattern matching, not actual virus detection
- **No code execution prevention**: Code files are saved but not executed
- **No sandboxing**: Files saved to disk (but in isolated directory)

For production use, consider adding:
- ClamAV integration for virus scanning
- Sandboxing for code execution
- Deep archive scanning

## See Also

- `SECURITY.md` - Complete security documentation
- `server.py` - Download manager with validation
- `security.py` - Security validation logic

