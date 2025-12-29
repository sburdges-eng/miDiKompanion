"""
Security Module for Web Parser

Validates downloads and prevents malicious files/code from being saved.
"""

import re
import magic
import hashlib
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from urllib.parse import urlparse
import mimetypes

# Try to import python-magic, fallback to mimetypes if not available
try:
    import magic as magic_lib
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False


# =============================================================================
# Security Configuration
# =============================================================================

# Whitelist of safe file extensions for downloads
SAFE_EXTENSIONS = {
    # Text formats
    '.txt', '.md', '.markdown', '.rst', '.org',
    # Document formats (read-only, no macros)
    '.pdf', '.epub',
    # Data formats
    '.json', '.jsonl', '.csv', '.tsv', '.xml',
    # Archive formats (will be extracted and validated)
    '.zip', '.tar', '.tar.gz', '.tgz',
    # Code (for training data, but will be scanned)
    '.py', '.js', '.ts', '.cpp', '.c', '.h', '.hpp', '.java', '.go', '.rs',
    '.rb', '.php', '.swift', '.kt', '.scala', '.sh', '.bash',
    # Web formats
    '.html', '.htm', '.css',
    # No executable extensions allowed
}

# Blacklist of dangerous extensions (never allow)
# Note: .sh, .bash, .js are in SAFE_EXTENSIONS but will be scanned carefully
DANGEROUS_EXTENSIONS = {
    '.exe', '.bat', '.cmd', '.com', '.scr', '.vbs', '.jar',
    '.dll', '.so', '.dylib', '.app', '.deb', '.rpm', '.msi', '.pkg',
    '.zsh', '.ps1', '.psm1', '.psd1',
}

# Maximum file size (100 MB default)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

# Maximum content length from Content-Type header
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB

# Suspicious patterns in file content
SUSPICIOUS_PATTERNS = [
    # Executable signatures
    re.compile(rb'MZ\0\0', re.IGNORECASE),  # Windows PE
    re.compile(rb'\x7fELF'),  # Linux ELF
    re.compile(rb'PK\x03\x04'),  # ZIP (could contain executables)
    # Script injection patterns
    re.compile(rb'eval\s*\(', re.IGNORECASE),
    re.compile(rb'exec\s*\(', re.IGNORECASE),
    re.compile(rb'system\s*\(', re.IGNORECASE),
    re.compile(rb'shell_exec', re.IGNORECASE),
    re.compile(rb'base64_decode', re.IGNORECASE),
    # Network patterns
    re.compile(rb'curl\s+http', re.IGNORECASE),
    re.compile(rb'wget\s+http', re.IGNORECASE),
    re.compile(rb'nc\s+-l', re.IGNORECASE),  # netcat listener
    # Suspicious URLs
    re.compile(rb'http://[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'),
    # Base64 encoded suspicious content
    re.compile(rb'[A-Za-z0-9+/]{100,}={0,2}'),  # Long base64 strings
]

# Allowed MIME types
ALLOWED_MIME_TYPES = {
    'text/plain', 'text/html', 'text/css', 'text/markdown', 'text/xml',
    'application/json', 'application/xml', 'application/pdf',
    'application/zip', 'application/x-tar', 'application/gzip',
    'text/x-python', 'text/javascript', 'text/x-c', 'text/x-c++',
    'application/x-sh', 'text/x-shellscript',
}

# Quarantine directory for suspicious files
QUARANTINE_DIR = Path.home() / ".mcp_web_parser" / "quarantine"


# =============================================================================
# Security Validators
# =============================================================================

class SecurityError(Exception):
    """Security validation error."""
    pass


def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate URL before processing.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        parsed = urlparse(url)
        
        # Must have scheme
        if not parsed.scheme:
            return False, "URL must have a scheme (http/https)"
        
        # Only allow http/https
        if parsed.scheme not in ('http', 'https'):
            return False, f"Unsupported scheme: {parsed.scheme} (only http/https allowed)"
        
        # Must have netloc
        if not parsed.netloc:
            return False, "URL must have a domain"
        
        # Check for suspicious patterns
        suspicious_domains = ['localhost', '127.0.0.1', '0.0.0.0']
        if any(domain in parsed.netloc.lower() for domain in suspicious_domains):
            return False, f"Suspicious domain detected: {parsed.netloc}"
        
        return True, None
    
    except Exception as e:
        return False, f"Invalid URL format: {str(e)}"


def validate_file_extension(filename: str) -> Tuple[bool, Optional[str]]:
    """
    Validate file extension against whitelist/blacklist.
    
    Returns:
        (is_valid, error_message)
    """
    path = Path(filename)
    ext = path.suffix.lower()
    
    # Check blacklist first (absolute blocks)
    if ext in DANGEROUS_EXTENSIONS:
        return False, f"Dangerous extension not allowed: {ext}"
    
    # Check whitelist
    if ext not in SAFE_EXTENSIONS:
        return False, f"Extension not in whitelist: {ext}"
    
    # Extensions in SAFE_EXTENSIONS but potentially risky (.sh, .bash, .js)
    # are allowed but will be scanned carefully in scan_file_content()
    return True, None


def validate_content_type(content_type: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate Content-Type header.
    
    Returns:
        (is_valid, error_message)
    """
    if not content_type:
        return True, None  # Allow if not specified
    
    # Extract MIME type (remove charset, etc.)
    mime_type = content_type.split(';')[0].strip().lower()
    
    # Check if it's in allowed list
    if mime_type in ALLOWED_MIME_TYPES:
        return True, None
    
    # Check if it starts with allowed prefix
    for allowed in ALLOWED_MIME_TYPES:
        if mime_type.startswith(allowed.split('/')[0] + '/'):
            return True, None
    
    return False, f"Content-Type not allowed: {content_type}"


def validate_file_size(file_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate file size.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        size = file_path.stat().st_size
        if size > MAX_FILE_SIZE:
            return False, f"File too large: {size} bytes (max: {MAX_FILE_SIZE})"
        return True, None
    except Exception as e:
        return False, f"Error checking file size: {str(e)}"


def scan_file_content(file_path: Path, sample_size: int = 8192) -> Tuple[bool, Optional[str], List[str]]:
    """
    Scan file content for suspicious patterns.
    
    Args:
        file_path: Path to file to scan
        sample_size: Number of bytes to read for initial scan
    
    Returns:
        (is_safe, error_message, warnings)
    """
    warnings = []
    
    try:
        # Read first chunk for pattern matching
        with open(file_path, 'rb') as f:
            sample = f.read(sample_size)
        
        # Check for suspicious patterns
        for pattern in SUSPICIOUS_PATTERNS:
            if pattern.search(sample):
                pattern_name = pattern.pattern.decode('utf-8', errors='ignore')[:50]
                warnings.append(f"Suspicious pattern detected: {pattern_name}")
        
        # For text files, do additional checks
        try:
            text_sample = sample.decode('utf-8', errors='ignore')
            
            # Check for suspicious function calls
            dangerous_functions = ['eval', 'exec', 'system', 'shell_exec', '__import__']
            for func in dangerous_functions:
                if func in text_sample.lower():
                    warnings.append(f"Potentially dangerous function call: {func}")
        
        except UnicodeDecodeError:
            # Binary file, that's okay for some types
            pass
        
        # If we have warnings, file is suspicious but not necessarily malicious
        # Return warnings but don't block (user can review)
        return True, None, warnings
    
    except Exception as e:
        return False, f"Error scanning file: {str(e)}", []


def detect_file_type(file_path: Path) -> Optional[str]:
    """
    Detect actual file type using magic numbers.
    
    Returns:
        MIME type or None if detection fails
    """
    if not MAGIC_AVAILABLE:
        # Fallback to extension-based detection
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type
    
    try:
        mime = magic_lib.Magic(mime=True)
        return mime.from_file(str(file_path))
    except Exception:
        return None


def validate_download(
    url: str,
    content_type: Optional[str],
    content_length: Optional[int],
    file_path: Path,
) -> Tuple[bool, Optional[str], List[str]]:
    """
    Comprehensive validation of a download.
    
    Args:
        url: Source URL
        content_type: Content-Type header
        content_length: Content-Length header
        file_path: Path to downloaded file
    
    Returns:
        (is_valid, error_message, warnings)
    """
    warnings = []
    
    # 1. Validate URL
    url_valid, url_error = validate_url(url)
    if not url_valid:
        return False, f"URL validation failed: {url_error}", []
    
    # 2. Validate Content-Length header
    if content_length and content_length > MAX_CONTENT_LENGTH:
        return False, f"Content-Length too large: {content_length} bytes", []
    
    # 3. Validate file extension
    ext_valid, ext_error = validate_file_extension(file_path.name)
    if not ext_valid:
        return False, f"File extension validation failed: {ext_error}", []
    
    # 4. Validate Content-Type
    ct_valid, ct_error = validate_content_type(content_type)
    if not ct_valid:
        warnings.append(f"Content-Type validation warning: {ct_error}")
        # Don't block, just warn
    
    # 5. Validate file size
    if file_path.exists():
        size_valid, size_error = validate_file_size(file_path)
        if not size_valid:
            return False, f"File size validation failed: {size_error}", []
    
    # 6. Detect actual file type
    detected_type = detect_file_type(file_path) if file_path.exists() else None
    if detected_type and content_type:
        # Verify Content-Type matches actual file type
        if detected_type != content_type.split(';')[0].strip().lower():
            warnings.append(
                f"Content-Type mismatch: header says '{content_type}' but file is '{detected_type}'"
            )
    
    # 7. Scan file content
    if file_path.exists():
        content_safe, content_error, content_warnings = scan_file_content(file_path)
        if not content_safe:
            return False, f"Content scan failed: {content_error}", []
        warnings.extend(content_warnings)
    
    return True, None, warnings


def quarantine_file(file_path: Path, reason: str) -> Path:
    """
    Move file to quarantine directory.
    
    Args:
        file_path: Path to file to quarantine
        reason: Reason for quarantine
    
    Returns:
        Path to quarantined file
    """
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create unique filename with reason hash
    reason_hash = hashlib.md5(reason.encode()).hexdigest()[:8]
    quarantine_name = f"{file_path.stem}_{reason_hash}{file_path.suffix}"
    quarantine_path = QUARANTINE_DIR / quarantine_name
    
    # Move file
    file_path.rename(quarantine_path)
    
    # Log quarantine reason
    log_file = QUARANTINE_DIR / "quarantine.log"
    with open(log_file, "a") as f:
        f.write(f"{quarantine_path.name}|{reason}|{file_path}\n")
    
    return quarantine_path


def get_security_summary() -> Dict[str, Any]:
    """Get security configuration summary."""
    return {
        "max_file_size": MAX_FILE_SIZE,
        "max_content_length": MAX_CONTENT_LENGTH,
        "safe_extensions_count": len(SAFE_EXTENSIONS),
        "dangerous_extensions_count": len(DANGEROUS_EXTENSIONS),
        "allowed_mime_types_count": len(ALLOWED_MIME_TYPES),
        "suspicious_patterns_count": len(SUSPICIOUS_PATTERNS),
        "magic_available": MAGIC_AVAILABLE,
        "quarantine_directory": str(QUARANTINE_DIR),
    }

