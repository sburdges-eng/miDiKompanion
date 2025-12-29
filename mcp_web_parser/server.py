"""
MCP Web Parser - Parallel Web Scraping & Training Data Collection

Exposes web parsing, preview, and download functionality as MCP tools.
Designed for parallel execution across 4 GPT Codex instances for training data collection.
"""

import json
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urljoin
from datetime import datetime
import hashlib

try:
    import requests
    from bs4 import BeautifulSoup
    import markdownify
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    # Create stubs for type checking
    requests = None
    BeautifulSoup = None
    markdownify = None

import logging

# Simple logging for web parser (can be enhanced later)
log_info = lambda category, msg: logging.info(f"[{category}] {msg}")
log_error = lambda category, msg: logging.error(f"[{category}] {msg}")

class DebugCategory:
    MCP = "mcp"

# Import security module
try:
    from .security import (
        validate_download,
        quarantine_file,
        get_security_summary,
        SecurityError,
    )
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    # Stubs for when security module not available
    def validate_download(*args, **kwargs):
        return True, None, []
    def quarantine_file(path, reason):
        return path
    def get_security_summary():
        return {"error": "Security module not available"}
    SecurityError = Exception


# =============================================================================
# Configuration
# =============================================================================

# Storage paths
DATA_ROOT = Path.home() / ".mcp_web_parser"
PARSED_DATA_DIR = DATA_ROOT / "parsed"
DOWNLOAD_DIR = DATA_ROOT / "downloads"
PREVIEW_DIR = DATA_ROOT / "previews"
METADATA_FILE = DATA_ROOT / "metadata.json"

# Parallel execution settings
MAX_WORKERS = 4  # For 4 parallel GPT Codex instances
RATE_LIMIT_SECONDS = 1.0  # Minimum delay between requests per domain
REQUEST_TIMEOUT = 30  # Seconds

# User agent for respectful scraping
USER_AGENT = "MCP-WebParser/1.0 (Training Data Collection)"


# =============================================================================
# Data Models
# =============================================================================

class ParsedPage:
    """Represents a parsed web page with extracted content."""
    
    def __init__(
        self,
        url: str,
        title: str,
        content: str,
        markdown: str,
        metadata: Dict[str, Any],
        timestamp: Optional[str] = None,
    ):
        self.url = url
        self.title = title
        self.content = content
        self.markdown = markdown
        self.metadata = metadata
        self.timestamp = timestamp or datetime.now().isoformat()
        self.url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "markdown": self.markdown,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "url_hash": self.url_hash,
        }
    
    def save(self, output_dir: Path) -> Path:
        """Save parsed page to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_file = output_dir / f"{self.url_hash}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Save markdown separately for easy reading
        md_file = output_dir / f"{self.url_hash}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(f"# {self.title}\n\n")
            f.write(f"**URL:** {self.url}\n\n")
            f.write(f"**Parsed:** {self.timestamp}\n\n")
            f.write("---\n\n")
            f.write(self.markdown)
        
        return json_file


class DownloadTask:
    """Represents a file download task."""
    
    def __init__(
        self,
        url: str,
        destination: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.url = url
        self.destination = destination or self._generate_destination()
        self.metadata = metadata or {}
        self.status = "pending"
        self.error: Optional[str] = None
        self.timestamp = datetime.now().isoformat()
    
    def _generate_destination(self) -> Path:
        """Generate destination path from URL."""
        parsed = urlparse(self.url)
        filename = Path(parsed.path).name or "download"
        if not filename or "." not in filename:
            filename += ".html"  # Default extension
        
        # Create subdirectory based on domain
        domain = parsed.netloc.replace(".", "_")
        download_dir = DOWNLOAD_DIR / domain
        download_dir.mkdir(parents=True, exist_ok=True)
        
        return download_dir / filename


# =============================================================================
# Web Parser
# =============================================================================

class WebParser:
    """Parses web pages and extracts structured content."""
    
    def __init__(self, rate_limit: float = RATE_LIMIT_SECONDS):
        self.rate_limit = rate_limit
        self._last_request: Dict[str, float] = {}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
    
    def _respect_rate_limit(self, domain: str) -> None:
        """Wait if necessary to respect rate limiting."""
        import time
        last_time = self._last_request.get(domain, 0)
        elapsed = time.time() - last_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request[domain] = time.time()
    
    def parse_url(self, url: str) -> Optional[ParsedPage]:
        """Parse a single URL and extract content."""
        if not DEPENDENCIES_AVAILABLE:
            return None
        
        try:
            # Rate limiting
            domain = urlparse(url).netloc
            self._respect_rate_limit(domain)
            
            # Fetch page
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract title
            title = soup.find("title")
            title_text = title.get_text(strip=True) if title else "Untitled"
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract main content
            main_content = soup.find("main") or soup.find("article") or soup.find("body")
            if main_content:
                content_text = main_content.get_text(separator="\n", strip=True)
            else:
                content_text = soup.get_text(separator="\n", strip=True)
            
            # Convert to markdown
            markdown_content = markdownify.markdownify(
                str(main_content) if main_content else str(soup),
                heading_style="ATX",
            )
            
            # Extract metadata
            metadata = {
                "domain": domain,
                "content_length": len(content_text),
                "markdown_length": len(markdown_content),
                "links": [a.get("href") for a in soup.find_all("a", href=True)],
                "images": [img.get("src") for img in soup.find_all("img", src=True)],
                "meta_description": self._extract_meta(soup, "description"),
                "meta_keywords": self._extract_meta(soup, "keywords"),
            }
            
            return ParsedPage(
                url=url,
                title=title_text,
                content=content_text,
                markdown=markdown_content,
                metadata=metadata,
            )
        
        except Exception as e:
            log_error(DebugCategory.MCP, f"Error parsing {url}: {e}")
            return None
    
    def _extract_meta(self, soup: BeautifulSoup, name: str) -> Optional[str]:
        """Extract meta tag content."""
        meta = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": f"og:{name}"})
        return meta.get("content") if meta else None


class ParallelParser:
    """Manages parallel parsing across multiple workers."""
    
    def __init__(self, max_workers: int = MAX_WORKERS):
        self.max_workers = max_workers
        self.parser = WebParser()
    
    def parse_urls(self, urls: List[str]) -> List[ParsedPage]:
        """Parse multiple URLs in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(self.parser.parse_url, url): url
                for url in urls
            }
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    parsed = future.result()
                    if parsed:
                        results.append(parsed)
                except Exception as e:
                    log_error(DebugCategory.MCP, f"Error parsing {url}: {e}")
        
        return results


# =============================================================================
# Download Manager
# =============================================================================

class DownloadManager:
    """Manages file downloads for training data with security validation."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
    
    def download_file(
        self,
        url: str,
        destination: Optional[Path] = None,
        validate: bool = True,
    ) -> Optional[Path]:
        """
        Download a file from URL with security validation.
        
        Args:
            url: URL to download
            destination: Optional destination path
            validate: Whether to perform security validation (default: True)
        
        Returns:
            Path to downloaded file, or None if download/validation failed
        """
        try:
            task = DownloadTask(url, destination)
            task.status = "downloading"
            
            # Get response with headers
            response = self.session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            response.raise_for_status()
            
            content_type = response.headers.get("Content-Type")
            content_length = response.headers.get("Content-Length")
            content_length_int = int(content_length) if content_length else None
            
            # Ensure destination directory exists
            task.destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Download to temporary file first
            temp_file = task.destination.with_suffix(task.destination.suffix + ".tmp")
            
            # Download with progress
            with open(temp_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Security validation
            if validate:
                is_valid, error_msg, warnings = validate_download(
                    url, content_type, content_length_int, temp_file
                )
                
                if not is_valid:
                    # Move to quarantine
                    quarantine_path = quarantine_file(temp_file, error_msg or "Validation failed")
                    log_error(DebugCategory.MCP, f"Download failed validation: {url} → {quarantine_path}")
                    return None
                
                # Log warnings but allow download
                if warnings:
                    for warning in warnings:
                        log_info(DebugCategory.MCP, f"Security warning for {url}: {warning}")
            
            # Move temp file to final destination
            temp_file.rename(task.destination)
            task.status = "completed"
            return task.destination
        
        except Exception as e:
            log_error(DebugCategory.MCP, f"Error downloading {url}: {e}")
            return None
    
    def download_parallel(self, urls: List[str]) -> List[Path]:
        """Download multiple files in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {
                executor.submit(self.download_file, url): url
                for url in urls
            }
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    path = future.result()
                    if path:
                        results.append(path)
                except Exception as e:
                    log_error(DebugCategory.MCP, f"Error downloading {url}: {e}")
        
        return results


# =============================================================================
# Metadata Manager
# =============================================================================

class MetadataManager:
    """Manages metadata for parsed pages and downloads."""
    
    def __init__(self, metadata_file: Path = METADATA_FILE):
        self.metadata_file = metadata_file
        self.metadata: Dict[str, Any] = self._load()
    
    def _load(self) -> Dict[str, Any]:
        """Load metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "parsed_pages": [],
            "downloads": [],
            "statistics": {
                "total_parsed": 0,
                "total_downloaded": 0,
                "last_updated": None,
            },
        }
    
    def save(self):
        """Save metadata to disk."""
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def add_parsed_page(self, parsed: ParsedPage):
        """Add parsed page to metadata."""
        entry = {
            "url": parsed.url,
            "url_hash": parsed.url_hash,
            "title": parsed.title,
            "timestamp": parsed.timestamp,
            "metadata": parsed.metadata,
        }
        self.metadata["parsed_pages"].append(entry)
        self.metadata["statistics"]["total_parsed"] += 1
        self.metadata["statistics"]["last_updated"] = datetime.now().isoformat()
        self.save()
    
    def add_download(self, url: str, path: Path):
        """Add download to metadata."""
        entry = {
            "url": url,
            "path": str(path),
            "timestamp": datetime.now().isoformat(),
        }
        self.metadata["downloads"].append(entry)
        self.metadata["statistics"]["total_downloaded"] += 1
        self.metadata["statistics"]["last_updated"] = datetime.now().isoformat()
        self.save()


# =============================================================================
# Global Instances
# =============================================================================

_parser = ParallelParser()
_download_manager = DownloadManager()
_metadata = MetadataManager()


def get_mcp_tools() -> List[Dict[str, Any]]:
    """Get MCP tool definitions for web parsing."""
    return [
        {
            "name": "web_parse_url",
            "description": "Parse a single URL and extract structured content (text, markdown, metadata). Returns parsed content ready for training data.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to parse",
                    },
                    "save": {
                        "type": "boolean",
                        "description": "Save parsed content to disk (default: true)",
                        "default": True,
                    },
                },
                "required": ["url"],
            },
        },
        {
            "name": "web_parse_batch",
            "description": "Parse multiple URLs in parallel (optimized for 4 parallel GPT Codex instances). Returns list of parsed pages.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of URLs to parse in parallel",
                    },
                    "save": {
                        "type": "boolean",
                        "description": "Save parsed content to disk (default: true)",
                        "default": True,
                    },
                },
                "required": ["urls"],
            },
        },
        {
            "name": "web_preview_url",
            "description": "Preview a URL without full parsing. Returns title, description, and basic metadata for quick assessment.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to preview",
                    },
                },
                "required": ["url"],
            },
        },
        {
            "name": "web_download_file",
            "description": "Download a file from URL for training data collection. Supports parallel downloads.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of file to download",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Optional destination path (default: auto-generated)",
                    },
                },
                "required": ["url"],
            },
        },
        {
            "name": "web_download_batch",
            "description": "Download multiple files in parallel (optimized for 4 parallel GPT Codex instances).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file URLs to download in parallel",
                    },
                },
                "required": ["urls"],
            },
        },
        {
            "name": "web_get_statistics",
            "description": "Get statistics about parsed pages and downloads.",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "web_get_security_info",
            "description": "Get security configuration and validation status.",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "web_list_parsed",
            "description": "List all parsed pages with metadata.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 100)",
                        "default": 100,
                    },
                },
            },
        },
    ]


def handle_tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP tool calls."""
    try:
        if name == "web_parse_url":
            url = arguments["url"]
            save = arguments.get("save", True)
            
            parsed = _parser.parser.parse_url(url)
            if not parsed:
                return {"error": f"Failed to parse {url}"}
            
            if save:
                output_file = parsed.save(PARSED_DATA_DIR)
                _metadata.add_parsed_page(parsed)
                return {
                    "success": True,
                    "url": parsed.url,
                    "title": parsed.title,
                    "url_hash": parsed.url_hash,
                    "output_file": str(output_file),
                    "content_length": len(parsed.content),
                    "markdown_length": len(parsed.markdown),
                    "metadata": parsed.metadata,
                }
            else:
                return {
                    "success": True,
                    "url": parsed.url,
                    "title": parsed.title,
                    "content": parsed.content,
                    "markdown": parsed.markdown,
                    "metadata": parsed.metadata,
                }
        
        elif name == "web_parse_batch":
            urls = arguments["urls"]
            save = arguments.get("save", True)
            
            parsed_pages = _parser.parse_urls(urls)
            
            results = []
            for parsed in parsed_pages:
                if save:
                    output_file = parsed.save(PARSED_DATA_DIR)
                    _metadata.add_parsed_page(parsed)
                    results.append({
                        "url": parsed.url,
                        "title": parsed.title,
                        "url_hash": parsed.url_hash,
                        "output_file": str(output_file),
                    })
                else:
                    results.append({
                        "url": parsed.url,
                        "title": parsed.title,
                        "content": parsed.content,
                        "markdown": parsed.markdown,
                    })
            
            return {
                "success": True,
                "total": len(urls),
                "parsed": len(parsed_pages),
                "results": results,
            }
        
        elif name == "web_preview_url":
            url = arguments["url"]
            
            if not DEPENDENCIES_AVAILABLE:
                return {"error": "Dependencies not available. Install: pip install requests beautifulsoup4 markdownify"}
            
            # Quick preview without full parsing
            try:
                response = requests.get(url, timeout=10, headers={"User-Agent": USER_AGENT})
                soup = BeautifulSoup(response.text, "html.parser")
                
                title = soup.find("title")
                title_text = title.get_text(strip=True) if title else "Untitled"
                
                meta_desc = soup.find("meta", attrs={"name": "description"})
                description = meta_desc.get("content") if meta_desc else None
                
                return {
                    "success": True,
                    "url": url,
                    "title": title_text,
                    "description": description,
                    "status_code": response.status_code,
                    "content_type": response.headers.get("Content-Type"),
                }
            except Exception as e:
                return {"error": str(e)}
        
        elif name == "web_download_file":
            url = arguments["url"]
            destination = Path(arguments["destination"]) if arguments.get("destination") else None
            
            path = _download_manager.download_file(url, destination)
            if path:
                _metadata.add_download(url, path)
                return {
                    "success": True,
                    "url": url,
                    "destination": str(path),
                    "size_bytes": path.stat().st_size if path.exists() else 0,
                }
            else:
                return {"error": f"Failed to download {url}"}
        
        elif name == "web_download_batch":
            urls = arguments["urls"]
            
            paths = _download_manager.download_parallel(urls)
            
            results = []
            for path in paths:
                _metadata.add_download(urls[paths.index(path)], path)
                results.append({
                    "url": urls[paths.index(path)],
                    "destination": str(path),
                    "size_bytes": path.stat().st_size if path.exists() else 0,
                })
            
            return {
                "success": True,
                "total": len(urls),
                "downloaded": len(paths),
                "results": results,
            }
        
        elif name == "web_get_statistics":
            result = {
                "success": True,
                "statistics": _metadata.metadata["statistics"],
                "data_directory": str(DATA_ROOT),
                "parsed_directory": str(PARSED_DATA_DIR),
                "download_directory": str(DOWNLOAD_DIR),
            }
            if SECURITY_AVAILABLE:
                result["security"] = get_security_summary()
            return result
        
        elif name == "web_get_security_info":
            if not SECURITY_AVAILABLE:
                return {
                    "error": "Security module not available. Install: pip install python-magic-bin (optional)"
                }
            return {
                "success": True,
                "security_config": get_security_summary(),
                "quarantine_directory": str(Path.home() / ".mcp_web_parser" / "quarantine"),
            }
        
        elif name == "web_list_parsed":
            limit = arguments.get("limit", 100)
            pages = _metadata.metadata["parsed_pages"][-limit:]
            return {
                "success": True,
                "total": len(_metadata.metadata["parsed_pages"]),
                "returned": len(pages),
                "pages": pages,
            }
        
        else:
            return {"error": f"Unknown tool: {name}"}
    
    except Exception as e:
        log_error(DebugCategory.MCP, f"Tool error: {name}: {e}")
        return {"error": str(e)}


# =============================================================================
# MCP Server (stdio transport)
# =============================================================================

def run_server():
    """Run the MCP server using stdio transport."""
    log_info(DebugCategory.MCP, "MCP Web Parser server starting")
    
    tools = get_mcp_tools()
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            request = json.loads(line)
            method = request.get("method", "")
            params = request.get("params", {})
            req_id = request.get("id")
            
            response = {"jsonrpc": "2.0", "id": req_id}
            
            if method == "initialize":
                response["result"] = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                    },
                    "serverInfo": {
                        "name": "mcp-web-parser",
                        "version": "1.0.0",
                    },
                }
            
            elif method == "tools/list":
                response["result"] = {"tools": tools}
            
            elif method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments", {})
                result = handle_tool_call(tool_name, tool_args)
                response["result"] = {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2),
                        }
                    ],
                }
            
            elif method == "notifications/initialized":
                continue
            
            else:
                response["error"] = {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                }
            
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        
        except json.JSONDecodeError as e:
            log_error(DebugCategory.MCP, f"JSON decode error: {e}")
        except Exception as e:
            log_error(DebugCategory.MCP, f"Server error: {e}")


if __name__ == "__main__":
    run_server()

