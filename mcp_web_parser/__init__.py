"""
MCP Web Parser - Parallel Web Scraping & Training Data Collection

MCP server for web parsing, preview, and download operations.
Optimized for parallel execution across multiple GPT Codex instances.
"""

__version__ = "1.0.0"

from .server import (
    get_mcp_tools,
    handle_tool_call,
    run_server,
    WebParser,
    ParallelParser,
    DownloadManager,
    MetadataManager,
    ParsedPage,
    DownloadTask,
)

__all__ = [
    "get_mcp_tools",
    "handle_tool_call",
    "run_server",
    "WebParser",
    "ParallelParser",
    "DownloadManager",
    "MetadataManager",
    "ParsedPage",
    "DownloadTask",
]

