"""
FastMCP server that exposes DAiW tools over the Model Context Protocol.

This module keeps the MCP dependency optional so the core library can be
installed without pulling in uvicorn/httpx. Importing this file succeeds
even if `mcp` is not installed; attempting to build or run the server raises
an informative error instead.
"""

from __future__ import annotations

from typing import Literal

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - exercised via unit tests that patch availability
    FastMCP = None  # type: ignore[misc]

from .tools import register_all_tools


class MCPNotAvailableError(RuntimeError):
    """Raised when the optional MCP dependency is not installed."""


def build_server(name: str = "daiw-mcp") -> "FastMCP":
    """
    Build a FastMCP server and register all DAiW tools.

    Args:
        name: Human-readable server name reported to MCP clients.

    Returns:
        Configured FastMCP instance.
    """

    if FastMCP is None:
        raise MCPNotAvailableError(
            "The `mcp` package is not installed. Install DAiW with the MCP extras:\n"
            "    pip install -e .[mcp]\n"
            "or install the upstream package directly:\n"
            "    pip install mcp\n"
        )

    server = FastMCP(name)
    register_all_tools(server)
    return server


def run(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """
    Convenience entry point that builds the server and starts it.

    Args:
        transport: How the MCP server should communicate with the client.
    """

    server = build_server()
    server.run(transport=transport)


if __name__ == "__main__":
    run()

