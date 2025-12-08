"""
MCP Server - Main entry point for DAiW Model Context Protocol server.

This server exposes DAiW's music analysis and generation tools to AI assistants
via the Model Context Protocol.
"""

import asyncio
import sys
from typing import Any, Sequence

# MCP SDK imports - try different import patterns
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    try:
        # Alternative import pattern
        from mcp import Server, stdio_server, Tool, TextContent
        MCP_AVAILABLE = True
    except ImportError:
        MCP_AVAILABLE = False
        print("Warning: MCP SDK not installed. Install with: pip install mcp")
        print("MCP server will not be available.")


# Import tool modules
from daiw_mcp.tools import (
    harmony_tools,
    groove_tools,
    intent_tools,
    audio_tools,
    teaching_tools,
)


def create_server() -> Server:
    """Create and configure the MCP server with all tools."""
    if not MCP_AVAILABLE:
        raise ImportError("MCP SDK not available. Install with: pip install mcp")
    
    server = Server("daiw-music-brain")
    
    # Register all tool modules
    harmony_tools.register_tools(server)
    groove_tools.register_tools(server)
    intent_tools.register_tools(server)
    audio_tools.register_tools(server)
    teaching_tools.register_tools(server)
    
    return server


async def main():
    """Main entry point for MCP server."""
    if not MCP_AVAILABLE:
        print("Error: MCP SDK not installed.")
        print("Install with: pip install mcp")
        sys.exit(1)
    
    server = create_server()
    
    # Run server with stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())

