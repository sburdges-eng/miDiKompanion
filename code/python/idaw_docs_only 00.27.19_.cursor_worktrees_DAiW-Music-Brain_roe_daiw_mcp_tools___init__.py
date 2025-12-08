"""
MCP Tools - Tool modules for DAiW MCP server.

Each module provides a set of related tools that AI assistants can use.
"""

# Import all tool modules for registration
from daiw_mcp.tools import (
    harmony_tools,
    groove_tools,
    intent_tools,
    audio_tools,
    teaching_tools,
)

__all__ = [
    "harmony_tools",
    "groove_tools",
    "intent_tools",
    "audio_tools",
    "teaching_tools",
]

