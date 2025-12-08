"""
Tool registration helpers.

Each tool module exposes a `register_tools(server)` function that receives a
FastMCP server instance and attaches its handlers.
"""

from __future__ import annotations

from importlib import import_module
from typing import Iterable, Protocol


class _FastMCPServer(Protocol):
    """Structural type describing the methods we need from FastMCP."""

    def tool(self, *args, **kwargs):  # pragma: no cover - interface proxy
        ...


TOOL_MODULES: Iterable[str] = (
    "daiw_mcp.tools.harmony",
    "daiw_mcp.tools.groove",
    "daiw_mcp.tools.intent",
    "daiw_mcp.tools.therapy",
)


def register_all_tools(server: _FastMCPServer) -> None:
    """
    Import each tool module and register its handlers with the MCP server.
    """

    for module_name in TOOL_MODULES:
        module = import_module(module_name)
        register = getattr(module, "register_tools", None)
        if register is None:
            raise RuntimeError(f"Tool module '{module_name}' is missing register_tools()")
        register(server)

