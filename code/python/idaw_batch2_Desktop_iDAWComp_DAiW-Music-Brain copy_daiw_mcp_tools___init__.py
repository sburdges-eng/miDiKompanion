"""
MCP Tools - Tool modules for DAiW MCP server.

Each module provides a set of related tools that AI assistants can use.
This package auto-discovers any `*.py` file exposing a `register_tools(server)`
function, so adding new tool groups requires zero boilerplate in server.py.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import List

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from mcp.server import Server

_TOOL_MODULE_CACHE: List[ModuleType] | None = None


def iter_tool_modules() -> List[ModuleType]:
    """Discover and import all tool modules in this package."""

    global _TOOL_MODULE_CACHE
    if _TOOL_MODULE_CACHE is not None:
        return _TOOL_MODULE_CACHE

    modules: List[ModuleType] = []
    prefix = f"{__name__}."

    for module_info in pkgutil.iter_modules(__path__, prefix):  # type: ignore[name-defined]
        module = importlib.import_module(module_info.name)
        if hasattr(module, "register_tools"):
            modules.append(module)

    _TOOL_MODULE_CACHE = modules
    return modules


def register_all_tools(server: "Server") -> None:
    """Register every tool module with the MCP server."""

    for module in iter_tool_modules():
        module.register_tools(server)  # type: ignore[attr-defined]


__all__ = ["iter_tool_modules", "register_all_tools"]