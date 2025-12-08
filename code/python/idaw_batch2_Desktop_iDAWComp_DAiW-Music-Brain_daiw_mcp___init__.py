"""
DAiW MCP Bridge.

This package exposes the DAiW Music Brain capabilities over the
Model Context Protocol (MCP) so that external copilots can call
into the toolkit using standardized tools.

Usage:
    from daiw_mcp.server import build_server
    server = build_server()
    server.run()
"""

from __future__ import annotations

from .server import build_server

__all__ = ["build_server"]

