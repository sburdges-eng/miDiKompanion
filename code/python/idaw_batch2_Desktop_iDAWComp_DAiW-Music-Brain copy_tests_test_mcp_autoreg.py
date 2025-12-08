"""
Tests for the MCP auto-registration utilities.

These tests intentionally avoid importing the heavy MCP SDK; they only verify
the discovery helpers in `daiw_mcp.tools`.
"""

from daiw_mcp import tools as mcp_tools


def test_iter_tool_modules_discovers_expected_modules():
    modules = mcp_tools.iter_tool_modules()
    names = {module.__name__.split(".")[-1] for module in modules}

    expected = {
        "harmony_tools",
        "groove_tools",
        "intent_tools",
        "audio_tools",
        "teaching_tools",
    }

    assert expected.issubset(names), f"Missing tool modules: {expected - names}"


def test_iter_tool_modules_returns_cached_list():
    first = mcp_tools.iter_tool_modules()
    second = mcp_tools.iter_tool_modules()
    assert first is second  # cache should return identical list object

