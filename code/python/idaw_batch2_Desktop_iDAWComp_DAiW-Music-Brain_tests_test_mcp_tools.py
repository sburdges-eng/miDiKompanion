"""
Smoke tests for the MCP tool layer.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from daiw_mcp.tools import harmony, intent, therapy


def run(coro):
    return asyncio.run(coro)


def test_analyze_chords_tool(monkeypatch):
    async def fake_run_sync(func, midi_path, include_sections):
        return {"key": "C", "chords": ["Cmaj7"], "roman_numerals": ["I"]}

    monkeypatch.setattr(harmony, "run_sync", fake_run_sync)
    midi_data = base64.b64encode(b"fake-midi").decode("utf-8")
    result = run(harmony.analyze_chords_tool(midi_base64=midi_data))
    assert result["key"] == "C"
    assert result["chords"]


def test_analyze_progression_tool():
    result = run(harmony.analyze_progression_tool("F-C-Am-Dm"))
    assert "key" in result
    assert "mode" in result


def test_diagnose_chords_tool(monkeypatch):
    async def fake_run_sync(func, progression):
        return {
            "key": "F",
            "mode": "major",
            "issues": ["No dominant resolution"],
            "suggestions": ["Try C7 before F"],
        }

    monkeypatch.setattr(harmony, "run_sync", fake_run_sync)
    result = run(harmony.diagnose_chords_tool("F-C-Am-Dm"))
    assert result["issues"]
    assert result["suggestions"]


def test_generate_harmony_basic_tool():
    result = run(
        harmony.generate_harmony_tool(
            intent_json=None,
            key="F",
            mode="major",
            pattern="I-V-vi-IV",
            include_midi=False,
        )
    )
    assert "harmony" in result
    assert result["harmony"]["chords"]


def test_create_intent_tool():
    template = run(intent.create_intent_tool("My Song"))
    assert template["title"] == "My Song"
    assert template["song_root"]["core_event"]


def test_validate_intent_tool_success():
    template = run(intent.create_intent_tool("Temp"))
    intent_json = json.dumps(template)
    result = run(intent.validate_intent_tool(intent_json))
    assert result["valid"]


def test_suggest_rulebreaks_tool():
    result = run(intent.suggest_rulebreaks_tool("grief"))
    assert result["emotion"] == "grief"


@pytest.mark.skip(reason="Therapy session requires heavy resources on CI")
def test_therapy_session_tool():
    result = run(therapy.therapy_session_tool("I feel lost", include_midi=False))
    assert "plan" in result

