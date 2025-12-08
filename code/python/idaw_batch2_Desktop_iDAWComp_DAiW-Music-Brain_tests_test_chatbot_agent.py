"""
Unit tests for the offline chatbot agent scaffolding.
"""

from pathlib import Path

from music_brain.chatbot.agent import AgentConfig, ChatAgent


def make_agent():
    config = AgentConfig(
        model_path=Path("fake-model.gguf"),
        system_prompt="You are an offline DAiW helper.",
    )
    return ChatAgent(config)


def test_chat_agent_history_appends():
    agent = make_agent()
    reply = agent.chat("hello there")
    assert "[offline placeholder]" in reply
    assert len(agent.history) >= 3  # system, user, assistant


def test_chat_agent_tool_trigger():
    agent = make_agent()
    reply = agent.chat("Can you auto-tune these vocals?")
    assert "auto-tune" in reply.lower()

