"""Tests for agent base class and skill definitions."""

from sbfa.agents.base import BaseAgent, Skill
from sbfa.mcp.types import ToolResult


class MockAgent(BaseAgent):
    """Mock agent for testing BaseAgent interface."""

    async def generate(self, prompt, context=None, tool_results=None):
        parts = [prompt]
        if context:
            parts.extend(context)
        if tool_results:
            parts.extend(tr.tool_name for tr in tool_results)
        return " | ".join(parts)

    async def stream(self, prompt, context=None, tool_results=None):
        yield await self.generate(prompt, context, tool_results)


def test_base_agent_init():
    agent = MockAgent(
        name="test",
        provider="mock",
        model="mock-v1",
        skills=[Skill(name="test", description="Test skill", tags=["test"])],
    )
    assert agent.name == "test"
    assert agent.provider == "mock"
    assert agent.model == "mock-v1"
    assert len(agent.skills) == 1


def test_to_agent_card():
    agent = MockAgent(
        name="test",
        provider="mock",
        model="mock-v1",
        skills=[Skill(name="test", description="Test skill", tags=["test"])],
    )
    card = agent.to_agent_card()
    assert card.name == "test"
    assert card.provider == "mock"
    assert card.version == "0.3.0"


def test_agent_card_camel_case():
    agent = MockAgent(
        name="test",
        provider="mock",
        model="mock-v1",
        skills=[Skill(name="coding", description="Code", tags=["code"])],
    )
    card = agent.to_agent_card()
    json_data = card.to_json()

    # A2A v0.3 requires camelCase
    assert "protocolVersion" in json_data
    assert "defaultInputModes" in json_data
    assert "defaultOutputModes" in json_data
    # snake_case must NOT be in JSON output
    assert "protocol_version" not in json_data
    assert "default_input_modes" not in json_data


async def test_generate_with_tool_results():
    agent = MockAgent(
        name="test",
        provider="mock",
        model="mock-v1",
        skills=[],
    )
    tool_results = [ToolResult(tool_name="search", result="found it")]
    result = await agent.generate("hello", tool_results=tool_results)
    assert "search" in result
