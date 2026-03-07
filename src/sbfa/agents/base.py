"""Base agent abstract class with MCP ToolResult support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator

from pydantic import BaseModel

from sbfa.mcp.types import ToolResult

if TYPE_CHECKING:
    from sbfa.a2a.agent_card import AgentCard


class Skill(BaseModel):
    """Agent skill definition for A2A AgentCard."""

    name: str
    description: str
    tags: list[str] = []


class BaseAgent(ABC):
    """Abstract base class for all AI agents.

    All agents share a common interface supporting:
    - Structured MCP tool results (not flattened strings)
    - RAG context injection
    - A2A AgentCard conversion
    """

    name: str
    provider: str
    model: str
    skills: list[Skill]

    def __init__(self, name: str, provider: str, model: str, skills: list[Skill]) -> None:
        self.name = name
        self.provider = provider
        self.model = model
        self.skills = skills

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        context: list[str] | None = None,
        tool_results: list[ToolResult] | None = None,
    ) -> str:
        """Generate a response from the AI model."""
        ...

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        context: list[str] | None = None,
        tool_results: list[ToolResult] | None = None,
    ) -> AsyncIterator[str]:
        """Stream a response from the AI model."""
        ...

    def to_agent_card(self) -> AgentCard:
        """Convert this agent to an A2A v0.3 AgentCard (camelCase)."""
        from sbfa.a2a.agent_card import AgentCard, Capabilities

        return AgentCard(
            name=self.name,
            description=f"{self.provider} agent using {self.model}",
            url="",
            provider=self.provider,
            model=self.model,
            skills=self.skills,
            capabilities=Capabilities(),
        )
