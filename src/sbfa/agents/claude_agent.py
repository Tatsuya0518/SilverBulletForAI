"""Claude agent using Anthropic SDK."""

from __future__ import annotations

from typing import AsyncIterator

import anthropic

from sbfa.agents.base import BaseAgent, Skill
from sbfa.config import settings
from sbfa.mcp.types import ToolResult


class ClaudeAgent(BaseAgent):
    """Claude Sonnet 4.6 agent - excels at coding, logical reasoning, long text analysis."""

    def __init__(self) -> None:
        super().__init__(
            name="claude",
            provider="anthropic",
            model=settings.models.claude_model,
            skills=[
                Skill(name="coding", description="Code generation and review", tags=["code", "debug", "refactor"]),
                Skill(name="reasoning", description="Logical and analytical reasoning", tags=["reasoning", "analysis"]),
                Skill(name="long_text", description="Long document analysis", tags=["summarize", "analysis"]),
            ],
        )
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    def _build_messages(
        self,
        prompt: str,
        context: list[str] | None = None,
        tool_results: list[ToolResult] | None = None,
    ) -> list[dict]:
        parts: list[str] = []
        if context:
            parts.append("Context:\n" + "\n---\n".join(context))
        if tool_results:
            for tr in tool_results:
                prefix = "[ERROR] " if tr.is_error else ""
                parts.append(f"Tool '{tr.tool_name}': {prefix}{tr.result}")
        parts.append(prompt)
        return [{"role": "user", "content": "\n\n".join(parts)}]

    async def generate(
        self,
        prompt: str,
        context: list[str] | None = None,
        tool_results: list[ToolResult] | None = None,
    ) -> str:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=self._build_messages(prompt, context, tool_results),
        )
        return response.content[0].text

    async def stream(
        self,
        prompt: str,
        context: list[str] | None = None,
        tool_results: list[ToolResult] | None = None,
    ) -> AsyncIterator[str]:
        async with self._client.messages.stream(
            model=self.model,
            max_tokens=4096,
            messages=self._build_messages(prompt, context, tool_results),
        ) as stream:
            async for text in stream.text_stream:
                yield text
