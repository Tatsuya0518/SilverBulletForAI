"""OpenAI agent using OpenAI SDK."""

from __future__ import annotations

from typing import AsyncIterator

import openai

from sbfa.agents.base import BaseAgent, Skill
from sbfa.config import settings
from sbfa.mcp.types import ToolResult


class OpenAIAgent(BaseAgent):
    """GPT-5.4 agent - excels at general tasks, function calling, structured output."""

    def __init__(self) -> None:
        super().__init__(
            name="openai",
            provider="openai",
            model=settings.models.openai_model,
            skills=[
                Skill(name="general", description="General-purpose task completion", tags=["general", "writing"]),
                Skill(name="function_calling", description="Structured function calls", tags=["function_calling", "structured"]),
                Skill(name="structured_output", description="JSON and structured data generation", tags=["structured", "json"]),
            ],
        )
        self._client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

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
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt, context, tool_results),
        )
        return response.choices[0].message.content or ""

    async def stream(
        self,
        prompt: str,
        context: list[str] | None = None,
        tool_results: list[ToolResult] | None = None,
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt, context, tool_results),
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
