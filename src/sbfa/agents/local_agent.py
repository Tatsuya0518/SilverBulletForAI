"""Local LLM agent using OpenAI-compatible API (e.g., Ollama)."""

from __future__ import annotations

from typing import AsyncIterator

import openai

from sbfa.agents.base import BaseAgent, Skill
from sbfa.config import settings
from sbfa.mcp.types import ToolResult


class LocalAgent(BaseAgent):
    """Local LLM agent - for privacy-sensitive or offline tasks."""

    def __init__(self) -> None:
        super().__init__(
            name="local",
            provider="local",
            model=settings.models.local_model_name,
            skills=[
                Skill(name="privacy", description="Privacy-sensitive processing", tags=["privacy", "offline"]),
                Skill(name="general", description="General tasks without API calls", tags=["general"]),
            ],
        )
        self._client = openai.AsyncOpenAI(
            base_url=settings.models.local_model_endpoint,
            api_key="not-needed",
        )

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
