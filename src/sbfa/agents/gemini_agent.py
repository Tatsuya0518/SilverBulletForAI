"""Gemini agent using Google GenAI SDK."""

from __future__ import annotations

from typing import AsyncIterator

from google import genai

from sbfa.agents.base import BaseAgent, Skill
from sbfa.config import settings
from sbfa.mcp.types import ToolResult


class GeminiAgent(BaseAgent):
    """Gemini 2.5 Flash agent - excels at multimodal, fast responses, large context."""

    def __init__(self) -> None:
        super().__init__(
            name="gemini",
            provider="google",
            model=settings.models.gemini_model,
            skills=[
                Skill(name="multimodal", description="Image and video understanding", tags=["multimodal", "vision"]),
                Skill(name="fast_response", description="Quick task completion", tags=["speed", "general"]),
                Skill(name="large_context", description="Processing very long documents", tags=["large_context"]),
            ],
        )
        self._client = genai.Client(api_key=settings.google_api_key)

    def _build_prompt(
        self,
        prompt: str,
        context: list[str] | None = None,
        tool_results: list[ToolResult] | None = None,
    ) -> str:
        parts: list[str] = []
        if context:
            parts.append("Context:\n" + "\n---\n".join(context))
        if tool_results:
            for tr in tool_results:
                prefix = "[ERROR] " if tr.is_error else ""
                parts.append(f"Tool '{tr.tool_name}': {prefix}{tr.result}")
        parts.append(prompt)
        return "\n\n".join(parts)

    async def generate(
        self,
        prompt: str,
        context: list[str] | None = None,
        tool_results: list[ToolResult] | None = None,
    ) -> str:
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=self._build_prompt(prompt, context, tool_results),
        )
        return response.text

    async def stream(
        self,
        prompt: str,
        context: list[str] | None = None,
        tool_results: list[ToolResult] | None = None,
    ) -> AsyncIterator[str]:
        async for chunk in await self._client.aio.models.generate_content_stream(
            model=self.model,
            contents=self._build_prompt(prompt, context, tool_results),
        ):
            if chunk.text:
                yield chunk.text
