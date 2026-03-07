"""Embedding generation with pluggable providers.

Default: OpenAI text-embedding-3-small (1536 dimensions).
Switchable to Ollama, Gemini Embedding, etc.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import openai

from sbfa.config import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract embedding provider interface."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...

    @abstractmethod
    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        ...


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self) -> None:
        self._client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding.model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def embed_single(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]


class OllamaEmbedding(EmbeddingProvider):
    """Ollama-compatible embedding provider (OpenAI API compatible)."""

    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "nomic-embed-text") -> None:
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key="not-needed")
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def embed_single(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]


def get_embedding_provider() -> EmbeddingProvider:
    """Factory function to get the configured embedding provider."""
    provider = settings.embedding.provider.lower()
    if provider == "openai":
        return OpenAIEmbedding()
    if provider == "ollama":
        return OllamaEmbedding()
    raise ValueError(f"Unknown embedding provider: {provider}")
