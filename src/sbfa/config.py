"""Configuration management - loads settings from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class SurrealConfig:
    url: str = os.getenv("SURREAL_URL", "ws://localhost:8000/rpc")
    namespace: str = os.getenv("SURREAL_NAMESPACE", "sbfa")
    database: str = os.getenv("SURREAL_DATABASE", "sbfa")
    user: str = os.getenv("SURREAL_USER", "root")
    password: str = os.getenv("SURREAL_PASS", "root")


@dataclass(frozen=True)
class ModelConfig:
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.4")
    local_model_endpoint: str = os.getenv("LOCAL_MODEL_ENDPOINT", "http://localhost:11434/v1")
    local_model_name: str = os.getenv("LOCAL_MODEL_NAME", "llama3.2")


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))


@dataclass(frozen=True)
class ServerConfig:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8080"))


@dataclass(frozen=True)
class Settings:
    surreal: SurrealConfig = field(default_factory=SurrealConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")


settings = Settings()
