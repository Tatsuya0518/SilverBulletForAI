"""A2A v0.3 AgentCard schema with camelCase serialization and JWS signature support.

A2A specification requires:
- All JSON field names MUST use camelCase (not snake_case)
- AgentCard discovery at /.well-known/agent-card.json
- Optional JWS signature (RFC 7515)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from sbfa.agents.base import Skill


class Capabilities(BaseModel):
    """Agent capabilities declaration."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    streaming: bool = True
    push_notifications: bool = False
    state_transition_history: bool = False


class AuthConfig(BaseModel):
    """Authentication configuration for agent access."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    schemes: list[str] = []
    credentials: str | None = None


class AgentCardSignature(BaseModel):
    """JWS signature for AgentCard integrity verification (RFC 7515).

    The AgentCard content is canonicalized using JCS (RFC 8785) before signing.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    protected: str  # Base64url-encoded JWS Protected Header
    signature: str  # Base64url-encoded signature value
    header: dict | None = None  # Optional JWS Unprotected Header


class AgentCard(BaseModel):
    """A2A v0.3 AgentCard with camelCase JSON serialization.

    Discovery endpoint: GET /.well-known/agent-card.json
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    name: str
    description: str
    url: str
    version: str = "0.3.0"
    protocol_version: str = "0.3.0"
    provider: str
    model: str
    skills: list[Skill]
    capabilities: Capabilities = Capabilities()
    default_input_modes: list[str] = ["text"]
    default_output_modes: list[str] = ["text"]
    auth: AuthConfig | None = None
    signature: AgentCardSignature | None = None

    def to_json(self) -> dict:
        """Serialize to camelCase dict for A2A protocol compliance."""
        return self.model_dump(by_alias=True, exclude_none=True)

    def to_db_record(self) -> dict:
        """Serialize to snake_case dict for SurrealDB storage."""
        return self.model_dump(by_alias=False)
