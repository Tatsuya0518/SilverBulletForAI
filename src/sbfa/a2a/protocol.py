"""A2A JSON-RPC 2.0 messaging protocol.

A2A v0.3 supports three transports (JSON-RPC, gRPC, HTTP/REST).
This implementation uses JSON-RPC 2.0 as the primary transport.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class TaskState(str, Enum):
    """A2A task states (SCREAMING_SNAKE_CASE per spec)."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"


class Role(str, Enum):
    """Message roles."""

    USER = "user"
    AGENT = "agent"


class Part(BaseModel):
    """Content part within a message."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    type: str = "text"
    text: str | None = None
    data: Any | None = None
    mime_type: str | None = None


class Message(BaseModel):
    """A2A message."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    role: Role
    parts: list[Part]
    context_id: str | None = None


class Task(BaseModel):
    """A2A task representation."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    context_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    state: TaskState = TaskState.SUBMITTED
    messages: list[Message] = []


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request."""

    jsonrpc: str = "2.0"
    method: str
    params: dict[str, Any] = {}
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 response."""

    jsonrpc: str = "2.0"
    result: Any | None = None
    error: dict[str, Any] | None = None
    id: str
