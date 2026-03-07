"""MCP type definitions for structured tool results.

These types are defined early (Phase 3) to stabilize the BaseAgent interface.
Agents receive ToolResult objects instead of flattened strings, preserving
structured data from MCP tool calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


@dataclass
class ToolResult:
    """Structured result from an MCP tool call."""

    tool_name: str
    result: Any
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolInfo(BaseModel):
    """Description of an available MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any] = {}


class MCPServerConfig(BaseModel):
    """Configuration for connecting to an MCP server."""

    name: str
    command: str
    args: list[str] = []
    env: dict[str, str] = {}


class AgentMCPBinding(BaseModel):
    """Binding between an agent and its allowed MCP tools."""

    agent_name: str
    mcp_servers: list[MCPServerConfig] = []
    allowed_tools: list[str] = []
