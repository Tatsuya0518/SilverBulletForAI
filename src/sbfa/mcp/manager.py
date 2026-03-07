"""MCP server connection manager.

Manages lifecycle of MCP server connections and provides
structured tool call interface for agents.
"""

from __future__ import annotations

import logging
from typing import Any

from sbfa.mcp.types import AgentMCPBinding, MCPServerConfig, ToolInfo, ToolResult

logger = logging.getLogger(__name__)


class MCPManager:
    """Manages MCP server connections and tool calls.

    Concrete MCP server selection is deferred, but the interface is
    defined now to prevent BaseAgent interface changes later.
    """

    def __init__(self) -> None:
        self._connections: dict[str, Any] = {}
        self._bindings: dict[str, AgentMCPBinding] = {}

    async def connect(self, server_config: MCPServerConfig) -> None:
        """Connect to an MCP server."""
        logger.info("Connecting to MCP server '%s'", server_config.name)
        # TODO: Implement actual MCP connection using mcp SDK
        self._connections[server_config.name] = server_config

    async def disconnect(self, server_name: str | None = None) -> None:
        """Disconnect from one or all MCP servers."""
        if server_name:
            self._connections.pop(server_name, None)
        else:
            self._connections.clear()

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """Call a tool on a connected MCP server.

        Returns a structured ToolResult preserving the tool's output format.
        """
        # TODO: Implement actual MCP tool call
        logger.info("MCP tool call: %s(%s)", tool_name, args)
        return ToolResult(
            tool_name=tool_name,
            result=f"MCP tool '{tool_name}' not yet connected",
            is_error=True,
        )

    async def list_tools(self) -> list[ToolInfo]:
        """List all available tools across connected MCP servers."""
        # TODO: Implement actual tool listing
        return []

    def bind_agent(self, binding: AgentMCPBinding) -> None:
        """Bind MCP servers and allowed tools to an agent."""
        self._bindings[binding.agent_name] = binding

    def get_binding(self, agent_name: str) -> AgentMCPBinding | None:
        """Get the MCP binding for an agent."""
        return self._bindings.get(agent_name)
