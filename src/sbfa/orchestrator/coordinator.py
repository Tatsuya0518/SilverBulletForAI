"""Multi-agent coordination - sequential and parallel execution with RAG context."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from sbfa.agents.base import BaseAgent
from sbfa.mcp.types import ToolResult
from sbfa.orchestrator.router import TaskRouter
from sbfa.rag.retrieval import RAGRetriever

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 30


@dataclass
class TaskResult:
    """Result of a task execution."""

    agent_name: str
    output: str
    latency_ms: int
    success: bool
    error: str | None = None


class Coordinator:
    """Orchestrates multi-agent task execution with RAG context injection."""

    def __init__(
        self,
        router: TaskRouter,
        retriever: RAGRetriever | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._router = router
        self._retriever = retriever
        self._timeout_s = timeout_s

    async def _get_rag_context(self, task: str) -> list[str]:
        """Fetch RAG context for the task if retriever is available."""
        if not self._retriever:
            return []
        try:
            return await self._retriever.search_as_context(task, top_k=5)
        except Exception:
            logger.warning("RAG retrieval failed, proceeding without context", exc_info=True)
            return []

    async def _execute_with_agent(
        self,
        agent: BaseAgent,
        prompt: str,
        context: list[str] | None = None,
        tool_results: list[ToolResult] | None = None,
    ) -> TaskResult:
        """Execute a task with a single agent, tracking latency."""
        start = time.monotonic()
        try:
            output = await asyncio.wait_for(
                agent.generate(prompt, context=context, tool_results=tool_results),
                timeout=self._timeout_s,
            )
            latency_ms = int((time.monotonic() - start) * 1000)
            self._router.update_stats(agent.name, latency_ms=latency_ms)
            return TaskResult(
                agent_name=agent.name,
                output=output,
                latency_ms=latency_ms,
                success=True,
            )
        except asyncio.TimeoutError:
            latency_ms = int((time.monotonic() - start) * 1000)
            self._router.update_stats(agent.name, error=True)
            return TaskResult(
                agent_name=agent.name,
                output="",
                latency_ms=latency_ms,
                success=False,
                error=f"Timeout after {self._timeout_s}s",
            )
        except Exception as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            self._router.update_stats(agent.name, error=True)
            return TaskResult(
                agent_name=agent.name,
                output="",
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )

    async def execute(self, task: str) -> TaskResult:
        """Execute a task with automatic agent selection and fallback.

        Tries agents in order of routing score. Falls back to next agent on failure.
        """
        context = await self._get_rag_context(task)
        scored_agents = self._router.select_agent(task)

        if not scored_agents:
            return TaskResult(
                agent_name="none",
                output="",
                latency_ms=0,
                success=False,
                error="No agents registered",
            )

        for scored in scored_agents:
            logger.info(
                "Trying agent '%s' (score=%.3f) for task",
                scored.agent.name,
                scored.score,
            )
            result = await self._execute_with_agent(scored.agent, task, context=context)
            if result.success:
                return result
            logger.warning(
                "Agent '%s' failed: %s. Trying next.",
                scored.agent.name,
                result.error,
            )

        return TaskResult(
            agent_name="all_failed",
            output="",
            latency_ms=0,
            success=False,
            error="All agents failed",
        )

    async def execute_parallel(self, task: str, agent_names: list[str]) -> list[TaskResult]:
        """Execute the same task on multiple agents in parallel."""
        context = await self._get_rag_context(task)
        agents = [
            agent
            for name in agent_names
            if (agent := self._router.get_agent(name)) is not None
        ]

        coros = [
            self._execute_with_agent(agent, task, context=context)
            for agent in agents
        ]
        return list(await asyncio.gather(*coros))

    async def execute_sequential(self, tasks: list[str]) -> list[TaskResult]:
        """Execute tasks sequentially, passing each output as context to the next."""
        results: list[TaskResult] = []
        accumulated_context: list[str] = []

        for task in tasks:
            rag_context = await self._get_rag_context(task)
            full_context = rag_context + accumulated_context

            scored_agents = self._router.select_agent(task)
            if not scored_agents:
                results.append(TaskResult(
                    agent_name="none", output="", latency_ms=0,
                    success=False, error="No agents registered",
                ))
                continue

            result = await self._execute_with_agent(
                scored_agents[0].agent, task, context=full_context,
            )
            results.append(result)

            if result.success:
                accumulated_context.append(result.output)

        return results
