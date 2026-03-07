"""Task router with weighted scoring algorithm.

Routing algorithm:
1. Classify task by keywords + intent (code/multimodal/general/reasoning)
2. Compute skill tag matching score for each agent
3. Weighted score: skill_match * 0.5 + (1/latency_avg) * 0.3 + (1/cost_per_token) * 0.2

Returns agents sorted by score (descending). The Coordinator handles
fallback execution (timeout, API errors) by trying agents in order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sbfa.agents.base import BaseAgent

logger = logging.getLogger(__name__)

TASK_CATEGORIES = {
    "code": ["code", "debug", "refactor", "programming", "function", "class", "api"],
    "multimodal": ["image", "video", "audio", "vision", "photo", "picture"],
    "reasoning": ["reasoning", "analysis", "logic", "math", "proof", "summarize"],
    "general": ["general", "writing", "translate", "explain", "help"],
}


@dataclass
class AgentStats:
    """Runtime statistics for an agent used in routing decisions."""

    avg_latency_ms: float = 1000.0
    cost_per_token: float = 0.001
    last_used_ts: float = 0.0
    error_count: int = 0


@dataclass
class ScoredAgent:
    """Agent with its computed routing score."""

    agent: BaseAgent
    score: float
    category_match: str = ""


class TaskRouter:
    """Routes tasks to the optimal agent using weighted scoring."""

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}
        self._stats: dict[str, AgentStats] = {}

    def register_agent(self, agent: BaseAgent) -> None:
        self._agents[agent.name] = agent
        self._stats[agent.name] = AgentStats()

    def get_agent(self, name: str) -> BaseAgent | None:
        """Get a registered agent by name."""
        return self._agents.get(name)

    def classify_task(self, task: str) -> str:
        """Classify task into a category based on keywords."""
        task_lower = task.lower()
        best_category = "general"
        best_score = 0

        for category, keywords in TASK_CATEGORIES.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > best_score:
                best_score = score
                best_category = category

        return best_category

    def _compute_score(self, agent: BaseAgent, category: str) -> float:
        """Compute weighted routing score for an agent."""
        skill_match = 0.0
        for skill in agent.skills:
            if any(tag in TASK_CATEGORIES.get(category, []) for tag in skill.tags):
                skill_match += 1.0

        if agent.skills:
            skill_match /= len(agent.skills)

        stats = self._stats.get(agent.name, AgentStats())
        latency_score = 1.0 / max(stats.avg_latency_ms, 1.0)
        cost_score = 1.0 / max(stats.cost_per_token, 0.0001)

        # Normalize scores to [0, 1] range approximately
        latency_score = min(latency_score * 1000, 1.0)
        cost_score = min(cost_score * 0.001, 1.0)

        return skill_match * 0.5 + latency_score * 0.3 + cost_score * 0.2

    def select_agent(self, task: str) -> list[ScoredAgent]:
        """Select agents ranked by score for the given task.

        Returns all agents sorted by score (descending) for fallback support.
        """
        category = self.classify_task(task)

        scored = [
            ScoredAgent(
                agent=agent,
                score=self._compute_score(agent, category),
                category_match=category,
            )
            for agent in self._agents.values()
        ]

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    def update_stats(
        self,
        agent_name: str,
        latency_ms: float | None = None,
        cost: float | None = None,
        error: bool = False,
    ) -> None:
        """Update agent statistics after a task execution."""
        stats = self._stats.setdefault(agent_name, AgentStats())
        if latency_ms is not None:
            stats.avg_latency_ms = (stats.avg_latency_ms + latency_ms) / 2
        if cost is not None:
            stats.cost_per_token = cost
        if error:
            stats.error_count += 1
