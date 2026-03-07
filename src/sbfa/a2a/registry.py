"""Agent registry - stores and discovers AgentCards via SurrealDB."""

from __future__ import annotations

import logging

from sbfa.a2a.agent_card import AgentCard
from sbfa.db.client import SurrealClient

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Manages AgentCard registration, discovery, and skill-based search."""

    def __init__(self, db: SurrealClient) -> None:
        self._db = db

    async def register(self, card: AgentCard) -> str:
        """Register an AgentCard in SurrealDB.

        Returns the record ID.
        """
        record = await self._db.create("agent_card", card.to_db_record())
        logger.info("Registered agent '%s' as %s", card.name, record.get("id"))
        return record.get("id", "")

    async def get(self, agent_id: str) -> AgentCard | None:
        """Get an AgentCard by ID."""
        result = await self._db.select(agent_id)
        if not result:
            return None
        data = result if isinstance(result, dict) else result[0]
        return AgentCard.model_validate(data)

    async def list_all(self) -> list[AgentCard]:
        """List all active AgentCards."""
        results = await self._db.query(
            "SELECT * FROM agent_card WHERE status = 'active';"
        )
        cards = []
        for batch in results:
            if isinstance(batch, dict) and "result" in batch:
                for record in batch["result"]:
                    cards.append(AgentCard.model_validate(record))
            elif isinstance(batch, list):
                for record in batch:
                    cards.append(AgentCard.model_validate(record))
        return cards

    async def find_by_skill(self, tag: str) -> list[AgentCard]:
        """Find agents that have a skill matching the given tag."""
        results = await self._db.query(
            "SELECT * FROM agent_card WHERE status = 'active' "
            "AND skills[*].tags CONTAINS $tag;",
            {"tag": tag},
        )
        cards = []
        for batch in results:
            if isinstance(batch, dict) and "result" in batch:
                for record in batch["result"]:
                    cards.append(AgentCard.model_validate(record))
            elif isinstance(batch, list):
                for record in batch:
                    cards.append(AgentCard.model_validate(record))
        return cards

    async def deactivate(self, agent_id: str) -> None:
        """Deactivate an agent by setting status to 'inactive'."""
        await self._db.update(agent_id, {"status": "inactive"})
        logger.info("Deactivated agent %s", agent_id)
