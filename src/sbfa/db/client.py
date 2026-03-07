"""SurrealDB connection client with authentication support."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from surrealdb import Surreal

from sbfa.config import settings

logger = logging.getLogger(__name__)


class SurrealClient:
    """Async SurrealDB client wrapper with auth and namespace management."""

    def __init__(self) -> None:
        self._db: Surreal | None = None

    async def connect(self) -> None:
        cfg = settings.surreal
        self._db = Surreal(cfg.url)
        await self._db.connect()
        await self._db.signin({"username": cfg.user, "password": cfg.password})
        await self._db.use(cfg.namespace, cfg.database)
        logger.info("Connected to SurrealDB at %s", cfg.url)

    async def disconnect(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
            logger.info("Disconnected from SurrealDB")

    @property
    def db(self) -> Surreal:
        if self._db is None:
            raise RuntimeError("SurrealDB client not connected. Call connect() first.")
        return self._db

    async def query(self, sql: str, vars: dict[str, Any] | None = None) -> list[dict]:
        return await self.db.query(sql, vars or {})

    async def create(self, table: str, data: dict[str, Any]) -> dict:
        return await self.db.create(table, data)

    async def select(self, thing: str) -> list[dict] | dict:
        return await self.db.select(thing)

    async def update(self, thing: str, data: dict[str, Any]) -> dict:
        return await self.db.update(thing, data)

    async def delete(self, thing: str) -> None:
        await self.db.delete(thing)


@asynccontextmanager
async def get_db():
    """Context manager for database connections."""
    client = SurrealClient()
    await client.connect()
    try:
        yield client
    finally:
        await client.disconnect()


# Singleton for app-wide usage
db_client = SurrealClient()
