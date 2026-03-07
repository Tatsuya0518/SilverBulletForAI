"""SurrealDB schema definitions and migration management.

Schema follows A2A v0.3 conventions:
- SurrealDB stores fields in snake_case internally
- Python Pydantic models handle camelCase conversion for A2A protocol compliance
- HNSW index uses TYPE F32, EFC 500, M 16, DEFER for optimal RAG performance
"""

from __future__ import annotations

import logging

from sbfa.db.client import SurrealClient

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
-- ===========================================
-- Agent Card (A2A v0.3)
-- ===========================================
DEFINE TABLE agent_card SCHEMAFULL;
DEFINE FIELD name ON agent_card TYPE string;
DEFINE FIELD description ON agent_card TYPE string;
DEFINE FIELD url ON agent_card TYPE string;
DEFINE FIELD provider ON agent_card TYPE string;
DEFINE FIELD model ON agent_card TYPE string;
DEFINE FIELD version ON agent_card TYPE string DEFAULT '0.3.0';
DEFINE FIELD skills ON agent_card TYPE array<object>;
DEFINE FIELD capabilities ON agent_card TYPE object;
DEFINE FIELD default_input_modes ON agent_card TYPE array<string> DEFAULT ['text'];
DEFINE FIELD default_output_modes ON agent_card TYPE array<string> DEFAULT ['text'];
DEFINE FIELD auth ON agent_card TYPE option<object>;
DEFINE FIELD signature ON agent_card TYPE option<object>;
DEFINE FIELD status ON agent_card TYPE string DEFAULT 'active';
DEFINE FIELD created_at ON agent_card TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON agent_card TYPE datetime DEFAULT time::now();

-- ===========================================
-- RAG Document
-- ===========================================
DEFINE TABLE rag_document SCHEMAFULL;
DEFINE FIELD title ON rag_document TYPE string;
DEFINE FIELD source ON rag_document TYPE string;
DEFINE FIELD content_hash ON rag_document TYPE string;
DEFINE FIELD metadata ON rag_document TYPE option<object>;
DEFINE FIELD created_at ON rag_document TYPE datetime DEFAULT time::now();

-- ===========================================
-- RAG Chunk (with vector embedding)
-- TYPE F32, EFC 500, M 16, DEFER for production RAG
-- ===========================================
DEFINE TABLE rag_chunk SCHEMAFULL;
DEFINE FIELD document ON rag_chunk TYPE record<rag_document>;
DEFINE FIELD content ON rag_chunk TYPE string;
DEFINE FIELD embedding ON rag_chunk TYPE array<float>;
DEFINE FIELD chunk_index ON rag_chunk TYPE int;
DEFINE FIELD metadata ON rag_chunk TYPE option<object>;
DEFINE INDEX idx_chunk_embedding ON rag_chunk FIELDS embedding
  HNSW DIMENSION 1536 TYPE F32 DIST COSINE EFC 500 M 16;

-- ===========================================
-- Task History (summaries only, no full text)
-- ===========================================
DEFINE TABLE task_history SCHEMAFULL;
DEFINE FIELD task ON task_history TYPE string;
DEFINE FIELD assigned_agent ON task_history TYPE record<agent_card>;
DEFINE FIELD input_summary ON task_history TYPE string;
DEFINE FIELD input_token_count ON task_history TYPE int;
DEFINE FIELD output_summary ON task_history TYPE option<string>;
DEFINE FIELD output_token_count ON task_history TYPE option<int>;
DEFINE FIELD status ON task_history TYPE string;
DEFINE FIELD cost ON task_history TYPE option<float>;
DEFINE FIELD latency_ms ON task_history TYPE option<int>;
DEFINE FIELD created_at ON task_history TYPE datetime DEFAULT time::now();
"""


async def apply_schema(db: SurrealClient) -> None:
    """Apply the full database schema."""
    for statement in SCHEMA_SQL.strip().split(";"):
        statement = statement.strip()
        if statement and not statement.startswith("--"):
            await db.query(statement + ";")
    logger.info("Schema applied successfully")


async def rebuild_indexes(db: SurrealClient) -> None:
    """Rebuild HNSW indexes after server restart.

    SurrealDB HNSW indexes are in-memory structures.
    They must be rebuilt after server restart to restore search capability.
    """
    await db.query("REBUILD INDEX idx_chunk_embedding ON rag_chunk;")
    logger.info("HNSW index rebuilt: idx_chunk_embedding")


async def initialize_database(db: SurrealClient) -> None:
    """Apply schema and rebuild indexes."""
    await apply_schema(db)
    await rebuild_indexes(db)
