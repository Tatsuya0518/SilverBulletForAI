"""SurrealDB RAG store - manages chunk and vector storage.

Uses HNSW index with TYPE F32, EFC 500, M 16, DEFER for production RAG.
Supports REBUILD INDEX for post-restart recovery.
"""

from __future__ import annotations

import logging
from typing import Any

from sbfa.db.client import SurrealClient
from sbfa.rag.embeddings import EmbeddingProvider
from sbfa.rag.ingestion import compute_content_hash, fast_chunk_text

logger = logging.getLogger(__name__)


class RAGStore:
    """Manages document ingestion and chunk storage in SurrealDB."""

    def __init__(self, db: SurrealClient, embedding_provider: EmbeddingProvider) -> None:
        self._db = db
        self._embedder = embedding_provider

    async def ingest_document(
        self,
        title: str,
        source: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> str:
        """Ingest a document: chunk, embed, and store.

        Full text is NOT stored in rag_document - only content_hash.

        Returns the document record ID.
        """
        content_hash = compute_content_hash(content)

        doc_record = await self._db.create("rag_document", {
            "title": title,
            "source": source,
            "content_hash": content_hash,
            "metadata": metadata or {},
        })
        doc_id = doc_record.get("id", "")

        chunks = fast_chunk_text(content, chunk_size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            logger.warning("No chunks generated for document '%s'", title)
            return doc_id

        embeddings = await self._embedder.embed(chunks)

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            await self._db.create("rag_chunk", {
                "document": doc_id,
                "content": chunk,
                "embedding": embedding,
                "chunk_index": i,
                "metadata": metadata or {},
            })

        logger.info("Ingested '%s': %d chunks stored", title, len(chunks))
        return doc_id

    async def rebuild_index(self) -> None:
        """Rebuild HNSW index (required after server restart)."""
        await self._db.query("REBUILD INDEX idx_chunk_embedding ON rag_chunk;")
        logger.info("HNSW index rebuilt")
