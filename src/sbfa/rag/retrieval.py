"""RAG retrieval - cosine similarity search using SurrealDB HNSW index."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sbfa.db.client import SurrealClient
from sbfa.rag.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with chunk content and similarity score."""

    content: str
    score: float
    chunk_index: int
    document_id: str
    metadata: dict


class RAGRetriever:
    """Performs cosine similarity search against RAG chunks."""

    def __init__(self, db: SurrealClient, embedding_provider: EmbeddingProvider) -> None:
        self._db = db
        self._embedder = embedding_provider

    async def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Search for relevant chunks using cosine similarity.

        Args:
            query: The search query text.
            top_k: Number of top results to return.

        Returns:
            List of RetrievalResult ordered by similarity score (descending).
        """
        query_embedding = await self._embedder.embed_single(query)

        results = await self._db.query(
            """
            SELECT
                content,
                chunk_index,
                document,
                metadata,
                vector::similarity::cosine(embedding, $query_embedding) AS score
            FROM rag_chunk
            ORDER BY score DESC
            LIMIT $top_k;
            """,
            {"query_embedding": query_embedding, "top_k": top_k},
        )

        retrieval_results: list[RetrievalResult] = []
        for batch in results:
            records = batch.get("result", []) if isinstance(batch, dict) else batch if isinstance(batch, list) else []
            for record in records:
                retrieval_results.append(
                    RetrievalResult(
                        content=record.get("content", ""),
                        score=record.get("score", 0.0),
                        chunk_index=record.get("chunk_index", 0),
                        document_id=str(record.get("document", "")),
                        metadata=record.get("metadata", {}),
                    )
                )
        return retrieval_results

    async def search_as_context(self, query: str, top_k: int = 5) -> list[str]:
        """Search and return results as context strings for agent injection."""
        results = await self.search(query, top_k)
        return [r.content for r in results]
