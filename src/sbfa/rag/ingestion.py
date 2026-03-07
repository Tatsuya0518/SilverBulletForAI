"""Document ingestion pipeline with chunking support.

Uses DEFER index for high-volume concurrent writes.
Full text is not stored in rag_document - only content_hash for reference.
"""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """Split text into overlapping chunks.

    This is the Python fallback implementation.
    When available, the Rust sbfa_core.fast_chunk_text is used instead.

    Args:
        text: The text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    if overlap >= chunk_size:
        overlap = 0

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of document content for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# Try to use Rust implementation for performance
try:
    from sbfa_core import fast_chunk_text  # type: ignore[import-not-found]

    logger.info("Using Rust fast_chunk_text implementation")
except ImportError:
    fast_chunk_text = chunk_text
    logger.info("Using Python chunk_text fallback")
