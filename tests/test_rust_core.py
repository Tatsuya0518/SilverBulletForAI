"""Tests for Rust core module (sbfa_core).

These tests verify both the Rust implementation (if available)
and the Python fallback.
"""

import pytest

from sbfa.rag.ingestion import chunk_text, fast_chunk_text


def _rust_available() -> bool:
    try:
        import sbfa_core  # noqa: F401
        return True
    except ImportError:
        return False


def test_python_chunk_text():
    """Test Python fallback chunking."""
    result = chunk_text("hello world", chunk_size=512, overlap=64)
    assert len(result) == 1


def test_fast_chunk_text_fallback():
    """fast_chunk_text should work whether Rust module is available or not."""
    result = fast_chunk_text("hello world", chunk_size=512, overlap=64)
    assert len(result) == 1


def test_fast_chunk_text_long():
    text = "word " * 10000
    result = fast_chunk_text(text, chunk_size=512, overlap=64)
    assert len(result) > 1


@pytest.mark.skipif(
    not _rust_available(),
    reason="Rust sbfa_core module not built",
)
def test_rust_cosine_similarity():
    from sbfa_core import cosine_similarity

    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6


@pytest.mark.skipif(
    not _rust_available(),
    reason="Rust sbfa_core module not built",
)
def test_rust_batch_similarity():
    from sbfa_core import batch_cosine_similarity

    query = [1.0, 0.0]
    candidates = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
    results = batch_cosine_similarity(query, candidates, 2)
    assert len(results) == 2
    assert results[0][0] == 0  # most similar
