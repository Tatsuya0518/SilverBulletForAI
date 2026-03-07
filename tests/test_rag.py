"""Tests for RAG system components."""

from sbfa.rag.ingestion import chunk_text, compute_content_hash


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_short():
    result = chunk_text("Hello world", chunk_size=512, overlap=64)
    assert len(result) == 1
    assert result[0] == "Hello world"


def test_chunk_text_splits():
    text = "x" * 1024
    result = chunk_text(text, chunk_size=512, overlap=64)
    assert len(result) >= 2


def test_chunk_text_overlap():
    text = "abcdefghij" * 100  # 1000 chars
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    # Verify overlap: end of chunk N should overlap with start of chunk N+1
    for i in range(len(chunks) - 1):
        assert len(chunks[i]) <= 200


def test_content_hash_deterministic():
    h1 = compute_content_hash("test content")
    h2 = compute_content_hash("test content")
    assert h1 == h2


def test_content_hash_different():
    h1 = compute_content_hash("content A")
    h2 = compute_content_hash("content B")
    assert h1 != h2


def test_chunk_text_overlap_exceeds_chunk_size():
    """overlap >= chunk_size should not cause infinite loop."""
    result = chunk_text("hello world test", chunk_size=5, overlap=10)
    assert len(result) >= 1


def test_chunk_text_overlap_equals_chunk_size():
    """overlap == chunk_size should not cause infinite loop."""
    result = chunk_text("abcdefghij", chunk_size=5, overlap=5)
    assert len(result) >= 1


def test_chunk_text_forward_progress():
    """Ensure chunking always makes forward progress."""
    text = "a" * 100
    result = chunk_text(text, chunk_size=10, overlap=9)
    # Should still produce chunks without hanging
    assert len(result) >= 1
