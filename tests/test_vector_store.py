"""Tests for src/ingestion/vector_store.py (Task 1.4).

All tests use a temporary directory for Chroma persistence so they are
isolated, deterministic, and leave no artefacts in the working tree.

A real LocalEmbedder generates embeddings so no test relies on mocked
vectors — correctness of the store is verified end-to-end.
"""

import pytest
from pathlib import Path
from typing import List

from src.schema import Document, Chunk
from src.ingestion.embedder import LocalEmbedder
from src.ingestion.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def embedder() -> LocalEmbedder:
    """Shared embedder — model already cached from test_embedder.py run."""
    return LocalEmbedder(model_id=MODEL_ID)


def _make_chunks(n: int = 3) -> List[Chunk]:
    """Generate ``n`` deterministic test chunks."""
    texts = [
        "The self-healing RAG system corrects retrieval errors automatically.",
        "ChromaDB persists vector embeddings on the local filesystem.",
        "LangGraph orchestrates stateful retrieval feedback loops.",
        "sentence-transformers encode text into dense float vectors.",
        "Query reformulation improves recall when initial retrieval fails.",
    ]
    return [
        Chunk(
            id=f"test_chunk_{i}",
            document_id="test_doc",
            content=texts[i % len(texts)],
            chunk_index=i,
            metadata={"source": "test.txt", "file_type": "txt"},
        )
        for i in range(n)
    ]


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    """Fresh VectorStore backed by a temporary directory per test."""
    return VectorStore(persist_path=str(tmp_path), collection_name="test_collection")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_store_initializes_empty(store: VectorStore) -> None:
    """A freshly created store must contain zero documents."""
    assert store.count() == 0


def test_upsert_adds_chunks(
    store: VectorStore, embedder: LocalEmbedder
) -> None:
    """Upserting chunks must increase the document count."""
    chunks = _make_chunks(3)
    embeddings = embedder.embed_chunks(chunks)
    store.upsert(chunks, embeddings)
    assert store.count() == 3


def test_upsert_is_idempotent(
    store: VectorStore, embedder: LocalEmbedder
) -> None:
    """Upserting the same chunks twice must not duplicate entries."""
    chunks = _make_chunks(3)
    embeddings = embedder.embed_chunks(chunks)
    store.upsert(chunks, embeddings)
    store.upsert(chunks, embeddings)
    assert store.count() == 3


def test_upsert_empty_raises(store: VectorStore) -> None:
    """Upserting an empty list must raise ValueError."""
    with pytest.raises(ValueError):
        store.upsert([], [])


def test_upsert_length_mismatch_raises(
    store: VectorStore, embedder: LocalEmbedder
) -> None:
    """Mismatched chunks/embeddings lengths must raise ValueError."""
    chunks = _make_chunks(2)
    embeddings = embedder.embed_chunks(chunks[:1])  # only one embedding
    with pytest.raises(ValueError):
        store.upsert(chunks, embeddings)


def test_query_returns_results(
    store: VectorStore, embedder: LocalEmbedder
) -> None:
    """Query must return a non-empty list of result dicts."""
    chunks = _make_chunks(3)
    embeddings = embedder.embed_chunks(chunks)
    store.upsert(chunks, embeddings)

    query_vec = embedder.embed_query("self-healing retrieval augmented generation")
    results = store.query(query_vec, n_results=2)

    assert len(results) == 2
    for r in results:
        assert "id" in r
        assert "content" in r
        assert "metadata" in r
        assert "distance" in r


def test_query_result_ids_are_valid(
    store: VectorStore, embedder: LocalEmbedder
) -> None:
    """Returned IDs must match IDs from the upserted chunks."""
    chunks = _make_chunks(3)
    embeddings = embedder.embed_chunks(chunks)
    store.upsert(chunks, embeddings)
    chunk_ids = {c.id for c in chunks}

    query_vec = embedder.embed_query("ChromaDB vector storage")
    results = store.query(query_vec, n_results=3)

    for r in results:
        assert r["id"] in chunk_ids


def test_query_top_result_is_most_relevant(
    store: VectorStore, embedder: LocalEmbedder
) -> None:
    """The first result must have the smallest (best) distance score."""
    chunks = _make_chunks(3)
    embeddings = embedder.embed_chunks(chunks)
    store.upsert(chunks, embeddings)

    query_vec = embedder.embed_query("ChromaDB persists vector embeddings")
    results = store.query(query_vec, n_results=3)

    distances = [r["distance"] for r in results]
    assert distances == sorted(distances), "Results must be ordered by ascending distance."


def test_query_empty_store_raises(store: VectorStore, embedder: LocalEmbedder) -> None:
    """Querying an empty store must raise ValueError."""
    query_vec = embedder.embed_query("empty store query")
    with pytest.raises(ValueError):
        store.query(query_vec, n_results=1)


def test_query_invalid_n_results_raises(
    store: VectorStore, embedder: LocalEmbedder
) -> None:
    """n_results < 1 must raise ValueError."""
    chunks = _make_chunks(1)
    embeddings = embedder.embed_chunks(chunks)
    store.upsert(chunks, embeddings)
    query_vec = embedder.embed_query("test")
    with pytest.raises(ValueError):
        store.query(query_vec, n_results=0)


def test_query_n_results_capped_at_collection_size(
    store: VectorStore, embedder: LocalEmbedder
) -> None:
    """Requesting more results than exist must not raise — return all that exist."""
    chunks = _make_chunks(2)
    embeddings = embedder.embed_chunks(chunks)
    store.upsert(chunks, embeddings)

    query_vec = embedder.embed_query("retrieval")
    results = store.query(query_vec, n_results=100)
    assert len(results) == 2


def test_persistence_across_instances(
    tmp_path: Path, embedder: LocalEmbedder
) -> None:
    """Data upserted by one VectorStore instance must be readable by another
    instance pointing to the same path (persistence check)."""
    chunks = _make_chunks(3)

    store_a = VectorStore(persist_path=str(tmp_path), collection_name="persist_test")
    embeddings = embedder.embed_chunks(chunks)
    store_a.upsert(chunks, embeddings)
    del store_a  # close first instance

    store_b = VectorStore(persist_path=str(tmp_path), collection_name="persist_test")
    assert store_b.count() == 3


def test_upsert_chunk_with_empty_metadata(store: VectorStore, embedder: LocalEmbedder) -> None:
    """Regression test (BUG-001): chunks with empty metadata dict can be upserted cleanly."""
    chunks = [
        Chunk(
            id="empty_meta_chunk",
            document_id="doc_empty",
            content="Text without custom metadata attributes.",
            chunk_index=0,
            metadata={},
        )
    ]
    embeddings = embedder.embed_chunks(chunks)
    store.upsert(chunks, embeddings)
    assert store.count() == 1

    results = store.query(embeddings[0], n_results=1)
    assert len(results) == 1
    assert results[0]["id"] == "empty_meta_chunk"
    assert results[0]["metadata"]["document_id"] == "doc_empty"
    assert str(results[0]["metadata"]["chunk_index"]) == "0"
