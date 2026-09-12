"""Tests for src/ingestion/embedder.py (Task 1.3).

All tests use the real sentence-transformers model to maintain deterministic,
device-independent behaviour without mocking the model weights.

Note: The first test run will download ``all-MiniLM-L6-v2`` (~90 MB) to the
HuggingFace cache.  Subsequent runs are fully offline.
"""

import pytest
from src.schema import Document, Chunk
from src.ingestion.embedder import LocalEmbedder


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # known output dimension for all-MiniLM-L6-v2


@pytest.fixture(scope="module")
def embedder() -> LocalEmbedder:
    """Single embedder instance shared across all tests in this module."""
    return LocalEmbedder(model_id=MODEL_ID)


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    doc = Document(
        id="doc_embed_test",
        content="Sentence transformers encode text into dense vectors.",
        metadata={"source": "test.txt", "file_type": "txt"},
    )
    return [
        Chunk(
            id="chunk_0",
            document_id=doc.id,
            content="Sentence transformers encode text into dense vectors.",
            chunk_index=0,
            metadata=doc.metadata.copy(),
        ),
        Chunk(
            id="chunk_1",
            document_id=doc.id,
            content="ChromaDB stores these vectors for similarity search.",
            chunk_index=1,
            metadata=doc.metadata.copy(),
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_embedder_initializes(embedder: LocalEmbedder) -> None:
    """Embedder must load the model without raising."""
    assert embedder.model_id == MODEL_ID
    assert embedder._model is not None


def test_embed_chunks_returns_correct_count(
    embedder: LocalEmbedder, sample_chunks: list[Chunk]
) -> None:
    """One embedding vector must be returned per input chunk."""
    embeddings = embedder.embed_chunks(sample_chunks)
    assert len(embeddings) == len(sample_chunks)


def test_embed_chunks_correct_dimension(
    embedder: LocalEmbedder, sample_chunks: list[Chunk]
) -> None:
    """Each embedding must have the expected dimensionality (384 for MiniLM-L6-v2)."""
    embeddings = embedder.embed_chunks(sample_chunks)
    for vec in embeddings:
        assert len(vec) == EMBEDDING_DIM


def test_embed_chunks_returns_floats(
    embedder: LocalEmbedder, sample_chunks: list[Chunk]
) -> None:
    """Embedding values must be plain Python floats, not numpy scalars."""
    embeddings = embedder.embed_chunks(sample_chunks)
    for vec in embeddings:
        for val in vec:
            assert isinstance(val, float)


def test_embed_chunks_empty_raises(embedder: LocalEmbedder) -> None:
    """Passing an empty chunk list must raise ValueError."""
    with pytest.raises(ValueError):
        embedder.embed_chunks([])


def test_embed_query_returns_correct_dimension(embedder: LocalEmbedder) -> None:
    """A single query embedding must have the expected dimensionality."""
    vec = embedder.embed_query("What is self-healing RAG?")
    assert len(vec) == EMBEDDING_DIM


def test_embed_query_returns_floats(embedder: LocalEmbedder) -> None:
    """Query embedding values must be plain Python floats."""
    vec = embedder.embed_query("What is self-healing RAG?")
    for val in vec:
        assert isinstance(val, float)


def test_embed_query_empty_raises(embedder: LocalEmbedder) -> None:
    """An empty (whitespace-only) query must raise ValueError."""
    with pytest.raises(ValueError):
        embedder.embed_query("   ")


def test_embed_chunks_deterministic(
    embedder: LocalEmbedder, sample_chunks: list[Chunk]
) -> None:
    """Embedding the same chunks twice must produce identical vectors."""
    emb1 = embedder.embed_chunks(sample_chunks)
    emb2 = embedder.embed_chunks(sample_chunks)
    assert emb1 == emb2


def test_embed_query_deterministic(embedder: LocalEmbedder) -> None:
    """Embedding the same query twice must produce identical vectors."""
    query = "Self-healing retrieval augmented generation"
    assert embedder.embed_query(query) == embedder.embed_query(query)


def test_different_texts_produce_different_embeddings(
    embedder: LocalEmbedder, sample_chunks: list[Chunk]
) -> None:
    """Semantically different chunks must not produce identical vectors."""
    embeddings = embedder.embed_chunks(sample_chunks)
    assert embeddings[0] != embeddings[1]
