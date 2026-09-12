"""Tests for Retriever component (Task 2.1 & 2.5)."""

import pytest
from unittest.mock import MagicMock

from src.ingestion.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.schema import Chunk, RetrievalResult


@pytest.fixture
def empty_store(tmp_path):
    """Fixture providing an empty VectorStore."""
    return VectorStore(persist_path=str(tmp_path / "chroma_empty"), collection_name="test_empty")


@pytest.fixture
def populated_store(tmp_path):
    """Fixture providing a VectorStore populated with 3 deterministic chunks."""
    store = VectorStore(persist_path=str(tmp_path / "chroma_pop"), collection_name="test_pop")
    chunks = [
        Chunk(
            id="doc1_c0",
            document_id="doc1",
            content="Self-Healing RAG dynamically evaluates retrieved context and answer quality.",
            chunk_index=0,
            metadata={"source": "overview.md", "topic": "architecture"},
        ),
        Chunk(
            id="doc1_c1",
            document_id="doc1",
            content="Sentence transformers run locally on CPU or GPU with deterministic embeddings.",
            chunk_index=1,
            metadata={"source": "overview.md", "topic": "embeddings"},
        ),
        Chunk(
            id="doc2_c0",
            document_id="doc2",
            content="Chroma operates as an embedded vector database storing dense embeddings.",
            chunk_index=0,
            metadata={"source": "database.md", "topic": "storage"},
        ),
    ]
    # Simple deterministic orthogonal-ish 3D embeddings for unit testing
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    store.upsert(chunks, embeddings)
    return store, chunks


def test_retrieve_empty_store_returns_empty_list(empty_store):
    """Empty vector store returns an empty list safely without throwing unhandled exceptions."""
    mock_embedder = MagicMock()
    retriever = Retriever(vector_store=empty_store, embedder=mock_embedder)

    results = retriever.retrieve("Any query here")
    assert results == []
    # Embedder shouldn't even need to be called if store is empty
    mock_embedder.embed_query.assert_not_called()


@pytest.mark.parametrize("invalid_query", ["", "   ", "\n\t", None])
def test_retrieve_invalid_query_raises_value_error(empty_store, invalid_query):
    """Invalid or blank queries raise ValueError."""
    retriever = Retriever(vector_store=empty_store, embedder=MagicMock())
    with pytest.raises(ValueError, match="Query must be a non-empty string"):
        retriever.retrieve(invalid_query)


def test_retrieve_invalid_top_k_raises_value_error(empty_store):
    """top_k < 1 raises ValueError."""
    retriever = Retriever(vector_store=empty_store, embedder=MagicMock())
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.retrieve("Valid query", top_k=0)


def test_retrieve_relevant_query_and_metadata_preservation(populated_store):
    """Relevant query preserves chunk ID, content, metadata, distance, and similarity."""
    store, _ = populated_store

    mock_embedder = MagicMock()
    # Mock query embedding aligned with doc1_c0 ([1.0, 0.0, 0.0])
    mock_embedder.embed_query.return_value = [1.0, 0.0, 0.0]

    retriever = Retriever(vector_store=store, embedder=mock_embedder)
    results = retriever.retrieve("How does Self-Healing RAG evaluate answers?", top_k=2)

    assert len(results) == 2
    top_result = results[0]
    assert isinstance(top_result, RetrievalResult)
    assert top_result.id == "doc1_c0"
    assert "Self-Healing RAG dynamically evaluates" in top_result.content
    assert top_result.metadata["source"] == "overview.md"
    assert top_result.metadata["topic"] == "architecture"
    assert isinstance(top_result.distance, float)
    assert isinstance(top_result.similarity, float)
    # Cosine distance to exact vector should be ~0.0, similarity ~1.0
    assert abs(top_result.distance) < 1e-5
    assert abs(top_result.similarity - 1.0) < 1e-5


def test_retrieve_top_k_behavior(populated_store):
    """Retriever strictly respects configurable top_k."""
    store, _ = populated_store

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.5, 0.5, 0.5]

    retriever = Retriever(vector_store=store, embedder=mock_embedder)

    res_1 = retriever.retrieve("query", top_k=1)
    assert len(res_1) == 1

    res_3 = retriever.retrieve("query", top_k=3)
    assert len(res_3) == 3

    # top_k larger than collection count returns all available
    res_10 = retriever.retrieve("query", top_k=10)
    assert len(res_10) == 3
