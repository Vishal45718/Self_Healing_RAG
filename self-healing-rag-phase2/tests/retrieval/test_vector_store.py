"""
Tests use a deterministic, network-free fake embedder rather than the
real sentence-transformers model. This keeps unit tests fast and
offline (standard practice — not a sandbox workaround), while the
production code path (embedder.embed_texts, exercised by
test_embedder_real_model.py) uses the real model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.retrieval.vector_store import VectorStore  # noqa: E402
from src.chunking.chunker import Chunk  # noqa: E402

VOCAB = ["eiffel", "tower", "paris", "france", "python", "programming", "language", "code"]


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-words vector: presence of vocab terms.
    Same text -> same vector. Semantically related text (shared words)
    -> closer vectors. Good enough to test retrieval behavior without
    a real model."""
    vectors = []
    for text in texts:
        lowered = text.lower()
        vectors.append([1.0 if term in lowered else 0.0 for term in VOCAB])
    return vectors


def make_chunk(chunk_id: str, content: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="doc", content=content, chunk_index=0, metadata={"source": "test"})


def make_store(tmp_path) -> VectorStore:
    return VectorStore(collection_name="test_collection", persist_dir=tmp_path, embed_fn=fake_embed)


def test_add_and_query_returns_relevant_chunk(tmp_path):
    store = make_store(tmp_path)
    store.add_chunks(
        [
            make_chunk("c1", "The Eiffel Tower is located in Paris, France."),
            make_chunk("c2", "Python is a popular programming language."),
        ]
    )
    results = store.query("Tell me about programming language code", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "c2"


def test_query_returns_top_k_ordered_by_distance(tmp_path):
    store = make_store(tmp_path)
    store.add_chunks(
        [
            make_chunk("c1", "Eiffel Tower Paris France"),
            make_chunk("c2", "Python programming language code"),
            make_chunk("c3", "France Paris Eiffel"),
        ]
    )
    results = store.query("Eiffel Tower France Paris", top_k=2)
    assert len(results) == 2
    # both Paris-related chunks should outrank the unrelated programming chunk
    ids = {r.chunk_id for r in results}
    assert ids == {"c1", "c3"}
    assert results[0].distance <= results[1].distance


def test_empty_store_query_returns_empty_list(tmp_path):
    store = make_store(tmp_path)
    assert store.query("anything") == []


def test_add_empty_chunk_list_is_noop(tmp_path):
    store = make_store(tmp_path)
    store.add_chunks([])
    assert store.count() == 0


def test_metadata_preserved_through_retrieval(tmp_path):
    store = make_store(tmp_path)
    chunk = Chunk(
        chunk_id="c1",
        doc_id="doc1",
        content="Python programming language",
        chunk_index=2,
        metadata={"source": "doc1.txt", "chunk_index": 2},
    )
    store.add_chunks([chunk])
    results = store.query("Python programming", top_k=1)
    assert results[0].metadata["source"] == "doc1.txt"
    assert results[0].metadata["chunk_index"] == 2


def test_reset_clears_collection(tmp_path):
    store = make_store(tmp_path)
    store.add_chunks([make_chunk("c1", "Python programming language")])
    assert store.count() == 1
    store.reset()
    assert store.count() == 0
