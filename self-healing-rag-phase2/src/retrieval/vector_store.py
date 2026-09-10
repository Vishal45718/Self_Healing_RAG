"""
Vector store.

Single Chroma collection (persistent, local). No multi-vector-DB
abstraction — see DECISIONS.md D-002. The embedding function is a
constructor parameter (defaults to the real sentence-transformers
embedder) purely so tests can inject a fake, network-free embedder;
this is not a provider-abstraction layer, just standard dependency
injection for testability.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import chromadb
from chromadb.api.types import EmbeddingFunction

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import settings  # noqa: E402
from src.chunking.chunker import Chunk  # noqa: E402
from src.embeddings.embedder import embed_texts  # noqa: E402

EmbeddingFn = Callable[[list[str]], list[list[float]]]


@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    distance: float  # lower = more similar (cosine distance)


class _EmbeddingFunctionAdapter(EmbeddingFunction):
    """Adapts our embed_texts(list[str]) signature to Chroma's expected
    EmbeddingFunction interface. Subclassing (rather than duck-typing)
    matters: Chroma's base class supplies embed_query() -> __call__()
    fallback and result normalization that queries depend on."""

    def __init__(self, embed_fn: EmbeddingFn):
        self._embed_fn = embed_fn

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - Chroma's expected param name
        return self._embed_fn(list(input))

    @staticmethod
    def name() -> str:
        return "custom_embedding_function"

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "_EmbeddingFunctionAdapter":
        return _EmbeddingFunctionAdapter(embed_texts)

    def get_config(self) -> dict[str, Any]:
        return {}


class VectorStore:
    def __init__(
        self,
        collection_name: str = "self_healing_rag",
        persist_dir: Path | None = None,
        embed_fn: EmbeddingFn = embed_texts,
    ):
        persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection_name = collection_name
        self._embedding_fn_adapter = _EmbeddingFunctionAdapter(embed_fn)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn_adapter,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    def query(self, query_text: str, top_k: int = settings.RETRIEVAL_TOP_K) -> list[RetrievedChunk]:
        if self.collection.count() == 0:
            return []
        results = self.collection.query(query_texts=[query_text], n_results=min(top_k, self.collection.count()))
        retrieved = []
        for chunk_id, content, metadata, distance in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            retrieved.append(
                RetrievedChunk(chunk_id=chunk_id, content=content, metadata=metadata, distance=distance)
            )
        return retrieved

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self._collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_fn_adapter,
            metadata={"hnsw:space": "cosine"},
        )
