"""Chroma persistent vector store integration (Task 1.4).

Manages a local ChromaDB collection for storing and querying chunk embeddings.

Design rationale:
- Uses ``chromadb.PersistentClient`` as mandated by AGENTS.md §3.2.
- Collection name is configurable; defaults to ``self_healing_rag``.
- Upsert semantics avoid duplicate storage across multiple pipeline runs
  (leveraging deterministic chunk IDs from D-006).
- No unnecessary abstraction layers (AGENTS.md §3.1).
"""

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb import Collection

from config.settings import settings
from src.schema import Chunk

logger = logging.getLogger(__name__)

_DEFAULT_COLLECTION = "self_healing_rag"


class VectorStore:
    """Wraps a ChromaDB persistent collection for chunk storage and retrieval.

    Lifecycle
    ---------
    Instantiate once per application run.  The underlying Chroma client
    persists data to ``settings.chroma_directory`` automatically on every
    write.

    Example
    -------
    >>> store = VectorStore()
    >>> store.upsert(chunks, embeddings)
    >>> results = store.query(query_embedding, n_results=5)
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        collection_name: str = _DEFAULT_COLLECTION,
    ) -> None:
        """Initialise the ChromaDB client and get-or-create the collection.

        Args:
            persist_path: Directory for Chroma data files.  Defaults to
                          ``settings.chroma_directory``.
            collection_name: Name of the ChromaDB collection to use.
        """
        resolved_path = str(
            persist_path if persist_path is not None else settings.chroma_directory
        )
        logger.info(
            "Initialising ChromaDB PersistentClient at '%s', collection '%s'.",
            resolved_path,
            collection_name,
        )
        self._client: chromadb.ClientAPI = chromadb.PersistentClient(
            path=resolved_path
        )
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready. Collection '%s' has %d existing documents.",
            collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Add or update chunks and their embeddings in the collection.

        Uses Chroma's upsert to avoid duplicates across repeated pipeline runs
        (deterministic IDs from D-006 guarantee idempotence).

        Args:
            chunks: Document chunks to store.
            embeddings: Pre-computed embedding vectors, one per chunk.
                        Must be the same length as ``chunks``.

        Raises:
            ValueError: If ``chunks`` is empty or lengths differ.
        """
        if not chunks:
            raise ValueError("Cannot upsert an empty list of chunks.")
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        ids: List[str] = [c.id for c in chunks]
        documents: List[str] = [c.content for c in chunks]
        metadatas: List[Dict[str, Any]] = [
            {
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
                **{k: str(v) for k, v in c.metadata.items()},
            }
            for c in chunks
        ]

        logger.debug("Upserting %d chunks into collection.", len(ids))
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunks. Collection total: %d.", len(ids), self._collection.count())

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-``n_results`` chunks nearest to ``query_embedding``.

        Args:
            query_embedding: Dense vector for the query.
            n_results: Number of nearest neighbours to return.

        Returns:
            A list of result dicts, each containing:
            ``{"id": str, "content": str, "metadata": dict, "distance": float}``
            Ordered by ascending cosine distance (most similar first).

        Raises:
            ValueError: If the collection is empty or ``n_results`` < 1.
        """
        if n_results < 1:
            raise ValueError("n_results must be at least 1.")
        if self._collection.count() == 0:
            raise ValueError("Cannot query an empty collection.")

        actual_n = min(n_results, self._collection.count())
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_n,
            include=["documents", "metadatas", "distances"],
        )

        results: List[Dict[str, Any]] = []
        for idx in range(len(raw["ids"][0])):
            results.append(
                {
                    "id": raw["ids"][0][idx],
                    "content": raw["documents"][0][idx],
                    "metadata": raw["metadatas"][0][idx],
                    "distance": raw["distances"][0][idx],
                }
            )
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of documents currently stored in the collection."""
        return self._collection.count()

    def delete_collection(self) -> None:
        """Delete the entire collection.  Intended for testing teardown only."""
        name = self._collection.name
        self._client.delete_collection(name)
        logger.warning("Deleted collection '%s'.", name)
