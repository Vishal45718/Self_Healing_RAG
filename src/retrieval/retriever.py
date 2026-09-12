"""Retriever component for Self-Healing RAG (Task 2.1).

Coordinates query embedding via LocalEmbedder and similarity search via VectorStore,
returning structured RetrievalResult instances.
"""

import logging
from typing import List, Optional

from src.ingestion.embedder import LocalEmbedder
from src.ingestion.vector_store import VectorStore
from src.schema import RetrievalResult

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant document chunks from the Chroma vector store for a given query."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[LocalEmbedder] = None,
    ) -> None:
        """Initialise Retriever with vector store and embedding model.

        Args:
            vector_store: VectorStore instance. If None, instantiates a default VectorStore.
            embedder: LocalEmbedder instance. If None, instantiates a default LocalEmbedder.
        """
        self.vector_store = vector_store if vector_store is not None else VectorStore()
        self.embedder = embedder if embedder is not None else LocalEmbedder()

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve the top-k most similar chunks for a natural-language query.

        Args:
            query: The user query string.
            top_k: Number of nearest chunks to retrieve. Must be >= 1.

        Returns:
            A list of RetrievalResult objects ordered by relevance (lowest distance first).
            Returns an empty list if the vector store contains no documents.

        Raises:
            ValueError: If query is empty, whitespace, or invalid, or if top_k < 1.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if self.vector_store.count() == 0:
            logger.warning("Vector store is empty; returning 0 retrieval results.")
            return []

        logger.debug("Embedding query: %s", query[:50])
        query_embedding = self.embedder.embed_query(query)

        raw_results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
        )

        structured_results: List[RetrievalResult] = []
        for item in raw_results:
            dist = float(item["distance"])
            sim = round(1.0 - dist, 6)
            structured_results.append(
                RetrievalResult(
                    id=item["id"],
                    content=item["content"],
                    metadata=item["metadata"],
                    distance=dist,
                    similarity=sim,
                )
            )

        logger.info("Retrieved %d chunks for query.", len(structured_results))
        return structured_results
