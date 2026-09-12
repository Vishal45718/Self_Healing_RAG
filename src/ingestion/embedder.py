"""Local embedding pipeline using sentence-transformers.

Encodes document chunks into dense vector representations for downstream
Chroma vector storage (Task 1.4).

Design rationale: see DECISIONS.md D-004.
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from src.schema import Chunk

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """Generates dense vector embeddings for Chunk objects using a local model.

    Uses ``sentence_transformers.SentenceTransformer`` directly as mandated by
    the project's architecture guidelines (AGENTS.md §3.2 — standard interfaces,
    no unnecessary wrappers).
    """

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Load the SentenceTransformer model.

        Args:
            model_id: Hugging Face model identifier or local path.
                      Defaults to ``all-MiniLM-L6-v2`` (D-004).
        """
        logger.info("Loading embedding model: %s", model_id)
        self.model_id: str = model_id
        self._model: SentenceTransformer = SentenceTransformer(model_id)
        logger.info("Embedding model loaded successfully.")

    def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        """Embed a list of Chunk objects.

        Args:
            chunks: Document chunks whose ``content`` field will be embedded.

        Returns:
            A list of embedding vectors (one per chunk), where each vector is a
            ``list[float]``.  Order matches the input list order.

        Raises:
            ValueError: If ``chunks`` is empty.
        """
        if not chunks:
            raise ValueError("Cannot embed an empty list of chunks.")

        texts: List[str] = [chunk.content for chunk in chunks]
        logger.debug("Embedding %d chunks with model '%s'.", len(texts), self.model_id)

        # convert_to_numpy=True returns a numpy array; .tolist() gives plain Python floats
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string.

        Args:
            query: The raw query text to embed.

        Returns:
            A single embedding vector as a ``list[float]``.

        Raises:
            ValueError: If ``query`` is an empty string.
        """
        if not query.strip():
            raise ValueError("Cannot embed an empty query string.")

        logger.debug("Embedding query of length %d.", len(query))
        embedding = self._model.encode([query], convert_to_numpy=True)
        return embedding[0].tolist()
