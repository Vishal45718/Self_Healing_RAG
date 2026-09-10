"""
Embeddings.

Thin wrapper around sentence-transformers. One model, config-driven
(see config/settings.EMBEDDING_MODEL) — no multi-backend abstraction.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import settings  # noqa: E402

_model = None  # lazy-loaded singleton, avoids reloading weights per call


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # imported lazily

        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using the configured sentence-transformers model."""
    if not texts:
        return []
    model = get_model()
    return model.encode(texts, convert_to_numpy=True).tolist()
