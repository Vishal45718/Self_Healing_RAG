"""
Integration test for the real embedder (src/embeddings/embedder.py).

This exercises the actual sentence-transformers model, which requires
downloading weights from huggingface.co on first run. If that network
access isn't available (e.g. a restricted sandbox), the test skips
itself rather than failing the whole suite — everything else in this
project is tested against a deterministic fake embedder and does not
depend on this.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.embeddings.embedder import embed_texts  # noqa: E402


def _model_available() -> bool:
    try:
        embed_texts(["connectivity check"])
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _model_available(), reason="sentence-transformers model unreachable (no network to huggingface.co)")
def test_embed_texts_returns_vectors_of_consistent_dimension():
    vectors = embed_texts(["hello world", "another sentence"])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
    assert len(vectors[0]) > 0


@pytest.mark.skipif(not _model_available(), reason="sentence-transformers model unreachable (no network to huggingface.co)")
def test_embed_texts_is_deterministic():
    v1 = embed_texts(["The Eiffel Tower is in Paris."])
    v2 = embed_texts(["The Eiffel Tower is in Paris."])
    assert v1 == v2


def test_embed_empty_list_returns_empty_list():
    # No model load needed — should short-circuit
    assert embed_texts([]) == []
