"""Tests for BaselineRAG component (Phase 7)."""

import pytest
from unittest.mock import MagicMock

from src.baseline.baseline_rag import BaselineRAG, BaselineResult
from src.generation.generator import GenerationResult
from src.critic.schema import CriticEvaluation, CriticVerdict
from src.schema import RetrievalResult


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    chunk = RetrievalResult(
        id="chunk_test_1",
        content="Raft requires a quorum of floor(N/2) + 1 nodes.",
        distance=0.1,
        similarity=0.9,
    )
    retriever.retrieve.return_value = [chunk]
    return retriever


@pytest.fixture
def mock_generator():
    generator = MagicMock()
    generator.generate.return_value = GenerationResult(
        answer="The quorum size is 3 for 5 nodes.",
        model_id="test-model",
        provider="test-provider",
    )
    return generator


@pytest.fixture
def mock_critic():
    critic = MagicMock()
    critic.evaluate.return_value = CriticEvaluation(
        verdict=CriticVerdict.PASS,
        is_retrieval_sufficient=True,
        is_generation_grounded=True,
        reasoning="Answer is fully supported.",
    )
    return critic


def test_baseline_rag_successful_invoke(mock_retriever, mock_generator, mock_critic):
    """Verify BaselineRAG executes single-pass retrieval and generation with critic."""
    rag = BaselineRAG(
        retriever=mock_retriever,
        generator=mock_generator,
        critic=mock_critic,
    )

    result = rag.invoke("What is the quorum size?")

    assert isinstance(result, BaselineResult)
    assert result.query == "What is the quorum size?"
    assert len(result.retrieved_chunks) == 1
    assert result.retrieved_chunks[0].id == "chunk_test_1"
    assert result.answer == "The quorum size is 3 for 5 nodes."
    assert result.model_id == "test-model"
    assert result.provider == "test-provider"
    assert result.critic_evaluation is not None
    assert result.critic_evaluation.verdict == CriticVerdict.PASS

    # Verify single-pass calls
    mock_retriever.retrieve.assert_called_once_with(query="What is the quorum size?", top_k=5)
    mock_generator.generate.assert_called_once()
    mock_critic.evaluate.assert_called_once()


def test_baseline_rag_without_critic(mock_retriever, mock_generator):
    """Verify BaselineRAG works when no critic is configured."""
    rag = BaselineRAG(
        retriever=mock_retriever,
        generator=mock_generator,
        critic=None,
    )

    result = rag.invoke("What is the quorum size?")

    assert isinstance(result, BaselineResult)
    assert result.critic_evaluation is None
    mock_retriever.retrieve.assert_called_once()
    mock_generator.generate.assert_called_once()


@pytest.mark.parametrize("invalid_query", ["", "   ", None, 123])
def test_baseline_rag_invalid_query_raises(mock_retriever, mock_generator, invalid_query):
    """Verify invalid queries raise ValueError."""
    rag = BaselineRAG(retriever=mock_retriever, generator=mock_generator)
    with pytest.raises(ValueError, match="Query must be a non-empty string"):
        rag.invoke(invalid_query)  # type: ignore


def test_baseline_rag_empty_retrieval_handling(mock_generator, mock_critic):
    """Verify BaselineRAG handles empty vector store / empty retrieval safely."""
    empty_retriever = MagicMock()
    empty_retriever.retrieve.return_value = []

    mock_generator.generate.return_value = GenerationResult(
        answer="The provided context is insufficient to answer this question.",
        model_id="test-model",
        provider="test-provider",
    )
    mock_critic.evaluate.return_value = CriticEvaluation(
        verdict=CriticVerdict.FAIL,
        is_retrieval_sufficient=False,
        is_generation_grounded=True,
        reasoning="Empty context.",
    )

    rag = BaselineRAG(
        retriever=empty_retriever,
        generator=mock_generator,
        critic=mock_critic,
    )

    result = rag.invoke("Unknown question")

    assert len(result.retrieved_chunks) == 0
    assert "insufficient" in result.answer.lower()
    assert result.critic_evaluation.verdict == CriticVerdict.FAIL
