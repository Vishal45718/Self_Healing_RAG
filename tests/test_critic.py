"""Tests for Critic component (Task 2.4 & 2.5)."""

import json
from unittest.mock import MagicMock
import pytest

from src.critic.critic import Critic
from src.critic.schema import CriticEvaluation, CriticFailureReason, CriticVerdict
from src.schema import RetrievalResult


def _build_mock_response(json_payload: dict):
    """Helper to mock an InferenceClient chat_completion response returning json_payload."""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(json_payload)
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


def test_critic_grounded_and_sufficient_answer_passes():
    """Grounded answer with sufficient evidence produces PASS verdict."""
    payload = {
        "verdict": "PASS",
        "failure_reason": None,
        "is_retrieval_sufficient": True,
        "is_generation_grounded": True,
        "unsupported_claims": [],
        "reasoning": "The retrieved context fully supports the answer.",
    }
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = _build_mock_response(payload)

    critic = Critic(client=mock_client)
    evaluation = critic.evaluate(
        query="Where does Chroma store data?",
        context="Chroma operates as an embedded database storing parquet files on local disk.",
        answer="Chroma stores parquet files locally on disk.",
    )

    assert isinstance(evaluation, CriticEvaluation)
    assert evaluation.verdict == CriticVerdict.PASS
    assert evaluation.failure_reason is None
    assert evaluation.is_retrieval_sufficient is True
    assert evaluation.is_generation_grounded is True
    assert len(evaluation.unsupported_claims) == 0


def test_critic_insufficient_evidence_fails_with_retrieval_insufficient():
    """Evidence lacking the required answer fails with retrieval_insufficient."""
    payload = {
        "verdict": "FAIL",
        "failure_reason": "retrieval_insufficient",
        "is_retrieval_sufficient": False,
        "is_generation_grounded": True,
        "unsupported_claims": [],
        "reasoning": "The evidence does not mention any GPU requirements.",
    }
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = _build_mock_response(payload)

    critic = Critic(client=mock_client)
    evaluation = critic.evaluate(
        query="What GPU is required?",
        context="Sentence transformers run locally on CPU.",
        answer="The provided context does not mention GPU requirements.",
    )

    assert evaluation.verdict == CriticVerdict.ABSTAIN
    assert evaluation.failure_reason is None
    assert evaluation.is_retrieval_sufficient is False


def test_critic_unsupported_answer_fails_with_generation_ungrounded():
    """Answer with hallucinated or unsupported claims fails with generation_ungrounded."""
    payload = {
        "verdict": "FAIL",
        "failure_reason": "generation_ungrounded",
        "is_retrieval_sufficient": True,
        "is_generation_grounded": False,
        "unsupported_claims": ["Sentence transformers require 64GB of RAM."],
        "reasoning": "The RAM requirement of 64GB is not present in the retrieved evidence.",
    }
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = _build_mock_response(payload)

    critic = Critic(client=mock_client)
    evaluation = critic.evaluate(
        query="What are sentence transformers?",
        context="Sentence transformers run locally on CPU or GPU with deterministic embeddings.",
        answer="Sentence transformers run on CPU and require 64GB of RAM.",
    )

    assert evaluation.verdict == CriticVerdict.FAIL
    assert evaluation.failure_reason == CriticFailureReason.GENERATION_UNGROUNDED
    assert evaluation.is_generation_grounded is False
    assert "Sentence transformers require 64GB of RAM." in evaluation.unsupported_claims


def test_critic_markdown_wrapped_json_parsing():
    """Critic handles response wrapped in markdown code fence correctly."""
    payload = {
        "verdict": "PASS",
        "failure_reason": None,
        "is_retrieval_sufficient": True,
        "is_generation_grounded": True,
        "unsupported_claims": [],
        "reasoning": "Context supports answer.",
    }
    mock_choice = MagicMock()
    mock_choice.message.content = f"Here is the evaluation:\n```json\n{json.dumps(payload)}\n```\nDone."
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = mock_resp

    critic = Critic(client=mock_client)
    evaluation = critic.evaluate(query="Q", context="C", answer="A")
    assert evaluation.verdict == CriticVerdict.PASS


def test_critic_malformed_json_raises_value_error():
    """Non-JSON text returned by LLM raises ValueError."""
    mock_choice = MagicMock()
    mock_choice.message.content = "I am an LLM and I forgot how to output JSON."
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = mock_resp

    critic = Critic(client=mock_client)
    with pytest.raises(ValueError, match="could not be parsed as valid JSON"):
        critic.evaluate(query="Q", context="C", answer="A")


def test_critic_schema_validation_error():
    """JSON missing mandatory fields raises ValueError due to schema validation failure."""
    invalid_payload = {
        "verdict": "MAYBE",  # Invalid enum value
        "reasoning": "Incomplete evaluation",
    }
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = _build_mock_response(invalid_payload)

    critic = Critic(client=mock_client)
    with pytest.raises(ValueError, match="failed schema validation"):
        critic.evaluate(query="Q", context="C", answer="A")


def test_critic_missing_credentials_raises_value_error():
    """Critic raises ValueError when token is missing and no client is supplied."""
    critic = Critic(client=None, token="")
    with pytest.raises(ValueError, match="Hugging Face API token is required"):
        critic.evaluate(query="Q", context="C", answer="A")


@pytest.mark.parametrize("invalid_arg", ["", "   ", None])
def test_critic_invalid_inputs_raise_value_error(invalid_arg):
    """Empty query or answer raises ValueError."""
    critic = Critic(client=MagicMock())
    with pytest.raises(ValueError, match="Query must be a non-empty string"):
        critic.evaluate(query=invalid_arg, context="C", answer="A")

    with pytest.raises(ValueError, match="Answer must be a non-empty string"):
        critic.evaluate(query="Valid Q", context="C", answer=invalid_arg)

def test_critic_empty_context_shortcut():
    """Empty context immediately returns retrieval_insufficient without calling LLM."""
    mock_client = MagicMock()
    critic = Critic(client=mock_client)
    
    evaluation = critic.evaluate(
        query="What is X?",
        context=[],
        answer="I do not know."
    )
    
    assert evaluation.verdict == CriticVerdict.FAIL
    assert evaluation.failure_reason == CriticFailureReason.RETRIEVAL_INSUFFICIENT
    assert evaluation.is_retrieval_sufficient is False
    assert evaluation.is_generation_grounded is True
    # Ensure LLM was not called
    mock_client.chat_completion.assert_not_called()

def test_critic_abstention_shortcut():
    """Abstention answer immediately returns ABSTAIN without calling LLM."""
    mock_client = MagicMock()
    critic = Critic(client=mock_client)
    
    evaluation = critic.evaluate(
        query="What is X?",
        context="Some context that doesn't mention X.",
        answer="I cannot answer based on the provided context."
    )
    
    assert evaluation.verdict == CriticVerdict.ABSTAIN
    assert evaluation.failure_reason is None
    assert evaluation.is_retrieval_sufficient is False
    assert evaluation.is_generation_grounded is True
    # Ensure LLM was not called
    mock_client.chat_completion.assert_not_called()
