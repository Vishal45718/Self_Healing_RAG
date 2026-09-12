"""Tests for Baseline Generator component (Task 2.2, 2.3 & 2.5)."""

from unittest.mock import MagicMock
import pytest

from src.generation.generator import GenerationResult, Generator
from src.generation.prompts import format_rag_messages
from src.schema import RetrievalResult


def test_prompt_formatting_includes_context_and_query():
    """Prompt messages include the user query and all retrieved context items."""
    results = [
        RetrievalResult(
            id="chunk_1",
            content="Vector embeddings capture semantic relationships.",
            metadata={"source": "doc1.txt"},
            distance=0.1,
            similarity=0.9,
        ),
        RetrievalResult(
            id="chunk_2",
            content="Chroma persists embeddings on disk.",
            metadata={"source": "doc2.txt"},
            distance=0.2,
            similarity=0.8,
        ),
    ]
    query = "How do vector embeddings work?"

    messages = format_rag_messages(query=query, context=results)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "ONLY the provided retrieved context" in messages[0]["content"]

    user_msg = messages[1]["content"]
    assert "Question: How do vector embeddings work?" in user_msg
    assert "chunk_1" in user_msg
    assert "Vector embeddings capture semantic relationships." in user_msg
    assert "chunk_2" in user_msg
    assert "Chroma persists embeddings on disk." in user_msg


def test_successful_generation_with_mocked_client():
    """Generator produces structured GenerationResult from successful inference."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Vector embeddings represent text as dense numerical vectors."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.model_dump.return_value = {"id": "chatcmpl-123"}
    mock_client.chat_completion.return_value = mock_response

    generator = Generator(
        client=mock_client,
        model_id="test-model",
        provider="together",
    )

    context = [
        RetrievalResult(
            id="c1",
            content="Vector embeddings represent text as dense numerical vectors.",
            metadata={},
            distance=0.05,
            similarity=0.95,
        )
    ]

    result = generator.generate(query="What are embeddings?", context=context)

    assert isinstance(result, GenerationResult)
    assert result.answer == "Vector embeddings represent text as dense numerical vectors."
    assert result.model_id == "test-model"
    assert result.provider == "together"
    mock_client.chat_completion.assert_called_once()


def test_missing_credentials_raises_value_error():
    """Generator raises ValueError when token is missing and no client is supplied."""
    generator = Generator(client=None, token="")
    with pytest.raises(ValueError, match="Hugging Face API token is required"):
        generator.generate(query="Valid query", context="Some non-empty context")


def test_empty_context_handling_returns_insufficient_message_safely():
    """Empty context (list or empty string) safely returns insufficient context without calling API."""
    mock_client = MagicMock()
    generator = Generator(client=mock_client)

    # Test empty list
    result1 = generator.generate(query="What is LangGraph?", context=[])
    assert "insufficient to answer" in result1.answer.lower()
    mock_client.chat_completion.assert_not_called()

    # Test blank string
    result2 = generator.generate(query="What is LangGraph?", context="   ")
    assert "insufficient to answer" in result2.answer.lower()
    mock_client.chat_completion.assert_not_called()


@pytest.mark.parametrize("invalid_query", ["", "   ", None])
def test_invalid_query_raises_value_error(invalid_query):
    """Empty or invalid query string raises ValueError."""
    generator = Generator(client=MagicMock())
    with pytest.raises(ValueError, match="Query must be a non-empty string"):
        generator.generate(query=invalid_query, context="Some context")


def test_inference_failure_raises_runtime_error():
    """Inference failures (e.g. network/500 errors) are caught and raise RuntimeError."""
    mock_client = MagicMock()
    mock_client.chat_completion.side_effect = ConnectionError("Could not reach API")

    generator = Generator(client=mock_client)
    with pytest.raises(RuntimeError, match="Generation inference failed"):
        generator.generate(query="What is RAG?", context="Valid non-empty context")


def test_malformed_response_handling():
    """Malformed responses (missing choices or empty content) raise ValueError."""
    mock_client = MagicMock()

    # Response with empty choices list
    empty_choices_resp = MagicMock()
    empty_choices_resp.choices = []
    mock_client.chat_completion.return_value = empty_choices_resp

    generator = Generator(client=mock_client)
    with pytest.raises(ValueError, match="Malformed or empty response"):
        generator.generate(query="What is RAG?", context="Some context")

    # Response with None content
    none_content_choice = MagicMock()
    none_content_choice.message.content = ""
    none_content_resp = MagicMock()
    none_content_resp.choices = [none_content_choice]
    mock_client.chat_completion.return_value = none_content_resp

    with pytest.raises(ValueError, match="empty content received"):
        generator.generate(query="What is RAG?", context="Some context")
