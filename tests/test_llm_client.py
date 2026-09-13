"""Unit tests for LLM client abstractions (GeminiLLMClient, HFInferenceAdapter, get_llm_client).

All tests are completely offline, fast, and deterministic, relying on mocked SDK objects.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.generation.llm_client import (
    ChatChoice,
    ChatCompletionResponse,
    ChatMessage,
    GeminiLLMClient,
    HFInferenceAdapter,
    get_llm_client,
)


def test_gemini_client_missing_key_raises_value_error():
    """GeminiLLMClient raises ValueError when no API key is provided or configured."""
    client = GeminiLLMClient(api_key="")
    with pytest.raises(ValueError, match="Gemini API key is required"):
        client.chat_completion(messages=[{"role": "user", "content": "Hello"}])


def test_gemini_client_successful_completion():
    """GeminiLLMClient translates chat messages into Gemini generate_content call."""
    mock_genai_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Grounded response from Gemini."
    mock_response.model_dump.return_value = {"text": "Grounded response from Gemini."}
    mock_genai_client.models.generate_content.return_value = mock_response

    gemini_client = GeminiLLMClient(
        api_key="mock-key",
        default_model="gemini-3.6-flash",
        client=mock_genai_client,
    )

    messages = [
        {"role": "system", "content": "You are a factual assistant."},
        {"role": "user", "content": "What is Python?"},
    ]

    result = gemini_client.chat_completion(
        messages=messages,
        model="gemini-3.6-flash",
        temperature=0.0,
        max_tokens=500,
    )

    assert isinstance(result, ChatCompletionResponse)
    assert len(result.choices) == 1
    assert result.choices[0].message.content == "Grounded response from Gemini."

    # Verify call arguments passed to generate_content
    mock_genai_client.models.generate_content.assert_called_once()
    call_kwargs = mock_genai_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.6-flash"
    assert call_kwargs["contents"] == "What is Python?"
    config = call_kwargs["config"]
    assert config.system_instruction == "You are a factual assistant."
    assert config.temperature == 0.0
    assert config.max_output_tokens == 500


def test_gemini_client_fallback_parts_parsing():
    """GeminiLLMClient correctly falls back to candidates.parts when response.text is empty."""
    mock_genai_client = MagicMock()
    mock_part = MagicMock()
    mock_part.text = "Answer from candidates part."
    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part]
    mock_response = MagicMock(text=None, candidates=[mock_candidate])
    mock_genai_client.models.generate_content.return_value = mock_response

    gemini_client = GeminiLLMClient(
        api_key="mock-key",
        client=mock_genai_client,
    )

    result = gemini_client.chat_completion(
        messages=[{"role": "user", "content": "Test"}],
    )
    assert result.choices[0].message.content == "Answer from candidates part."


def test_gemini_client_error_wrapping():
    """Gemini client wraps SDK errors into RuntimeError with clear message."""
    mock_genai_client = MagicMock()
    mock_genai_client.models.generate_content.side_effect = Exception("Quota exceeded")

    gemini_client = GeminiLLMClient(
        api_key="mock-key",
        default_model="gemini-3.6-flash",
        client=mock_genai_client,
    )

    with pytest.raises(RuntimeError, match="Gemini inference failed for model 'gemini-3.6-flash'"):
        gemini_client.chat_completion(messages=[{"role": "user", "content": "Hi"}])


def test_hf_adapter_missing_token_raises_value_error():
    """HFInferenceAdapter raises ValueError when no token is provided."""
    adapter = HFInferenceAdapter(token="")
    with pytest.raises(ValueError, match="Hugging Face API token is required"):
        adapter.chat_completion(messages=[{"role": "user", "content": "Hello"}])


def test_hf_adapter_delegation():
    """HFInferenceAdapter delegates directly to wrapped client chat_completion."""
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = "raw-response"

    adapter = HFInferenceAdapter(token="mock-token", client=mock_client)
    res = adapter.chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        model="test-model",
        temperature=0.5,
        max_tokens=100,
    )

    assert res == "raw-response"
    mock_client.chat_completion.assert_called_once_with(
        messages=[{"role": "user", "content": "Hello"}],
        model="test-model",
        temperature=0.5,
        max_tokens=100,
    )


def test_get_llm_client_factory():
    """get_llm_client returns pre-configured client, Gemini client, or HF adapter."""
    # 1. Pre-configured client passthrough
    mock_client = MagicMock()
    assert get_llm_client(client=mock_client) is mock_client

    # 2. Gemini provider client
    client_gemini = get_llm_client(
        provider="gemini",
        token="test-api-key",
        model_id="gemini-3.6-flash",
    )
    assert isinstance(client_gemini, GeminiLLMClient)
    assert client_gemini.default_model == "gemini-3.6-flash"

    # 3. HF provider client
    client_hf = get_llm_client(
        provider="together",
        token="test-hf-token",
        model_id="meta-llama/Llama-3.3-70B-Instruct",
    )
    assert isinstance(client_hf, HFInferenceAdapter)
    assert client_hf.provider == "together"

    # 4. Unsupported provider raises ValueError
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm_client(provider="unsupported_provider", token="tok")
