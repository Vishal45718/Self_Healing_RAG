"""LLM client abstraction for Self-Healing RAG.

Provides a unified interface across different LLM providers (Google Gemini and
Hugging Face Inference Providers) while maintaining backward compatibility with
existing InferenceClient mock patterns.
"""

from abc import ABC, abstractmethod
import logging
import os
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """Encapsulates a message inside a chat choice."""

    content: str = Field(description="Text content of the message.")


class ChatChoice(BaseModel):
    """Encapsulates a choice in a chat completion response."""

    message: ChatMessage = Field(description="The message object.")


class ChatCompletionResponse(BaseModel):
    """Standard response object compatible with InferenceClient response schema."""

    choices: List[ChatChoice] = Field(description="List of choices generated.")
    raw_response: Optional[Dict[str, Any]] = Field(
        default=None, description="Raw provider response if available."
    )


class BaseLLMClient(ABC):
    """Abstract interface for LLM chat completion clients."""

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatCompletionResponse:
        """Execute chat completion and return standard ChatCompletionResponse."""
        pass


class GeminiLLMClient(BaseLLMClient):
    """Google Gemini client implemented using the official google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.api_key = (
            api_key
            if api_key is not None
            else (settings.gemini_api_key or os.getenv("GEMINI_API_KEY"))
        )
        self.default_model = (
            default_model if default_model is not None else settings.gemini_model_id
        )
        if client is not None:
            self._client = client
        else:
            if not self.api_key:
                logger.warning("No Gemini API key provided or found in settings.")
                self._client = None
            else:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required but not configured. "
                "Set GEMINI_API_KEY in your environment or .env file."
            )
        from google import genai

        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatCompletionResponse:
        """Execute chat completion via google-genai SDK."""
        from google.genai import types

        active_model = model or self.default_model

        system_instruction: Optional[str] = None
        user_parts: List[str] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                if system_instruction:
                    system_instruction += f"\n\n{content}"
                else:
                    system_instruction = content
            elif role == "user":
                user_parts.append(content)
            elif role == "assistant":
                user_parts.append(f"Assistant: {content}")

        combined_user_content = "\n\n".join(user_parts) if user_parts else ""

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        client = self._get_client()
        max_attempts = 5
        base_delay = 2.0
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.models.generate_content(
                    model=active_model,
                    contents=combined_user_content,
                    config=config,
                )
                break
            except Exception as e:
                err_msg = str(e)
                is_rate_limit = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg
                if is_rate_limit and attempt < max_attempts:
                    sleep_time = base_delay * (2 ** (attempt - 1)) + 2.0
                    logger.warning(
                        "Gemini rate limit (429) encountered on attempt %d/%d for '%s'. Backing off for %.1fs...",
                        attempt,
                        max_attempts,
                        active_model,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error("Gemini inference failure on model '%s': %s", active_model, e)
                    raise RuntimeError(
                        f"Gemini inference failed for model '{active_model}': {e}"
                    ) from e

        if not response:
            raise ValueError("Empty response received from Gemini API.")

        text_content: Optional[str] = None
        if hasattr(response, "text") and response.text:
            text_content = response.text
        elif (
            hasattr(response, "candidates")
            and response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            parts_text = [
                p.text
                for p in response.candidates[0].content.parts
                if getattr(p, "text", None)
            ]
            if parts_text:
                text_content = "".join(parts_text)

        if text_content is None or not text_content.strip():
            raise ValueError("Malformed response: empty text received from Gemini API.")

        choice = ChatChoice(message=ChatMessage(content=text_content.strip()))
        raw_dict = None
        if hasattr(response, "model_dump") and callable(response.model_dump):
            dumped = response.model_dump()
            if isinstance(dumped, dict):
                raw_dict = dumped
        return ChatCompletionResponse(
            choices=[choice],
            raw_response=raw_dict,
        )


class HFInferenceAdapter(BaseLLMClient):
    """Adapter wrapping Hugging Face InferenceClient into BaseLLMClient."""

    def __init__(
        self,
        token: Optional[str] = None,
        provider: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.token = token if token is not None else settings.hf_token
        self.provider = provider if provider is not None else settings.hf_provider
        if client is not None:
            self._client = client
        else:
            if not self.token:
                logger.warning("No Hugging Face token provided or found in settings.")
                self._client = None
            else:
                from huggingface_hub import InferenceClient

                self._client = InferenceClient(
                    provider=self.provider,  # type: ignore[arg-type]
                    token=self.token,
                )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.token:
            raise ValueError(
                "Hugging Face API token is required but not configured. "
                "Set HF_TOKEN in your environment or .env file."
            )
        from huggingface_hub import InferenceClient

        self._client = InferenceClient(
            provider=self.provider,  # type: ignore[arg-type]
            token=self.token,
        )
        return self._client

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Any:
        client = self._get_client()
        active_model = model or settings.llm_model_id
        return client.chat_completion(
            messages=messages,
            model=active_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def get_llm_client(
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    token: Optional[str] = None,
    client: Optional[Any] = None,
) -> BaseLLMClient:
    """Factory creating the appropriate LLM client abstraction.

    Ensures Generator, Critic, and graph nodes do not duplicate provider
    resolution, credential verification, or client construction logic.

    Args:
        provider: Provider name ("gemini", "huggingface", "together", etc.).
        model_id: Model ID string.
        token: API key or token string.
        client: Pre-configured or mock client.

    Returns:
        Instance of BaseLLMClient (or provided mock client).

    Raises:
        ValueError: If required credentials are missing or provider is unsupported.
    """
    if client is not None:
        return client

    active_provider = (
        provider if provider is not None else settings.llm_provider
    ).lower()

    if active_provider == "gemini":
        api_key = (
            token
            if token is not None
            else (settings.gemini_api_key or os.getenv("GEMINI_API_KEY"))
        )
        if not api_key:
            raise ValueError(
                "Gemini API key is required for generation but not configured. "
                "Set GEMINI_API_KEY in your environment or pass a configured client."
            )
        return GeminiLLMClient(
            api_key=api_key,
            default_model=model_id or settings.gemini_model_id,
        )
    elif active_provider in ("huggingface", "together", "featherless", "sambanova"):
        hf_token = token if token is not None else settings.hf_token
        if not hf_token:
            raise ValueError(
                "Hugging Face API token is required for generation but not configured. "
                "Set HF_TOKEN in your environment or pass a configured client."
            )
        hf_provider = (
            active_provider
            if active_provider != "huggingface"
            else settings.hf_provider
        )
        return HFInferenceAdapter(
            token=hf_token,
            provider=hf_provider,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: '{active_provider}'")
