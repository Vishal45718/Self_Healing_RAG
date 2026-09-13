"""Baseline generation component for Self-Healing RAG (Task 2.2).

Produces answers grounded in retrieved context using Hugging Face InferenceClient.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from config.settings import settings
from src.generation.llm_client import BaseLLMClient, get_llm_client
from src.generation.prompts import RAG_SYSTEM_PROMPT, format_rag_messages
from src.schema import RetrievalResult

logger = logging.getLogger(__name__)


class GenerationResult(BaseModel):
    """Structured result of the baseline generation step."""

    answer: str = Field(description="The generated answer text.")
    model_id: str = Field(description="The model used for generation.")
    provider: str = Field(description="The inference provider used.")
    raw_response: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional raw response data from the inference provider.",
    )


class Generator:
    """Generates answers grounded strictly in retrieved context."""

    def __init__(
        self,
        client: Optional[Any] = None,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
        token: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        """Initialise the Generator.

        Args:
            client: Optional pre-configured client/adapter (useful for testing).
            model_id: LLM model ID. Defaults to provider active model in settings.
            provider: LLM provider ("gemini", "huggingface", "together", etc.).
            token: API key or token. Defaults to provider-specific token from settings.
            temperature: Sampling temperature. Defaults to 0.0 for deterministic answers.
            max_tokens: Maximum tokens in generated response.
        """
        self.provider = provider if provider is not None else settings.llm_provider
        if model_id is not None:
            self.model_id = model_id
        else:
            if self.provider.lower() == "gemini":
                self.model_id = settings.gemini_model_id
            else:
                self.model_id = settings.llm_model_id

        self.token = token
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = client

    def _get_client(self) -> Any:
        """Return the active LLM client or raise ValueError if credentials missing."""
        if self._client is not None:
            return self._client

        self._client = get_llm_client(
            provider=self.provider,
            model_id=self.model_id,
            token=self.token,
            client=None,
        )
        return self._client

    def generate(
        self,
        query: str,
        context: Union[List[RetrievalResult], str],
    ) -> GenerationResult:
        """Generate an answer grounded in the retrieved context.

        Args:
            query: The user's query string.
            context: List of RetrievalResult objects or raw context text.

        Returns:
            GenerationResult containing the generated answer, model_id, and provider.

        Raises:
            ValueError: If query is empty/whitespace, credentials missing, or response malformed.
            RuntimeError: If inference fails during API execution.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        # Handle empty retrieved context gracefully without unnecessary API calls
        is_empty_context = False
        if isinstance(context, str) and not context.strip():
            is_empty_context = True
        elif isinstance(context, list) and len(context) == 0:
            is_empty_context = True

        if is_empty_context:
            logger.info("Empty context provided to generator. Returning insufficient context answer.")
            return GenerationResult(
                answer="The provided context is insufficient to answer this question.",
                model_id=self.model_id,
                provider=self.provider,
                raw_response=None,
            )

        client = self._get_client()
        messages = format_rag_messages(query=query, context=context)

        logger.debug(
            "Invoking model '%s' via provider '%s' with %d messages.",
            self.model_id,
            self.provider,
            len(messages),
        )

        try:
            response = client.chat_completion(
                messages=messages,  # type: ignore[arg-type]
                model=self.model_id,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.error("Inference failure on model '%s': %s", self.model_id, e)
            raise RuntimeError(f"Generation inference failed for model '{self.model_id}': {e}") from e

        # Validate response structure
        if not response or not hasattr(response, "choices") or not response.choices:
            raise ValueError("Malformed or empty response received from inference provider.")

        first_choice = response.choices[0]
        if not hasattr(first_choice, "message") or first_choice.message is None:
            raise ValueError("Malformed response: missing message in first choice.")

        answer_text = first_choice.message.content
        if answer_text is None or not answer_text.strip():
            raise ValueError("Malformed response: empty content received in message.")

        return GenerationResult(
            answer=answer_text.strip(),
            model_id=self.model_id,
            provider=self.provider,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )
