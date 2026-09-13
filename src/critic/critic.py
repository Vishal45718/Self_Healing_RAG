"""Critic component for Self-Healing RAG (Task 2.4).

Evaluates retrieval sufficiency and generation groundedness using structured output.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

from config.settings import settings
from src.critic.prompts import format_critic_messages
from src.critic.schema import CriticEvaluation, CriticFailureReason, CriticVerdict
from src.generation.llm_client import get_llm_client
from src.schema import RetrievalResult

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract and parse JSON object from model output text.

    Handles optional markdown code fences and extraneous leading/trailing text.
    """
    clean_text = text.strip()

    # If wrapped in markdown code fence, extract the content
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
    if code_block_match:
        clean_text = code_block_match.group(1).strip()

    # Find the outermost JSON object if there's surrounding text
    start_idx = clean_text.find("{")
    end_idx = clean_text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        clean_text = clean_text[start_idx : end_idx + 1]

    try:
        return json.loads(clean_text)
    except Exception as e:
        raise ValueError(f"Critic output could not be parsed as valid JSON: {text}") from e


class Critic:
    """Evaluates whether retrieved evidence is sufficient and generated answers are grounded."""

    def __init__(
        self,
        client: Optional[Any] = None,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
        token: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        """Initialise the Critic.

        Args:
            client: Optional pre-configured client/adapter (useful for testing).
            model_id: LLM model ID. Defaults to provider active model in settings.
            provider: LLM provider ("gemini", "huggingface", "together", etc.).
            token: API key or token. Defaults to provider-specific token from settings.
            temperature: Sampling temperature. Defaults to 0.0 for deterministic evaluation.
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

    def evaluate(
        self,
        query: str,
        context: Union[List[RetrievalResult], str],
        answer: str,
    ) -> CriticEvaluation:
        """Evaluate the generated answer against the retrieved evidence.

        Args:
            query: The user query.
            context: The retrieved context evidence.
            answer: The generated answer to evaluate.

        Returns:
            CriticEvaluation containing verdict, failure_reason, sufficiency, groundedness,
            unsupported claims, and reasoning.

        Raises:
            ValueError: If query or answer is empty, or if critic output fails validation.
            RuntimeError: If the inference call fails.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Answer must be a non-empty string.")

        if not context:
            logger.info("Shortcut: Empty context, returning retrieval_insufficient.")
            return CriticEvaluation(
                verdict=CriticVerdict.FAIL,
                failure_reason=CriticFailureReason.RETRIEVAL_INSUFFICIENT,
                is_retrieval_sufficient=False,
                is_generation_grounded=True,
                unsupported_claims=[],
                reasoning="Context is empty, automatic retrieval failure.",
            )

        abstention_keywords = [
            "i don't know",
            "i cannot answer",
            "does not contain sufficient information",
            "provided context does not",
            "no information is provided",
            "i do not have enough information",
            "i can't answer",
            "insufficient information",
            "does not mention"
        ]
        ans_lower = answer.lower()
        if any(k in ans_lower for k in abstention_keywords):
            logger.info("Shortcut: Generator abstained, returning ABSTAIN verdict.")
            return CriticEvaluation(
                verdict=CriticVerdict.ABSTAIN,
                failure_reason=None,
                is_retrieval_sufficient=False,
                is_generation_grounded=True,
                unsupported_claims=[],
                reasoning="The generator safely abstained from answering due to insufficient context.",
            )

        client = self._get_client()
        messages = format_critic_messages(query=query, context=context, answer=answer)

        logger.debug(
            "Invoking Critic with model '%s' via provider '%s'.",
            self.model_id,
            self.provider,
        )

        try:
            response = client.chat_completion(
                messages=messages,  # type: ignore[arg-type]
                model=self.model_id,
                temperature=self.temperature,
            )
        except Exception as e:
            logger.error("Critic inference failure on model '%s': %s", self.model_id, e)
            raise RuntimeError(f"Critic inference failed for model '{self.model_id}': {e}") from e

        if not response or not hasattr(response, "choices") or not response.choices:
            raise ValueError("Malformed or empty response received from critic inference.")

        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("Critic returned empty response content.")

        parsed_json = _extract_json(content)

        try:
            evaluation = CriticEvaluation.model_validate(parsed_json)
        except ValidationError as e:
            logger.error("Critic response failed schema validation: %s", e)
            raise ValueError(f"Critic response failed schema validation: {e}") from e

        # Normalize verdict and failure_reason consistency
        if evaluation.verdict == CriticVerdict.ABSTAIN:
            evaluation.failure_reason = None
        elif evaluation.is_retrieval_sufficient and evaluation.is_generation_grounded:
            if evaluation.verdict != CriticVerdict.PASS:
                evaluation.verdict = CriticVerdict.PASS
            evaluation.failure_reason = None
        else:
            if evaluation.verdict != CriticVerdict.FAIL:
                evaluation.verdict = CriticVerdict.FAIL
            if not evaluation.failure_reason:
                if not evaluation.is_retrieval_sufficient:
                    evaluation.failure_reason = CriticFailureReason.RETRIEVAL_INSUFFICIENT
                else:
                    evaluation.failure_reason = CriticFailureReason.GENERATION_UNGROUNDED

        logger.info(
            "Critic evaluation completed: verdict=%s, failure_reason=%s",
            evaluation.verdict.value,
            evaluation.failure_reason.value if evaluation.failure_reason else "None",
        )
        return evaluation
