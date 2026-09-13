"""Live integration tests for Google Gemini inference.

This module contains integration tests against the live Google Gemini API using
the official google-genai SDK.
It automatically skips when:
  - GEMINI_API_KEY is not configured or available in the environment.
  - The API key is invalid, lacks quota/permissions, or service is unavailable.
    This is an environment/credential condition, not a code defect.

It does NOT skip on arbitrary assertion failures or internal logic errors.
"""

import os
from typing import Optional
import pytest
from google.genai.errors import APIError, ClientError

from config.settings import settings
from src.critic.critic import Critic
from src.critic.schema import CriticVerdict
from src.generation.generator import Generator


def _check_runtime_error_for_gemini_auth_skip(exc: Exception) -> None:
    """Re-raise as pytest.skip if any underlying cause is an auth or quota error.

    Inspects the entire __cause__ / __context__ chain and converts credential/quota conditions
    (e.g., INVALID_ARGUMENT, UNAUTHENTICATED, PERMISSION_DENIED, RESOURCE_EXHAUSTED, API key not valid)
    into a pytest skip. All other runtime errors are re-raised as-is.
    """
    curr: Optional[BaseException] = exc
    while curr is not None:
        if isinstance(curr, (ClientError, APIError)):
            status_code = getattr(curr, "code", None)
            message = str(curr).lower()
            if (
                status_code in (400, 401, 403, 429)
                or "api key not valid" in message
                or "unauthenticated" in message
                or "permission_denied" in message
                or "resource_exhausted" in message
                or "quota" in message
                or "invalid_argument" in message
            ):
                pytest.skip(
                    f"Gemini live inference skipped due to API credential/quota limitation: {curr}"
                )
        message = str(curr).lower()
        if (
            "api key not valid" in message
            or "unauthenticated" in message
            or "permission_denied" in message
            or "resource_exhausted" in message
        ):
            pytest.skip(
                f"Gemini live inference skipped due to API credential/quota limitation: {curr}"
            )
        curr = curr.__cause__ or getattr(curr, "__context__", None)
    raise exc


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") and not settings.gemini_api_key,
    reason="Google Gemini live inference integration test skipped: GEMINI_API_KEY is not set.",
)
def test_live_gemini_generation_and_critic_pipeline():
    """Verify live generation and critic against Google Gemini (using default gemini-3.6-flash)."""
    generator = Generator(provider="gemini")
    query = "What is the capital of France?"
    context = "France is a country in Western Europe. Its capital and largest city is Paris."

    # 1. Test live generation
    try:
        gen_result = generator.generate(query=query, context=context)
    except RuntimeError as exc:
        _check_runtime_error_for_gemini_auth_skip(exc)
        raise

    assert gen_result is not None
    assert "Paris" in gen_result.answer
    assert gen_result.provider == "gemini"

    # 2. Test live critic on the generated answer
    critic = Critic(provider="gemini")
    try:
        crit_result = critic.evaluate(
            query=query, context=context, answer=gen_result.answer
        )
    except RuntimeError as exc:
        _check_runtime_error_for_gemini_auth_skip(exc)
        raise

    assert crit_result is not None
    assert crit_result.verdict == CriticVerdict.PASS
    assert crit_result.is_generation_grounded is True
    assert crit_result.is_retrieval_sufficient is True
