"""Live integration tests for Hugging Face inference (Task 2.5).

This module contains integration tests against the live Hugging Face Inference API.
It automatically skips when:
  - HF_TOKEN is not configured or available in the environment.
  - The configured HF_TOKEN lacks Inference Provider permissions (HTTP 403).
    This is an environment/credential condition, not a code defect. The HF Free
    tier token requires "Make calls to the serverless Inference API" permission
    and may additionally need a subscription for third-party providers such as
    Together AI. A 403 from the provider is therefore an expected environment
    limitation that must not be reported as a code failure.

It does NOT skip on arbitrary network failures or unexpected errors; those are
propagated as genuine test failures.
"""

import pytest
from huggingface_hub.errors import HfHubHTTPError

from config.settings import settings
from src.critic.critic import Critic
from src.critic.schema import CriticVerdict
from src.generation.generator import Generator


def _check_runtime_error_for_auth_skip(exc: RuntimeError) -> None:
    """Re-raise as pytest.skip if the underlying cause is a 403 Forbidden.

    The production code wraps HfHubHTTPError inside RuntimeError. This helper
    inspects the __cause__ chain and converts ONLY the specific
    "insufficient permissions" condition (HTTP 403) into a pytest skip.
    All other runtime errors are re-raised as-is so they surface as real
    test failures.

    Args:
        exc: The RuntimeError raised by Generator.generate or Critic.evaluate.
    """
    cause = exc.__cause__
    if isinstance(cause, HfHubHTTPError):
        status = getattr(getattr(cause, "response", None), "status_code", None)
        if status == 403:
            pytest.skip(
                "Hugging Face live inference skipped: the configured HF_TOKEN does not "
                "have sufficient Inference Provider permissions (HTTP 403 Forbidden). "
                "This is an environment/credential limitation, not a code defect. "
                "Grant the token 'Make calls to the serverless Inference API' permission "
                "or use a PRO subscription token to run this test."
            )
    raise exc


@pytest.mark.integration
@pytest.mark.skipif(
    not settings.hf_token,
    reason="Hugging Face live inference integration test skipped: HF_TOKEN is not set.",
)
def test_live_hf_generation_and_critic_pipeline():
    """Verify live generation and critic against the configured Hugging Face provider."""
    generator = Generator(provider="huggingface")
    query = "What is the capital of France?"
    context = "France is a country in Western Europe. Its capital and largest city is Paris."

    # 1. Test live generation
    try:
        gen_result = generator.generate(query=query, context=context)
    except RuntimeError as exc:
        _check_runtime_error_for_auth_skip(exc)
        raise  # unreachable but satisfies type checkers

    assert gen_result is not None
    assert "Paris" in gen_result.answer

    # 2. Test live critic on the generated answer
    critic = Critic(provider="huggingface")
    try:
        crit_result = critic.evaluate(query=query, context=context, answer=gen_result.answer)
    except RuntimeError as exc:
        _check_runtime_error_for_auth_skip(exc)
        raise  # unreachable but satisfies type checkers

    assert crit_result is not None
    assert crit_result.verdict == CriticVerdict.PASS
    assert crit_result.is_generation_grounded is True
    assert crit_result.is_retrieval_sufficient is True
