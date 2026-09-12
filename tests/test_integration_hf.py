"""Live integration tests for Hugging Face inference (Task 2.5).

This module contains integration tests against the live Hugging Face Inference API.
It automatically skips when HF_TOKEN is not configured or available.
"""

import pytest

from config.settings import settings
from src.critic.critic import Critic
from src.critic.schema import CriticVerdict
from src.generation.generator import Generator


@pytest.mark.integration
@pytest.mark.skipif(
    not settings.hf_token,
    reason="Hugging Face live inference integration test skipped: HF_TOKEN is not set.",
)
def test_live_hf_generation_and_critic_pipeline():
    """Verify live generation and critic against the configured Hugging Face provider."""
    # 1. Test live generation
    generator = Generator()
    query = "What is the capital of France?"
    context = "France is a country in Western Europe. Its capital and largest city is Paris."

    gen_result = generator.generate(query=query, context=context)
    assert gen_result is not None
    assert "Paris" in gen_result.answer

    # 2. Test live critic on the generated answer
    critic = Critic()
    crit_result = critic.evaluate(query=query, context=context, answer=gen_result.answer)
    assert crit_result is not None
    assert crit_result.verdict == CriticVerdict.PASS
    assert crit_result.is_generation_grounded is True
    assert crit_result.is_retrieval_sufficient is True
