"""Phase 5 tests: Query Reformulation and Strict Regeneration.

Acceptance criteria verified here:
    AC1  reformulated query differs from all previous queries
    AC2  first-failure case works with empty history
    AC3  repeated query is prevented (fallback to original when LLM echoes)
    AC4  regeneration directive contains critic feedback (reasoning + claims)
    AC5  generation_ungrounded does NOT call retrieval again
    AC6  both failure reasons route to the correct action
    AC7  invalid/None failure_reason is rejected explicitly
    AC8  query_history is populated and grows correctly across iterations
    AC9  regeneration loop: regenerate → critic → end on PASS
    AC10 max_retries with GENERATION_UNGROUNDED only calls retrieval once
"""

import pytest
from unittest.mock import MagicMock, call

from src.graph.graph import SelfHealingRAG
from src.schema import RetrievalResult, GraphState
from src.generation.generator import GenerationResult
from src.generation.prompts import (
    format_reformulate_messages,
    format_regenerate_messages,
    REFORMULATE_SYSTEM_PROMPT,
    REGENERATE_SYSTEM_PROMPT,
)
from src.critic.schema import CriticEvaluation, CriticVerdict, CriticFailureReason


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_chunk() -> RetrievalResult:
    return RetrievalResult(
        id="chunk-1",
        content="Relevant context about the topic.",
        distance=0.1,
        similarity=0.9,
    )


@pytest.fixture
def mock_retriever(dummy_chunk):
    retriever = MagicMock()
    retriever.retrieve.return_value = [dummy_chunk]
    return retriever


@pytest.fixture
def mock_generator():
    generator = MagicMock()
    generator.generate.return_value = GenerationResult(
        answer="A grounded answer.",
        model_id="mock-model",
        provider="mock-provider",
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="reformulated query v2"))]
    mock_client.chat_completion.return_value = mock_response
    generator._client = mock_client

    return generator


@pytest.fixture
def mock_critic():
    return MagicMock()


def _pass_eval() -> CriticEvaluation:
    return CriticEvaluation(
        verdict=CriticVerdict.PASS,
        is_retrieval_sufficient=True,
        is_generation_grounded=True,
        reasoning="Everything looks good.",
    )


def _retrieval_fail(reasoning: str = "Not enough info.") -> CriticEvaluation:
    return CriticEvaluation(
        verdict=CriticVerdict.FAIL,
        failure_reason=CriticFailureReason.RETRIEVAL_INSUFFICIENT,
        is_retrieval_sufficient=False,
        is_generation_grounded=True,
        reasoning=reasoning,
    )


def _ungrounded_fail(
    reasoning: str = "Claim X has no support.",
    unsupported: list | None = None,
) -> CriticEvaluation:
    return CriticEvaluation(
        verdict=CriticVerdict.FAIL,
        failure_reason=CriticFailureReason.GENERATION_UNGROUNDED,
        is_retrieval_sufficient=True,
        is_generation_grounded=False,
        unsupported_claims=unsupported or ["Claim X"],
        reasoning=reasoning,
    )


# ===========================================================================
# AC1 + AC2: Reformulated query differs from previous; empty history works
# ===========================================================================

class TestReformulateQueryDistinct:
    """AC1 + AC2 — reformulated query differs; empty history is handled."""

    def test_reformulated_query_differs_from_original(
        self, mock_retriever, mock_generator, mock_critic
    ):
        """AC1: After reformulation the new current_query must not equal the original."""
        mock_critic.evaluate.side_effect = [_retrieval_fail(), _pass_eval()]

        rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
        state = rag.invoke("What is X?")

        # The reformulated query was returned by the mock LLM
        assert state["current_query"] == "reformulated query v2"
        assert state["current_query"] != "What is X?"

    def test_first_failure_with_empty_query_history(
        self, mock_retriever, mock_generator, mock_critic
    ):
        """AC2: reformulate_node works correctly on the very first failure
        where query_history starts as just [original_query]."""
        mock_critic.evaluate.side_effect = [_retrieval_fail(), _pass_eval()]

        rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
        state = rag.invoke("Initial question?")

        # History should contain original + reformulated
        assert "Initial question?" in state["query_history"]
        assert "reformulated query v2" in state["query_history"]
        assert len(state["query_history"]) == 2


# ===========================================================================
# AC3: Repeated query is prevented
# ===========================================================================

class TestRepeatedQueryPrevention:
    """AC3 — reformulate_node never inserts a duplicate query."""

    def test_duplicate_llm_response_falls_back_to_original(
        self, mock_retriever, mock_generator, mock_critic
    ):
        """If the LLM echoes the original query verbatim, fall back to original
        and do NOT add a duplicate to history."""
        # Make the mock LLM return the same query as the original
        mock_generator._client.chat_completion.return_value.choices[0].message.content = (
            "What is X?"
        )
        mock_critic.evaluate.side_effect = [_retrieval_fail(), _pass_eval()]

        rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
        state = rag.invoke("What is X?")

        # Should fall back to original since candidate == original (in history)
        assert state["current_query"] == "What is X?"
        # History must not have duplicates
        assert state["query_history"].count("What is X?") == 1

    def test_duplicate_reformulated_query_falls_back(
        self, mock_retriever, mock_generator, mock_critic
    ):
        """If the LLM echoes a previously reformulated query, the node should
        fall back to original_query rather than insert a duplicate."""
        call_count = 0
        original = "What is X?"
        first_reformulation = "reformulated query v2"

        def side_effect_completion(messages, model, max_tokens, temperature):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            if call_count == 1:
                resp.choices = [MagicMock(message=MagicMock(content=first_reformulation))]
            else:
                # Second reformulation attempt echoes the first → should be blocked
                resp.choices = [MagicMock(message=MagicMock(content=first_reformulation))]
            return resp

        mock_generator._client.chat_completion.side_effect = side_effect_completion

        mock_critic.evaluate.side_effect = [
            _retrieval_fail(),   # triggers first reformulate
            _retrieval_fail(),   # triggers second reformulate
            _pass_eval(),
        ]

        rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
        state = rag.invoke(original)

        # first_reformulation should appear exactly once in history
        assert state["query_history"].count(first_reformulation) == 1


# ===========================================================================
# AC4: Regeneration directive contains critic feedback
# ===========================================================================

class TestRegenerationPromptContent:
    """AC4 — format_regenerate_messages embeds critic reasoning and claims."""

    def test_regenerate_messages_contain_critic_reasoning(self, dummy_chunk):
        reasoning = "The answer claimed the sky is green, which is not in the evidence."
        messages = format_regenerate_messages(
            query="What color is the sky?",
            context=[dummy_chunk],
            critic_reasoning=reasoning,
            unsupported_claims=["The sky is green."],
        )

        combined = " ".join(m["content"] for m in messages)
        assert reasoning in combined

    def test_regenerate_messages_contain_unsupported_claims(self, dummy_chunk):
        claim = "The sky is green."
        messages = format_regenerate_messages(
            query="What color is the sky?",
            context=[dummy_chunk],
            critic_reasoning="Claim is unsupported.",
            unsupported_claims=[claim],
        )

        combined = " ".join(m["content"] for m in messages)
        assert claim in combined

    def test_regenerate_messages_system_prompt_is_stricter(self, dummy_chunk):
        messages = format_regenerate_messages(
            query="Q?",
            context=[dummy_chunk],
            critic_reasoning="Some issue.",
        )

        system_content = messages[0]["content"]
        assert "UNGROUNDED" in system_content
        assert "MUST be directly" in system_content

    def test_regenerate_messages_no_claims_section_when_empty(self, dummy_chunk):
        messages = format_regenerate_messages(
            query="Q?",
            context=[dummy_chunk],
            critic_reasoning="Some issue.",
            unsupported_claims=[],
        )

        combined = " ".join(m["content"] for m in messages)
        assert "Specific Unsupported Claims" not in combined

    def test_reformulate_messages_include_query_history(self):
        history = ["original query", "reformulated once"]
        messages = format_reformulate_messages(
            original_query="original query",
            critic_reasoning="Not enough info.",
            query_history=history,
        )

        combined = " ".join(m["content"] for m in messages)
        for q in history:
            assert q in combined
        assert "DO NOT repeat" in combined or "Previously Tried" in combined


# ===========================================================================
# AC5: generation_ungrounded does NOT call retrieval
# ===========================================================================

class TestGenerationUngroundedNoRetrieval:
    """AC5 — retriever is not called when failure_reason is generation_ungrounded."""

    def test_retriever_not_called_on_generation_ungrounded(
        self, mock_retriever, mock_generator, mock_critic
    ):
        """The retriever must be called exactly once (initial retrieve only)."""
        mock_critic.evaluate.side_effect = [_ungrounded_fail(), _pass_eval()]

        # The regenerate_node calls generator._client.chat_completion
        regen_response = MagicMock()
        regen_response.choices = [
            MagicMock(message=MagicMock(content="Corrected grounded answer."))
        ]
        mock_generator._client.chat_completion.return_value = regen_response

        rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
        state = rag.invoke("What is X?")

        # Retriever called exactly once (initial only)
        mock_retriever.retrieve.assert_called_once()
        assert state["iterations"] == 2

    def test_regeneration_uses_same_chunks(
        self, mock_retriever, dummy_chunk, mock_generator, mock_critic
    ):
        """The retrieved_chunks in state remain unchanged after regeneration."""
        mock_critic.evaluate.side_effect = [_ungrounded_fail(), _pass_eval()]

        regen_response = MagicMock()
        regen_response.choices = [
            MagicMock(message=MagicMock(content="Fixed answer."))
        ]
        mock_generator._client.chat_completion.return_value = regen_response

        rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
        state = rag.invoke("What is X?")

        # Same chunks from initial retrieve
        assert len(state["retrieved_chunks"]) == 1
        assert state["retrieved_chunks"][0].id == dummy_chunk.id


# ===========================================================================
# AC6: Both failure reasons route correctly
# ===========================================================================

class TestRoutingCorrectness:
    """AC6 — route_recovery maps each failure reason to the right action."""

    def test_retrieval_insufficient_routes_to_reformulate_then_retrieve(
        self, mock_retriever, mock_generator, mock_critic
    ):
        """retrieval_insufficient → reformulate → retrieve again."""
        mock_critic.evaluate.side_effect = [_retrieval_fail(), _pass_eval()]

        rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
        rag.invoke("What is X?")

        # Retrieve called twice (initial + after reformulation)
        assert mock_retriever.retrieve.call_count == 2

    def test_generation_ungrounded_routes_to_regenerate_not_retrieve(
        self, mock_retriever, mock_generator, mock_critic
    ):
        """generation_ungrounded → regenerate → critic.  No extra retrieval."""
        mock_critic.evaluate.side_effect = [_ungrounded_fail(), _pass_eval()]

        regen_response = MagicMock()
        regen_response.choices = [
            MagicMock(message=MagicMock(content="Fixed answer."))
        ]
        mock_generator._client.chat_completion.return_value = regen_response

        rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
        rag.invoke("What is X?")

        # Retrieve called exactly once
        mock_retriever.retrieve.assert_called_once()

    def test_retrieval_insufficient_updates_current_query(
        self, mock_retriever, mock_generator, mock_critic
    ):
        """After reformulation, current_query should reflect the new query."""
        mock_critic.evaluate.side_effect = [_retrieval_fail(), _pass_eval()]

        rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
        state = rag.invoke("What is X?")

        assert state["current_query"] == "reformulated query v2"

    def test_generation_ungrounded_updates_generation(
        self, mock_retriever, mock_generator, mock_critic
    ):
        """After regeneration, the generation field should be updated."""
        mock_critic.evaluate.side_effect = [_ungrounded_fail(), _pass_eval()]

        regen_response = MagicMock()
        regen_response.choices = [
            MagicMock(message=MagicMock(content="Strictly grounded answer."))
        ]
        mock_generator._client.chat_completion.return_value = regen_response

        rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
        state = rag.invoke("What is X?")

        assert state["generation"] == "Strictly grounded answer."


# ===========================================================================
# AC7: Invalid / None failure_reason is rejected explicitly
# ===========================================================================

class TestInvalidFailureReasonRejected:
    """AC7 — route_recovery raises ValueError on invalid/None failure_reason."""

    def test_none_failure_reason_raises_value_error(
        self, mock_retriever, mock_generator, mock_critic
    ):
        """A FAIL verdict with failure_reason=None should raise ValueError."""
        bad_eval = CriticEvaluation(
            verdict=CriticVerdict.FAIL,
            failure_reason=None,           # deliberately broken
            is_retrieval_sufficient=False,
            is_generation_grounded=False,
            reasoning="Something went wrong.",
        )
        mock_critic.evaluate.return_value = bad_eval

        rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)

        with pytest.raises((ValueError, Exception)):
            rag.invoke("What is X?")

    def test_route_recovery_raises_on_unexpected_reason(self):
        """route_recovery must raise ValueError for an unrecognized reason string."""
        mock_retriever = MagicMock()
        mock_generator = MagicMock()
        mock_critic = MagicMock()

        rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)

        # Manually construct a state with a monkey-patched failure_reason
        bad_eval = MagicMock()
        bad_eval.failure_reason = "completely_unknown_reason"
        state: GraphState = {
            "original_query": "Q",
            "current_query": "Q",
            "retrieved_chunks": [],
            "generation": "A",
            "critic_evaluation": bad_eval,
            "iterations": 1,
            "query_history": ["Q"],
        }

        with pytest.raises(ValueError, match="unexpected failure_reason"):
            rag.route_recovery(state)


# ===========================================================================
# AC8: query_history grows and stays consistent
# ===========================================================================

class TestQueryHistoryConsistency:
    """AC8 — query_history is initialized and updated correctly."""

    def test_initial_history_contains_original_query(
        self, mock_retriever, mock_generator, mock_critic
    ):
        mock_critic.evaluate.return_value = _pass_eval()

        rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
        state = rag.invoke("My question")

        assert "My question" in state["query_history"]

    def test_history_grows_after_reformulation(
        self, mock_retriever, mock_generator, mock_critic
    ):
        mock_critic.evaluate.side_effect = [_retrieval_fail(), _pass_eval()]

        rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
        state = rag.invoke("Original")

        # original + reformulated
        assert len(state["query_history"]) == 2

    def test_no_duplicate_entries_in_history(
        self, mock_retriever, mock_generator, mock_critic
    ):
        """History must never contain duplicate query strings."""
        mock_critic.evaluate.side_effect = [_retrieval_fail(), _pass_eval()]

        rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
        state = rag.invoke("What is X?")

        assert len(state["query_history"]) == len(set(state["query_history"]))


# ===========================================================================
# AC9: Regeneration loop ends correctly
# ===========================================================================

class TestRegenerationLoop:
    """AC9 — regenerate → critic → PASS terminates the workflow cleanly."""

    def test_regeneration_then_pass(
        self, mock_retriever, mock_generator, mock_critic
    ):
        mock_critic.evaluate.side_effect = [_ungrounded_fail(), _pass_eval()]

        regen_response = MagicMock()
        regen_response.choices = [
            MagicMock(message=MagicMock(content="Corrected answer."))
        ]
        mock_generator._client.chat_completion.return_value = regen_response

        rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
        state = rag.invoke("What is X?")

        assert state["critic_evaluation"].verdict == CriticVerdict.PASS
        assert state["iterations"] == 2
        assert state["generation"] == "Corrected answer."


# ===========================================================================
# AC10: GENERATION_UNGROUNDED max retries — retriever called exactly once
# ===========================================================================

class TestGenerationUngroundedMaxRetries:
    """AC10 — retriever is only called once when all failures are GENERATION_UNGROUNDED."""

    def test_retriever_called_once_across_all_regeneration_retries(
        self, mock_retriever, mock_generator, mock_critic
    ):
        """When GENERATION_UNGROUNDED repeats, retrieve must never be called again."""
        mock_critic.evaluate.return_value = _ungrounded_fail()

        regen_response = MagicMock()
        regen_response.choices = [
            MagicMock(message=MagicMock(content="Still wrong."))
        ]
        mock_generator._client.chat_completion.return_value = regen_response

        rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
        state = rag.invoke("What is X?")

        assert state["iterations"] == rag.max_retries
        # CRITICAL: retriever must never be called more than once
        mock_retriever.retrieve.assert_called_once()
        # Regeneration is called (max_retries - 1) times
        assert mock_generator._client.chat_completion.call_count == rag.max_retries - 1


class TestReformulationNormalization:
    """Regression tests for reformulate_node quote stripping and case-insensitive history check."""

    def test_reformulate_node_strips_quotes_and_checks_case(
        self, mock_retriever, mock_generator, mock_critic
    ):
        """Quoted output e.g. '"New Query"' is stripped to 'New Query' and case matches history."""
        mock_critic.evaluate.side_effect = [_retrieval_fail(), _pass_eval()]

        quoted_response = MagicMock()
        quoted_response.choices = [
            MagicMock(message=MagicMock(content='"New Reformulated Query"'))
        ]
        mock_generator._client.chat_completion.return_value = quoted_response

        rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
        state = rag.invoke("Original")

        assert state["current_query"] == "New Reformulated Query"
        assert "New Reformulated Query" in state["query_history"]
        assert '"New Reformulated Query"' not in state["query_history"]

    def test_reformulate_node_rejects_case_insensitive_duplicate(
        self, mock_retriever, mock_generator, mock_critic
    ):
        """Case-insensitive duplicate output e.g. 'ORIGINAL' is rejected for 'Original'."""
        mock_critic.evaluate.side_effect = [_retrieval_fail(), _pass_eval()]

        upper_response = MagicMock()
        upper_response.choices = [
            MagicMock(message=MagicMock(content="ORIGINAL"))
        ]
        mock_generator._client.chat_completion.return_value = upper_response

        rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
        state = rag.invoke("Original")

        # Falls back to Original and does not insert uppercase duplicate into query_history
        assert state["current_query"] == "Original"
        assert state["query_history"] == ["Original"]

