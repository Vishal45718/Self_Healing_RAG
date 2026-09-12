import pytest
from unittest.mock import MagicMock

from src.graph.graph import SelfHealingRAG
from src.schema import RetrievalResult
from src.generation.generator import GenerationResult
from src.critic.schema import CriticEvaluation, CriticVerdict, CriticFailureReason


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    # Mocking retrieve to return some dummy chunks
    chunk = RetrievalResult(
        id="chunk1",
        content="Dummy context for testing.",
        distance=0.1,
        similarity=0.9
    )
    retriever.retrieve.return_value = [chunk]
    return retriever

@pytest.fixture
def mock_generator():
    generator = MagicMock()
    # Mocking generate method
    generator.generate.return_value = GenerationResult(
        answer="This is a generated answer based on context.",
        model="mock-model",
        usage={"total_tokens": 10}
    )
    
    # Mocking _client for reformulation
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="reformulated query"))]
    mock_client.chat_completion.return_value = mock_response
    generator._client = mock_client
    
    return generator

@pytest.fixture
def mock_critic():
    critic = MagicMock()
    return critic

def test_graph_pass_on_first_try(mock_retriever, mock_generator, mock_critic):
    """Test scenario: Critic passes the first generation."""
    mock_critic.evaluate.return_value = CriticEvaluation(
        verdict=CriticVerdict.PASS,
        is_retrieval_sufficient=True,
        is_generation_grounded=True,
        reasoning="All good."
    )
    
    rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
    state = rag.invoke("What is X?")
    
    # Should only take 1 iteration
    assert state["iterations"] == 1
    assert state["critic_evaluation"].verdict == CriticVerdict.PASS
    assert state["current_query"] == "What is X?"
    assert state["generation"] == "This is a generated answer based on context."
    assert len(state["retrieved_chunks"]) == 1
    
    # Verify calls
    mock_retriever.retrieve.assert_called_once_with("What is X?")
    mock_generator.generate.assert_called_once()
    mock_critic.evaluate.assert_called_once()
    mock_generator._client.chat_completion.assert_not_called() # No reformulation

def test_graph_fail_and_recover(mock_retriever, mock_generator, mock_critic):
    """Test scenario: Critic fails first attempt, passes second attempt."""
    
    # Setup critic to fail then pass
    fail_eval = CriticEvaluation(
        verdict=CriticVerdict.FAIL,
        failure_reason=CriticFailureReason.RETRIEVAL_INSUFFICIENT,
        is_retrieval_sufficient=False,
        is_generation_grounded=True,
        reasoning="Need more info."
    )
    pass_eval = CriticEvaluation(
        verdict=CriticVerdict.PASS,
        is_retrieval_sufficient=True,
        is_generation_grounded=True,
        reasoning="All good now."
    )
    mock_critic.evaluate.side_effect = [fail_eval, pass_eval]
    
    rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
    state = rag.invoke("What is X?")
    
    # Should take 2 iterations
    assert state["iterations"] == 2
    assert state["critic_evaluation"].verdict == CriticVerdict.PASS
    
    # After reformulation, current_query should be updated
    assert state["current_query"] == "reformulated query"
    
    # Verify calls
    assert mock_retriever.retrieve.call_count == 2
    mock_retriever.retrieve.assert_any_call("What is X?")
    mock_retriever.retrieve.assert_any_call("reformulated query")
    
    assert mock_generator.generate.call_count == 2
    assert mock_critic.evaluate.call_count == 2
    mock_generator._client.chat_completion.assert_called_once()

def test_graph_max_retries_reached(mock_retriever, mock_generator, mock_critic):
    """Test scenario: Critic fails continuously until max retries reached."""
    
    fail_eval = CriticEvaluation(
        verdict=CriticVerdict.FAIL,
        failure_reason=CriticFailureReason.GENERATION_UNGROUNDED,
        is_retrieval_sufficient=True,
        is_generation_grounded=False,
        reasoning="Ungrounded claim."
    )
    # Always fail
    mock_critic.evaluate.return_value = fail_eval
    
    rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
    
    # Run the graph
    state = rag.invoke("What is X?")
    
    # Should end at max_retries (default 3 from settings)
    assert state["iterations"] == rag.max_retries
    assert state["critic_evaluation"].verdict == CriticVerdict.FAIL
    
    # Verify calls
    assert mock_retriever.retrieve.call_count == rag.max_retries
    assert mock_generator.generate.call_count == rag.max_retries
    assert mock_critic.evaluate.call_count == rag.max_retries
    assert mock_generator._client.chat_completion.call_count == rag.max_retries - 1

def test_reformulation_fallback(mock_retriever, mock_generator, mock_critic):
    """Test scenario: Reformulation API call fails, falls back to original query."""
    fail_eval = CriticEvaluation(
        verdict=CriticVerdict.FAIL,
        failure_reason=CriticFailureReason.RETRIEVAL_INSUFFICIENT,
        is_retrieval_sufficient=False,
        is_generation_grounded=True,
        reasoning="Failed."
    )
    pass_eval = CriticEvaluation(
        verdict=CriticVerdict.PASS,
        is_retrieval_sufficient=True,
        is_generation_grounded=True,
        reasoning="Passed."
    )
    mock_critic.evaluate.side_effect = [fail_eval, pass_eval]
    
    # Force an exception on reformulation
    mock_generator._client.chat_completion.side_effect = Exception("API Error")
    
    rag = SelfHealingRAG(mock_retriever, mock_generator, mock_critic)
    state = rag.invoke("Original query")
    
    # Should recover gracefully but current_query remains unchanged
    assert state["iterations"] == 2
    assert state["current_query"] == "Original query"
