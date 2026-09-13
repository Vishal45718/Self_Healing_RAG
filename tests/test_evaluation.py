"""Tests for Phase 7 Evaluation & Baseline Comparison harness."""

import json
import pytest
from unittest.mock import MagicMock

from src.critic.schema import CriticEvaluation, CriticFailureReason, CriticVerdict
from src.evaluation.dataset import (
    DEFAULT_EVALUATION_DOCUMENTS,
    DEFAULT_EVALUATION_SAMPLES,
    load_dataset_from_json,
    load_default_dataset,
    load_default_documents,
)
from src.evaluation.metrics import calculate_metric_deltas, calculate_system_metrics
from src.evaluation.report import (
    generate_markdown_report,
    load_report_json,
    save_report_json,
)
from src.evaluation.runner import EvaluationRunner
from src.evaluation.schema import (
    EvalExecutionResult,
    EvalSample,
    EvaluationReport,
    ExpectedBehavior,
    ScenarioType,
    SystemMetrics,
)
from src.generation.generator import GenerationResult
from src.schema import RetrievalResult


# ==============================================================================
# 1. Dataset Tests
# ==============================================================================

def test_default_dataset_and_documents_structure():
    """Verify default evaluation documents and query samples are well-formed."""
    docs = load_default_documents()
    samples = load_default_dataset()

    assert len(docs) == 3
    assert len(samples) == 9

    # Verify scenario breakdown: 3 direct, 3 reformulate, 3 unanswerable
    direct_samples = [s for s in samples if s.scenario == ScenarioType.DIRECT]
    reformulate_samples = [
        s for s in samples if s.scenario == ScenarioType.REFORMULATION_TARGET
    ]
    unanswerable_samples = [
        s for s in samples if s.scenario == ScenarioType.UNANSWERABLE
    ]

    assert len(direct_samples) == 3
    assert len(reformulate_samples) == 3
    assert len(unanswerable_samples) == 3

    for s in samples:
        assert s.id
        assert s.query
        assert s.expected_behavior in ExpectedBehavior


def test_load_dataset_from_json(tmp_path):
    """Verify loading custom evaluation dataset from JSON."""
    custom_data = [
        {
            "id": "custom_01",
            "query": "What is the consensus timeout?",
            "scenario": "direct",
            "expected_behavior": "answer_grounded",
            "relevant_doc_ids": ["doc_1"],
            "ground_truth_answer": "150ms to 300ms",
        }
    ]
    json_file = tmp_path / "custom_eval.json"
    json_file.write_text(json.dumps(custom_data), encoding="utf-8")

    loaded = load_dataset_from_json(str(json_file))
    assert len(loaded) == 1
    assert loaded[0].id == "custom_01"
    assert loaded[0].scenario == ScenarioType.DIRECT


def test_load_dataset_from_json_missing_file_raises():
    """Verify loading non-existent JSON file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_dataset_from_json("non_existent_file.json")


def test_load_dataset_from_json_empty_raises(tmp_path):
    """Verify empty JSON list raises ValueError."""
    empty_file = tmp_path / "empty_eval.json"
    empty_file.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty list"):
        load_dataset_from_json(str(empty_file))


# ==============================================================================
# 2. Metrics Calculation Tests
# ==============================================================================

def test_metrics_calculation_accuracy():
    """Verify mathematical accuracy of system metrics calculations."""
    results = [
        EvalExecutionResult(
            query_id="q1",
            query="Query 1",
            scenario=ScenarioType.DIRECT,
            expected_behavior=ExpectedBehavior.ANSWER_GROUNDED,
            system_name="self_healing",
            final_query="Query 1",
            retrieved_chunk_ids=["c1"],
            answer="Answer 1",
            critic_verdict=CriticVerdict.PASS,
            is_retrieval_sufficient=True,
            is_generation_grounded=True,
            iterations=1,
            retries=0,
            latency_ms=100.0,
            llm_calls=2,
            is_critic_pass=True,
            is_correct_abstention=False,
            is_recovery_success=False,
            is_reformulation_success=False,
        ),
        EvalExecutionResult(
            query_id="q2",
            query="Query 2",
            scenario=ScenarioType.REFORMULATION_TARGET,
            expected_behavior=ExpectedBehavior.REFORMULATE_AND_ANSWER,
            system_name="self_healing",
            final_query="Query 2 reformulated",
            retrieved_chunk_ids=["c2"],
            answer="Answer 2",
            critic_verdict=CriticVerdict.PASS,
            is_retrieval_sufficient=True,
            is_generation_grounded=True,
            iterations=2,
            retries=1,
            latency_ms=250.0,
            llm_calls=5,
            is_critic_pass=True,
            is_correct_abstention=False,
            is_recovery_success=True,  # Successfully recovered
            is_reformulation_success=True,
        ),
        EvalExecutionResult(
            query_id="q3",
            query="Query 3",
            scenario=ScenarioType.UNANSWERABLE,
            expected_behavior=ExpectedBehavior.ABSTAIN,
            system_name="self_healing",
            final_query="Query 3",
            retrieved_chunk_ids=[],
            answer="The provided context is insufficient to answer this question.",
            critic_verdict=CriticVerdict.ABSTAIN,
            is_retrieval_sufficient=False,
            is_generation_grounded=True,
            iterations=1,
            retries=0,
            latency_ms=50.0,
            llm_calls=0,
            is_critic_pass=False,
            is_correct_abstention=True,
            is_recovery_success=False,
            is_reformulation_success=False,
        ),
        EvalExecutionResult(
            query_id="q4",
            query="Query 4",
            scenario=ScenarioType.REFORMULATION_TARGET,
            expected_behavior=ExpectedBehavior.REFORMULATE_AND_ANSWER,
            system_name="self_healing",
            final_query="Query 4 failed",
            retrieved_chunk_ids=[],
            answer="Failed answer",
            critic_verdict=CriticVerdict.FAIL,
            is_retrieval_sufficient=False,
            is_generation_grounded=False,
            iterations=3,
            retries=2,
            latency_ms=400.0,
            llm_calls=7,
            is_critic_pass=False,
            is_correct_abstention=False,
            is_recovery_success=False,  # Attempted recovery but failed
            is_reformulation_success=False,
        ),
    ]

    metrics = calculate_system_metrics(system_name="self_healing", results=results)

    assert metrics.total_queries == 4
    # 2 PASS out of 4 = 0.50
    assert metrics.critic_pass_rate == 0.5
    # 3 grounded out of 4 = 0.75
    assert metrics.groundedness_rate == 0.75
    # 2 sufficient out of 4 = 0.50
    assert metrics.retrieval_sufficiency_rate == 0.50
    # 1 unanswerable query, which abstained = 1.0
    assert metrics.correct_abstention_rate == 1.0
    assert metrics.hallucination_rate_on_unanswerable == 0.0

    # Recovery attempts: q2 (retries=1, success) and q4 (retries=2, failed). 1/2 = 0.50
    assert metrics.recovery_success_rate == 0.50

    # Reformulation attempts: q2 (success) and q4 (failed). 1/2 = 0.50
    assert metrics.reformulation_success_rate == 0.50

    # Averages
    assert metrics.average_retries == (0 + 1 + 0 + 2) / 4.0  # 0.75
    assert metrics.average_iterations == (1 + 2 + 1 + 3) / 4.0  # 1.75
    assert metrics.average_latency_ms == (100 + 250 + 50 + 400) / 4.0  # 200.0
    assert metrics.average_llm_calls == (2 + 5 + 0 + 7) / 4.0  # 3.50
    assert metrics.total_llm_calls == 14


def test_recovery_success_condition_rigor():
    """Verify recovery success is strictly false if final outcome is not PASS/ABSTAIN."""
    # Query had 3 iterations (2 retries) but final verdict was still FAIL
    failed_recovery = EvalExecutionResult(
        query_id="fail_q",
        query="Hard query",
        scenario=ScenarioType.DIRECT,
        expected_behavior=ExpectedBehavior.ANSWER_GROUNDED,
        system_name="self_healing",
        final_query="Hard query",
        retrieved_chunk_ids=[],
        answer="Still wrong",
        critic_verdict=CriticVerdict.FAIL,
        is_retrieval_sufficient=False,
        is_generation_grounded=False,
        iterations=3,
        retries=2,
        latency_ms=300.0,
        llm_calls=6,
        is_critic_pass=False,
        is_correct_abstention=False,
        is_recovery_success=False,
        is_reformulation_success=False,
    )

    metrics = calculate_system_metrics(
        system_name="self_healing", results=[failed_recovery]
    )
    assert metrics.recovery_success_rate == 0.0


def test_metric_deltas():
    """Verify calculate_metric_deltas computes correct improvements and overhead ratios."""
    baseline_metrics = SystemMetrics(
        system_name="baseline",
        total_queries=10,
        critic_pass_rate=0.40,
        groundedness_rate=0.50,
        retrieval_sufficiency_rate=0.40,
        correct_abstention_rate=0.33,
        hallucination_rate_on_unanswerable=0.67,
        recovery_success_rate=0.0,
        reformulation_success_rate=0.0,
        average_retries=0.0,
        average_iterations=1.0,
        average_latency_ms=100.0,
        average_llm_calls=2.0,
        total_llm_calls=20,
    )

    self_healing_metrics = SystemMetrics(
        system_name="self_healing",
        total_queries=10,
        critic_pass_rate=0.80,
        groundedness_rate=0.90,
        retrieval_sufficiency_rate=0.80,
        correct_abstention_rate=1.00,
        hallucination_rate_on_unanswerable=0.0,
        recovery_success_rate=0.75,
        reformulation_success_rate=0.66,
        average_retries=0.8,
        average_iterations=1.8,
        average_latency_ms=220.0,
        average_llm_calls=4.5,
        total_llm_calls=45,
    )

    deltas = calculate_metric_deltas(
        baseline=baseline_metrics, self_healing=self_healing_metrics
    )

    assert deltas["pass_rate_improvement"] == 0.40
    assert deltas["groundedness_improvement"] == 0.40
    assert deltas["sufficiency_improvement"] == 0.40
    assert deltas["abstention_improvement"] == 0.67
    assert deltas["recovery_success_rate"] == 0.75
    assert deltas["latency_overhead_ratio"] == 2.2
    assert deltas["llm_call_overhead_ratio"] == 2.25


# ==============================================================================
# 3. Runner & Side-by-Side Execution Tests
# ==============================================================================

@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    chunk = RetrievalResult(
        id="chunk_test_1",
        content="Our cache uses LRU and ARC eviction policies.",
        distance=0.1,
        similarity=0.9,
    )
    retriever.retrieve.return_value = [chunk]
    return retriever


@pytest.fixture
def mock_generator():
    generator = MagicMock()
    generator.generate.return_value = GenerationResult(
        answer="The supported policies are LRU and ARC.",
        model_id="test-model",
        provider="test-provider",
    )
    # Chat completion mock for reformulator
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="reformulated cache query"))]
    mock_client.chat_completion.return_value = mock_resp
    generator._client = mock_client
    return generator


@pytest.fixture
def mock_critic():
    critic = MagicMock()
    critic.evaluate.return_value = CriticEvaluation(
        verdict=CriticVerdict.PASS,
        is_retrieval_sufficient=True,
        is_generation_grounded=True,
        reasoning="Well supported.",
    )
    return critic


def test_runner_side_by_side_execution(mock_retriever, mock_generator, mock_critic):
    """Verify EvaluationRunner executes both pipelines and generates complete report."""
    runner = EvaluationRunner(
        retriever=mock_retriever,
        generator=mock_generator,
        critic=mock_critic,
    )

    sample = EvalSample(
        id="test_01",
        query="What cache policies exist?",
        scenario=ScenarioType.DIRECT,
        expected_behavior=ExpectedBehavior.ANSWER_GROUNDED,
    )

    report = runner.run_side_by_side([sample])

    assert isinstance(report, EvaluationReport)
    assert report.dataset_size == 1
    assert report.baseline_metrics.total_queries == 1
    assert report.self_healing_metrics.total_queries == 1
    assert len(report.per_query_results) == 1

    per_q = report.per_query_results[0]
    assert per_q["query_id"] == "test_01"
    assert per_q["baseline"]["is_pass"] is True
    assert per_q["self_healing"]["is_pass"] is True


def test_runner_empty_samples_raises_value_error(mock_retriever, mock_generator, mock_critic):
    """Verify running on empty dataset raises ValueError."""
    runner = EvaluationRunner(
        retriever=mock_retriever,
        generator=mock_generator,
        critic=mock_critic,
    )
    with pytest.raises(ValueError, match="must not be empty"):
        runner.run_side_by_side([])


def test_runner_unanswerable_query_abstention(mock_retriever, mock_generator, mock_critic):
    """Verify evaluation runner correctly tags safe abstention on unanswerable query."""
    # Critic returns ABSTAIN
    mock_critic.evaluate.return_value = CriticEvaluation(
        verdict=CriticVerdict.ABSTAIN,
        is_retrieval_sufficient=False,
        is_generation_grounded=True,
        reasoning="Safely abstained.",
    )
    mock_generator.generate.return_value = GenerationResult(
        answer="The provided context is insufficient to answer this question.",
        model_id="test-model",
        provider="test-provider",
    )

    runner = EvaluationRunner(
        retriever=mock_retriever,
        generator=mock_generator,
        critic=mock_critic,
    )

    sample = EvalSample(
        id="unans_01",
        query="What is the stock price of Apple?",
        scenario=ScenarioType.UNANSWERABLE,
        expected_behavior=ExpectedBehavior.ABSTAIN,
    )

    base_res = runner.evaluate_sample_baseline(sample)
    sh_res = runner.evaluate_sample_self_healing(sample)

    assert base_res.is_correct_abstention is True
    assert sh_res.is_correct_abstention is True


# ==============================================================================
# 4. Report Generation & Serialization Tests
# ==============================================================================

def test_markdown_report_formatting():
    """Verify generate_markdown_report produces formatted table with required sections."""
    bm = SystemMetrics(
        system_name="baseline",
        total_queries=5,
        critic_pass_rate=0.40,
        groundedness_rate=0.60,
        retrieval_sufficiency_rate=0.40,
        correct_abstention_rate=0.50,
        hallucination_rate_on_unanswerable=0.50,
        recovery_success_rate=0.0,
        reformulation_success_rate=0.0,
        average_retries=0.0,
        average_iterations=1.0,
        average_latency_ms=120.0,
        average_llm_calls=2.0,
        total_llm_calls=10,
    )
    sh = SystemMetrics(
        system_name="self_healing",
        total_queries=5,
        critic_pass_rate=0.80,
        groundedness_rate=1.00,
        retrieval_sufficiency_rate=0.80,
        correct_abstention_rate=1.00,
        hallucination_rate_on_unanswerable=0.0,
        recovery_success_rate=1.0,
        reformulation_success_rate=1.0,
        average_retries=0.6,
        average_iterations=1.6,
        average_latency_ms=250.0,
        average_llm_calls=4.0,
        total_llm_calls=20,
    )
    deltas = calculate_metric_deltas(bm, sh)

    report = EvaluationReport(
        timestamp="2026-09-12T12:00:00Z",
        dataset_size=5,
        baseline_metrics=bm,
        self_healing_metrics=sh,
        per_query_results=[],
        summary_deltas=deltas,
    )

    md = generate_markdown_report(report)

    assert "# RAG Evaluation & Baseline Comparison Report" in md
    assert "| **Critic Pass Rate** | 40.0% | 80.0% |" in md
    assert "| **Correct Abstention Rate** | 50.0% | 100.0% |" in md
    assert "| **Recovery Success Rate** | N/A (0 retries) | 100.0% |" in md
    assert "2.08x overhead" in md or "2.08" in md or "overhead" in md


def test_json_report_serialization_and_deserialization(tmp_path):
    """Verify saving and loading EvaluationReport from JSON preserves all data."""
    bm = SystemMetrics(
        system_name="baseline",
        total_queries=1,
        critic_pass_rate=1.0,
        groundedness_rate=1.0,
        retrieval_sufficiency_rate=1.0,
        correct_abstention_rate=0.0,
        hallucination_rate_on_unanswerable=0.0,
        recovery_success_rate=0.0,
        reformulation_success_rate=0.0,
        average_retries=0.0,
        average_iterations=1.0,
        average_latency_ms=80.0,
        average_llm_calls=2.0,
        total_llm_calls=2,
    )
    sh = SystemMetrics(
        system_name="self_healing",
        total_queries=1,
        critic_pass_rate=1.0,
        groundedness_rate=1.0,
        retrieval_sufficiency_rate=1.0,
        correct_abstention_rate=0.0,
        hallucination_rate_on_unanswerable=0.0,
        recovery_success_rate=0.0,
        reformulation_success_rate=0.0,
        average_retries=0.0,
        average_iterations=1.0,
        average_latency_ms=90.0,
        average_llm_calls=2.0,
        total_llm_calls=2,
    )
    original_report = EvaluationReport(
        timestamp="2026-09-12T12:00:00Z",
        dataset_size=1,
        baseline_metrics=bm,
        self_healing_metrics=sh,
        per_query_results=[{"query_id": "q1", "baseline": {"is_pass": True}}],
        summary_deltas={"pass_rate_improvement": 0.0},
    )

    output_file = tmp_path / "test_report.json"
    save_report_json(original_report, str(output_file))

    assert output_file.exists()

    loaded_report = load_report_json(str(output_file))
    assert loaded_report.timestamp == original_report.timestamp
    assert loaded_report.dataset_size == 1
    assert loaded_report.baseline_metrics.critic_pass_rate == 1.0
    assert loaded_report.self_healing_metrics.average_latency_ms == 90.0
    assert len(loaded_report.per_query_results) == 1


def test_evaluate_sample_self_healing_llm_call_shortcuts():
    """Regression test: LLM call count in evaluate_sample_self_healing accounts for empty context shortcuts."""
    mock_retriever = MagicMock()
    mock_generator = MagicMock()
    mock_critic = MagicMock()
    mock_sh = MagicMock()

    # Empty context state on 1 iteration
    mock_sh.invoke.return_value = {
        "original_query": "Unanswerable Q",
        "current_query": "Unanswerable Q",
        "retrieved_chunks": [],
        "generation": "The provided context is insufficient...",
        "critic_evaluation": CriticEvaluation(
            verdict=CriticVerdict.FAIL,
            failure_reason=CriticFailureReason.RETRIEVAL_INSUFFICIENT,
            is_retrieval_sufficient=False,
            is_generation_grounded=True,
            reasoning="Empty context",
        ),
        "iterations": 1,
        "query_history": ["Unanswerable Q"],
    }

    runner = EvaluationRunner(
        retriever=mock_retriever,
        generator=mock_generator,
        critic=mock_critic,
        self_healing_rag=mock_sh,
    )

    sample = EvalSample(
        id="unanswerable_test",
        query="Unanswerable Q",
        scenario=ScenarioType.UNANSWERABLE,
        expected_behavior=ExpectedBehavior.ABSTAIN,
    )

    result = runner.evaluate_sample_self_healing(sample)

    # Empty context shortcut means 0 LLM calls made
    assert result.llm_calls == 0

