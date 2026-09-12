"""Evaluation metric calculations and delta summaries for Phase 7."""

from typing import Any, Dict, List
from src.evaluation.schema import (
    EvalExecutionResult,
    ScenarioType,
    SystemMetrics,
)


def calculate_system_metrics(
    system_name: str,
    results: List[EvalExecutionResult],
) -> SystemMetrics:
    """Calculate aggregated evaluation metrics for a single RAG pipeline run.

    Args:
        system_name: Name identifier for the system (e.g., 'baseline' or 'self_healing').
        results: List of execution results for the dataset queries.

    Returns:
        SystemMetrics containing computed rates, averages, and counts.

    Raises:
        ValueError: If results list is empty.
    """
    if not results:
        raise ValueError("Cannot calculate metrics for an empty list of results.")

    total_queries = len(results)

    # 1. Critic Pass & Groundedness Rates
    critic_pass_count = sum(1 for r in results if r.is_critic_pass)
    grounded_count = sum(1 for r in results if r.is_generation_grounded)
    sufficient_count = sum(1 for r in results if r.is_retrieval_sufficient)

    critic_pass_rate = round(critic_pass_count / total_queries, 4)
    groundedness_rate = round(grounded_count / total_queries, 4)
    retrieval_sufficiency_rate = round(sufficient_count / total_queries, 4)

    # 2. Abstention & Hallucination Rates on Unanswerable Queries
    unanswerable_results = [
        r for r in results if r.scenario == ScenarioType.UNANSWERABLE
    ]
    if unanswerable_results:
        correct_abstentions = sum(
            1 for r in unanswerable_results if r.is_correct_abstention
        )
        correct_abstention_rate = round(
            correct_abstentions / len(unanswerable_results), 4
        )
        # Hallucination = not abstained AND not grounded
        hallucinations = sum(
            1
            for r in unanswerable_results
            if not r.is_correct_abstention and not r.is_generation_grounded
        )
        hallucination_rate_on_unanswerable = round(
            hallucinations / len(unanswerable_results), 4
        )
    else:
        correct_abstention_rate = 0.0
        hallucination_rate_on_unanswerable = 0.0

    # 3. Recovery Success Rate (Queries that entered recovery and succeeded)
    recovery_attempts = [r for r in results if r.retries > 0]
    if recovery_attempts:
        successful_recoveries = sum(
            1 for r in recovery_attempts if r.is_recovery_success
        )
        recovery_success_rate = round(
            successful_recoveries / len(recovery_attempts), 4
        )
    else:
        recovery_success_rate = 0.0

    # 4. Reformulation Success Rate
    reformulation_attempts = [
        r for r in results if r.scenario == ScenarioType.REFORMULATION_TARGET or (r.final_query != r.query)
    ]
    if reformulation_attempts:
        successful_reformulations = sum(
            1 for r in reformulation_attempts if r.is_reformulation_success
        )
        reformulation_success_rate = round(
            successful_reformulations / len(reformulation_attempts), 4
        )
    else:
        reformulation_success_rate = 0.0

    # 5. Iterations, Retries, Latency & LLM Call Counts
    total_retries = sum(r.retries for r in results)
    total_iterations = sum(r.iterations for r in results)
    total_latency_ms = sum(r.latency_ms for r in results)
    total_llm_calls = sum(r.llm_calls for r in results)

    average_retries = round(total_retries / total_queries, 2)
    average_iterations = round(total_iterations / total_queries, 2)
    average_latency_ms = round(total_latency_ms / total_queries, 2)
    average_llm_calls = round(total_llm_calls / total_queries, 2)

    return SystemMetrics(
        system_name=system_name,
        total_queries=total_queries,
        critic_pass_rate=critic_pass_rate,
        groundedness_rate=groundedness_rate,
        retrieval_sufficiency_rate=retrieval_sufficiency_rate,
        correct_abstention_rate=correct_abstention_rate,
        hallucination_rate_on_unanswerable=hallucination_rate_on_unanswerable,
        recovery_success_rate=recovery_success_rate,
        reformulation_success_rate=reformulation_success_rate,
        average_retries=average_retries,
        average_iterations=average_iterations,
        average_latency_ms=average_latency_ms,
        average_llm_calls=average_llm_calls,
        total_llm_calls=total_llm_calls,
    )


def calculate_metric_deltas(
    baseline: SystemMetrics,
    self_healing: SystemMetrics,
) -> Dict[str, Any]:
    """Calculate comparative metric deltas between Baseline and Self-Healing.

    Positive deltas in quality metrics (pass rate, groundedness, abstention) indicate improvement.
    Ratio deltas in cost metrics (latency, LLM calls) indicate overhead factors.
    """
    pass_rate_diff = round(
        self_healing.critic_pass_rate - baseline.critic_pass_rate, 4
    )
    groundedness_diff = round(
        self_healing.groundedness_rate - baseline.groundedness_rate, 4
    )
    sufficiency_diff = round(
        self_healing.retrieval_sufficiency_rate - baseline.retrieval_sufficiency_rate,
        4,
    )
    abstention_diff = round(
        self_healing.correct_abstention_rate - baseline.correct_abstention_rate, 4
    )

    latency_ratio = (
        round(self_healing.average_latency_ms / baseline.average_latency_ms, 2)
        if baseline.average_latency_ms > 0
        else 1.0
    )
    llm_call_ratio = (
        round(self_healing.average_llm_calls / baseline.average_llm_calls, 2)
        if baseline.average_llm_calls > 0
        else 1.0
    )

    return {
        "pass_rate_improvement": pass_rate_diff,
        "groundedness_improvement": groundedness_diff,
        "sufficiency_improvement": sufficiency_diff,
        "abstention_improvement": abstention_diff,
        "recovery_success_rate": self_healing.recovery_success_rate,
        "reformulation_success_rate": self_healing.reformulation_success_rate,
        "latency_overhead_ratio": latency_ratio,
        "llm_call_overhead_ratio": llm_call_ratio,
        "average_retries_needed": self_healing.average_retries,
    }
