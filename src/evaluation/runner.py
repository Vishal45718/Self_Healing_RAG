"""Evaluation runner executing Baseline vs Self-Healing pipelines side-by-side."""

from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional

from src.baseline.baseline_rag import BaselineRAG, BaselineResult
from src.critic.critic import Critic
from src.critic.schema import CriticEvaluation, CriticVerdict
from src.evaluation.metrics import calculate_metric_deltas, calculate_system_metrics
from src.evaluation.schema import (
    EvalExecutionResult,
    EvalSample,
    EvaluationReport,
    ExpectedBehavior,
    ScenarioType,
    SystemMetrics,
)
from src.generation.generator import Generator
from src.graph.graph import SelfHealingRAG
from src.retrieval.retriever import Retriever
from src.schema import GraphState

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Coordinates side-by-side execution and comparative measurement of Baseline vs Self-Healing RAG."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        critic: Critic,
        baseline_rag: Optional[BaselineRAG] = None,
        self_healing_rag: Optional[SelfHealingRAG] = None,
    ) -> None:
        """Initialise EvaluationRunner with shared components.

        Args:
            retriever: Shared Retriever instance (guarantees identical vector store/embeddings).
            generator: Shared Generator instance.
            critic: Shared Critic instance.
            baseline_rag: Optional custom BaselineRAG instance.
            self_healing_rag: Optional custom SelfHealingRAG instance.
        """
        self.retriever = retriever
        self.generator = generator
        self.critic = critic

        self.baseline_rag = (
            baseline_rag
            if baseline_rag is not None
            else BaselineRAG(
                retriever=self.retriever,
                generator=self.generator,
                critic=self.critic,
            )
        )
        self.self_healing_rag = (
            self_healing_rag
            if self_healing_rag is not None
            else SelfHealingRAG(
                retriever=self.retriever,
                generator=self.generator,
                critic=self.critic,
            )
        )

    def evaluate_sample_baseline(self, sample: EvalSample) -> EvalExecutionResult:
        """Run a single evaluation query through Baseline RAG.

        Args:
            sample: The test query sample.

        Returns:
            EvalExecutionResult with execution metrics.
        """
        t0 = time.perf_counter()
        baseline_res: BaselineResult = self.baseline_rag.invoke(query=sample.query)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        critic_eval = baseline_res.critic_evaluation
        if critic_eval is None:
            # If critic was not invoked inside baseline, run critic evaluation now
            critic_eval = self.critic.evaluate(
                query=sample.query,
                context=baseline_res.retrieved_chunks,
                answer=baseline_res.answer,
            )

        verdict = critic_eval.verdict
        is_pass = verdict == CriticVerdict.PASS

        # Check correct abstention for unanswerable queries
        is_abstention = False
        if sample.scenario == ScenarioType.UNANSWERABLE:
            ans_lower = baseline_res.answer.lower()
            is_abstention = (
                verdict == CriticVerdict.ABSTAIN
                or "insufficient" in ans_lower
                or "cannot answer" in ans_lower
                or "does not contain" in ans_lower
            )

        # Baseline: 1 generator LLM call + 1 critic LLM call (unless shortcut taken)
        llm_calls = 2
        if not baseline_res.retrieved_chunks:
            # Generator empty context shortcut (0) + Critic empty context shortcut (0)
            llm_calls = 0
        elif is_abstention and verdict == CriticVerdict.ABSTAIN:
            # Critic abstention shortcut used
            llm_calls = 1

        retrieved_ids = [c.id for c in baseline_res.retrieved_chunks]

        return EvalExecutionResult(
            query_id=sample.id,
            query=sample.query,
            scenario=sample.scenario,
            expected_behavior=sample.expected_behavior,
            system_name="baseline",
            final_query=sample.query,
            retrieved_chunk_ids=retrieved_ids,
            answer=baseline_res.answer,
            critic_verdict=verdict,
            critic_failure_reason=critic_eval.failure_reason,
            is_retrieval_sufficient=critic_eval.is_retrieval_sufficient,
            is_generation_grounded=critic_eval.is_generation_grounded,
            unsupported_claims=critic_eval.unsupported_claims,
            iterations=1,
            retries=0,
            latency_ms=latency_ms,
            llm_calls=llm_calls,
            is_critic_pass=is_pass,
            is_correct_abstention=is_abstention,
            is_recovery_success=False,  # Baseline never performs recovery
            is_reformulation_success=False,  # Baseline never reformulates
        )

    def evaluate_sample_self_healing(
        self, sample: EvalSample
    ) -> EvalExecutionResult:
        """Run a single evaluation query through SelfHealingRAG.

        Args:
            sample: The test query sample.

        Returns:
            EvalExecutionResult with recovery, retry, and latency metrics.
        """
        t0 = time.perf_counter()
        state: GraphState = self.self_healing_rag.invoke(query=sample.query)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        critic_eval: Optional[CriticEvaluation] = state.get("critic_evaluation")
        if critic_eval is None:
            raise ValueError(
                f"SelfHealingRAG finished with critic_evaluation=None for query {sample.id}"
            )

        verdict = critic_eval.verdict
        is_pass = verdict == CriticVerdict.PASS
        iterations = state.get("iterations", 1)
        retries = max(0, iterations - 1)
        query_history = state.get("query_history", [sample.query])
        final_query = state.get("current_query", sample.query)
        final_answer = state.get("generation", "") or ""

        # Check correct abstention
        is_abstention = False
        if sample.scenario == ScenarioType.UNANSWERABLE:
            ans_lower = final_answer.lower()
            is_abstention = (
                verdict == CriticVerdict.ABSTAIN
                or "insufficient" in ans_lower
                or "cannot answer" in ans_lower
                or "does not contain" in ans_lower
            )

        # Recovery success:
        # A query is a recovery success IF AND ONLY IF:
        # 1. Recovery was attempted (retries > 0, meaning attempt 1 failed)
        # 2. Final outcome satisfies expected success condition (PASS, or ABSTAIN for unanswerable)
        is_recovery_success = False
        if retries > 0:
            if sample.scenario == ScenarioType.UNANSWERABLE:
                is_recovery_success = is_abstention
            else:
                is_recovery_success = is_pass

        # Reformulation success:
        # Query was reformulated (query_history > 1) AND subsequent retrieval was sufficient
        is_reformulation_success = False
        if len(query_history) > 1:
            is_reformulation_success = critic_eval.is_retrieval_sufficient and is_pass

        # Calculate LLM call count across all iterations
        # Iteration 1: generate (1) + critic (1)
        # Iteration k > 1: recovery action (1 reformulate or 1 regenerate) + (1 generate if reformulate) + critic (1)
        reformulations_count = max(0, len(query_history) - 1)
        regenerations_count = max(0, retries - reformulations_count)
        llm_calls = (
            iterations  # Critic calls
            + 1  # Initial generator call
            + (reformulations_count * 2)  # Reformulator + Re-generator
            + regenerations_count  # Regenerator
        )

        retrieved_ids = [c.id for c in state.get("retrieved_chunks", [])]

        return EvalExecutionResult(
            query_id=sample.id,
            query=sample.query,
            scenario=sample.scenario,
            expected_behavior=sample.expected_behavior,
            system_name="self_healing",
            final_query=final_query,
            retrieved_chunk_ids=retrieved_ids,
            answer=final_answer,
            critic_verdict=verdict,
            critic_failure_reason=critic_eval.failure_reason,
            is_retrieval_sufficient=critic_eval.is_retrieval_sufficient,
            is_generation_grounded=critic_eval.is_generation_grounded,
            unsupported_claims=critic_eval.unsupported_claims,
            iterations=iterations,
            retries=retries,
            latency_ms=latency_ms,
            llm_calls=llm_calls,
            is_critic_pass=is_pass,
            is_correct_abstention=is_abstention,
            is_recovery_success=is_recovery_success,
            is_reformulation_success=is_reformulation_success,
        )

    def run_side_by_side(
        self, samples: List[EvalSample]
    ) -> EvaluationReport:
        """Execute all evaluation samples across Baseline and Self-Healing RAG.

        Args:
            samples: List of evaluation test samples.

        Returns:
            EvaluationReport containing per-query comparisons, system metrics, and deltas.

        Raises:
            ValueError: If samples list is empty.
        """
        if not samples:
            raise ValueError("Evaluation samples list must not be empty.")

        logger.info(
            "Starting side-by-side evaluation of %d samples across Baseline vs Self-Healing RAG.",
            len(samples),
        )

        baseline_results: List[EvalExecutionResult] = []
        self_healing_results: List[EvalExecutionResult] = []
        per_query_comparison: List[Dict[str, Any]] = []

        for sample in samples:
            logger.debug("Evaluating sample %s: %s", sample.id, sample.query[:40])

            # Run Baseline RAG
            base_res = self.evaluate_sample_baseline(sample)
            baseline_results.append(base_res)

            # Run Self-Healing RAG
            sh_res = self.evaluate_sample_self_healing(sample)
            self_healing_results.append(sh_res)

            # Build side-by-side comparison dict
            per_query_comparison.append(
                {
                    "query_id": sample.id,
                    "query": sample.query,
                    "scenario": sample.scenario.value,
                    "expected_behavior": sample.expected_behavior.value,
                    "baseline": {
                        "verdict": base_res.critic_verdict.value if base_res.critic_verdict else None,
                        "is_pass": base_res.is_critic_pass,
                        "is_sufficient": base_res.is_retrieval_sufficient,
                        "is_grounded": base_res.is_generation_grounded,
                        "is_abstention": base_res.is_correct_abstention,
                        "latency_ms": base_res.latency_ms,
                        "llm_calls": base_res.llm_calls,
                        "answer_snippet": base_res.answer[:80],
                    },
                    "self_healing": {
                        "verdict": sh_res.critic_verdict.value if sh_res.critic_verdict else None,
                        "is_pass": sh_res.is_critic_pass,
                        "is_sufficient": sh_res.is_retrieval_sufficient,
                        "is_grounded": sh_res.is_generation_grounded,
                        "is_abstention": sh_res.is_correct_abstention,
                        "is_recovery_success": sh_res.is_recovery_success,
                        "is_reformulation_success": sh_res.is_reformulation_success,
                        "iterations": sh_res.iterations,
                        "retries": sh_res.retries,
                        "latency_ms": sh_res.latency_ms,
                        "llm_calls": sh_res.llm_calls,
                        "final_query": sh_res.final_query,
                        "answer_snippet": sh_res.answer[:80],
                    },
                }
            )

        # Compute aggregate metrics
        baseline_metrics: SystemMetrics = calculate_system_metrics(
            system_name="baseline", results=baseline_results
        )
        self_healing_metrics: SystemMetrics = calculate_system_metrics(
            system_name="self_healing", results=self_healing_results
        )
        summary_deltas = calculate_metric_deltas(
            baseline=baseline_metrics, self_healing=self_healing_metrics
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Evaluation complete. Baseline Pass Rate: %.1f%% | Self-Healing Pass Rate: %.1f%% | Recovery Success: %.1f%%",
            baseline_metrics.critic_pass_rate * 100,
            self_healing_metrics.critic_pass_rate * 100,
            self_healing_metrics.recovery_success_rate * 100,
        )

        return EvaluationReport(
            timestamp=timestamp,
            dataset_size=len(samples),
            baseline_metrics=baseline_metrics,
            self_healing_metrics=self_healing_metrics,
            per_query_results=per_query_comparison,
            summary_deltas=summary_deltas,
        )
