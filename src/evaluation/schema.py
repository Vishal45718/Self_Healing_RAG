"""Schema definitions for Phase 7 Evaluation & Baseline Comparison."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.critic.schema import CriticFailureReason, CriticVerdict


class ScenarioType(str, Enum):
    """Evaluation test query scenario category."""

    DIRECT = "direct"
    REFORMULATION_TARGET = "reformulation_target"
    UNANSWERABLE = "unanswerable"


class ExpectedBehavior(str, Enum):
    """Expected target behavior for an evaluation query."""

    ANSWER_GROUNDED = "answer_grounded"
    REFORMULATE_AND_ANSWER = "reformulate_and_answer"
    ABSTAIN = "abstain"


class EvalSample(BaseModel):
    """Represents a single query item in the evaluation dataset."""

    id: str = Field(description="Unique identifier for the evaluation query.")
    query: str = Field(description="The user input query string.")
    scenario: ScenarioType = Field(description="The scenario category.")
    expected_behavior: ExpectedBehavior = Field(
        description="The ideal target behavior of the system."
    )
    relevant_doc_ids: List[str] = Field(
        default_factory=list,
        description="IDs of documents in the corpus containing relevant evidence.",
    )
    ground_truth_answer: Optional[str] = Field(
        default=None,
        description="Optional reference ground-truth factual answer.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Brief description of the test case intent.",
    )


class EvalExecutionResult(BaseModel):
    """Execution trace and metrics for a single query on a single RAG pipeline."""

    query_id: str = Field(description="ID of the evaluated sample.")
    query: str = Field(description="Original query string.")
    scenario: ScenarioType = Field(description="Scenario category of the sample.")
    expected_behavior: ExpectedBehavior = Field(description="Expected target behavior.")
    system_name: str = Field(
        description="System evaluated ('baseline' or 'self_healing')."
    )
    final_query: str = Field(description="Final query used for retrieval.")
    retrieved_chunk_ids: List[str] = Field(
        default_factory=list,
        description="List of chunk IDs retrieved in the final retrieval pass.",
    )
    answer: str = Field(description="Final generated answer string.")
    critic_verdict: Optional[CriticVerdict] = Field(
        default=None,
        description="Final critic verdict (PASS, FAIL, ABSTAIN).",
    )
    critic_failure_reason: Optional[CriticFailureReason] = Field(
        default=None,
        description="Critic failure reason if verdict is FAIL.",
    )
    is_retrieval_sufficient: bool = Field(
        default=False,
        description="Whether the retrieved context was deemed sufficient.",
    )
    is_generation_grounded: bool = Field(
        default=False,
        description="Whether the generated answer was grounded in evidence.",
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="List of factual claims flagged as unsupported by critic.",
    )
    iterations: int = Field(
        default=1,
        description="Total iterations/critic evaluations performed.",
    )
    retries: int = Field(
        default=0,
        description="Number of retry/recovery attempts (iterations - 1).",
    )
    latency_ms: float = Field(
        default=0.0,
        description="Total wall-clock latency in milliseconds.",
    )
    llm_calls: int = Field(
        default=1,
        description="Total count of LLM calls (generator + critic + recovery).",
    )

    # Derived boolean flags for scoring
    is_critic_pass: bool = Field(
        default=False,
        description="True if final verdict is PASS.",
    )
    is_correct_abstention: bool = Field(
        default=False,
        description="True if query is UNANSWERABLE and system safely abstained.",
    )
    is_recovery_success: bool = Field(
        default=False,
        description="True if query initially failed, underwent recovery, and concluded with PASS/ABSTAIN matching expected behavior.",
    )
    is_reformulation_success: bool = Field(
        default=False,
        description="True if query was reformulated and subsequently achieved retrieval sufficiency or PASS.",
    )


class SystemMetrics(BaseModel):
    """Aggregated evaluation metrics for a specific RAG pipeline system."""

    system_name: str = Field(description="System name ('baseline' or 'self_healing').")
    total_queries: int = Field(description="Total number of evaluated queries.")
    critic_pass_rate: float = Field(
        description="Proportion of queries achieving CriticVerdict.PASS (0.0 to 1.0)."
    )
    groundedness_rate: float = Field(
        description="Proportion of answers evaluated as fully grounded (0.0 to 1.0)."
    )
    retrieval_sufficiency_rate: float = Field(
        description="Proportion of queries with sufficient retrieval (0.0 to 1.0)."
    )
    correct_abstention_rate: float = Field(
        description="Proportion of unanswerable queries correctly abstained (0.0 to 1.0)."
    )
    hallucination_rate_on_unanswerable: float = Field(
        description="Proportion of unanswerable queries where an ungrounded answer was generated (0.0 to 1.0)."
    )
    recovery_success_rate: float = Field(
        description="Proportion of queries requiring recovery that successfully recovered (0.0 to 1.0)."
    )
    reformulation_success_rate: float = Field(
        description="Proportion of reformulated queries that successfully recovered (0.0 to 1.0)."
    )
    average_retries: float = Field(
        description="Average number of retries per query."
    )
    average_iterations: float = Field(
        description="Average number of iterations per query."
    )
    average_latency_ms: float = Field(
        description="Average total latency in milliseconds."
    )
    average_llm_calls: float = Field(
        description="Average number of LLM invocations per query."
    )
    total_llm_calls: int = Field(
        description="Total LLM invocations across all queries."
    )


class EvaluationReport(BaseModel):
    """Complete comparative evaluation report."""

    timestamp: str = Field(description="ISO timestamp of evaluation execution.")
    dataset_size: int = Field(description="Total number of evaluation samples.")
    baseline_metrics: SystemMetrics = Field(
        description="Aggregated metrics for Baseline RAG."
    )
    self_healing_metrics: SystemMetrics = Field(
        description="Aggregated metrics for Self-Healing RAG."
    )
    per_query_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-sample side-by-side execution details.",
    )
    summary_deltas: Dict[str, Any] = Field(
        default_factory=dict,
        description="Comparative deltas and ratios between Self-Healing and Baseline.",
    )
