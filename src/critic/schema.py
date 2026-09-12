"""Schema definitions for Critic evaluations (Task 2.4)."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class CriticVerdict(str, Enum):
    """Overall verdict of the critic evaluation."""

    PASS = "PASS"
    FAIL = "FAIL"
    ABSTAIN = "ABSTAIN"


class CriticFailureReason(str, Enum):
    """Specific category of failure identified by the critic."""

    RETRIEVAL_INSUFFICIENT = "retrieval_insufficient"
    GENERATION_UNGROUNDED = "generation_ungrounded"


class CriticEvaluation(BaseModel):
    """Structured output of a Critic evaluation."""

    verdict: CriticVerdict = Field(
        description="Overall verdict: PASS if both sufficient and grounded, FAIL if hallucinated or insufficient, ABSTAIN if safely refused to answer."
    )
    failure_reason: Optional[CriticFailureReason] = Field(
        default=None,
        description="Specific failure category: 'retrieval_insufficient' or 'generation_ungrounded'. None if PASS.",
    )
    is_retrieval_sufficient: bool = Field(
        description="True if the retrieved evidence is relevant and sufficient to answer the query."
    )
    is_generation_grounded: bool = Field(
        description="True if every factual claim in the answer is directly supported by the evidence."
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Specific factual claims made in the answer that lack evidence in the context.",
    )
    reasoning: str = Field(
        description="Concise reasoning justifying the verdict and evaluations.",
    )
