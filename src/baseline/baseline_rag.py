"""Baseline RAG implementation (Phase 7).

Provides a standard single-pass Retrieve -> Generate pipeline for objective
comparison against SelfHealingRAG.
"""

import logging
import time
from typing import List, Optional
from pydantic import BaseModel, Field

from src.retrieval.retriever import Retriever
from src.generation.generator import Generator, GenerationResult
from src.critic.critic import Critic
from src.critic.schema import CriticEvaluation
from src.schema import RetrievalResult

logger = logging.getLogger(__name__)


class BaselineResult(BaseModel):
    """Structured output of a Baseline RAG execution."""

    query: str = Field(description="The input user query.")
    retrieved_chunks: List[RetrievalResult] = Field(
        default_factory=list,
        description="The document chunks retrieved in the single pass.",
    )
    answer: str = Field(description="The generated answer text.")
    model_id: str = Field(description="The model ID used for generation.")
    provider: str = Field(description="The inference provider used.")
    critic_evaluation: Optional[CriticEvaluation] = Field(
        default=None,
        description="Optional critic evaluation of the baseline answer.",
    )


class BaselineRAG:
    """Standard single-pass RAG pipeline (Retrieve -> Generate).

    Used as the control/baseline system in Phase 7 evaluations.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        critic: Optional[Critic] = None,
    ) -> None:
        """Initialise BaselineRAG.

        Args:
            retriever: Configured Retriever instance.
            generator: Configured Generator instance.
            critic: Optional Critic instance for evaluating baseline outputs.
        """
        self.retriever = retriever
        self.generator = generator
        self.critic = critic

    def invoke(self, query: str, top_k: int = 5) -> BaselineResult:
        """Execute single-pass retrieval and generation for a query.

        Args:
            query: The user query string.
            top_k: Number of nearest chunks to retrieve.

        Returns:
            BaselineResult containing retrieved chunks, generated answer, and optional critic evaluation.

        Raises:
            ValueError: If query is empty or whitespace.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        logger.debug("Baseline RAG: retrieving chunks for query: %s", query[:50])
        chunks = self.retriever.retrieve(query=query, top_k=top_k)

        logger.debug("Baseline RAG: generating answer from %d chunks", len(chunks))
        gen_result: GenerationResult = self.generator.generate(
            query=query, context=chunks
        )

        critic_eval: Optional[CriticEvaluation] = None
        if self.critic is not None:
            logger.debug("Baseline RAG: evaluating generation with Critic")
            critic_eval = self.critic.evaluate(
                query=query,
                context=chunks,
                answer=gen_result.answer,
            )

        return BaselineResult(
            query=query,
            retrieved_chunks=chunks,
            answer=gen_result.answer,
            model_id=gen_result.model_id,
            provider=gen_result.provider,
            critic_evaluation=critic_eval,
        )
