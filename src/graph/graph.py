"""LangGraph workflow for Self-Healing RAG (Phase 3 + Phase 5).

Phase 5 additions:
- query_history in GraphState tracks all queries tried (no repeat guarantee).
- recover_node: single entry-point that routes based on critic failure_reason.
  * retrieval_insufficient -> reformulate_node (re-retrieves with new query)
  * generation_ungrounded  -> regenerate_node (uses same chunks, no new retrieval)
  * unexpected reason      -> raises ValueError explicitly (never fails silently).
- reformulate_node: uses query_history + critic reasoning to produce a
  distinct reformulated query; falls back to original_query on LLM errors
  but never inserts a duplicate into query_history.
- regenerate_node: builds a strict regeneration directive from critic
  feedback + unsupported_claims; calls the generator without re-retrieving.
"""

from typing import Any, Dict, List, Literal

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from config.settings import settings
from src.schema import GraphState
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator
from src.critic.critic import Critic
from src.critic.schema import CriticFailureReason, CriticVerdict
from src.generation.prompts import (
    format_reformulate_messages,
    format_regenerate_messages,
)


class SelfHealingRAG:
    """Orchestrates the Self-Healing RAG pipeline using LangGraph."""

    def __init__(self, retriever: Retriever, generator: Generator, critic: Critic):
        """Initialize the workflow with required components."""
        self.retriever = retriever
        self.generator = generator
        self.critic = critic
        self.max_retries = settings.max_retries
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        """Construct the LangGraph state machine.

        Graph topology (Phase 3 + Phase 5):
            retrieve -> generate -> critic -> [end | recover]
            recover  -> [reformulate | regenerate]
            reformulate -> retrieve            (re-retrieval path)
            regenerate  -> critic              (re-evaluation, no new retrieval)
        """
        workflow = StateGraph(GraphState)

        # Nodes
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("critic", self.critic_node)
        workflow.add_node("recover", self.recover_node)
        workflow.add_node("reformulate", self.reformulate_node)
        workflow.add_node("regenerate", self.regenerate_node)

        # Entry point
        workflow.set_entry_point("retrieve")

        # Fixed edges
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "critic")

        # Conditional routing from critic: end or enter recovery
        workflow.add_conditional_edges(
            "critic",
            self.should_continue,
            {
                "end": END,
                "recover": "recover",
            },
        )

        # Conditional routing from recover: reformulate or regenerate
        workflow.add_conditional_edges(
            "recover",
            self.route_recovery,
            {
                "reformulate": "reformulate",
                "regenerate": "regenerate",
            },
        )

        # After reformulation: back to retrieve (new query, new chunks)
        workflow.add_edge("reformulate", "retrieve")

        # After regeneration: back to critic (same chunks, new answer)
        workflow.add_edge("regenerate", "critic")

        return workflow.compile()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def retrieve_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: embed and retrieve document chunks for current_query."""
        query = state["current_query"]
        docs = self.retriever.retrieve(query)
        return {"retrieved_chunks": docs}

    def generate_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: generate an answer grounded in retrieved chunks."""
        query = state["original_query"]
        context = state["retrieved_chunks"]
        result = self.generator.generate(query, context)
        return {"generation": result.answer}

    def critic_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: evaluate the current generation against retrieved evidence."""
        query = state["original_query"]
        context = state["retrieved_chunks"]
        generation = state["generation"]

        evaluation = self.critic.evaluate(
            query=query,
            context=context,
            answer=generation,
        )

        current_iterations = state.get("iterations", 0) + 1

        return {
            "critic_evaluation": evaluation,
            "iterations": current_iterations,
        }

    def recover_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: single recovery entry point — performs no action itself.

        Its only job is to be the target of the conditional edge from critic
        and then immediately route (via ``route_recovery``) to either
        ``reformulate`` or ``regenerate``.  This keeps routing logic clean
        and testable.

        Returns an empty dict (no state mutation at this node).
        """
        return {}

    def reformulate_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: reformulate the query for retrieval_insufficient failures.

        Uses query_history + critic reasoning to produce a new, distinct
        query.  If the LLM call fails or returns a duplicate, falls back
        to the original query (never crashes the pipeline).

        Returns:
            Updates to current_query and query_history.
        """
        original_query: str = state["original_query"]
        current_query: str = state["current_query"]
        evaluation = state["critic_evaluation"]
        query_history: List[str] = list(state.get("query_history", []))

        # Ensure current query is tracked before reformulating
        if current_query not in query_history:
            query_history.append(current_query)

        messages = format_reformulate_messages(
            original_query=original_query,
            critic_reasoning=evaluation.reasoning,
            query_history=query_history,
        )

        new_query = original_query  # safe fallback
        try:
            client = self.generator._client or self.generator._get_client()
            response = client.chat_completion(
                messages=messages,
                model=self.generator.model_id,
                max_tokens=150,
                temperature=0.7,
            )
            candidate = response.choices[0].message.content.strip()
            if (candidate.startswith('"') and candidate.endswith('"')) or (
                candidate.startswith("'") and candidate.endswith("'")
            ):
                candidate = candidate[1:-1].strip()

            # Guarantee: never return an empty query or a repeated query (case-insensitive)
            lower_history = {q.lower() for q in query_history}
            if candidate and candidate.lower() not in lower_history:
                new_query = candidate

        except Exception:
            # Reformulation failure is not fatal; use fallback
            pass

        # Record the new (or fallback) query in history
        if new_query not in query_history:
            query_history.append(new_query)

        return {
            "current_query": new_query,
            "query_history": query_history,
        }

    def regenerate_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: regenerate the answer for generation_ungrounded failures.

        IMPORTANT: This node does NOT call the retriever.  It reuses the
        same retrieved_chunks already present in the state and sends a
        stricter directive to the generator, embedding the critic's
        reasoning and the specific unsupported claims.

        Returns:
            Updated generation in state.
        """
        original_query: str = state["original_query"]
        context = state["retrieved_chunks"]
        evaluation = state["critic_evaluation"]

        messages = format_regenerate_messages(
            query=original_query,
            context=context,
            critic_reasoning=evaluation.reasoning,
            unsupported_claims=evaluation.unsupported_claims,
        )

        new_answer = state["generation"]  # safe fallback
        try:
            client = self.generator._client or self.generator._get_client()
            response = client.chat_completion(
                messages=messages,
                model=self.generator.model_id,
                max_tokens=1024,
                temperature=0.0,
            )
            candidate = response.choices[0].message.content.strip()
            if candidate:
                new_answer = candidate
        except Exception:
            # Regeneration failure is not fatal; retain previous answer
            pass

        return {"generation": new_answer}

    # ------------------------------------------------------------------
    # Routing functions
    # ------------------------------------------------------------------

    def should_continue(
        self, state: GraphState
    ) -> Literal["end", "recover"]:
        """Route after critic: end on PASS/ABSTAIN or max retries, recover otherwise."""
        evaluation = state["critic_evaluation"]
        iterations = state["iterations"]

        # Terminal verdicts
        if evaluation.verdict in (CriticVerdict.PASS, CriticVerdict.ABSTAIN):
            return "end"

        # Exhausted retries
        if iterations >= self.max_retries:
            return "end"

        return "recover"

    def route_recovery(
        self, state: GraphState
    ) -> Literal["reformulate", "regenerate"]:
        """Route within recovery: map failure_reason to the correct action.

        Raises:
            ValueError: If failure_reason is None or an unrecognised value.
                        Never guesses silently.
        """
        evaluation = state["critic_evaluation"]
        reason = evaluation.failure_reason

        if reason is None:
            raise ValueError(
                "route_recovery called with failure_reason=None on a FAIL verdict. "
                "This indicates a Critic normalization bug."
            )

        if reason == CriticFailureReason.RETRIEVAL_INSUFFICIENT:
            return "reformulate"

        if reason == CriticFailureReason.GENERATION_UNGROUNDED:
            return "regenerate"

        raise ValueError(
            f"route_recovery encountered unexpected failure_reason: {reason!r}. "
            "Add handling for this reason before extending the graph."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invoke(self, query: str) -> GraphState:
        """Run the self-healing workflow for a given query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        initial_state: GraphState = {
            "original_query": query,
            "current_query": query,
            "retrieved_chunks": [],
            "generation": None,
            "critic_evaluation": None,
            "iterations": 0,
            "query_history": [query],  # seed history with original query
        }
        return self.graph.invoke(initial_state)
