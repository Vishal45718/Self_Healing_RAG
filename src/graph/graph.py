"""LangGraph workflow for Self-Healing RAG (Phase 3)."""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from config.settings import settings
from src.schema import GraphState
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator
from src.critic.critic import Critic
from src.critic.schema import CriticVerdict
from src.generation.prompts import format_reformulate_messages


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
        """Construct the LangGraph state machine."""
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("critic", self.critic_node)
        workflow.add_node("reformulate", self.reformulate_node)

        # Set entry point
        workflow.set_entry_point("retrieve")

        # Define basic edges
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "critic")
        workflow.add_edge("reformulate", "retrieve")

        # Define conditional routing from critic
        workflow.add_conditional_edges(
            "critic",
            self.should_continue,
            {
                "end": END,
                "reformulate": "reformulate"
            }
        )

        return workflow.compile()

    def retrieve_node(self, state: GraphState) -> Dict[str, Any]:
        """Node for retrieving documents."""
        query = state["current_query"]
        docs = self.retriever.retrieve(query)
        return {"retrieved_chunks": docs}

    def generate_node(self, state: GraphState) -> Dict[str, Any]:
        """Node for generating an answer based on retrieved documents."""
        query = state["original_query"] # The generator should ideally answer the original query
        context = state["retrieved_chunks"]
        generation = self.generator.generate(query, context)
        return {"generation": generation.answer}

    def critic_node(self, state: GraphState) -> Dict[str, Any]:
        """Node for evaluating the generation."""
        query = state["original_query"]
        context = state["retrieved_chunks"]
        generation = state["generation"]
        
        evaluation = self.critic.evaluate(
            query=query,
            context=context,
            generation=generation
        )
        
        # Increment iterations here
        current_iterations = state.get("iterations", 0) + 1
        
        return {
            "critic_evaluation": evaluation,
            "iterations": current_iterations
        }

    def reformulate_node(self, state: GraphState) -> Dict[str, Any]:
        """Node for reformulating the query if the critic fails."""
        original_query = state["original_query"]
        evaluation = state["critic_evaluation"]
        
        messages = format_reformulate_messages(
            original_query=original_query,
            critic_reasoning=evaluation.reasoning
        )
        
        # We can use the Generator's underlying client directly or we can use the Generator 
        # to just pass the prompt. Let's use the client directly since Generator expects RAG context.
        # But wait, Generator has `_client`. Let's just use `_client.chat_completion`.
        # However, Generator might be mocked in tests. If mocked, it might not have the same interface if we mock generate().
        # Actually, `Generator` takes `client`. I should use `self.generator._client` or maybe add a method to `Generator`.
        # Wait, for simplicity, I can just use `self.generator._client` and handle mock setup in tests.
        try:
            response = self.generator._client.chat_completion(
                messages=messages,
                model=settings.llm_model_id,
                max_tokens=150,
                temperature=0.7,
            )
            new_query = response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback in case of reformulation failure
            new_query = original_query
            
        return {"current_query": new_query}

    def should_continue(self, state: GraphState) -> Literal["end", "reformulate"]:
        """Determine whether to end the workflow or retry/reformulate."""
        evaluation = state["critic_evaluation"]
        iterations = state["iterations"]

        if evaluation.verdict == CriticVerdict.PASS:
            return "end"
        
        if iterations >= self.max_retries:
            return "end"
            
        return "reformulate"

    def invoke(self, query: str) -> GraphState:
        """Run the self-healing workflow for a given query."""
        initial_state = {
            "original_query": query,
            "current_query": query,
            "retrieved_chunks": [],
            "generation": None,
            "critic_evaluation": None,
            "iterations": 0
        }
        return self.graph.invoke(initial_state)
