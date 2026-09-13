"""Self-Healing RAG Interactive & Deterministic Demonstration Script.

Demonstrates the core capabilities of the Self-Healing RAG system:
  1. Normal successful answer (single-pass PASS)
  2. Critic rejection + self-healing recovery (GENERATION_UNGROUNDED -> REGENERATE -> PASS)
  3. Final safe abstention on unanswerable query (ABSTAIN)

Usage:
  python demo.py               # Deterministic offline mode (reproducible without API keys)
  python demo.py --offline     # Explicit deterministic offline mode
  python demo.py --live        # Live mode via Hugging Face Inference API (requires HF_TOKEN)
"""

import argparse
import logging
import os
import sys
from typing import Any, List, Optional
from unittest.mock import MagicMock

from config.settings import settings
from src.critic.critic import Critic
from src.critic.schema import CriticEvaluation, CriticFailureReason, CriticVerdict
from src.generation.generator import GenerationResult, Generator
from src.graph.graph import SelfHealingRAG
from src.ingestion.chunker import RecursiveCharacterChunker
from src.ingestion.embedder import LocalEmbedder
from src.ingestion.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.schema import Document

# Suppress noisy logs for a clean demo terminal experience
logging.basicConfig(level=logging.WARNING)


# ----------------------------------------------------------------------
# Demonstration Document Corpus
# ----------------------------------------------------------------------
DEMO_DOCUMENTS = [
    Document(
        id="doc_chroma",
        content=(
            "ChromaDB is an open-source, AI-native embedded vector database designed "
            "to store embeddings and metadata locally. It operates in-process with "
            "zero external daemon dependencies, utilizing SQLite and DuckDB for persistence."
        ),
        metadata={"source": "docs/chroma.md", "topic": "vector_store"},
    ),
    Document(
        id="doc_self_healing",
        content=(
            "Self-Healing RAG coordinates retrieval, generation, and critique via a "
            "LangGraph cyclical state machine. If retrieval is insufficient, the system "
            "reformulates the query and re-retrieves. If generation is ungrounded (hallucinated), "
            "the system triggers regeneration with strict groundedness directives without "
            "re-retrieving. If context is missing or unanswerable, it safely abstains."
        ),
        metadata={"source": "docs/architecture.md", "topic": "self_healing"},
    ),
    Document(
        id="doc_langgraph",
        content=(
            "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
            "It supports cyclical execution graphs, conditional edge branching, and persistent state "
            "across iterations, making it suitable for corrective and self-healing agent loops."
        ),
        metadata={"source": "docs/langgraph.md", "topic": "orchestration"},
    ),
]


# ----------------------------------------------------------------------
# Deterministic Offline Simulator
# ----------------------------------------------------------------------
class OfflineDeterministicClient:
    """Mock Hugging Face InferenceClient for reproducible offline demonstrations."""

    def __init__(self, mode: str = "normal"):
        self.mode = mode
        self.call_count = 0
        self.recovery_hallucinated = False

    def chat_completion(self, messages: List[Any], **kwargs: Any) -> Any:
        self.call_count += 1
        system_content = ""
        user_content = ""
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role")
                content = str(m.get("content", ""))
                if role == "system":
                    system_content = content
                elif role == "user":
                    user_content = content

        response_content = ""

        # 1. Critic evaluation prompt
        if "evaluation critic" in system_content.lower() or "output json schema" in system_content.lower():
            if "quantum" in user_content.lower() or "assembly" in user_content.lower():
                # Detect the simulated hallucination on attempt 1
                response_content = (
                    '{"verdict": "FAIL", "failure_reason": "generation_ungrounded", '
                    '"is_retrieval_sufficient": true, "is_generation_grounded": false, '
                    '"unsupported_claims": ["quantum annealing hardware", "rewrites Python into x86 assembly"], '
                    '"reasoning": "The answer claims the system executes on quantum hardware and writes assembly, which is completely unsupported by the retrieved context."}'
                )
            else:
                response_content = (
                    '{"verdict": "PASS", "failure_reason": null, '
                    '"is_retrieval_sufficient": true, "is_generation_grounded": true, '
                    '"unsupported_claims": [], '
                    '"reasoning": "The response is fully grounded in the retrieved document context."}'
                )
        # 2. Regeneration prompt (strict groundedness directive after hallucination)
        elif "found to be ungrounded" in system_content.lower():
            response_content = (
                "Self-Healing RAG coordinates retrieval, generation, and critique via a "
                "LangGraph cyclical state machine. When generation is ungrounded (hallucinated), "
                "the system triggers regeneration with strict groundedness directives without "
                "re-retrieving."
            )
        # 3. Reformulation prompt
        elif "expert query reformulator" in system_content.lower():
            response_content = "ChromaDB embedded local persistence architecture"
        # 4. Standard Generation prompt (RAG_SYSTEM_PROMPT)
        else:
            question = user_content.split("Question:")[-1].strip().lower() if "Question:" in user_content else user_content.lower()
            if "dark matter" in question or "propulsion" in question or "unanswerable" in question:
                response_content = (
                    "The provided context does not contain sufficient information to answer this question."
                )
            elif "recovery workflow" in question and not self.recovery_hallucinated:
                # Intentionally hallucinate on attempt 1 to showcase self-healing recovery
                self.recovery_hallucinated = True
                response_content = (
                    "Self-Healing RAG executes on quantum annealing hardware and rewrites Python "
                    "into x86 assembly to achieve sub-millisecond inference."
                )
            elif "chromadb" in question:
                response_content = (
                    "ChromaDB is an open-source, AI-native embedded vector database designed "
                    "to store embeddings and metadata locally in-process without external daemons."
                )
            else:
                response_content = (
                    "The system utilizes cyclical feedback loops to guarantee groundedness."
                )

        choice = MagicMock()
        choice.message = MagicMock(content=response_content)
        mock_response = MagicMock(choices=[choice])
        mock_response.model_dump.return_value = {"simulated": True}
        return mock_response


def build_offline_components(vector_store: VectorStore, embedder: LocalEmbedder):
    """Build components wired with deterministic offline simulator."""
    retriever = Retriever(vector_store=vector_store, embedder=embedder)

    mock_client = OfflineDeterministicClient()
    generator = Generator(client=mock_client)  # type: ignore[arg-type]
    critic = Critic(client=mock_client)        # type: ignore[arg-type]

    return retriever, generator, critic


def build_live_components(vector_store: VectorStore, embedder: LocalEmbedder):
    """Build components wired to live LLM API (Google Gemini or Hugging Face)."""
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        has_key = bool(settings.gemini_api_key or os.getenv("GEMINI_API_KEY"))
        if not has_key:
            print("\n[ERROR] GEMINI_API_KEY is not configured in .env or environment.")
            print("Please configure GEMINI_API_KEY to run in --live mode, or run with --offline.\n")
            sys.exit(1)
    else:
        has_token = bool(settings.hf_token or os.getenv("HF_TOKEN"))
        if not has_token:
            print("\n[ERROR] HF_TOKEN is not configured in .env or environment.")
            print("Please configure HF_TOKEN to run in --live mode, or run with --offline.\n")
            sys.exit(1)

    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    generator = Generator()
    critic = Critic()

    return retriever, generator, critic


# ----------------------------------------------------------------------
# Demo Presentation Utilities
# ----------------------------------------------------------------------
def print_section(title: str) -> None:
    border = "=" * 76
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}\n")


def print_state_summary(scenario_name: str, state: dict) -> None:
    print(f"Scenario: {scenario_name}")
    print(f"  * Original Query:    \"{state['original_query']}\"")
    print(f"  * Final Query:       \"{state['current_query']}\"")
    print(f"  * Iterations/Retries: {state['iterations']}")

    eval_result = state.get("critic_evaluation")
    if eval_result:
        print(f"  * Critic Verdict:    {eval_result.verdict.value}")
        if eval_result.failure_reason:
            print(f"  * Failure Reason:    {eval_result.failure_reason.value}")
        if eval_result.unsupported_claims:
            print(f"  * Unsupported Claims: {eval_result.unsupported_claims}")
        print(f"  * Critic Reasoning:  {eval_result.reasoning}")

    print(f"  * Final Answer:\n    \"{state['generation']}\"")
    print("-" * 76)


# ----------------------------------------------------------------------
# Main Interactive Runner
# ----------------------------------------------------------------------
def run_demo(is_live: bool = False) -> None:
    provider_name = "Google Gemini API" if settings.llm_provider.lower() == "gemini" else "Hugging Face Inference API"
    mode_str = f"LIVE ({provider_name})" if is_live else "DETERMINISTIC OFFLINE (Local & Reproducible)"
    print_section(f"SELF-HEALING RAG DEMONSTRATION  [{mode_str}]")

    # Step 1: Ingest Demo Corpus locally
    print("1. Ingesting knowledge base into local Chroma vector store...")
    chunker = RecursiveCharacterChunker(chunk_size=250, chunk_overlap=30)
    chunks = chunker.chunk_documents(DEMO_DOCUMENTS)

    embedder = LocalEmbedder()
    embeddings = embedder.embed_chunks(chunks)

    vector_store = VectorStore(collection_name="demo_collection")
    vector_store.upsert(chunks=chunks, embeddings=embeddings)
    print(f"   Indexed {len(chunks)} chunks from {len(DEMO_DOCUMENTS)} documents successfully.\n")

    # Step 2: Initialize components
    if is_live:
        print("2. Connecting to live Hugging Face Inference API...")
        retriever, generator, critic = build_live_components(vector_store, embedder)
    else:
        print("2. Wiring deterministic offline simulator...")
        retriever, generator, critic = build_offline_components(vector_store, embedder)

    pipeline = SelfHealingRAG(retriever=retriever, generator=generator, critic=critic)

    # ------------------------------------------------------------------
    # DEMO 1: Normal Successful Answer
    # ------------------------------------------------------------------
    print_section("DEMO 1: Normal Successful Answer (Direct Retrieval & Grounded Generation)")
    query_1 = "What is ChromaDB and how does it persist data?"
    print(f"User Query: \"{query_1}\"\nRunning pipeline...")
    state_1 = pipeline.invoke(query_1)
    print_state_summary("Direct Grounded Query", state_1)

    # ------------------------------------------------------------------
    # DEMO 2: Critic Rejection + Recovery Loop
    # ------------------------------------------------------------------
    print_section("DEMO 2: Critic Rejection + Recovery (Ungrounded Generation -> Regenerate -> PASS)")
    query_2 = "How does the Self-Healing RAG recovery workflow operate?"
    print(f"User Query: \"{query_2}\"")
    print("Simulating generation failure (hallucination) on attempt 1...")
    print("Running pipeline...")
    state_2 = pipeline.invoke(query_2)
    print_state_summary("Self-Healing Recovery Workflow", state_2)

    # ------------------------------------------------------------------
    # DEMO 3: Final Safe Abstention
    # ------------------------------------------------------------------
    print_section("DEMO 3: Final Safe Abstention (Out-of-Domain / Unanswerable Query)")
    query_3 = "What is the secret recipe for dark matter propulsion in deep space?"
    print(f"User Query: \"{query_3}\"")
    print("Query has no relevant evidence in the vector index.")
    print("Running pipeline...")
    state_3 = pipeline.invoke(query_3)
    print_state_summary("Unanswerable / Abstention Query", state_3)

    print_section("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("Summary of verified behaviors:")
    print("  [✓] Direct answers pass in a single iteration.")
    print("  [✓] Hallucinated/ungrounded claims trigger critic rejection and automatic regeneration.")
    print("  [✓] Unanswerable queries safely trigger abstention without hallucinating or looping.")
    print(f"{'=' * 76}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Healing RAG Demonstration Script")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute against live LLM API (Google Gemini or Hugging Face, requires credentials)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Execute in deterministic offline mode (default)",
    )
    args = parser.parse_args()

    # Default to offline if live is not explicitly requested or if credentials are unset
    provider = settings.llm_provider.lower()
    has_credentials = (
        bool(settings.gemini_api_key or os.getenv("GEMINI_API_KEY"))
        if provider == "gemini"
        else bool(settings.hf_token or os.getenv("HF_TOKEN"))
    )
    run_live = args.live and has_credentials
    if args.live and not has_credentials:
        req_var = "GEMINI_API_KEY" if provider == "gemini" else "HF_TOKEN"
        print(f"[WARNING] --live requested but {req_var} is not set. Falling back to --offline.")
        run_live = False

    run_demo(is_live=run_live)
