"""CLI runner for Phase 7 Baseline vs Self-Healing evaluation benchmark."""

import argparse
import logging
import sys
from pathlib import Path

from src.critic.critic import Critic
from src.evaluation.dataset import load_default_dataset, load_default_documents
from src.evaluation.report import generate_markdown_report, save_report_json
from src.evaluation.runner import EvaluationRunner
from src.generation.generator import Generator
from src.ingestion.chunker import RecursiveCharacterChunker
from src.ingestion.embedder import LocalEmbedder
from src.ingestion.vector_store import VectorStore
from src.retrieval.retriever import Retriever

logger = logging.getLogger("eval_benchmark")


def run_evaluation_benchmark(
    output_path: str = "evaluation_results.json",
    collection_name: str = "eval_benchmark_collection",
) -> None:
    """Execute the full side-by-side evaluation benchmark.

    Args:
        output_path: Filepath where JSON evaluation report will be written.
        collection_name: ChromaDB collection name for benchmark isolation.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("\n" + "=" * 70)
    print("  SELF-HEALING RAG vs BASELINE RAG: PHASE 7 EVALUATION BENCHMARK")
    print("=" * 70 + "\n")

    # 1. Ingestion of evaluation corpus
    print("1. Ingesting evaluation document corpus...")
    docs = load_default_documents()
    chunker = RecursiveCharacterChunker()
    chunks = chunker.chunk_documents(docs)

    embedder = LocalEmbedder()
    embeddings = embedder.embed_chunks(chunks)

    # Use isolated vector store collection
    vector_store = VectorStore(collection_name=collection_name)
    vector_store.upsert(chunks=chunks, embeddings=embeddings)
    print(f"   Indexed {len(chunks)} chunks from {len(docs)} documents into collection '{collection_name}'.\n")

    # 2. Component Initialization
    print("2. Initializing shared pipeline components...")
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    generator = Generator()
    critic = Critic()

    runner = EvaluationRunner(
        retriever=retriever,
        generator=generator,
        critic=critic,
    )

    # 3. Running Benchmark
    dataset = load_default_dataset()
    print(f"3. Executing side-by-side evaluation across {len(dataset)} queries...")
    report = runner.run_side_by_side(dataset)

    # 4. Save JSON Report
    save_report_json(report, output_path)
    print(f"\n4. Saved structured evaluation results to: {output_path}")

    # 5. Display Markdown Summary Report
    print("\n5. Summary Report:\n")
    md_report = generate_markdown_report(report)
    print(md_report)
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Phase 7 Baseline vs Self-Healing RAG evaluation benchmark."
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Path for JSON output report (default: evaluation_results.json)",
    )
    parser.add_argument(
        "--collection",
        default="eval_benchmark_collection",
        help="Chroma collection name (default: eval_benchmark_collection)",
    )
    args = parser.parse_args()

    run_evaluation_benchmark(
        output_path=args.output,
        collection_name=args.collection,
    )
