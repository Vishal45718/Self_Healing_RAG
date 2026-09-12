"""Evaluation dataset definitions and test corpus loader for Phase 7."""

import json
from pathlib import Path
from typing import List, Optional

from src.schema import Document
from src.evaluation.schema import EvalSample, ExpectedBehavior, ScenarioType


# Standard reference corpus for reproducible evaluation
DEFAULT_EVALUATION_DOCUMENTS: List[Document] = [
    Document(
        id="doc_distributed_cache",
        content=(
            "Distributed Caching System Architecture Overview:\n"
            "Our distributed cache cluster uses consistent hashing to partition keys across storage nodes. "
            "It supports three primary eviction policies: Least Recently Used (LRU), Least Frequently Used (LFU), "
            "and Adaptive Replacement Cache (ARC). The default eviction strategy is LRU with a sliding expiration TTL of 3600 seconds. "
            "Cache nodes communicate heartbeats via a gossip protocol every 500 milliseconds. "
            "Read-through and write-behind caching modes are supported to synchronize updates with the persistent backing database."
        ),
        metadata={"source": "cache_spec.md", "file_type": "md", "category": "infrastructure"},
    ),
    Document(
        id="doc_raft_consensus",
        content=(
            "Raft Consensus Protocol Implementation Details:\n"
            "The cluster maintains high availability and linearizable consistency using the Raft consensus algorithm. "
            "A cluster of N nodes requires a strict quorum of floor(N/2) + 1 nodes to elect a leader or commit log entries. "
            "For example, in a 5-node cluster, the quorum size is 3 nodes, tolerating up to 2 simultaneous node failures. "
            "Leader election timeouts are randomized between 150ms and 300ms to avoid split votes. "
            "Log compaction is triggered when the Raft write-ahead log exceeds 64 megabytes, creating snapshot checkpoints to disk."
        ),
        metadata={"source": "raft_spec.md", "file_type": "md", "category": "consensus"},
    ),
    Document(
        id="doc_vector_index",
        content=(
            "Vector Search Indexing with HNSW:\n"
            "Vector embeddings are indexed using Hierarchical Navigable Small World (HNSW) graphs. "
            "The key construction hyperparameters are M (the maximum number of bi-directional links per node, set to 16 by default) "
            "and efConstruction (the size of the dynamic candidate list during graph building, set to 200). "
            "During runtime queries, the efSearch parameter controls the search accuracy versus speed trade-off; higher efSearch yields "
            "higher recall at the expense of search latency. Cosine similarity distance metric is used for all normalized embedding vectors."
        ),
        metadata={"source": "vector_indexing.md", "file_type": "md", "category": "search"},
    ),
]


# Curated evaluation queries covering all 3 required scenario types
DEFAULT_EVALUATION_SAMPLES: List[EvalSample] = [
    # 1. Direct / Answerable queries
    EvalSample(
        id="direct_01",
        query="What is the minimum quorum size for a Raft cluster of 5 nodes?",
        scenario=ScenarioType.DIRECT,
        expected_behavior=ExpectedBehavior.ANSWER_GROUNDED,
        relevant_doc_ids=["doc_raft_consensus"],
        ground_truth_answer="In a 5-node cluster, the quorum size is 3 nodes (floor(N/2) + 1).",
        description="Direct question on Raft quorum formula explicitly answered in doc_raft_consensus.",
    ),
    EvalSample(
        id="direct_02",
        query="What are the key configuration hyperparameters for HNSW vector index construction?",
        scenario=ScenarioType.DIRECT,
        expected_behavior=ExpectedBehavior.ANSWER_GROUNDED,
        relevant_doc_ids=["doc_vector_index"],
        ground_truth_answer="The key construction hyperparameters are M (default 16) and efConstruction (default 200).",
        description="Direct question on HNSW construction parameters in doc_vector_index.",
    ),
    EvalSample(
        id="direct_03",
        query="Which cache eviction policies are supported by the distributed caching system?",
        scenario=ScenarioType.DIRECT,
        expected_behavior=ExpectedBehavior.ANSWER_GROUNDED,
        relevant_doc_ids=["doc_distributed_cache"],
        ground_truth_answer="The supported eviction policies are Least Recently Used (LRU), Least Frequently Used (LFU), and Adaptive Replacement Cache (ARC).",
        description="Direct question on cache eviction policies explicitly answered in doc_distributed_cache.",
    ),

    # 2. Vague / Reformulation-target queries
    EvalSample(
        id="reformulate_01",
        query="How do cluster nodes agree on updates and handle elections?",
        scenario=ScenarioType.REFORMULATION_TARGET,
        expected_behavior=ExpectedBehavior.REFORMULATE_AND_ANSWER,
        relevant_doc_ids=["doc_raft_consensus"],
        ground_truth_answer="Nodes use the Raft consensus algorithm, requiring a quorum of floor(N/2) + 1 nodes and randomized election timeouts.",
        description="Vague question about agreement that benefits from query reformulation to target Raft consensus.",
    ),
    EvalSample(
        id="reformulate_02",
        query="Memory cleanup and key expiration rules",
        scenario=ScenarioType.REFORMULATION_TARGET,
        expected_behavior=ExpectedBehavior.REFORMULATE_AND_ANSWER,
        relevant_doc_ids=["doc_distributed_cache"],
        ground_truth_answer="Memory cleanup is handled via LRU, LFU, or ARC eviction with a default sliding TTL of 3600 seconds.",
        description="Keyword fragment query that benefits from reformulation targeting cache eviction and TTL.",
    ),
    EvalSample(
        id="reformulate_03",
        query="Graph search speed vs accuracy tuning parameters",
        scenario=ScenarioType.REFORMULATION_TARGET,
        expected_behavior=ExpectedBehavior.REFORMULATE_AND_ANSWER,
        relevant_doc_ids=["doc_vector_index"],
        ground_truth_answer="Search accuracy vs speed is tuned using the runtime efSearch parameter in HNSW vector indexing.",
        description="Underspecified query requiring reformulation to link to HNSW efSearch parameter.",
    ),

    # 3. Unanswerable / Out-of-Domain queries
    EvalSample(
        id="unanswerable_01",
        query="What is the recommended pediatric dosage of Amoxicillin for acute otitis media?",
        scenario=ScenarioType.UNANSWERABLE,
        expected_behavior=ExpectedBehavior.ABSTAIN,
        relevant_doc_ids=[],
        ground_truth_answer=None,
        description="Medical query with no relevant documents in corpus; system should safely abstain.",
    ),
    EvalSample(
        id="unanswerable_02",
        query="What was the closing stock price of Apple Inc on January 15, 2024?",
        scenario=ScenarioType.UNANSWERABLE,
        expected_behavior=ExpectedBehavior.ABSTAIN,
        relevant_doc_ids=[],
        ground_truth_answer=None,
        description="Financial query with no evidence in corpus; system should safely abstain.",
    ),
    EvalSample(
        id="unanswerable_03",
        query="How do you configure quantum key distribution protocols with BB84 encoding?",
        scenario=ScenarioType.UNANSWERABLE,
        expected_behavior=ExpectedBehavior.ABSTAIN,
        relevant_doc_ids=[],
        ground_truth_answer=None,
        description="Quantum cryptography query absent from corpus; system should safely abstain.",
    ),
]


def load_default_documents() -> List[Document]:
    """Return a deep copy of the default evaluation document corpus."""
    return [doc.model_copy(deep=True) for doc in DEFAULT_EVALUATION_DOCUMENTS]


def load_default_dataset() -> List[EvalSample]:
    """Return a deep copy of the curated evaluation dataset samples."""
    return [sample.model_copy(deep=True) for sample in DEFAULT_EVALUATION_SAMPLES]


def load_dataset_from_json(path: str) -> List[EvalSample]:
    """Load evaluation samples from an external JSON file.

    Args:
        path: File path to a JSON file containing a list of sample objects.

    Returns:
        List of validated EvalSample instances.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON is invalid or empty.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset file not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Evaluation dataset JSON must contain a non-empty list of samples.")

    return [EvalSample.model_validate(item) for item in data]
