# Architectural Decision Records (ADR)

This document records the architectural and technology decisions made for the Self-Healing RAG project.

---

## D-001: Why LangGraph

- **Status**: Accepted
- **Context**: Self-Healing RAG requires dynamic, stateful, and cyclical workflows (retrieval -> evaluation/critique -> conditional query reformulation or hallucination checks -> regeneration).
- **Alternatives Considered**:
  - *Standard LangChain Chains / LCEL*: Great for linear Directed Acyclic Graphs (DAGs), but awkward and brittle when handling iterative loops, backtracking, and state persistence across retries.
  - *Custom Python State Machine*: Full flexibility, but requires writing boilerplate execution runtime, state checkpointing, edge routing, and tracing from scratch.
  - *LlamaIndex Workflows*: Functional event-driven workflows, but less direct control over granular state graphs and edge conditionals compared to LangGraph's explicit graph structure.
- **Decision & Rationale**: LangGraph provides first-class support for cyclical graphs, clear shared state schemas, conditional edge branching, and built-in checkpointing, making it the ideal framework to model the self-healing feedback loop cleanly.

---

## D-002: Why Chroma

- **Status**: Accepted
- **Context**: The system needs a vector database for embedding storage and similarity search that works out-of-the-box in local environments and CI without requiring dedicated server infrastructure.
- **Alternatives Considered**:
  - *FAISS*: High-performance vector search library, but lacks built-in metadata persistence, filtering capabilities, and CRUD operations out of the box.
  - *Qdrant / Milvus / Weaviate*: Feature-rich distributed vector databases, but require running Docker containers or managing external server processes.
  - *PostgreSQL + pgvector*: Powerful relational + vector store, but requires a running PostgreSQL instance and schema management.
- **Decision & Rationale**: Chroma operates as an embedded vector store backed by DuckDB/SQLite and local parquet files. It requires zero external daemon processes, installs cleanly via pip, and allows simple persistent local storage (`PersistentClient(path=...)`), keeping the development setup lightweight.

---

## D-003: Why Hugging Face Inference Providers

- **Status**: Accepted
- **Context**: The generation, critique, and reformulation stages require access to performant large language models (e.g. Llama 3.3, Qwen 2.5) without mandating high-end local GPU hardware or tightly coupling to proprietary single-vendor APIs.
- **Alternatives Considered**:
  - *Local Ollama / vLLM*: Requires local GPU infrastructure (16GB+ VRAM for quality 70B models) which may not be available on development machines.
  - *Proprietary OpenAI / Anthropic APIs*: High capability, but introduces single-vendor lock-in and precludes using open weights models.
  - *Direct Cloud Provider APIs (AWS Bedrock, Azure ML)*: Complex setup, cloud-specific SDKs, and billing overhead.
- **Decision & Rationale**: Hugging Face Inference Providers (via `huggingface_hub.InferenceClient`) offer a unified, vendor-agnostic interface to route requests to high-throughput providers (e.g. Together AI, Sambanova, Featherless, or Serverless HF Inference) using a single standard API format and common Hugging Face authentication tokens.

---

## D-004: Why sentence-transformers

- **Status**: Accepted
- **Context**: Generating dense vector embeddings for document chunks and user queries is a continuous, high-frequency operation in RAG pipelines.
- **Alternatives Considered**:
  - *Remote Embedding APIs (OpenAI, Cohere, HF Inference)*: Introduces network latency on every query and ingest batch, rate-limit constraints, and per-token API costs.
  - *Hugging Face Transformers raw pipeline*: Requires manual tensor batching, pooling logic, and device management.
  - *FastEmbed / ONNX Runtime*: Very fast, but slightly narrower model catalog and ecosystem integration.
- **Decision & Rationale**: `sentence-transformers` runs locally on CPU or GPU with excellent throughput for lightweight models (like `all-MiniLM-L6-v2` or `bge-small-en-v1.5`), provides deterministic results, has zero per-call cost or network latency, and integrates seamlessly with Chroma embedding functions.

---

## D-005: Why pypdf
- **Status**: Accepted
- **Context**: The `DocumentLoader` requires a lightweight dependency to parse content and extract page information from `.pdf` files.
- **Alternatives Considered**:
  - *PyMuPDF*: Very fast and robust, but uses AGPL licensing which could be restrictive depending on deployment.
  - *pdfplumber*: Excellent table extraction, but heavier footprint.
- **Decision & Rationale**: `pypdf` is a pure-python, actively maintained library with a permissive BSD-3 license that easily integrates into our ingestion pipeline for extracting text and basic metadata (such as page lengths and counts).

---

## D-006: Deterministic Chunk ID Generation
- **Status**: Accepted
- **Context**: Document chunks must have consistent, predictable identifiers to avoid duplicated embedding computations across multiple pipeline runs.
- **Decision & Rationale**: Instead of random UUIDs, chunk IDs are generated using a SHA-256 hash of `document_id`, `chunk_index`, and the chunk's text `content`. This guarantees identical text pieces from identical documents receive identical IDs in vector storage.

---

## D-007: Pydantic Schema Validation for Structured Critic Output
- **Status**: Accepted
- **Context**: The Critic component must evaluate retrieval sufficiency and generation groundedness and return structured, categorical outputs for conditional routing in the Phase 3 LangGraph self-healing loop.
- **Alternatives Considered**:
  - *Raw string parsing / regex matching*: Fragile and prone to false negatives or parsing crashes.
  - *Unvalidated JSON dicts*: Silently misses missing or corrupted fields from LLM outputs.
- **Decision & Rationale**: We use strict Pydantic schemas (`CriticEvaluation`, `CriticVerdict`, `CriticFailureReason`) to validate the LLM's parsed JSON output. Malformed responses or missing keys raise explicit validation errors rather than silently propagating corrupted state into downstream graph nodes.

---

## D-008: Dependency Injection for LLM Inference Clients
- **Status**: Accepted
- **Context**: Generation and Critic components depend on Hugging Face inference APIs, but unit tests must run deterministically, fast, and offline without requiring active API tokens or network access.
- **Alternatives Considered**:
  - *Monkeypatching global SDK functions*: Brittle and prone to test order dependency issues.
  - *Separate MockGenerator / MockCritic classes*: Duplicates logic and causes interface divergence.
- **Decision & Rationale**: `Generator` and `Critic` constructors accept an optional `client: Optional[InferenceClient] = None`. When omitted in production, they instantiate the official `huggingface_hub.InferenceClient` using centralized `config.settings`. When provided in tests, a mock client can simulate completions and errors deterministically.

---

## D-009: Separation of Retrieval Failure and Correct Abstention
- **Status**: Accepted
- **Context**: The `Critic` node receives a generated answer and verifies whether it is grounded in the provided retrieved context. However, the Generator may output a "safe" abstention response (e.g., "The provided context is insufficient") if the retrieval was empty or irrelevant.
- **Alternatives Considered**:
  - *Treating abstention as a retrieval failure*: Maps to `FAIL/retrieval_insufficient`, which might trigger a generic reformulation loop without recognizing the model's safe behavior.
  - *Treating abstention as hallucination*: Inaccurate, as the model did exactly what it was instructed to do (not hallucinate).
- **Decision & Rationale**: Introduce a distinct `ABSTAIN` verdict in the `CriticVerdict` schema alongside `PASS` and `FAIL`. This ensures the system explicitly recognizes when the LLM safely defers due to lack of evidence, enabling distinct behaviors (e.g. fast-failing the graph gracefully) instead of conflating it with hallucination or standard retrieval gaps.

---

## D-010: Dedicated `recover_node` as the Single Recovery Entry Point (Phase 5)
- **Status**: Accepted
- **Context**: Phase 5 adds two distinct recovery actions — query reformulation (for `retrieval_insufficient`) and strict answer regeneration (for `generation_ungrounded`). A clean dispatch mechanism was needed.
- **Alternatives Considered**:
  - *Direct conditional edges from `critic` to both `reformulate` and `regenerate`*: Structurally simpler but conflates routing logic into `should_continue`, which already handles the PASS/FAIL/max-retries decision. Two distinct concepts in one function.
  - *Inline dispatch inside `should_continue`*: Would require `should_continue` to return 4 literals (`"end"`, `"reformulate"`, `"regenerate"`, `"recover"`), violating single-responsibility.
- **Decision & Rationale**: Introduce a lightweight `recover_node` that performs no state mutation but serves as a clean separation point. `should_continue` decides *whether* to recover (FAIL + below max_retries) and `route_recovery` decides *how* to recover (based on `failure_reason`). This keeps each function focused and testable in isolation. `route_recovery` explicitly raises `ValueError` on `None` or unknown failure reasons — no silent fallbacks.

---

## D-011: query_history for Reformulation Deduplication (Phase 5)
- **Status**: Accepted
- **Context**: The reformulation loop could theoretically produce the same query the LLM had already tried, leading to infinite loops with identical retrieval results.
- **Alternatives Considered**:
  - *Hashing queries and storing a set*: Correct but slightly more complex serialization through TypedDict state.
  - *Comparing only the previous query*: Only prevents immediate repetition, not repetition of earlier queries in a multi-step loop.
- **Decision & Rationale**: Add `query_history: List[str]` to `GraphState`, seeded with `[original_query]` at invocation. `reformulate_node` checks that the LLM's candidate is not already in `query_history` before accepting it. If it is a duplicate or empty, it falls back to `original_query`. The LLM also receives the full history in its prompt (via `format_reformulate_messages`), reducing the chance of repetition at the source.
