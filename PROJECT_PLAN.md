# Self-Healing RAG: Master Project Plan

This document tracks the phased milestones, tasks, acceptance criteria, and completion status for the Self-Healing RAG system.

---

## Phase Overview

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | **Project Scaffold & Setup** | **COMPLETED** |
| **Phase 1** | **Ingestion & Vector Storage Pipeline** | **COMPLETED** |
| **Phase 2** | **Retrieval, Generation & Critic Components** | **COMPLETED** |
| Phase 3 | LangGraph Self-Healing Feedback Workflow | Pending |
| Phase 4 | End-to-End Evaluation & Hardening | Pending |

---

## Phase 0: Project Scaffold & Setup

### Objectives
Initialize project structure, virtual environment, dependencies, configuration system, architectural documentation, and verify environment connectivity.

### Tasks & Status

- [x] **0.1 Initialize repository structure**
  - [x] Create folder layout (`src/`, `tests/`, `config/`, `data/`, `logs/`)
  - [x] Create `pyproject.toml` with agreed core dependencies
  - [x] Create `.gitignore` and `.env.example`
  - [x] Create `README.md` stub
  - *Acceptance*:
    - [x] `pip install -e .` succeeds cleanly
    - [x] Folder structure matches agreed layout
    - [x] No unnecessary/unused folders created

- [x] **0.2 Create project-management files**
  - [x] Create `PROJECT_PLAN.md`
  - [x] Create `BUG_LOG.md`
  - [x] Create `DECISIONS.md` (D-001 through D-004)
  - [x] Create `AGENTS.md`
  - *Acceptance*:
    - [x] All four files exist with correct structure

- [x] **0.3 Config system**
  - [x] Create `config/settings.py` using `pydantic-settings`
  - [x] Centralize HF provider, LLM model ID, embedding model ID, Chroma path, retry limits
  - [x] Support environment variables / `.env`
  - [x] Avoid unnecessary provider factories or plugin registries
  - *Acceptance*:
    - [x] Settings load correctly from environment / defaults
    - [x] Secrets are never hardcoded

- [x] **0.4 Record initial architectural decisions**
  - [x] D-001: Why LangGraph
  - [x] D-002: Why Chroma
  - [x] D-003: Why Hugging Face Inference Providers
  - [x] D-004: Why sentence-transformers
  - *Acceptance*:
    - [x] All decisions concise, include alternatives considered and rationales

- [x] **0.5 Verify environment connectivity**
  - [x] Verify Hugging Face authentication/inference connectivity
  - [x] Verify Chroma creates and persists a local collection
  - [x] Use temporary smoke-test code only; cleaned up afterward
  - *Acceptance*:
    - [x] HF inference/hub check succeeds
    - [x] Chroma persistence check succeeds
    - [x] No credentials leaked or printed

---

## Definition of Done (Phase 0)

- [x] Repository scaffold exists
- [x] `pyproject.toml` installs successfully
- [x] Management files exist
- [x] Configuration works
- [x] HF inference/hub connectivity verified
- [x] Chroma persistence verified
- [x] Relevant documentation updated
- [x] No Phase 1+ implementation added
- [x] No unresolved critical/high issues introduced

---

## Phase 1: Ingestion & Vector Storage Pipeline

### Tasks & Status
- [x] **1.1 Document loading & parser support**
  - [x] Implemented `DocumentLoader` supporting TXT, MD, and PDF.
  - [x] Handled missing/corrupt/unsupported files without crashing.
  - [x] Preserved useful metadata (PDF page information, file type, source).
- [x] **1.2 Semantic chunking strategies**
  - [x] Implemented `RecursiveCharacterChunker`.
  - [x] Configurable chunk size and overlap in `settings.py`.
  - [x] Deterministic chunk ID generation using `hashlib.sha256`.
  - [x] Tests written to verify overlap and metadata preservation.
- [x] **1.3 Local embedding pipeline with `sentence-transformers`**
  - [x] Implemented `LocalEmbedder` in `src/ingestion/embedder.py`.
  - [x] `embed_chunks(chunks)` returns one `list[float]` per chunk (dim=384).
  - [x] `embed_query(query)` returns a single `list[float]`.
  - [x] Empty-input guards raise `ValueError`.
  - [x] Embeddings are deterministic across runs.
  - [x] 11 tests in `tests/test_embedder.py` — all passing.
- [x] **1.4 Chroma storage and persistent index management**
  - [x] Implemented `VectorStore` in `src/ingestion/vector_store.py`.
  - [x] Uses `chromadb.PersistentClient` with cosine similarity.
  - [x] `upsert(chunks, embeddings)` is idempotent via deterministic chunk IDs (D-006).
  - [x] `query(query_embedding, n_results)` returns ordered results with id/content/metadata/distance.
  - [x] Error guards: empty upsert, length mismatch, empty-store query, invalid n_results.
  - [x] Persistence verified across separate client instances.
  - [x] 12 tests in `tests/test_vector_store.py` — all passing.

---

## Definition of Done (Phase 1)

- [x] All four ingestion tasks implemented and tested
- [x] 39 total tests passing (16 Phase 0/1.1/1.2 + 11 embedder + 12 vector store)
- [x] No unresolved critical/high issues
- [x] No Phase 2+ implementation added

---

## Phase 2: Retrieval, Generation & Critic Components

### Tasks & Status
- [x] **2.1 Retrieval component on top of VectorStore and LocalEmbedder**
  - [x] Implemented `Retriever` in `src/retrieval/retriever.py`.
  - [x] Accepts natural-language query and embeds via `LocalEmbedder`.
  - [x] Returns structured `RetrievalResult` list preserving ID, content, metadata, distance, and similarity.
  - [x] Configurable `top_k` (defaults to 5, validated `>= 1`).
  - [x] Empty vector store safely returns `[]` without raising unhandled exceptions.
  - [x] Invalid/blank query input raises `ValueError`.
  - [x] 8 tests in `tests/test_retriever.py` — all passing.
- [x] **2.2 Baseline generation with Hugging Face InferenceClient**
  - [x] Implemented `Generator` in `src/generation/generator.py`.
  - [x] Uses official `huggingface_hub.InferenceClient` (D-003, D-008).
  - [x] Structured output via `GenerationResult`.
  - [x] Gracefully handles missing HF credentials, empty context, inference failures, and malformed responses.
  - [x] Never silently falls back to an unrelated model.
  - [x] 9 tests in `tests/test_generator.py` — all passing.
- [x] **2.3 Baseline RAG prompt design**
  - [x] Created `src/generation/prompts.py` with `RAG_SYSTEM_PROMPT` and `format_rag_messages`.
  - [x] Instructs model to rely strictly on retrieved context, avoid unsupported facts, and indicate insufficient context.
- [x] **2.4 Structured Critic component**
  - [x] Implemented `Critic` in `src/critic/critic.py` and schemas in `src/critic/schema.py`.
  - [x] Evaluates retrieval sufficiency and generation groundedness strictly against evidence.
  - [x] Structured output `CriticEvaluation` with `PASS`/`FAIL` verdict and `retrieval_insufficient`/`generation_ungrounded` reasons.
  - [x] Pydantic schema validation (D-007) and robust JSON extraction handling markdown blocks.
  - [x] 10 tests in `tests/test_critic.py` — all passing.
- [x] **2.5 Comprehensive tests & live smoke test**
  - [x] Offline unit tests for retrieval, generation, and critic using deterministic mocks/fixtures.
  - [x] Live HF smoke test in `tests/test_integration_hf.py` (skips cleanly when `HF_TOKEN` is unset).
  - [x] Regression test for BUG-001 (empty chunk metadata upsert in `VectorStore`).
  - [x] 67 passed, 1 skipped across the complete test suite.

---

## Definition of Done (Phase 2)

- [x] All five Phase 2 tasks implemented and tested
- [x] 67 total tests passing (+ 1 skipped live integration test)
- [x] BUG-001 diagnosed, logged, fixed, and verified with regression test
- [x] No LangGraph graph, retry loops, or query reformulations implemented (strictly adhering to Phase boundary)
- [x] No secrets leaked or printed

---

## Next Phase: Phase 3 — LangGraph Self-Healing Feedback Workflow
