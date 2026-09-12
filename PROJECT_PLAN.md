# Self-Healing RAG: Master Project Plan

This document tracks the phased milestones, tasks, acceptance criteria, and completion status for the Self-Healing RAG system.

---

## Phase Overview

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | **Project Scaffold & Setup** | **COMPLETED** |
| **Phase 1** | **Ingestion & Vector Storage Pipeline** | **COMPLETED** |
| **Phase 2** | **Retrieval, Generation & Critic Components** | **COMPLETED** |
| **Phase 3** | **LangGraph Self-Healing Feedback Workflow** | **COMPLETED** |
| **Phase 4** | **End-to-End Evaluation & Hardening** | **COMPLETED** |
| **Phase 5** | **Reformulation & Strict Regeneration** | **COMPLETED** |
| **Phase 7** | **Evaluation & Baseline Comparison** | **COMPLETED** |

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

---

## Phase 4: End-to-End Evaluation & Hardening (Refinement)

### Tasks & Status
- [x] **4.1 Implement Empty Retrieval Shortcut**
  - [x] Critic fast-fails with `retrieval_insufficient` if context is empty without calling LLM.
- [x] **4.2 Implement Correct-Abstention Shortcut**
  - [x] Critic fast-fails with `ABSTAIN` if generator safely abstains without calling LLM.
  - [x] Updated `CriticVerdict` schema to include `ABSTAIN` as a successful safe outcome.
- [x] **4.3 Fix Phase 3 Graph Bug**
  - [x] Fixed `critic_node` in `graph.py` passing `generation=...` instead of `answer=...`.
- [x] **4.4 Add Tests for Shortcuts**
  - [x] Empty context and abstention shortcuts tested and verified to bypass `InferenceClient`.
- [x] **4.5 Run and Verify Test Suite**
  - [x] Verified full regression suite passes (see 4.6 for integration test details).
- [x] **4.6 Harden Integration Test Skip Logic**
  - [x] Extended `test_integration_hf.py` to skip (not fail) when the configured HF token lacks Inference Provider permissions (HTTP 403).
  - [x] Implemented `_check_runtime_error_for_auth_skip()` helper that inspects the `__cause__` chain and only converts the exact `HfHubHTTPError` 403 condition to a `pytest.skip()`.
  - [x] All other `RuntimeError` causes remain genuine test failures.
  - [x] Final suite result: **73 passed, 1 skipped, 0 failed**.
  - [x] The 1 skipped test is `test_live_hf_generation_and_critic_pipeline` — an environment/credential limitation (HF token missing Inference Provider permission scope), confirmed as NOT a code defect.

---

## Phase 5: Reformulation & Strict Regeneration

### Objectives
Implement the two recovery actions for the self-healing feedback loop: query reformulation
(for `retrieval_insufficient`) and strict answer regeneration (for `generation_ungrounded`).
Add a single `recover` routing entry-point that dispatches to the correct action.
Guarantee no query is repeated and that `generation_ungrounded` never triggers re-retrieval.

### Tasks & Status
- [x] **5.1 Extend `GraphState` with `query_history`**
  - [x] Added `query_history: List[str]` to `GraphState` in `src/schema.py`.
  - [x] Seeded with `[original_query]` in `SelfHealingRAG.invoke()`.
  - [x] Compatible with all existing Phase 3 tests.

- [x] **5.2 Implement `recover_node` as the single recovery entry point**
  - [x] Added `recover` node to the LangGraph graph.
  - [x] Conditional edge `critic → recover` on FAIL below max_retries.
  - [x] `route_recovery()` dispatches to `reformulate` or `regenerate` based on `failure_reason`.
  - [x] `route_recovery()` raises `ValueError` explicitly on `None` or unknown `failure_reason`.

- [x] **5.3 Implement `reformulate_node` with query-history deduplication**
  - [x] Passes `query_history` to `format_reformulate_messages` so the LLM sees all prior queries.
  - [x] Prevents duplicate queries: if LLM echoes a known query, falls back to `original_query`.
  - [x] Returns a valid, non-empty query on every code path.
  - [x] Updates both `current_query` and `query_history` in state.

- [x] **5.4 Implement `regenerate_node` (no re-retrieval)**
  - [x] Reuses `state["retrieved_chunks"]` — does NOT call `self.retriever.retrieve()`.
  - [x] Calls `format_regenerate_messages()` embedding critic reasoning + unsupported_claims.
  - [x] Falls back to prior answer on LLM error; never crashes the pipeline.
  - [x] Routes back to `critic` node after regeneration (not to `retrieve`).

- [x] **5.5 Add `format_regenerate_messages` prompt**
  - [x] New `REGENERATE_SYSTEM_PROMPT` with strict "UNGROUNDED" framing.
  - [x] Embeds the same retrieved context, critic reasoning, and specific unsupported claims.
  - [x] Updated `format_reformulate_messages` to accept `query_history` parameter.

- [x] **5.6 Comprehensive Phase 5 test suite (`tests/test_reformulation.py`)**
  - [x] AC1: reformulated query differs from all previous queries.
  - [x] AC2: first-failure case works with empty (seed-only) history.
  - [x] AC3: repeated/duplicate query from LLM is rejected; no duplicate in history.
  - [x] AC4: regeneration directive contains critic reasoning and unsupported claims.
  - [x] AC5: `generation_ungrounded` does NOT call retrieval again (retrieve called exactly once).
  - [x] AC6: both failure reasons route to the correct action.
  - [x] AC7: `None`/unknown `failure_reason` raises `ValueError` explicitly.
  - [x] AC8: `query_history` grows correctly across iterations; no duplicates.
  - [x] AC9: regeneration → critic → PASS terminates the workflow correctly.
  - [x] AC10: all GENERATION_UNGROUNDED retries call retriever exactly once.
  - [x] **22 new tests in `tests/test_reformulation.py` — all passing.**

- [x] **5.7 Updated Phase 3 regression test (`tests/test_graph.py`)**
  - [x] `test_graph_max_retries_reached` updated to use `RETRIEVAL_INSUFFICIENT`
    (the reason that triggers re-retrieval), consistent with Phase 5 routing semantics.
  - [x] All 4 existing graph tests passing.

---

## Definition of Done (Phase 5)

- [x] All six Phase 5 tasks implemented and tested
- [x] 22 new Phase 5 tests in `tests/test_reformulation.py` — all passing
- [x] Full regression suite: **95 passed, 1 skipped, 0 failed**
- [x] `query_history` correctly seeds, grows, and prevents duplicates
- [x] `recover_node` is the single entry point for all recovery actions
- [x] `generation_ungrounded` path never calls retriever more than once
- [x] All invalid `failure_reason` values rejected with explicit `ValueError`
- [x] No new phases implemented; no existing components redesigned
- [x] `DECISIONS.md` updated with D-010
- [x] No secrets leaked or printed

---

## Phase 7: Evaluation & Baseline Comparison

### Objectives
Objectively evaluate and compare the performance, quality, and cost overhead of Baseline RAG
versus Self-Healing RAG across a reproducible evaluation dataset covering direct, vague,
and unanswerable queries.

### Tasks & Status
- [x] **7.1 Baseline RAG Runner Component (`src/baseline/baseline_rag.py`)**
  - [x] Implemented `BaselineRAG` single-pass retrieval and generation pipeline.
  - [x] Accepts same `Retriever`, `Generator`, and `Critic` components.
  - [x] Structured output via `BaselineResult`.
  - [x] 7 unit tests in `tests/test_baseline.py` — all passing.

- [x] **7.2 Evaluation Schema & Dataset (`src/evaluation/schema.py`, `src/evaluation/dataset.py`)**
  - [x] Pydantic schemas for `EvalSample`, `ScenarioType`, `ExpectedBehavior`, `EvalExecutionResult`, `SystemMetrics`, `EvaluationReport`.
  - [x] Curated default evaluation corpus (3 documents) and benchmark dataset (9 queries: 3 direct, 3 vague/reformulation, 3 unanswerable).
  - [x] JSON dataset loader and validator `load_dataset_from_json()`.

- [x] **7.3 Evaluation Engine & Metrics Calculator (`src/evaluation/runner.py`, `src/evaluation/metrics.py`)**
  - [x] `EvaluationRunner` coordinates side-by-side execution on identical vector store.
  - [x] Metrics distinguish: critic pass, groundedness, retrieval sufficiency, correct abstention, recovery success, reformulation success, average retries/iterations, latency (ms), and total LLM calls.
  - [x] Strict recovery condition: recovery is only counted as successful if an initial failure was subsequently resolved to a PASS/ABSTAIN matching expected behavior.
  - [x] `calculate_metric_deltas()` computes comparative deltas and overhead ratios.

- [x] **7.4 Result Serialization & Summary Reporter (`src/evaluation/report.py`, `src/evaluation/run_eval.py`)**
  - [x] `save_report_json()` and `load_report_json()` for structured evaluation persistence.
  - [x] `generate_markdown_report()` generates clean comparison table with impact summary.
  - [x] CLI runner script `src/evaluation/run_eval.py` for standalone benchmark execution.

- [x] **7.5 Evaluation Harness Unit Tests (`tests/test_evaluation.py`)**
  - [x] Dataset loading tests, custom JSON tests, error handling tests.
  - [x] Deterministic math tests for metric formulas and recovery success conditions.
  - [x] Side-by-side runner execution tests with mocked components.
  - [x] Report generation and JSON serialization round-trip tests.
  - [x] 12 tests in `tests/test_evaluation.py` — all passing.

---

## Definition of Done (Phase 7)

- [x] Baseline RAG and Self-Healing RAG execute against identical vector store collections
- [x] Metrics accurately distinguish critic pass, groundedness, abstention, recovery, retries, latency, and LLM calls
- [x] 19 new Phase 7 tests across `tests/test_baseline.py` and `tests/test_evaluation.py` — all passing
- [x] Full regression suite: **114 passed, 1 skipped, 0 failed**
- [x] Structured JSON serialization and formatted Markdown reporting implemented
- [x] Zero secret leakage or credential exposure
- [x] `DECISIONS.md` updated with D-012
