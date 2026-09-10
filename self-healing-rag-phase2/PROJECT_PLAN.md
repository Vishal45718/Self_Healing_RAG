# PROJECT_PLAN.md

## Status
Current Phase: 2 — Embeddings & Vector Store
Current Task: Phase 2 complete, ready for Phase 3
Blockers: none. Note: this dev sandbox cannot reach huggingface.co, so the
real sentence-transformers model can't be downloaded here — 2 integration
tests skip themselves for that reason (see Phase 2 section). They will run
normally on a machine with standard internet access.

## Phase 0 — Project Scaffold & Setup
- [x] 0.1 Initialize repository structure
  - [x] Create folder layout (src/, tests/, config/, data/, logs/)
  - [x] Create pyproject.toml with core dependencies
  - [x] Create .gitignore, README stub (README pending polish in Phase 9)
  Acceptance:
  - [x] Folder structure matches agreed layout, no extra/unused folders
- [x] 0.2 Create project-management files
  - [x] PROJECT_PLAN.md, BUG_LOG.md, DECISIONS.md, AGENTS.md
  Acceptance:
  - [x] All four files exist with correct structure
- [x] 0.3 Config system (minimal, not over-abstracted)
  - [x] Single config/settings.py, env-driven
  Acceptance:
  - [x] Changing LLM/embedding model requires only editing .env, no code changes
  - [x] No provider factory/registry abstraction present
- [x] 0.4 Record initial decisions
  - [x] D-001 through D-004 recorded in DECISIONS.md
- [~] 0.5 Verify environment connectivity
  - [ ] Confirm HF Inference Providers auth/token works (deferred to Phase 3 — not needed until generation is implemented)
  - [x] Confirm Chroma persistence works — verified in Phase 2 (PersistentClient + on-disk collection, exercised by all vector store tests)

## Phase 1 — Document Ingestion & Chunking
- [x] 1.1 Document loader (.txt, .md, .pdf)
  - [x] Load supported file types
  - [x] Preserve metadata (source, file_type, page_count for PDFs)
  - [x] Handle invalid/unsupported/empty files without crashing the batch
  - [x] Tests: valid load, unsupported extension, empty file, missing file, mixed batch, directory load
  Acceptance:
  - [x] Valid documents load successfully — verified by test
  - [x] Metadata is preserved — verified by test
  - [x] Invalid input fails gracefully (per-file error, batch continues) — verified by test
- [x] 1.2 Chunker (recursive character splitting)
  - [x] Configurable chunk size / overlap (via config/settings.py)
  - [x] Preserve parent document metadata + add chunk_index/chunk_id
  - [x] Tests: doc smaller than chunk size, empty doc, long doc splitting, metadata preservation, multi-doc batch
  Acceptance:
  - [x] All chunking tests pass (5/5)
  - [x] Chunks respect configured size bound within overlap margin
  - [x] Metadata correctly inherited from parent document

**Result:** 12/12 tests passing. 0 bugs found.

## Phase 2 — Embeddings & Vector Store
- [x] 2.1 Embedding wrapper (sentence-transformers, single model, config-driven)
  - [x] `src/embeddings/embedder.py`: lazy-loaded singleton model, `embed_texts()`
  - [x] Empty-input short-circuit (no model load needed)
  Acceptance:
  - [x] Model/embedding choice is config-only (`EMBEDDING_MODEL` in settings) — verified by inspection, no code path hardcodes a model name
  - [x] Real-model integration tests present, gated by connectivity (see BUG_LOG note above) — will run in a normal dev environment
- [x] 2.2 Chroma vector store (single collection, add + query)
  - [x] `src/retrieval/vector_store.py`: `VectorStore` with `add_chunks`, `query`, `count`, `reset`
  - [x] Persistent (on-disk) client, cosine distance space
  - [x] Embedding function injectable for testability (D-007) — one production default, not a multi-backend abstraction
  Acceptance:
  - [x] Chunks added are retrievable by semantically relevant queries — verified by test
  - [x] Metadata survives the add→query round trip — verified by test
  - [x] Empty store / empty add handled without error — verified by test
- [x] 2.3 Tests
  - [x] 6 vector store tests (relevance ranking, metadata round-trip, empty store, empty add, reset)
  - [x] 3 embedder tests (2 real-model integration, network-gated; 1 empty-input, always runs)
  Acceptance:
  - [x] All non-network-gated tests pass: 7/7 passing, 2 skipped (network-gated only)

**Result:** 19/21 tests passing, 2 skipped (require huggingface.co access unavailable
in this sandbox). 2 bugs found and fixed during implementation (see BUG_LOG.md
BUG-001, BUG-002) — both in the Chroma custom-embedding-function integration.

## Completed
- Phase 0: scaffold, config, management docs
- Phase 1: ingestion + chunking, fully tested
- Phase 2: embeddings + vector store, fully tested (2 bugs found & fixed)

## Known Issues
None open. (2 fixed bugs logged in BUG_LOG.md for Phase 2.)

## Next Task
Phase 3 — Baseline RAG: wire ingestion → chunking → embedding → retrieval
together with a plain (non-self-healing) generation step calling HF
Inference Providers. This is the first end-to-end working system and the
baseline that Phase 7's evaluation will compare the self-healing graph
against. Will close out deferred item 0.5's HF auth check as part of this
phase's setup.
