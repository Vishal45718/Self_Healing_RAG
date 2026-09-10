# PROJECT_PLAN.md

## Status
Current Phase: 1 — Document Ingestion & Chunking
Current Task: Phase 1 complete, ready for Phase 2
Blockers: none

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
  - [ ] Confirm Chroma persistence works (deferred to Phase 2)

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

## Completed
- Phase 0: scaffold, config, management docs (0.5 partially deferred — see above)
- Phase 1: ingestion + chunking, fully tested

## Known Issues
None.

## Next Task
Phase 2 — Embeddings + Vector Store (sentence-transformers + Chroma, single
collection, no multi-backend abstraction). Will also close out deferred
item 0.5 (Chroma persistence check) as part of Phase 2 setup.
