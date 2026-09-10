# Self-Healing RAG: Master Project Plan

This document tracks the phased milestones, tasks, acceptance criteria, and completion status for the Self-Healing RAG system.

---

## Phase Overview

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | **Project Scaffold & Setup** | **COMPLETED** |
| Phase 1 | Ingestion & Vector Storage Pipeline | Pending |
| Phase 2 | Retrieval, Generation & Critic Components | Pending |
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

## Next Phase: Phase 1 — Ingestion & Vector Storage Pipeline

- Task 1.1: Document loading & parser support
- Task 1.2: Semantic chunking strategies
- Task 1.3: Local embedding pipeline with `sentence-transformers`
- Task 1.4: Chroma storage and persistent index management
