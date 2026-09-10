# Agent Guidelines & Operational Protocols

This document establishes operating conventions, architectural boundaries, and coding standards for all AI agents working on the Self-Healing RAG codebase.

---

## 1. Operating Protocols

1. **Phase Discipline**: Only execute tasks belonging to the currently active phase. Never jump ahead to implement components of subsequent phases (e.g., do not implement retrieval or critic nodes during scaffolding).
2. **Inspect Before Changing**: Always inspect existing files and understand the current system state before introducing modifications. Never assume an empty workspace or blindly overwrite existing files.
3. **Verified Evidence**: Never fabricate test results, execution outputs, or acceptance confirmations. Always execute verifiable commands and capture output.
4. **Defect Tracking**: If a genuine defect or blocker is identified, immediately record it in [BUG_LOG.md](file:///home/jonsnow/Desktop/Self_Healing_RAG/BUG_LOG.md) before attempting workarounds.
5. **Plan Alignment**: Update [PROJECT_PLAN.md](file:///home/jonsnow/Desktop/Self_Healing_RAG/PROJECT_PLAN.md) strictly as tasks meet their formal acceptance criteria.

---

## 2. Security & Credentials

1. **Zero Secret Leakage**: Never hardcode API keys, tokens, or credentials in source code, configuration files, commit messages, or test fixtures.
2. **Environment Variables**: Load all sensitive configurations from `.env` via `config.settings.Settings`.
3. **No Printing Secrets**: Never log or print sensitive values (e.g. `HF_TOKEN`). When logging or reporting status, only verify and output boolean presence (e.g. `token_set: bool`).
4. **Git Hygiene**: Keep `.env` and local database files (such as `data/chroma/`) tracked in `.gitignore`.

---

## 3. Architecture & Code Quality

1. **Simplicity Over Abstraction**: Do not create speculative abstractions, factory registries, dynamic plugin managers, or unnecessary wrappers until explicitly mandated by project requirements.
2. **Standard Interfaces**: Use official SDK interfaces (e.g. `huggingface_hub.InferenceClient`, `chromadb.PersistentClient`, `sentence_transformers.SentenceTransformer`).
3. **Modern Python**: Write clean, typed Python 3.11+ code with explicit type annotations and docstrings.
4. **Test Cleanliness**: Do not commit ad-hoc, throwaway smoke tests to the permanent `tests/` directory. Permanent tests should be deterministic, fast, and structured for `pytest`.
