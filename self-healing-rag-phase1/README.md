# Self-Healing RAG

A RAG system that critiques its own answers for groundedness and
self-corrects via query reformulation and bounded retries, built as a
cyclic LangGraph workflow. Includes a baseline (non-self-healing) RAG
for side-by-side evaluation.

See PROJECT_PLAN.md for current status, DECISIONS.md for architectural
rationale, and AGENTS.md for agent working instructions.
