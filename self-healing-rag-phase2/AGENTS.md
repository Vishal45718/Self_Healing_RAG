# Instructions for Antigravity

Before modifying anything:
1. Read this file and PROJECT_PLAN.md.
2. Inspect the current repository state — do not assume prior context is accurate.
3. Identify the single current task from PROJECT_PLAN.md ("Current Task").
4. Make only the changes required for that task.
5. Run the relevant tests (not the full suite unless explicitly asked).
6. Fix failures before proceeding.
7. Update PROJECT_PLAN.md checkboxes to reflect actual, verified state.
8. Log any bug found in BUG_LOG.md, including root cause and a regression test.
9. Report: what changed, what was tested, what's next.

## Hard Rules
- No unrelated refactors — touch only what the current task requires.
- No marking a task or acceptance criterion complete without a passing test proving it.
- No implementing future phases ahead of the current one.
- Config changes (model/provider swaps) belong in `config/settings.py` / `.env`
  only — do not introduce new abstraction layers to "future proof" this.
- Keep Chroma as the only vector store implementation; keep HF Inference
  Providers as the only LLM provider. Do not add multi-backend abstraction
  unless PROJECT_PLAN.md explicitly adds that as a task.
