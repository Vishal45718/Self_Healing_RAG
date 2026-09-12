# Bug Log

This document records defects, issues, and unexpected behaviors encountered during the project lifecycle, along with diagnosis, root causes, and resolutions.

## Bug Registry

| ID | Status | Severity | Phase | Component | Description | Root Cause | Resolution |
|----|--------|----------|-------|-----------|-------------|------------|------------|
| BUG-001 | Fixed | Medium | Phase 2 | VectorStore | ChromaDB raises ValueError when upserting chunks with empty metadata `{}`. | `VectorStore.upsert` passed `{}` when `chunk.metadata` was empty, which Chroma rejects. | Include `document_id` and `chunk_index` from `Chunk` in the upsert metadata dict so it is never empty. |
| BUG-002 | Fixed | High | Phase 3 | Graph | TypeError in `critic_node` due to incorrect keyword argument `generation=generation` passed to `Critic.evaluate()`. | The `Critic.evaluate()` method signature expects `answer=...`, but `graph.py` passed `generation=...`. | Changed the kwarg in `src/graph/graph.py` from `generation=generation` to `answer=generation`. Verified via `test_graph.py` regression suite. |
| BUG-003 | Fixed | Medium | Phase 4 | Integration Test | `test_live_hf_generation_and_critic_pipeline` was FAILING (not skipping) when the configured `HF_TOKEN` lacked Inference Provider permissions (HTTP 403 Forbidden). | `@pytest.mark.skipif` only guards against a missing token. When a token is present but lacks the "Make calls to the serverless Inference API" scope, the provider returns 403, which was propagated as a hard test failure rather than a skip. This is an environment/credential limitation, not a code defect. | Added `_check_runtime_error_for_auth_skip()` helper in `tests/test_integration_hf.py` that inspects the `__cause__` chain of `RuntimeError`. If and only if the root cause is `HfHubHTTPError` with `status_code == 403`, the test is converted to `pytest.skip()`. All other errors remain genuine failures. |

## Severity Levels

- **Critical**: Blocks all progress, causes data corruption, or exposes sensitive credentials.
- **High**: Major functionality broken with no immediate workaround.
- **Medium**: Incorrect behavior with a known workaround.
- **Low**: Minor styling, typing, or non-blocking cosmetic defects.
- **Info**: Recorded observation or benign anomaly.
