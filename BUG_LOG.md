# Bug Log

This document records defects, issues, and unexpected behaviors encountered during the project lifecycle, along with diagnosis, root causes, and resolutions.

## Bug Registry

| ID | Status | Severity | Phase | Component | Description | Root Cause | Resolution |
|----|--------|----------|-------|-----------|-------------|------------|------------|
| BUG-001 | Fixed | Medium | Phase 2 | VectorStore | ChromaDB raises ValueError when upserting chunks with empty metadata `{}`. | `VectorStore.upsert` passed `{}` when `chunk.metadata` was empty, which Chroma rejects. | Include `document_id` and `chunk_index` from `Chunk` in the upsert metadata dict so it is never empty. |

## Severity Levels

- **Critical**: Blocks all progress, causes data corruption, or exposes sensitive credentials.
- **High**: Major functionality broken with no immediate workaround.
- **Medium**: Incorrect behavior with a known workaround.
- **Low**: Minor styling, typing, or non-blocking cosmetic defects.
- **Info**: Recorded observation or benign anomaly.
