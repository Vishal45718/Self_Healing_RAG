# BUG_LOG.md

## BUG-001
Phase: 2
Severity: Medium
Description: `VectorStore.query()` raised `AttributeError: '_EmbeddingFunctionAdapter'
object has no attribute 'embed_query'` when querying a collection.
Root Cause: Custom embedding function was a duck-typed class, not a subclass
of Chroma's `chromadb.api.types.EmbeddingFunction`. Chroma's query path calls
`embed_query()` on the embedding function directly (distinct from `__call__`,
used for documents), and only the base class supplies a default
`embed_query -> __call__` fallback. Duck-typing `__call__` alone was
insufficient.
Fix: Made `_EmbeddingFunctionAdapter` subclass `chromadb.api.types.EmbeddingFunction`
instead of being a plain class.
Status: Fixed
Regression Test: `tests/retrieval/test_vector_store.py::test_add_and_query_returns_relevant_chunk`
(and the other query-path tests in the same file) would fail without the fix.

## BUG-002
Phase: 2
Severity: Low
Description: Deprecation warning on every collection creation:
"Failed to register embedding function: _EmbeddingFunctionAdapter.name()
missing 1 required positional argument: 'self'".
Root Cause: Chroma's embedding-function registry calls `name()` as an
unbound/static lookup during config serialization; defining it as a regular
instance method fails that call path even though direct instance calls work
fine.
Fix: Changed `name()` to a `@staticmethod`, matching the pattern used by
Chroma's own built-in embedding functions (e.g. `DefaultEmbeddingFunction`).
Status: Fixed
Regression Test: Warning no longer appears in `pytest -v` output for
`tests/retrieval/test_vector_store.py`.

<!--
## BUG-001
Phase:
Severity: Low/Med/High/Critical
Description:
Root Cause:
Fix:
Status: Open/Fixed/Verified
Regression Test:
-->
