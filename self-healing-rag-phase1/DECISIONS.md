# DECISIONS.md

## D-001: Why LangGraph
Decision: Use LangGraph for the retrieve→generate→critic→reformulate workflow.
Rationale: The workflow is a cyclic state machine (bounded retry loop with
conditional routing), which is exactly what LangGraph models natively via
typed state + conditional edges. A plain chain (e.g. sequential LCEL) can't
express the cycle cleanly.
Alternatives Considered: Hand-rolled while-loop orchestration — rejected,
would reimplement what LangGraph already provides (state, routing, retry
bounding) with less clarity for a reviewer reading the code.
Date: Phase 0

## D-002: Why Chroma
Decision: Use Chroma as the single vector database for this project.
Rationale: Embedded/local, zero infra to stand up, good Python ergonomics —
appropriate for a resume-scale project. We are not building multi-vector-DB
abstraction; retrieval logic lives behind one small interface so a future
swap is possible, but only Chroma is implemented now.
Alternatives Considered: FAISS (more manual index management), Qdrant
(requires running a service) — both unnecessary at this scale.
Date: Phase 0

## D-003: Why Hugging Face Inference Providers (no Ollama)
Decision: Use HF Inference Providers via `huggingface_hub.InferenceClient`
for the LLM.
Rationale: This is HF's current, actively maintained multi-provider serving
layer (superseding the old single-provider serverless Inference API), keeps
the project cloud-based and reviewable without requiring local GPU/Ollama
setup, and keeps model/provider swappable via config.
Alternatives Considered: Ollama — rejected per project requirement to stay
API-based and not depend on local model hosting.
Date: Phase 0

## D-004: Why sentence-transformers for embeddings
Decision: Use local `sentence-transformers` (all-MiniLM-L6-v2 default) for
embeddings rather than a hosted embedding API.
Rationale: Free, fast, deterministic, no network dependency during heavy
iteration (ingestion/chunking/retrieval will be re-run constantly during
development). Swappable via config if a hosted embedding model is wanted
later.
Alternatives Considered: HF-hosted embedding endpoint — adds latency/cost
for no benefit at this stage.
Date: Phase 0

## D-005: Chunking strategy
Decision: Recursive character splitting (paragraph → sentence → hard
character boundary) with configurable size/overlap.
Rationale: Simplest strategy that behaves reasonably on general prose
without extra dependencies; sufficient for a portfolio-scale corpus. No
semantic/agentic chunking — would be over-engineering for this project's
scope.
Alternatives Considered: Fixed-size character chunking only (loses sentence
boundaries) — recursive splitting was chosen for only marginally more
complexity with meaningfully better chunk boundaries.
Date: Phase 1

## D-006: retrieval_insufficient vs generation_ungrounded
Decision: The critic will classify failures into two distinct reasons
rather than a single FAIL verdict.
Rationale: The correct recovery action differs — poor retrieval calls for
a broadened/rephrased query, while an ungrounded generation from adequate
context calls for a stricter grounding prompt on the same context. Collapsing
these would make reformulation blind to which problem it's solving.
Alternatives Considered: Single generic FAIL — rejected, discussed in
Phase 0 scoping as insufficient for meaningful self-healing behavior.
Date: Phase 0 (recorded ahead of Phase 4 implementation)
