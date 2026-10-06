# TASK-14 ingestion contract

Baseline: `87ad47a`. Status: implementation in progress.

## Decisions

- Splitters return original contiguous UTF-8 byte intervals in input `Document.Content`, in source order. `Chunk.InputSpan` names that input interval, not a retained representation. `SourceMapping` is sliced from the supplied document mapping; zero mapping stays absent. Source supports remain revision-bound and owned.
- Recursive and Markdown carry ranges throughout separator splitting and fixed overlap. Whitespace trimming adjusts ranges. Positions are never recovered by searching finished fragment text. Chunk sizes/overlap are rune counts, ranges are bytes.
- SentenceSegmenter becomes a context-aware byte-range port. Custom segmenters return valid ordered non-overlapping UTF-8 spans. Semantic groups retain the contiguous original range, including separators, rather than reconstructing text from sentence strings.
- Semantic cosine uses standard finite arithmetic with shape/zero-norm checks. Arbitrary finite nonzero float32 scales are accepted; no implicit unit-normalization requirement. All splitters reject cancellation before callbacks and inspect it in long loops.
- Chunk original content/mapping and generated Context remain separate. Projection requires an explicit host `IndexText` callback. `ProjectedDocument.Document` always retains original content/mapping for retrieval. Separate `IndexText`/`IndexMapping` feed the host encoder/index field. Unchanged index text retains original mapping. Changed index text receives derived/support-only mapping when supports exist; it never inherits exact coordinates. Unsupported provenance stays absent. BYOT metadata/identity projection remains host-owned.
- MapOrdered uses a child context. Any callback failure cancels siblings and new dispatch; all started cooperative callbacks and dispatcher finish before return. Ordinary failure returns no result slice; caller cancellation may retain the internal ordered partial result shape. Callback/host reference metadata is immutable unless explicitly cloned by the host. Cancellation is cooperative; no forced termination of arbitrary functions.
- Graph ingestion replaces Stage/Provider/raw Upsert facade with typed extraction → resolution → materialization composition. The result is an owned planned manifest + payload, handed to the existing lifecycle executor explicitly by the host. The orchestrator performs no index writes or publication. Raw graph storage capabilities remain independent.

## Mandatory requirements

Each row is mandatory; completeness requires every assertion in the row. Reviewer also checks the original task scope.

| ID | Requirement | Evidence |
|---|---|---|
| I01 | Original order/Index/ID, regression Recursive(8,2) long-first/TAIL | range regression |
| I02 | UTF-8/repeated text/separators/trimming/fixed overlap correct input ranges | adversarial/property cases |
| I03 | Source mapping sliced honestly; derived/support-only/unmapped precision preserved | mapping tests |
| I04 | Explicit index text policy, Context cannot become an exact source quote | projection regressions |
| I05 | Scale-invariant finite cosine, NaN/Inf threshold/vectors/shape/zero norm reject | numeric tests |
| I06 | Cooperative callbacks cancelled/joined, no new work after cancel, no goroutine outlives return | barriers/counters and race |
| I07 | Early/mid-loop cancellation, context checked before/after injected callbacks | deterministic tests |
| I08 | Split→project→index→retrieve→resolve returns original revision/span | actual integration |
| I09 | Revocation/deletion/new revision cannot resolve latest as old source | negative integration |
| I10 | Strict graph stage composition + actual managed lifecycle handoff | full example/integration |
| I11 | Partial stage failure never claims successful publication | adversarial lifecycle test |
| I12 | Consumers/PDF/layout/examples/docs updated, replaced APIs removed, BYOT and library boundaries retained | module tests/lint/diff |

No provider retry, model auto-selection, ontology, OCR, source retention storage or scheduling is introduced. Embedding port replacement belongs to TASK-15; this task updates its current consumer semantics.
