# TASK-14 correctness acceptance

Reviewer: `/root/task14_correctness`, independent from implementation.
Candidate: `d67ddf6b79ef66cb9c626fb337478e257b6c8515736b8a12df24aa92e1401dff`; baseline `87ad47ae3c64f2babc0bf7f2e6c9c5ed0ee072a1`. All candidate file hashes matched.

**Accepted: no confirmed open defects.**

Verified range propagation through Recursive/Markdown/Semantic, finite cosine for non-normalized float32, ordered segmenter ranges, revision-bound mapping and derived contextual precision. Verified MapOrdered cancellation and joining; retrieval callers retain their own partial-success semantics by returning branch errors within results. Verified typed extraction/resolution/materialization composition and explicit lifecycle handoff; interrupted staging cannot publish. BYOT and raw graph storage remain independent, without added runtime, ontology, retries or retention storage.

Independent checks passed: `go test -race ./chunking ./internal/parallel ./graphingest/... ./documents ./retrieval`; deterministic cancellation/barrier chunking tests with `-race -count=25`; internal parallel tests with `-race -count=10`.

Final additional actual PDF race log reviewed: parser/layout, projection/index/retrieve/scoped resolve, retained revision and durable publication all executed with PASS, without SKIP or race failures. Candidate hashes rechecked unchanged; acceptance reaffirmed. Initial optional SKIP log is not execution evidence.

Callbacks remain cooperative: the contract does not promise forced termination of host functions ignoring context.
