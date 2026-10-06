# Ragy: tensor / lexical / durablefs review

Scope: `tensor`, `tensor/query`, `tensor/persistent`, `lexical`, `lexical/managed`, `internal/durablefs`; reviewed working tree b63d5e19. No source changes. Targeted overlay reproduction only; full suite is parent responsibility.

## Confirmed defect

### T-F01 — P2: BM25Snapshot dispatches metadata callbacks after read cancellation

Locations: `lexical/bm25.go:398–420`, especially 409–415; `lexical/snapshot.go:20–21` explicitly promises “Binding freshness gates every callback”. Applies to raw BM25 and to snapshot/managed paths that call `BM25Index.retrieve`.

`filterScoredDocs` receives no context/read binding and calls `retrieval.MatchDocument(codec, doc, cond)` for the complete candidate list. If first Encode cancels context (or revokes host authority), later candidates still enter host codec. Final delivery correctly hides all results, but it cannot undo already performed host callbacks. This is a callback-boundary defect, NOT successful result leakage. A malicious callback is not required: ordinary cancellation/revocation during the first slow callback has same result.

Reproduced with an immutable scoped snapshot containing IDs a/b, both matching required tenant. Codec delegates ordinary JSON encoding, but cancels context on first query invocation. Observed: `calls=2 error=read protection failed results=0`, errors.Is(context.Canceled)=true.

Evidence: `/tmp/ragy-tensor-review/repro_test.go`, `/tmp/ragy-tensor-review/overlay.json`; command from ragy root:
`GOCACHE=/tmp/ragy-review-go-cache go test -overlay=/tmp/ragy-tensor-review/overlay.json -run '^TestReviewSnapshot' -v ./lexical`.
Diagnostic test PASS (asserts current faulty behavior); preliminary run failed only because diagnostic compared wrapped error by equality, then changed to errors.Is.

Fix: pass context and binding into filtering, check before and after every metadata callback; protection error must discard partial candidates and take precedence over ordinary callback failure. Preserve raw mutable index documented ownership contract. Do not add retries or an authority cache.

AAA acceptance: Arrange scoped snapshot with two candidates, custom codec first call cancels context / revokes authority. Act Retrieve. Assert only one Encode, no clone/delivery callback afterward, empty protected result and errors.Is cancellation / denial. Include managed adapter cache hit and miss. Add cancellation during first Encode returning ordinary error; protection wins.

## Architectural / naming / unusual behavior decisions (not additional confirmed bugs)

1. **Tensor is bounded only by candidate count**, not full compute. `tensor/maxsim.go:83–104`, `RerankOptions`: token count × query tokens × dimension determines work. Persistent payload bytes bound stored tensors, public MaxSim/Rerank accept arbitrarily large caller slices. Document resource model; optionally explicit token/operation budget in a lower-level scoring API only when needed, not scheduler/timeouts owned by ragy.
2. **Cancellation during tensor validation**: `Embedding.Validate` scans full matrix without ctx, `Rerank` validates every candidate before scoring. Check ctx between candidates/token rows for responsiveness. Not a validation correctness bug and no realistic float64 overflow found for finite float32 input.
3. **Native MaxSim negative and >1 scores correct**; preserve declared space and metric. Never clamp or normalize implicitly. `Embedding` comment currently says “normalized token matrix” although Dot is explicitly supported; clarify metric-dependent normalization.
4. **Candidate loss intentionally observable**, exact-within-candidates not exhaustive. Keep `CandidateIDs` and native ranks, no hidden full-index fallback. `tensor/query.Search` is appropriate retrieval composition, not an agent harness.
5. **Duplicate source references are explicitly deduplicated** in `tensor/query/search.go:project`; candidate budget counts backend docs while scoring counts unique refs. Document this difference and first-occurrence order, rather than replace with arbitrary dedup or silently claim all supplied docs were scored.
6. **Catalog work is broader than candidate budget**: tensor persistent reads and validates each pinned catalog, then scans descriptors to find explicit candidates. TopK/CandidateBudget bound payload/scoring, not catalog memory and iteration. Document this and profile actual selected-catalog count; avoid suggesting ANN scalability for the reference file adapter.
7. **MaxRecords overloaded** in persistent adapter: per-manifest Stage record count, query CandidateLimit, inventory key count and total observed records (`inventory_unix.go:41,75`). Split naming/config bounds if independent tuning needed. Lexical InventoryObserver(maxEntries,maxRecords) is clearer precedent. Existing limits fail explicitly, not corrupting data.
8. **Lexical cache bound counts snapshots**, not bytes. Each snapshot can duplicate a large filtered corpus; concurrent same-key misses independently build it. MaxCachedSnapshots is resident count only and concurrent transient builders can exceed it. Document; consider simple singleflight or byte budget only after measurement, no general cache framework.
9. **Lexical generation invalidates all scope caches** on any Stage/Cleanup. Simple safe choice; don't add fine-grained dependency graph absent measured bottleneck. Inventory checks on cache hits remain required.
10. **Repeated admission/build passes**: managed build clones admitted records, NewBM25Snapshot clones again and rechecks mandatory metadata; indexed raw docs then filtered again at query. Some repetition enforces ownership/freshness; consolidate only with executable fence tests and an internal validated constructor, never exported bypass flag.
11. **BM25 B=0 cannot mean no length normalization** because zero selects .75; documented explicitly, not hidden bug. Clean break could use optional config / explicit default constructor to support mathematically valid B=0. Same question for K1=0. Avoid pointers everywhere merely for stylistic uniformity.
12. **Raw BM25 mutable metadata is host-owned**, documented README. Don't claim thread safety covers caller mutations to map/pointer metadata. Snapshot/managed are the owning choices. Host tokenizer/codec must be stable/concurrent-safe; removal retokenizes original document and can fail if callback changes behavior.
13. **Cancellation inside BM25 CPU loops**: scoring/sorting all matches does not consult ctx; final gate correct but expensive common-term queries continue after timeout. Introduce periodic checks where cost merits it; TopK remains output count, not exhaustive work bound.
14. **Threshold semantics belong terminal composition** (`retrieval/postprocessor_chain.go:157–210`); raw BM25 accepts options but does not apply threshold, tensor persistent rejects threshold. Document consistent backend/terminal distinction; do not classify as another definite bug without contract decision. Explicitly reject irrelevant graph/vector options or document pass-through behavior.
15. **Persistent filesystem initialization** uses MkdirAll(root) but no ancestor-directory fsync (`storage_unix.go:110`); install fsyncs target root and files. Process-crash suite does not establish power-loss durability for newly created root hierarchy. Clarify initialization durability / require pre-provisioned durable root or implement synced mkdir chain if power-loss guarantee intended. Do not label real observed data loss: no crash-at-hardware test performed.
16. **Nonblocking flock intentional**, contention returns lifecycle.ErrConflict, no hidden retry. Name Lock documentation “process-owned lock” too broad on Linux flock open-file-description semantics; describe close-owned handle and supported local filesystem profile. Don't extend to network FS or platform emulation fallback.
17. **Persistent read failures are deliberately redacted** (`durablefs.QueryPayloadError`): preserves cancellation/deadline/protocol, ordinary errors -> unavailable. Keep privacy; expose safe observer categorization if diagnostics needed, never raw filesystem path in exported error.
18. **selectedManifest omits Retired flag** whereas managed confirmedVersion checks it. Current artifact inventory stripping plus eligibility after cleanup normally blocks retired payload anyway, so no bypass reproduced. Add explicit retired rejection for contract clarity and parity, plus focused test. Don't count as separate proven defect.
19. **Local root ownership is explicit boundary**. Files open through trusted host-controlled directories, no O_NOFOLLOW/root capability hardening. Not a traversal vulnerability in normal model; if threat model changes, use rooted filesystem APIs as distinct storage profile, not generic path sanitization patches.
20. **dense/tensor persistence duplication**: lifecycle/catalog/cleanup/durablefs mechanics overlap. A small internal generic helper may remove divergence (Retired/error mappings/bounds), but avoid public generic storage engine or callback pipeline that hides exact inventory gates. Keep type-specific embedding validation/scoring straightforward.

## Positive review conclusions

- Strong explicit pinned-publication contract; catalog payload checksum, identity and exact inventory admission before actual payload reader; no latest-version substitution.
- Namespace/target-derived root and shared target flock serialize Stage/Cleanup/retained reads; fsync/rename crash tests exist; failed stages remain explicit and inspectable.
- Mandatory attributes matched before payload reads; host Decode/CloneMeta callbacks in tensor loadCandidate are guarded before/after.
- Scores retain metric/space identity; ties stable, candidate budgets fail rather than silently truncate input.
- Managed lexical correctly documents volatility, does not pretend durable lifecycle ledger reconstructs missing payloads.
- Cache key incorporates binding/predicate/inventory/generation and rechecks ledger on hit; no obvious cross-scope result cache leak found.
- Reference BM25/native MaxSim/file adapters fit ragy retrieval boundaries. Model training, embedding service execution policies, retention scheduling, distributed transactions and agent workflows remain host/other libs.
