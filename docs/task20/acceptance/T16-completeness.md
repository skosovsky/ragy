# T16 independent completeness acceptance

Baseline `5775f4ee0b1aeb11feb634cbb480a670ace0c7ce`. Nonimplementing, read-only reviewer; counterpart acceptance report not read. **PASS: 5/5 criteria = 100%; 27/27 assigned source findings = 100%.** No missing required check or SKIP counted as PASS. Reviewed original master D44–D50 and tensor role items 01–20 against actual code, current guides, regression assertions and independently executed checks. User AGENTS instructions (Contract-first, BYOT, AAA) applied. T16 normative contract and consumer inventory are recorded in T16.md; implementation matches it.

| Criterion | Result | Evidence |
|---|---|---|
| T16.C01 | выполнено | All 27 dispositions below. Embedding.ValidateContext checks before/between/after rows; Rerank candidate admission and MaxSim pairs check context. Native Dot/NormalizedDot semantics preserved; actual negative -2 and raw Dot23 oracle assertions PASS. Token/dimension/CPU costs and indivisible row/sort work documented. |
| T16.C02 | выполнено | Candidate backend-document limits precede first-occurrence source.Reference dedup; pinned catalog discovery remains broader than payload scoring. All five MaxRecords units checked against Stage/readCatalog/query/inventory code. Cache resident-count/transient distinction retained. Optional copied BM25Parameters permits both zero parameters, nil defaults, invalid finite/range rejection and overflow suppression; all compiled consumers migrated and independently tested. |
| T16.C03 | выполнено | Stage/selected inventory/build/snapshot/delivery gates and separate metadata clones remain visible. Raw borrowing and host callback stability explicitly described. Cache keys retain generation/inventory; global invalidation and inventory confirmation on hits unchanged. Independent race includes actual mutation fence, stale build, bounded concurrent cache, callback miss/hit and clone revocation tests. |
| T16.C04 | выполнено | Both selectedManifest functions explicitly skip Retired, with live/retired/retired-duplicate parity tests. Durable guides require trusted preprovisioned roots/ancestors, distinguish process crash from hardware power loss, and explain unduplicated nonblocking local flock handles. QueryPayloadError preserves sentinel privacy contract. Type-specific persistence retained with rationale; narrow shared durablefs mechanics remain. |
| T16.C05 | выполнено | Independent fresh final affected seven-package race, including real process restart and each payload/catalog/installed/retired/removed child crash phase, PASS. Independent GOWORK=off conformance PASS. No CPU/cache/clone optimization occurred: algorithms unchanged, cancellation checkpoints added; no speed/allocation claim, so optimization before/after measurement is not applicable. |

## Independent verification

`go test -race -count=1 -v ./lexical ./lexical/managed ./tensor ./tensor/query ./tensor/persistent ./dense/persistent ./internal/durablefs`: exit0; [final race log](T16-completeness-final-race.log). Ran again after lint refactor and reviewed helper extraction/formatting. Earlier independently passing run retained in [initial race log](T16-completeness-race.log).

Both physical-crash parent tests execute and PASS all five actual subprocess phases. Both separate-process restart tests PASS. The unconfigured TestActualPhysicalCrashChild entry reports SKIP intentionally; this helper SKIP is not acceptance evidence. Parent tests spawn the helper with actual owned paths/environment and assert physical crash exit, reopen storage and verify recovery. No hardware power-loss claim.

From `examples/conformance`, `GOWORK=off go test -count=1 ./...`: exit0; [consumer log](T16-completeness-conformance.log). Graph comparison, recipe comparison and internal/task19 compiled migration consumers all PASS, alongside remaining conformance packages. Search for scalar K1/B consumers leaves only intentionally historical build-ignore docs/task12/audits/repro_bm25_nonfinite.go. Root lint final log inspected: T16-lint-fix.log reports 0 issues; preceding failed lint log is retained as attempt evidence, not PASS.

Whole-root legacy-word documentation blacklist failure previously recorded belongs T21; this scoped acceptance does not claim all-root green. Unrelated docs/task18/correctness 2.md excluded and untouched.

Refresh after nested consumer lint: exactly three Config literal whitespace layouts changed. Reversing only their multiline whitespace formatting reconstructed each previously accepted whole-file SHA256 exactly, proving no other byte or semantic changes. Final nested consumer lint log T16-consumer-lint-fix.log reports 0 issues. Existing independent full GOWORK=off conformance and affected race remain applicable; no redundant full rerun needed for verified whitespace-only formatting. The 26-file manifest below is refreshed to current bytes; prior three consumer hashes are superseded. Criteria/source conclusion remains 5/5 and 27/27 PASS.

## Source coverage

Every row below is verified against the original requirement and actual evidence, not merely presence of a trace entry.

| Source | Result | Verified result |
|---|---|---|
| D44 | выполнено (change) | Metric-dependent validation; cancellation rows/candidates/pairs; native -2/23 scores; explicit work units. |
| D45 | выполнено (retain) | Document budget before first-reference dedup; exact supplied universe, catalog scan and five distinct MaxRecords units documented. |
| D46 | выполнено (change) | Copied optional BM25 aggregate supports zeros/defaults; cache count/transient/generation and borrowed metadata boundaries explicit. |
| D47 | выполнено (change) | Repeated ownership/publication gates retained; score/result loops observe context; no cache/clone optimization. |
| D48 | выполнено (retain) | Preprovisioned durable ancestors, process crash/restart versus power loss separated; actual child phase tests PASS. |
| D49 | выполнено (change) | Local nonblocking unduplicated flock handle, trusted root boundary; explicit retired guards and parity tests. |
| D50 | выполнено (retain) | Type-specific validation/publication gates retained with rationale; shared durablefs remains narrow. |
| tensor:01 | выполнено (change) | Candidate count does not cap tokens/dimensions/CPU; matrix host bounds documented. |
| tensor:02 | выполнено (change) | ValidateContext row checks and Rerank candidate checks; deterministic cancellation before malformed rows PASS. |
| tensor:03 | выполнено (change) | Dot nonunit23 and both metrics negative -2 retained; normalized validation and no clamp. |
| tensor:04 | выполнено (retain) | CandidateIDs/budget/ranks preserved, exact-within-candidates contract, no full-index fallback. |
| tensor:05 | выполнено (retain) | Search.project seen[Reference] first occurrence; overflow before projection counts backend documents. |
| tensor:06 | выполнено (retain) | Persistent pinned catalogs/descriptor discovery explicitly broader than TopK and CandidateBudget. |
| tensor:07 | выполнено (retain) | Per Stage, catalog, candidate, inventory directory-key and aggregate-record ceilings documented against code. |
| tensor:08 | выполнено (change) | Resident snapshot count versus bytes and simultaneous transient builds documented; no unmeasured singleflight. |
| tensor:09 | выполнено (change) | Global generation invalidation and inventory recheck on hits retained; stale/cache mutation tests PASS. |
| tensor:10 | выполнено (retain) | Stage, retained selection, clone, snapshot and delivery gates remain separate; callback/fence tests PASS. |
| tensor:11 | выполнено (change) | Nil defaults versus explicit optional K1/B zeros; actual-score formula/ownership/range tests PASS. |
| tensor:12 | выполнено (change) | Raw borrowed immutable metadata; host stable concurrent tokenizer/codec; managed owning clone contract. |
| tensor:13 | выполнено (change) | Context checks in scoring/postings and ranking construction; sort/snapshot indivisibility explained. |
| tensor:14 | выполнено (retain) | Terminal threshold versus raw BM25 ignored threshold; irrelevant graph/vector documented; tensor options reject/clear explicit. |
| tensor:15 | выполнено (retain) | MkdirAll ancestor durability not implied; durable preprovisioned root and process-only proof explicit. |
| tensor:16 | выполнено (retain) | Nonblocking conflict/no retry; unduplicated close-owned handle and Linux open-file-description contract. |
| tensor:17 | выполнено (retain) | QueryPayloadError redacts ordinary storage details, preserves cancel/deadline/protocol; privacy regression PASS. |
| tensor:18 | выполнено (change) | Dense/tensor explicitly exclude retired single/duplicate selectors, focused parity regression PASS. |
| tensor:19 | выполнено (retain) | Trusted host roots, no hostile filesystem/symlink sandbox or networkFS promise. |
| tensor:20 | выполнено (retain) | Keep visible type-specific gates; shared narrow durablefs; no broad generic storage engine or optimization claim. |

## Accepted implementation/docs SHA256

26 changed implementation, tests, consumer and normative guide files. Mutable execution journals (backlog/plan/traceability), acceptance reports/logs and unrelated iCloud duplicate excluded from this implementation fingerprint; trace rows and criteria were independently reviewed above. Any change to these files invalidates this acceptance until reviewed again.

| File | SHA256 |
|---|---|
| dense/persistent/README.md | `c6539bcb4d729ce464a72b65d3ca4ad5aeb517244e7af788000cee934afef83c` |
| dense/persistent/query_unix.go | `da0ea06ce0a79a6ef6483390c5a2424d74827ac12efaafa496bfa68bfcc94cd2` |
| dense/persistent/retired_selection_unix_test.go | `e2d4daee8b5cb69115cdf1c1cdcec7b8375233db1e5949505520126fc13929e1` |
| docs/task20/T16.md | `d4cb70b5d339f3585e3cdeba18d252c0b1a80feaa8d9a51078d3ef0950d340d3` |
| examples/conformance/graph_comparison/baseline_unix.go | `92936c5469a7780347a3bf42ee7e1f51355ca9b43da9201b85140180bf2a9a5a` |
| examples/conformance/internal/task19/data.go | `72b4158b5e1def5ec98c1f76d49c646eb16dbea343f5391903a8df193ac1c018` |
| examples/conformance/recipe_comparison/capture.go | `52d21ae65854c897e8c3b663b1c4a6011f82877b1e45cfc1f468c71c49e321d1` |
| internal/durablefs/README.md | `ddd6b980d96a8a13abe9727ee6a0637ae5722cddc7eccca8928b5cdf079982a8` |
| internal/durablefs/files_unix.go | `c7e5dc94bf3681992579b35de982c0733928f726671336c492ced30a093743ec` |
| lexical/README.md | `5dfa34a249c8df93f77020ab023e974e04571ec750cfb966c89eee539068c879` |
| lexical/bm25.go | `90fb8802df5743daf2ab718870b2e1937f7abfd3319259a53cf270a1751ef8d3` |
| lexical/bm25_parameters_test.go | `19ada153d37365fd2516333a442a7fb4baa2510afacefbe5ad32b8f16dbb4fad` |
| lexical/bm25_scale_test.go | `5309972c4771037fa2489ffe71c9e75247c2a678eb65396aaa49c67d04f19fba` |
| lexical/cancellation_test.go | `250238c25eb7497c00e18f922a90d2b96e1555aef78ca89ecf138cfa446eb206` |
| lexical/managed/README.md | `bd3cd79634aad18af3210dbd5d4841ad85b58ff32d08155c483f1b08e51c6364` |
| lexical/managed/managed.go | `3a85686586a0aeb13aca09b03f29495966316ff7f4ee1f581fb31843a34c1263` |
| lexical/managed/parameters_unix_test.go | `d557517c81ddeeb3026365f3eab0bd5c9ce153c6ae30199b28fb7e13fc3fe86c` |
| tensor/README.md | `0bba4b5294d2580296e08b4f44573d86e74f61d9a4d2af5b3583ab4927804f4e` |
| tensor/cancellation_test.go | `c303d3fb44c509b00be9bc08c308d340572aabf1cba49ff5c75c736295dcfc8b` |
| tensor/maxsim.go | `bfe5807c4beba928a0c5b6e433474a9bd744ea6bbd657407a74cebeb060c0a88` |
| tensor/metric_test.go | `5fdfead81fc5ae029ff99aada2b8db72b3b528bbc39fa70fb049154f0fa58f0a` |
| tensor/persistent/README.md | `02467d4b119f660905685f833c45700cdaed474c017c6476ff0b7067b761dc93` |
| tensor/persistent/query_unix.go | `70292fbf9097851a7b6c1437e12475eca5fa4b46f79a463cd519248123e1f64e` |
| tensor/persistent/retired_selection_unix_test.go | `e2d4daee8b5cb69115cdf1c1cdcec7b8375233db1e5949505520126fc13929e1` |
| tensor/persistent/storage_unix.go | `014ae3ab2067e7b9f9e3f2b9562e444afcd2607454abbf20f6c812cd821a0104` |
| tensor/query/search.go | `e29865cbb44c85e761cdecb19b64aee6002349746d0671f3b20a5e9209066bea` |
