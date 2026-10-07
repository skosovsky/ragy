# T16 correctness acceptance

Baseline: `5775f4ee0b1aeb11feb634cbb480a670ace0c7ce`.
Independent read-only reviewer; no implementation participation or edits.
Final candidate was announced stable after root and nested consumer lint formatting. Current implementation,
AAA tests, normative T16 contract and all assigned original review requirements were
inspected directly. Verdict: **PASS — no unresolved correctness findings.**

## Criteria audit

| Criterion | Result | Evidence and checked behavior |
|---|---|---|
| C01 | PASS | `ValidateContext` checks entry/each row/exit; MaxSim and Rerank use it; candidate validation retains uniqueness, identity and compatible space checks. Native Dot=23 and negative=-2 tests; no clamping. Limits are candidate counts, not token/dimension bounds. |
| C02 | PASS | Candidate overflow precedes projection; projection deduplicates exact references in first-occurrence order. Persistent catalogs can exceed payload/scoring budget. All five MaxRecords units and resident versus transient cache bounds documented. Explicit zero/default parameter formula and detached pointer tests pass. |
| C03 | PASS | Raw constructor copies parameters; managed constructor separately copies parameters retained for later builds. Existing publication/read/clone/inventory gates are retained. Generation invalidation rejects stale builds; immutable raw metadata borrowing and host callback concurrency boundaries are explicit. |
| C04 | PASS | Both selectors exclude Retired, Tombstone and unpublished candidates, preserving exact pin identity and ambiguity behavior. Local nonblocking flock documentation accurately describes unduplicated handle/open-file-description lifetime. Trusted preprovisioned root, ancestor fsync limits and process-crash versus power-loss distinction stated. QueryPayloadError privacy/classification unchanged. Visible type-specific gates retained with rationale. |
| C05 | PASS | Independent fresh race passes for all seven affected packages, including crash parent drivers, separate-process restarts, retired selectors, callback/cache/inventory fences. No CPU/cache/clone optimization implemented or claimed; before/after optimization measurement is inapplicable. |

Five criteria reviewed; no unresolved correctness findings.

## Original-source audit

All 27 assigned sources have inspected dispositions and matching implementation or
bounded retained-behavior rationale, rather than relying on journal status.

| Source | Result | Checked outcome |
|---|---|---|
| D44 | PASS | Metric-dependent row validation and native scores; cooperative row/candidate/pair cancellation. |
| D45 | PASS | Backend-document versus deduplicated-reference budgets, broad catalog scan and five limit units explicit. |
| D46 | PASS | Optional owned aggregate permits valid zeros; resident/transient cache and borrowed raw ownership explicit. |
| D47 | PASS | CPU checkpoints added; separate admission/clone/freshness gates retained. |
| D48 | PASS | Preprovisioned local root/ancestors; no hardware power-loss inference from process crashes. |
| D49 | PASS | Handle-owned nonblocking local flock, trusted-root boundary and retired-selector parity. |
| D50 | PASS | Retain type-specific gates; narrow durablefs mechanics already shared; no generic public engine. |
| tensor:01 | PASS | Public candidate bound distinguished from matrix/operation resources. |
| tensor:02 | PASS | Request and scoring validation observe context between rows/candidates. |
| tensor:03 | PASS | Dot magnitudes, negative and greater-than-one native sums preserved. |
| tensor:04 | PASS | Candidate IDs/native ranks remain observable; no hidden full fallback. |
| tensor:05 | PASS | First-occurrence exact reference dedup after backend-document count limit. |
| tensor:06 | PASS | Catalog discovery broader than TopK; no ANN/exhaustive-recall claim. |
| tensor:07 | PASS | Distinct MaxRecords uses enumerated; aggregate inventory may fail despite individually valid stages. |
| tensor:08 | PASS | Snapshot resident count does not cap bytes or simultaneous builders. |
| tensor:09 | PASS | Generation-wide invalidation and hit inventory rechecks retained. |
| tensor:10 | PASS | Separate owned capture/build/admission/delivery boundaries retained. |
| tensor:11 | PASS | Explicit K1=0/B=0 accepted; nil selects defaults; independent formula tests. |
| tensor:12 | PASS | Raw borrowed immutable metadata; host callback stable/concurrent requirement. |
| tensor:13 | PASS | Score accumulation and ranking construction cancel without partial output; sorts/snapshot copy remain indivisible work. |
| tensor:14 | PASS | Threshold terminal semantics, BM25 ignores threshold, tensor rejects Threshold/Graph and clears candidate vector before target. |
| tensor:15 | PASS | Trusted preprovisioned durable ancestors; MkdirAll not represented as power-loss proof. |
| tensor:16 | PASS | Local nonblocking conflicts returned immediately; handle lifetime documented. |
| tensor:17 | PASS | Sentinel preservation and ordinary-storage-to-unavailable redaction unchanged and tested. |
| tensor:18 | PASS | Dense/tensor retired rejection and retired-duplicate beside live selector regressions. |
| tensor:19 | PASS | Trusted roots do not imply a hostile-filesystem sandbox. |
| tensor:20 | PASS | Existing narrow shared mechanics retained; no hidden type/publication gates. |

## Independent verification

- `go test -race -count=1 ./lexical ./lexical/managed ./tensor ./tensor/query ./tensor/persistent ./dense/persistent ./internal/durablefs`: exit 0 (`T16-correctness-race.log`).
- Same command with `-v` after helper extraction: exit 0 (`T16-correctness-final-race.log`). This log records both physical process-crash parent drivers passing payload/catalog/installed/retired/removed phases, both separate-process restart profiles, callback cancellation/revocation/failure matrices on cache misses and hits, inventory mutation fences and generation tests. Child-only helper skips are not counted as independent acceptance evidence; the parent drivers actually spawn and check the child exit/recovered state.
- `GOWORK=off go test -race -count=1` on those same seven packages after final candidate announcement: exit 0 (`T16-correctness-stable-race.log`).
- From `examples/conformance`, `go test -race -count=1 ./internal/task19 ./graph_comparison ./recipe_comparison`: exit 0, all three migrated consumer packages PASS (`T16-correctness-consumers.log`).
- `git diff --check`: exit 0.
- Reviewed compiled consumer migrations in all three conformance packages; only historical build-ignore task12 reproduction retains old scalar fields.

No whole-root/all-module acceptance or hardware power-loss claim. The previously
known whole-root documentation blacklist issue belongs T21; T22 retains final-wide
verification responsibility. No newly uncovered T16 issue is waived. Unrelated
`docs/task18/correctness 2.md` was not edited.

## Formatting refresh and repeated acceptance

Nested consumer lint formatted exactly three migrated Config literals. I reviewed
the actual diff again and independently reconstructed each previously accepted file
by collapsing that one multiline literal to its previous single line. All three
reconstructed files match their previous accepted SHA256 exactly; every other
manifest entry is byte-identical. This proves only whitespace changed, with no AST
or behavior change, so previous independent race executions remain applicable.
The root consumer formatter log reports exit 0 / 0 issues. Acceptance remains
**PASS**, and all 26 SHA256 entries below now identify the current final candidate.

## Current candidate SHA256 manifest

26 task implementation/test/current-document paths. Mutable backlog/plan/traceability
bookkeeping and generated acceptance logs are excluded from this candidate manifest.

```text
c6539bcb4d729ce464a72b65d3ca4ad5aeb517244e7af788000cee934afef83c  dense/persistent/README.md
da0ea06ce0a79a6ef6483390c5a2424d74827ac12efaafa496bfa68bfcc94cd2  dense/persistent/query_unix.go
e2d4daee8b5cb69115cdf1c1cdcec7b8375233db1e5949505520126fc13929e1  dense/persistent/retired_selection_unix_test.go
d4cb70b5d339f3585e3cdeba18d252c0b1a80feaa8d9a51078d3ef0950d340d3  docs/task20/T16.md
92936c5469a7780347a3bf42ee7e1f51355ca9b43da9201b85140180bf2a9a5a  examples/conformance/graph_comparison/baseline_unix.go
72b4158b5e1def5ec98c1f76d49c646eb16dbea343f5391903a8df193ac1c018  examples/conformance/internal/task19/data.go
52d21ae65854c897e8c3b663b1c4a6011f82877b1e45cfc1f468c71c49e321d1  examples/conformance/recipe_comparison/capture.go
ddd6b980d96a8a13abe9727ee6a0637ae5722cddc7eccca8928b5cdf079982a8  internal/durablefs/README.md
c7e5dc94bf3681992579b35de982c0733928f726671336c492ced30a093743ec  internal/durablefs/files_unix.go
5dfa34a249c8df93f77020ab023e974e04571ec750cfb966c89eee539068c879  lexical/README.md
90fb8802df5743daf2ab718870b2e1937f7abfd3319259a53cf270a1751ef8d3  lexical/bm25.go
19ada153d37365fd2516333a442a7fb4baa2510afacefbe5ad32b8f16dbb4fad  lexical/bm25_parameters_test.go
5309972c4771037fa2489ffe71c9e75247c2a678eb65396aaa49c67d04f19fba  lexical/bm25_scale_test.go
250238c25eb7497c00e18f922a90d2b96e1555aef78ca89ecf138cfa446eb206  lexical/cancellation_test.go
bd3cd79634aad18af3210dbd5d4841ad85b58ff32d08155c483f1b08e51c6364  lexical/managed/README.md
3a85686586a0aeb13aca09b03f29495966316ff7f4ee1f581fb31843a34c1263  lexical/managed/managed.go
d557517c81ddeeb3026365f3eab0bd5c9ce153c6ae30199b28fb7e13fc3fe86c  lexical/managed/parameters_unix_test.go
0bba4b5294d2580296e08b4f44573d86e74f61d9a4d2af5b3583ab4927804f4e  tensor/README.md
c303d3fb44c509b00be9bc08c308d340572aabf1cba49ff5c75c736295dcfc8b  tensor/cancellation_test.go
bfe5807c4beba928a0c5b6e433474a9bd744ea6bbd657407a74cebeb060c0a88  tensor/maxsim.go
5fdfead81fc5ae029ff99aada2b8db72b3b528bbc39fa70fb049154f0fa58f0a  tensor/metric_test.go
02467d4b119f660905685f833c45700cdaed474c017c6476ff0b7067b761dc93  tensor/persistent/README.md
70292fbf9097851a7b6c1437e12475eca5fa4b46f79a463cd519248123e1f64e  tensor/persistent/query_unix.go
e2d4daee8b5cb69115cdf1c1cdcec7b8375233db1e5949505520126fc13929e1  tensor/persistent/retired_selection_unix_test.go
014ae3ab2067e7b9f9e3f2b9562e444afcd2607454abbf20f6c812cd821a0104  tensor/persistent/storage_unix.go
e29865cbb44c85e761cdecb19b64aee6002349746d0671f3b20a5e9209066bea  tensor/query/search.go
```
