# T14 independent correctness acceptance

Baseline: `45ef3d6ded230578e8f7572c91ad89c1c6dd39fc`.

Verdict: **PASS**, no detected correctness findings. Independent nonimplementing reviewer; completeness report was not read. Criterion coverage **5/5 (100%)**; assigned-source coverage **29/29 (100%)**. No SKIP counts as PASS.

## Criterion audit

| Criterion | Result | Evidence and adversarial assessment |
|---|---|---|
| T14.C01 | PASS | Audited all nine master decisions and twenty graph-role items against original graph review and traceability rationales. Obsolete resolution roadmap removed; explicit actual package boundaries retained. No new orchestration engine or implicit policy. |
| T14.C02 | PASS | Cross-source globally existent endpoint remains insufficient in materialization relation closure; no invented locator. Resolver namespace/key hash equality still checks canonical Name; mention label/category distinction and stable reflexive kinds are host contracts. Read boundedRecord/boundedResult: input/result occurrences have separate shared budgets, traces exact; Marshal accepted-byte cap is post-allocation. FileStore build tags limit darwin/linux; root durability/unknown append statements remain conservative. |
| T14.C03 | PASS | Read run/call/refresh and Summary.Resolve: complete original input refreshed before/after maps/reduction and pre-dispatch; later Resolve retains selected citation supports only under original binding. New AAA regression verifies unselected revocation success, selected revocation suppression and delivered text persistence. Coverage is selected declared member sets; MissingCoverage returns at first incomplete community, without later map/reduce. |
| T14.C04 | PASS | Managed capacity rejects zero payload, pre-dedup admission accounting and lost inventory fail closed; no raw/current fallback. Expand seed-only policy follows zero-edge condition. Read Build validateResult and managed captureNode/captureEdge: normalized-view validation discards projection; Stage compacts labels and normalizes codec attributes while independently cloning typed Meta. Latest wording accurately distinguishes these. |
| T14.C05 | PASS | Existing five-type complete integration is runnable with real managed adapter and explicit Prepare/Stage/Publish/interrupted stage. Fresh race all eight affected packages passed; independent focused tests passed; lint zero issues. Resolver exact declared maxima and isolated actual union/refresh kernels measured independently. Algorithms unchanged, before/after optimization N/A; no throughput or peak-allocation claim. |

## Assigned-source audit

Every listed row has an explicit retain/change/contract disposition, rationale and existing evidence. Audit against the original source confirmed the outcome, rather than inferring completion solely from bookkeeping.

| Source | Result | Verified disposition and evidence |
|---|---|---|
| D23 | PASS | retain: graphingest/README.md; graphingest/composition.md |
| D24 | PASS | contract: graphingest/materialization/README.md; graphingest/resolution/README.md; graphingest/materialization/materialization_test.go |
| D25 | PASS | contract: graphingest/resolution/README.md; graphingest/resolution/history/README.md; graphingest/resolution/history/history.go |
| D26 | PASS | contract: graphingest/resolution/history/README.md; graphingest/resolution/history/filestore.go |
| D27 | PASS | retain: graphingest/resolution/scaling.md; docs/task20/acceptance/T14-bench.log |
| D29 | PASS | contract: recipe/graphsummary/README.md; recipe/graphsummary/citation_policy_test.go |
| D30 | PASS | contract: recipe/graphsummary/README.md; recipe/graphsummary/negative_test.go; recipe/graphsummary/citation_policy_test.go |
| D31 | PASS | retain: recipe/graphexpand/README.md; graph/managed/README.md; recipe/graphexpand/expand_test.go |
| D33 | PASS | contract: graphingest/materialization/README.md; graphingest/composition.md; graphingest/pipeline_integration_test.go |
| graph:01 | PASS | retain: graphingest/README.md; graphingest/composition.md |
| graph:02 | PASS | change: graphingest/resolution/README.md; graphingest/composition.md |
| graph:03 | PASS | contract: graphingest/materialization/README.md; graphingest/resolution/README.md; graphingest/materialization/materialization_test.go |
| graph:04 | PASS | contract: graphingest/materialization/README.md; graphingest/resolution/README.md; graphingest/materialization/materialization_test.go |
| graph:05 | PASS | contract: graphingest/resolution/README.md; graphingest/resolution/history/README.md; graphingest/resolution/history/history.go |
| graph:06 | PASS | contract: graphingest/materialization/README.md; graphingest/resolution/README.md; graphingest/materialization/materialization_test.go |
| graph:07 | PASS | contract: graphingest/resolution/README.md; graphingest/resolution/history/README.md; graphingest/resolution/history/history.go |
| graph:08 | PASS | contract: graphingest/resolution/README.md; graphingest/resolution/history/README.md; graphingest/resolution/history/history.go |
| graph:09 | PASS | contract: graphingest/resolution/history/README.md; graphingest/resolution/history/filestore.go |
| graph:10 | PASS | contract: graphingest/resolution/history/README.md; graphingest/resolution/history/filestore.go |
| graph:11 | PASS | retain: graphingest/resolution/scaling.md; docs/task20/acceptance/T14-bench.log |
| graph:13 | PASS | retain: graphingest/resolution/scaling.md; docs/task20/acceptance/T14-bench.log |
| graph:14 | PASS | contract: recipe/graphsummary/README.md; recipe/graphsummary/citation_policy_test.go |
| graph:15 | PASS | contract: recipe/graphsummary/README.md; recipe/graphsummary/negative_test.go; recipe/graphsummary/citation_policy_test.go |
| graph:16 | PASS | contract: recipe/graphsummary/README.md; recipe/graphsummary/negative_test.go; recipe/graphsummary/citation_policy_test.go |
| graph:17 | PASS | retain: recipe/graphexpand/README.md; graph/managed/README.md; recipe/graphexpand/expand_test.go |
| graph:18 | PASS | retain: recipe/graphexpand/README.md; graph/managed/README.md; recipe/graphexpand/expand_test.go |
| graph:19 | PASS | retain: recipe/graphexpand/README.md; graph/managed/README.md; recipe/graphexpand/expand_test.go |
| graph:21 | PASS | contract: graphingest/materialization/README.md; graphingest/composition.md; graphingest/pipeline_integration_test.go |
| graph:22 | PASS | contract: graphingest/materialization/README.md; graphingest/composition.md; graphingest/pipeline_integration_test.go |

## Independent checks

All commands exited 0 with fresh count=1 where applicable:

- `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 ./graphingest/... ./graph/managed ./recipe/graphsummary ./recipe/graphexpand`: T14-correctness-race.log, eight packages PASS.
- `GOCACHE=/private/tmp/ragy-task20-go-cache GOLANGCI_LINT_CACHE=/private/tmp/ragy-task20-lint-cache golangci-lint run --allow-serial-runners ./graphingest/... ./graph/managed ./recipe/graphsummary ./recipe/graphexpand`: T14-correctness-lint.log, 0 issues (existing linter deprecation warning).
- Fresh verbose race run of explicit managed handoff, selected citation policy and insufficient membership tests: T14-correctness-focused.log.
- `go test -run '^$' -bench DeclaredUpperBound -benchmem -benchtime=100ms -count=1 ./graphingest/resolution ./graphingest/resolution/history ./recipe/graphsummary` with same GOCACHE: T14-correctness-bench.log. All workloads PASS; timings vary with concurrent verification, consistent with documented nonisolated reference scope.
- `git diff --check`: PASS.

Local deterministic model fixture, current darwin/arm64 filesystem and affected-scope verification only; no live model/backend or all-module certification claimed. No optimization occurred, so no before/after speedup comparison exists or is required for this retained implementation.

## Accepted file SHA256

Bookkeeping, mutable journal and acceptance files excluded; every changed/new current guide, contract, GoDoc, test and benchmark below was inspected.

```text
e942cf94ae418240a4f1377411463f21a8a42601cbaec0f38224cb79b6487cf9  docs/contracts/remediation.md
285f63ee422d28b32fb6f97b8c53448816a8ab4476f0be47638409d83ca8e881  graphingest/README.md
ff1ef6e7cb45f14c40eb1a507ac2e6eac03fd4f2e1a23b8edb43f09d8046bc28  graphingest/composition.md
a57fd18f4bdb35e15e72fd2d224ebc55181627def40e870d60da470b28cf53cb  graphingest/materialization/README.md
34cb48010fe0755f2277329ae27a0219370b9b464d575d88ff485572095a716b  graphingest/resolution/README.md
376230d16a12345db2a62f93403077200178cd889e7f5262f2f6ee26eb53aa4e  graphingest/resolution/contracts.go
82c09af909c640ac54c40222ac0b2069610105996f1cf310493d24fae1933088  graphingest/resolution/history/README.md
f542b5f9f66305f02932fb24659f72c9c7f6d60443a12e2700cab6132f3a7546  graphingest/resolution/history/scaling_benchmark_test.go
4e0815b85334e666539aa361708635c1e621fc8f202ba10849713690de8b4741  graphingest/resolution/scaling.md
8cc8ee39c6779d1e1c251e1f4e64b6a8b72791f421323dc3bfcf8b75744f21a9  graphingest/resolution/scaling_benchmark_test.go
d827f6377bde0b385e0859ee8cc44573d67a662f149eb4b3c66b4f8f80133435  recipe/graphexpand/README.md
e3cb33d76cc562829d25dc99cbb919d8ccf1376094636cb0dfa9844fdfa4f03a  recipe/graphsummary/README.md
f83d35bf0b198156b0e2545a9f71a037b1716cea3a1b86390e5085eef76f83f0  recipe/graphsummary/citation_policy_test.go
11a53e9cb4ca5935392534bb1629664205a8e2a15c6dda13965f40fe157a5bcf  recipe/graphsummary/contracts.go
f402a617ed55699416c4db8fd83b09770b6d1b0cbeff5af206c7e7d2be1dcbd1  recipe/graphsummary/scaling_benchmark_test.go
d497f5eb7c0a8c0045e491c789b4a08f3953bc223c35873a923740834af13b9b  recipe/graphsummary/summary.go
```
