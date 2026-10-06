# T12 correctness — PASS, revision 3

Baseline HEAD `63e775d8e1faee624d04429a63e6958b1efa30a3`. Independent current unstaged/untracked implementation/contract review; final candidate includes README single-authority correction at line 408. No source edits, implementation delegation, commits or provider calls performed by acceptor. Supplied user AGENTS instructions applied; no filesystem AGENTS.md exists in cwd. Read master task20, T12.C01–C05, final normative retrieval contract, T12 decisions/inventory, original retrieval and arch-docs assigned source rows, implementation diff and behavioral tests. Prior FAIL reports and raw logs remain preserved.

**Verdict PASS. Correctness acceptance: no outstanding actionable errors in assigned scope. Criteria completion 5/5 (100%); assigned source coverage 35/35 (100%).** Neither percentage claims universal error absence or production readiness. No SKIP counted as PASS.

## Criteria and adversarial assessment

- C01: D01–D14, retrieval:01–18 and arch-docs:01/07/08 decisions have explicit retain/change/contract rationale and concrete evidence. Cache/copy behavior retained (O(capacity), owned copies and clone outside lock), no optimization/speedup claim; before/after benchmark N/A for unchanged algorithms. Counts/bytes/work boundaries explicit.
- C02: recorded inventory precedes façade deletion, Cohere/OTel migrated. QueryReranker lives in retrieval. Former ranking behavioral tests migrated, not removed. Result-shaped adapters translate declarative structure and dispatch into the execution engine; independent result pipeline algorithms removed.
- C03: returned zero Executed remains authoritative. Separate ResultSet, including empty, is sole payload authority. Partial marker policy stays distinct from error-carried payload authority; empty partial cannot Rescue, while aggregate all-error/no-observation remains ordinary and rescuable. Explicit DegradingMerger has no automatic raw-score fallback, validates required ports and blocks protection/wrapped cancellation/deadline. Joined/outer causes preserved, diagnostics synchronized and suppressed at protected delivery. Empty successful fusion stays empty. Seven R1 and three R2 independent counterexamples now permanent passing regressions.
- C04: entire known node config validates before host callbacks, direct and pipeline paths; nil/typed-nil required ports and Degrading pointer/value forms fail before branch I/O. Chain invalid sentinel anywhere is checked before earlier processors. Nil/typed-nil optional resolver explicitly defaults; custom ResolverProvider survives normalization. Threshold pre-chain and terminal pipeline stages tested; explicit score-scale checks retained. Business MergeKey different payload cannot import losing evidence/history; same payload unions support. Strict RRF and explicit GroupBy retained. Recipe Artifact values/Diagnostics owned; unsupported bridge field removed.
- C05: budget ledger scope, calls/unknown reservations/full accounting tuple and cooperative bounds documented. Fresh independent current-source race covers eight affected packages, planner/resilience examples and entire conformance module. Lint 0 and diff --check clean. Runtime callbacks and arbitrary BYOT metadata remain trusted stable/concurrency-safe inputs, no sandbox or memory-ceiling assertion.

Additional independent revision-3 overlay matrix covers wrapped cancellation and deadline with still-live outer context, fallback protection preserving sibling causes, pointer Degrading invalid required config before dispatch, empty authoritative postprocessing diagnostic without mutating original owner error, and typed-nil custom ResultSet normalization. Four tests/six cases PASS. This extends review beyond the prior fixed findings.

## Actual validation

1. `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 ./retrieval/... ./recipe/... ./adapters/cohere/rerank ./adapters/observability/otel ./examples/planner/... ./examples/resilience/... ./examples/conformance/...`: exit 0. `T12-correctness-r3-race.log`. All runnable tests PASS; packages listed `[no test files]` are not counted as test PASS.
2. `GOCACHE=/private/tmp/ragy-task20-go-cache GOLANGCI_LINT_CACHE=/private/tmp/ragy-task20-lint-cache golangci-lint run`: exit 0, 0 issues. `T12-correctness-r3-lint.log` (nonfailure exhaustruct deprecation warning preserved).
3. `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 -overlay=/private/tmp/t12-r3-independent-overlay.json ./retrieval -run '^TestReviewR3'`: exit 0. `T12-correctness-r3-adversarial.log`.
4. `git diff --check`: exit 0.

No all-root or paid/live-service claim. Historical root production-document blacklist failure belongs to T21. Docker/DB/service profiles unchanged and not rerun for composition-only cleanup. All current necessary changed-scope mechanical checks above completed, not inferred from earlier logs.

Overlay maps workspace `retrieval/r3_review_adversarial_test.go` to `/private/tmp/t12-r3-adversarial_test.go`. Exact fixture:

```go
package retrieval
import("context";"errors";"fmt";"testing";ragy "github.com/skosovsky/ragy";"github.com/skosovsky/ragy/access")
func TestReviewR3WrappedCancellationAndFallbackProtectionMatrix(t *testing.T){for _,stage:=range []string{"primary_cancel","primary_deadline","fallback_protection"}{t.Run(stage,func(t *testing.T){input:=NewResultSet([]Document[struct{}]{{ID:"a",Content:"private"}},nil);outer:=errors.New("outer");calls:=0;primary:=compositionMergerFunc(func(context.Context,...ResultSet[struct{}])(ResultSet[struct{}],error){cause:=ragy.ErrUnavailable;if stage=="primary_cancel"{cause=context.Canceled};if stage=="primary_deadline"{cause=context.DeadlineExceeded};return input,errors.Join(outer,fmt.Errorf("wrapped: %w",cause))});fallback:=compositionMergerFunc(func(context.Context,...ResultSet[struct{}])(ResultSet[struct{}],error){calls++;return input,access.Protect(ragy.ErrUnavailable)});out,err:=(&DegradingMerger[struct{}]{Primary:primary,Fallback:fallback}).Merge(t.Context(),input);want:=0;if stage=="fallback_protection"{want=1};if !out.IsEmpty()||calls!=want||!errors.Is(err,outer){t.Fatalf("len=%d calls=%d err=%v",out.Len(),calls,err)}})}}
func TestReviewR3PointerDegradingNilConfigNoDispatch(t *testing.T){calls:=0;root:=AggregateNode[struct{},struct{},NoExecutionMeta]{Nodes:[]ExecutionNode[struct{},struct{},NoExecutionMeta]{&reviewNode{calls:&calls}},Merger:&DegradingMerger[struct{}]{Primary:NewScoreMerger[struct{}](nil)}};out,err:=root.Execute(t.Context(),Query[struct{}]{Read:UnrestrictedRead(),Options:RetrieveOptions{TopK:1}},NoExecutionMeta{});if !out.IsEmpty()||calls!=0||!errors.Is(err,ragy.ErrInvalidArgument){t.Fatalf("calls=%d err=%v",calls,err)}}
func TestReviewR3PostprocessorEmptyResultRemainsAuthoritative(t *testing.T){stale:=NewResultSet([]Document[struct{}]{{ID:"stale",Content:"stale"}},nil);original:=&PartialFailureError[struct{}]{Errors:[]error{ragy.ErrUnavailable},Result:stale};sibling:=errors.New("sibling");err:=errors.Join(fmt.Errorf("outer: %w",original),sibling);out,resultErr:=DeliverRead(t.Context(),UnrestrictedRead(),NewResultSet[struct{}](nil,nil),err,nil);view,ok:=AsPartialFailure[struct{}](resultErr);if !out.IsEmpty()||!ok||!view.Result.IsEmpty()||!errors.Is(resultErr,original)||!errors.Is(resultErr,sibling)||original.Result.IsEmpty(){t.Fatalf("out=%v err=%v",out,resultErr)}}
func TestReviewR3NilCustomSetNormalizer(t *testing.T){var input *capabilityResultSet;out,err:=NormalizeRankOnlyResultSet[struct{}](input,LinearRankNormalizer{});if err!=nil||!out.IsEmpty(){t.Fatalf("out=%v err=%v",out,err)}}
```

## Final candidate fingerprint

SHA256 all changed implementation/contract/evidence files (46 paths), excluding acceptance and mutable backlog/traceability/plan bookkeeping. Deleted files explicitly marked. Combined manifest SHA256 `f97bb1401796be4f15cf017ab198d47976e34ca8633d6465566542760fd548e7`.

```text
63c7091f8274d4230bc231508024becf6cd4f79ffd85feab39e50496f9310782  README.md
202ffeac3441aaba87af2d3b87c1497dc7ba679b352e1ba6618b58ea1725e89b  adapters/cohere/rerank/client.go
067a92f82ed3d4e3f6dd74851b40b99cd7c1f0da16dccbfe0e27c769868664d4  adapters/observability/otel/otel.go
9f8e96dba40f0db07bb0771b2b927a3e923b89538fa5f6cc6b0736381b1f385b  adapters/observability/otel/otel_test.go
258fa48bab2f46a69dfc9de2846cd6b38b9f4e9642fc9cace7dc3b61491f7d8f  docs/contracts/remediation.md
13d80ad61ee4f9986f774a0f1b0f3c28319d9d6ec77d95329ead69dde9cc33e3  docs/task20/T12-consumers.md
918d784466ab439e5f8991c0c7e528688e28deb997734c211b220e2638a42f5d  docs/task20/T12.md
050119e830092bf4f2fa03781043869e83765b68856a9fa5da9b64ce67fa5594  internal/nilvalue/nil.go
DELETED  ranking/ranking.go
DELETED  ranking/ranking_test.go
0d52a46d61c820cdfa1562f1c4db6a6659c32a7c46eed3a157fef1925fe84ef6  recipe/README.md
0f645969004cdfb493c9207000aa252442bb02fa1e4d96f7bd5e337d185ef3ff  recipe/budget/README.md
8f0612a82716eda0670d93411af8c4ea921be952e9b9b101b6039e07312b8749  recipe/budget/budget.go
1646ee48d150684951acbc7fa6731e1517918ceeb5fc7371f9284e760bffb1c8  recipe/config_ownership_test.go
961e741e39d13e4139756d60b83e3b27406fa398438684d5d04d1e4b41630184  recipe/encoding.go
76907304a8e70dfbea3e720e957417d6496ba9a4d2aae367a3d578d087c69a04  recipe/run.go
feab1dd9a9da0e15ff3bc96e4a264ba069ad6b36917ccf3122d4aa67de6cae19  retrieval/access.go
86187e7c34179e523cfdf02668555a78f53df9b7b864122dcd7a194801a49b09  retrieval/admission.go
0841bfb1e69cf42616f33e35b60e6851ae123c42e9215f7a9850e61ffc21f6b6  retrieval/artifact.go
7107d2d669e70a25d736fbb3f6d5e0a80999fb937da0a51a0cb5703b3eac2fd0  retrieval/composition_adversarial_test.go
af8542d592ecf3707df90d1a592bfeb7ac6d086f3badc16beeaa4de32da28a30  retrieval/composition_config_test.go
8cf498a0ea9f1ddc789bdad8e2d167a6897e6a727a1762e037eacf054793313d  retrieval/composition_degradation_adversarial_test.go
210bc60fb102363839221f7696ad1918901a01084a71ecd5cab26667b0be53ae  retrieval/composition_empty_partial_test.go
84ec1fc671d9ab202efb7e2f0b2a65fc3d12ba8a31b03a968486fae6f9284554  retrieval/document.go
9489b4b632fb7d841a0d7194f1abfdeb122345709c434661a2c1ea043f5c3076  retrieval/document_source_test.go
1fafb41e3a68578f8a52110263f7934c516517120fbc7928d296917189d15fe4  retrieval/execution.go
706c143c7a520135335176584484862220f106d9cd92a1daea51833d1a8ca8f9  retrieval/execution_test.go
343239d0fe05894e9c0db7813f3643686542de120ec53713b223490c8d15b5f2  retrieval/fusion_failure.go
6b52f6f2d4d52e0d259daf9fe159816ba88e7f555891101e65301b9e2a1cfc50  retrieval/fusion_failure_test.go
c391f01bf0188f805810cef5657bf3b1ddfece795d95ac7959690849159627ac  retrieval/identity.go
480c2e255f220e91c4df9beffc4e92d1e7c394e773ec27422bb7518ab5a0efd3  retrieval/observation_execution_test.go
6b3d843c92b0efe8d5879872790855ad22adbbf3a9a870890284cf8af8471dce  retrieval/observation_legacy_test.go
980a7d7828a62bb09e2d64df80af349258aba3316c5bd8733082307145640b55  retrieval/orchestrator.go
ce896889741d5596b1d11d1cd6a003d92738f370672fabd5e46d55d24ed9adf5  retrieval/orchestrator_test.go
423144b17ac044b78d3b5d6dd95129f178cadc6434dac71c32234b63ea24e237  retrieval/partial_authority_test.go
9bb8a2b6bd87421c08a3c39b34fb4f6b2b14356381c1e544c6565cd19dd49b71  retrieval/partial_failure.go
74aeeb4a1bb9fff2323bc63a4452e655878383ff00912fdd667c862261577d19  retrieval/partial_failure_test.go
e2da4c47b60831609405046abf9d02feb42e8dad08b67dda7e1287372569507d  retrieval/postprocess.go
47b9fd21ecb476ddc71d6ce157d2db9a66e31d6bebbd2751ff8cc756c607cc4b  retrieval/postprocessor_chain.go
caf4fe947e7516acb198c0d63827de6d349498bf2293d8904398779ec8c165d5  retrieval/query_reranker.go
145fcaa5ed2adfcf8a48ff631160f87fe05e8ce8987ebd2d4fdbe12591c1c08f  retrieval/ranking_contract_test.go
f9352c31dd83f00590acddc8c760c1dc951f29667011efe678625455ab585330  retrieval/resultset.go
8967e1b41140eecf7b33b3072e8f514232c8cd3f88ac07e25545682c22bcdc2c  retrieval/route_switch.go
8bca99ce4c33e635b68c7966d39ccebe428367d9d9c3d6ecb47ac7fe3982f0c1  retrieval/rrf.go
93a2f87d9667f5c2f932f891b1f405ef5a4918b4c7fd834adab44684799af8e6  retrieval/score_merger.go
e4602e3b150cb14bbc7b3866b6e9e8862d01eef3a1526e3d767187bf50c105bc  retrieval/score_policy.go
```
