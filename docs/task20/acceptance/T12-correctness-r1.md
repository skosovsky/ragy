# T12 correctness — FAIL, revision 1

Baseline HEAD `63e775d8e1faee624d04429a63e6958b1efa30a3`; independent current unstaged/untracked diff review. No implementation edits, commits, provider calls or delegated work. No filesystem AGENTS.md exists in cwd; supplied user AGENTS instructions (Contract-First, AAA, adversarial contexts) applied. Read master task20, T12 criteria, final contract, T12 decision/inventory, original retrieval and arch-docs reports, changed source/tests.

Verdict: **FAIL**. Assigned design-source coverage reviewed: 35/35 (100%); task acceptance correctness is not achieved. Criteria C01/C02 reviewed satisfactory, C03/C04 blocked by reproducible failures below; C05 race/lint pass but no overall acceptance. No SKIP counted as PASS. Root-wide historical T09 production-document blacklist failure remains explicitly outside this task; no all-root or live-provider PASS claimed.

## Findings

1. **P2 — Conditional invokes predicate before rejecting missing Child.** `retrieval/execution.go:577–586`. Public direct ConditionalNode with nil Child and predicate returning false performs host callback then returns nil error. Required port validation must occur before predicate dispatch, even when skipped. Repro `TestReviewConditionalInvalidChildBeforePredicate`: err=nil, calls=1.
2. **P2 — Typed-nil secondary dispatch panic after primary callback.** `retrieval/execution.go:327–331,343–347` (Fallback), analogous Rescue 446–450/464. Direct Fallback with typed-nil custom node reaches its Execute and panics after primary executes. Entire selected config must reject before callbacks; Build currently catches this only for pipelines. Repro `TestReviewFallbackTypedNilSecondaryBeforePrimary`: panic, primary calls=1. InspectRead unrestricted path does not recursively validate.
3. **P2 — DegradingMerger required nil ports accepted by pipeline Build.** `retrieval/orchestrator.go:500–509`, `retrieval/execution.go:779`. Rebinding leaves nil Primary/Fallback and returns no config error; aggregate validates only outer Merger. Direct merger eventually rejects, after branch I/O. Pointer form also bypasses value rebinding. Repro `TestReviewDegradingInvalidPortsRemainInvalidAtBuild`: Build err=nil.
4. **P2 — Protection classification loses joined sibling/outer causes.** `retrieval/orchestrator.go:150–151,165–166` and `retrieval/fusion_failure.go:59–60`. `access.Protect` discovers and returns nested ProtectionError, replacing original joined graph. Aggregate child protection joined with ordinary sentinel makes errors.Is(sibling)=false. Preserve entire joined cause under fresh ProtectionError, suppress diagnostics independently. Repro `TestReviewProtectionPreservesJoinedCauses`: sibling lost.
5. **P2 — Direct DegradingMerger success path misses post-call cancellation.** `retrieval/fusion_failure.go:52–54`. Primary cancels parent and returns payload,nil; Merge releases that payload,nil despite documented cancellation suppression. Context check needs every return path (also error classification before fallback). Repro `TestReviewDegradePrimarySuccessCanceledSuppresses`: len=1 err=nil after cancel.
6. **P2 — Empty authoritative result plus PartialFailureError rescues.** `retrieval/orchestrator.go:15–17`, `retrieval/execution.go:459–469`. Separately returned empty set correctly cannot resurrect stale error payload, but partial error must remain non-rescuable by contract. Current code invokes secondary. Repro `TestReviewPartialEmptyNeverRescued`: secondary calls=1. Error-classification policy and payload authority must remain distinct.
7. **P2 — Typed-nil RoutePlanner accepted by Build, panics in direct Execute.** `retrieval/route_switch.go:292–302,686–688`. Nil function adapter inside interface is not ==nil; ordinary invalid configuration passes Build and invokes nil function. Repro `TestReviewTypedNilRoutePlanner`: Build err=nil, runtime panic. Audit Case/Default typed-nil checks and direct whole-tree validation before planner/record/predicate callbacks too.

## Independent checks

- `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 ./retrieval/... ./recipe/... ./adapters/cohere/rerank ./adapters/observability/otel`: exit 0, all eight scope packages PASS (`T12-correctness-race.log`).
- `GOCACHE=/private/tmp/ragy-task20-go-cache GOLANGCI_LINT_CACHE=/private/tmp/ragy-task20-lint-cache golangci-lint run`: exit 0, 0 issues (`T12-correctness-lint.log`).
- Independent seven-test adversarial overlay: exit 1, all seven fail (`T12-correctness-adversarial.log`). No source changes. Fixture and overlay below permit exact reproduction. Tests make ordinary public API calls; fusion helper test isolates shared error-graph behavior. No sleeps/network required.

```sh
GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 -overlay=/private/tmp/t12-independent-overlay.json ./retrieval -run '^TestReview'
```

Overlay maps absolute workspace `retrieval/review_adversarial_test.go` to `/private/tmp/t12-independent-adversarial_test.go`. Fixture:

```go
package retrieval
import("context";"errors";"testing";ragy "github.com/skosovsky/ragy";"github.com/skosovsky/ragy/access")
type reviewNode struct{calls *int; err error}
func(n *reviewNode) Execute(context.Context,Query[struct{}],NoExecutionMeta)(RetrievalResult[struct{},NoExecutionMeta],error){if n==nil {panic("typed nil dispatched")};*n.calls++;return emptyRetrievalResult[struct{}](nil,NoExecutionMeta{}),n.err}
func TestReviewConditionalInvalidChildBeforePredicate(t *testing.T){calls:=0;n:=ConditionalNode[struct{},struct{},NoExecutionMeta]{Predicate:func(Query[struct{}])bool{calls++;return false}};_,err:=n.Execute(t.Context(),Query[struct{}]{Read:UnrestrictedRead(),Options:RetrieveOptions{TopK:1}},NoExecutionMeta{});if !errors.Is(err,ragy.ErrInvalidArgument)||calls!=0{t.Fatalf("err=%v calls=%d",err,calls)}}
func TestReviewFallbackTypedNilSecondaryBeforePrimary(t *testing.T){calls:=0;var nilNode *reviewNode;n:=FallbackNode[struct{},struct{},NoExecutionMeta]{Primary:&reviewNode{calls:&calls},Secondary:nilNode};defer func(){if p:=recover();p!=nil{t.Fatalf("panic=%v primary calls=%d",p,calls)}}();_,err:=n.Execute(t.Context(),Query[struct{}]{Read:UnrestrictedRead(),Options:RetrieveOptions{TopK:1}},NoExecutionMeta{});if !errors.Is(err,ragy.ErrInvalidArgument)||calls!=0{t.Fatalf("err=%v calls=%d",err,calls)}}
func TestReviewDegradingInvalidPortsRemainInvalidAtBuild(t *testing.T){calls:=0;root:=AggregateNode[struct{},struct{},NoExecutionMeta]{Nodes:[]ExecutionNode[struct{},struct{},NoExecutionMeta]{&reviewNode{calls:&calls}},Merger:DegradingMerger[struct{}]{}};_,err:=NewExecutionPipelineBuilder[struct{},struct{},NoExecutionMeta]().WithRoot(root).Build();if !errors.Is(err,ragy.ErrInvalidArgument){t.Fatalf("invalid degrading config build err=%v",err)}}
func TestReviewProtectionPreservesJoinedCauses(t *testing.T){sibling:=errors.New("sibling");input:=NewResultSet([]Document[struct{}]{{ID:"a",Content:"a"}},nil);merger:=compositionMergerFunc(func(context.Context,...ResultSet[struct{}])(ResultSet[struct{}],error){return nil,nil});_,err:=finalizeAggregateRetrieve(t.Context(),DefaultResolver[struct{}](nil),merger,[]aggregateChildResult[struct{}]{{rs:input,err:errors.Join(access.Protect(ragy.ErrUnavailable),sibling)}});if !errors.Is(err,sibling){t.Fatalf("lost joined sibling: %v",err)}}

func TestReviewDegradePrimarySuccessCanceledSuppresses(t *testing.T){ctx,cancel:=context.WithCancel(t.Context());input:=NewResultSet([]Document[struct{}]{{ID:"a",Content:"a"}},nil);m:=DegradingMerger[struct{}]{Primary:compositionMergerFunc(func(context.Context,...ResultSet[struct{}])(ResultSet[struct{}],error){cancel();return input,nil}),Fallback:compositionMergerFunc(func(context.Context,...ResultSet[struct{}])(ResultSet[struct{}],error){panic("fallback")})};out,err:=m.Merge(ctx,input);if !out.IsEmpty()||!errors.Is(err,context.Canceled){t.Fatalf("payload len=%d err=%v",out.Len(),err)}}
func TestReviewPartialEmptyNeverRescued(t *testing.T){calls:=0;secondaryCalls:=0;partial:=&PartialFailureError[struct{}]{Errors:[]error{ragy.ErrUnavailable},Result:NewResultSet[struct{}](nil,nil)};n:=RescueNode[struct{},struct{},NoExecutionMeta]{Primary:&reviewNode{calls:&calls,err:partial},Secondary:&reviewNode{calls:&secondaryCalls}};_,err:=n.Execute(t.Context(),Query[struct{}]{Read:UnrestrictedRead(),Options:RetrieveOptions{TopK:1}},NoExecutionMeta{});if secondaryCalls!=0||!errors.Is(err,partial){t.Fatalf("secondary calls=%d err=%v",secondaryCalls,err)}}

func TestReviewTypedNilRoutePlanner(t *testing.T){var planner RoutePlannerFunc[struct{},string,struct{}];n:=RouteSwitchNode[struct{},string,struct{},struct{},NoExecutionMeta]{Planner:planner};_,buildErr:=NewExecutionPipelineBuilder[struct{},struct{},NoExecutionMeta]().WithRoot(n).Build();if !errors.Is(buildErr,ragy.ErrInvalidArgument){t.Errorf("build err=%v",buildErr)};defer func(){if p:=recover();p!=nil{t.Errorf("typed nil route planner panic: %v",p)}}();_,err:=n.Execute(t.Context(),Query[struct{}]{Read:UnrestrictedRead(),Options:RetrieveOptions{TopK:1}},NoExecutionMeta{});if !errors.Is(err,ragy.ErrInvalidArgument){t.Errorf("execute err=%v",err)}}
```

## Candidate fingerprints

SHA256 of every changed implementation/contract/evidence file, excluding acceptance and mutable bookkeeping. Deleted paths explicit. Combined manifest SHA256 `64d78c8ce796a921bfbc1bbab720eeef4282245a24b8fb0efa76e810fcbf770c`.

```text
1f8dc24fc4946fa19f2bf0868ab1d8a352e79c0e0c114112ba8c8c8790549389  README.md
202ffeac3441aaba87af2d3b87c1497dc7ba679b352e1ba6618b58ea1725e89b  adapters/cohere/rerank/client.go
067a92f82ed3d4e3f6dd74851b40b99cd7c1f0da16dccbfe0e27c769868664d4  adapters/observability/otel/otel.go
9f8e96dba40f0db07bb0771b2b927a3e923b89538fa5f6cc6b0736381b1f385b  adapters/observability/otel/otel_test.go
258fa48bab2f46a69dfc9de2846cd6b38b9f4e9642fc9cace7dc3b61491f7d8f  docs/contracts/remediation.md
13d80ad61ee4f9986f774a0f1b0f3c28319d9d6ec77d95329ead69dde9cc33e3  docs/task20/T12-consumers.md
a922431e9d8932dace2cd60de1dc3702c919333078a5c7e9d2946d8277ba1ee4  docs/task20/T12.md
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
0841bfb1e69cf42616f33e35b60e6851ae123c42e9215f7a9850e61ffc21f6b6  retrieval/artifact.go
af8542d592ecf3707df90d1a592bfeb7ac6d086f3badc16beeaa4de32da28a30  retrieval/composition_config_test.go
84ec1fc671d9ab202efb7e2f0b2a65fc3d12ba8a31b03a968486fae6f9284554  retrieval/document.go
9489b4b632fb7d841a0d7194f1abfdeb122345709c434661a2c1ea043f5c3076  retrieval/document_source_test.go
36090978ce5f5afe1859bc6186655558cb0abdf12e515e75cf2636066dfb8c1b  retrieval/execution.go
706c143c7a520135335176584484862220f106d9cd92a1daea51833d1a8ca8f9  retrieval/execution_test.go
87068e349b00b290de71f4adc61de491cb3cf4baaf0db24523495e3e27756146  retrieval/fusion_failure.go
c954cc169ee95dcadf4f792eb54b14df0c838b7f6e85ed01db70b8738f8ccdbe  retrieval/fusion_failure_test.go
c391f01bf0188f805810cef5657bf3b1ddfece795d95ac7959690849159627ac  retrieval/identity.go
2d4badfdbd86422fa8e4276da58955111d2f59097b6c7548040837d357480052  retrieval/observation_execution_test.go
6b3d843c92b0efe8d5879872790855ad22adbbf3a9a870890284cf8af8471dce  retrieval/observation_legacy_test.go
920015ffef3441d1a95757c97532b950493ec8876a4d3b2310734350759eaab6  retrieval/orchestrator.go
ce896889741d5596b1d11d1cd6a003d92738f370672fabd5e46d55d24ed9adf5  retrieval/orchestrator_test.go
423144b17ac044b78d3b5d6dd95129f178cadc6434dac71c32234b63ea24e237  retrieval/partial_authority_test.go
9bb8a2b6bd87421c08a3c39b34fb4f6b2b14356381c1e544c6565cd19dd49b71  retrieval/partial_failure.go
74aeeb4a1bb9fff2323bc63a4452e655878383ff00912fdd667c862261577d19  retrieval/partial_failure_test.go
e2da4c47b60831609405046abf9d02feb42e8dad08b67dda7e1287372569507d  retrieval/postprocess.go
27d4113b50407ffccf33ef75c8b0ced026e04f4a01937412425ea56e1f50d030  retrieval/postprocessor_chain.go
caf4fe947e7516acb198c0d63827de6d349498bf2293d8904398779ec8c165d5  retrieval/query_reranker.go
145fcaa5ed2adfcf8a48ff631160f87fe05e8ce8987ebd2d4fdbe12591c1c08f  retrieval/ranking_contract_test.go
f9352c31dd83f00590acddc8c760c1dc951f29667011efe678625455ab585330  retrieval/resultset.go
7515254ee17db3caa89ba6913d952d442666015cf27fb52d9c4b9c071d85d9a0  retrieval/route_switch.go
8bca99ce4c33e635b68c7966d39ccebe428367d9d9c3d6ecb47ac7fe3982f0c1  retrieval/rrf.go
93a2f87d9667f5c2f932f891b1f405ef5a4918b4c7fd834adab44684799af8e6  retrieval/score_merger.go
e4602e3b150cb14bbc7b3866b6e9e8862d01eef3a1526e3d767187bf50c105bc  retrieval/score_policy.go
```

Independent additional check: `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 ./examples/planner/... ./examples/resilience/... ./examples/conformance/...` exit 0, all runnable tests PASS; retry_embedder has no test files and is not counted as a test PASS. Log `T12-correctness-examples-conformance.log`. Compiled against revision-1 candidate before fixes. Verdict remains FAIL.
