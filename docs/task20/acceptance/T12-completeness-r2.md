# T12 independent completeness acceptance revision 2 — FAIL

Baseline `63e775d8e1faee624d04429a63e6958b1efa30a3`. Acceptor `/root/t12_completeness` did not implement or modify sources; only this report and its check log were written. Root AGENTS.md is absent; supplied AGENTS.md instructions applied. Read original task20 master, retrieval and arch-docs reports, backlog criteria, every assigned traceability row, implementation/contracts/diff, regressions and recorded candidate logs.

Completeness **80% (4/5)**. Assigned-source review coverage **100% (35/35)**: 14 master +18 retrieval +3 arch-docs rows inspected. Coverage means inspected, not accepted. No blocked criteria. Independent current-diff acceptance **FAIL**; do not commit as accepted.

| Criterion | Status | Actual evidence |
|---|---|---|
| T12.C01 | fulfilled | 35 trace rows have decisions/rationales and existing concrete evidence. D13 retains unchanged cache eviction/copies with explicit no-speedup measurement decision; before/after N/A is justified because no cache/copy optimization was implemented. |
| T12.C02 | fulfilled | T12-consumers inventory records baseline original imports; ranking removed, Cohere/OTel use retrieval ports, old ranking test coverage migrated. resultExecutionNode only translates syntax and resultPipeline delegates to common RequestExecutionPipeline; dispatch/planning algorithms removed. Custom node adapter is host port adaptation, not independent engine. |
| T12.C03 | fulfilled | Explicit fusion policy/observations, zero reset, separate result authority and wrapped/joined cause preservation are covered. Rescue and route rescue now reject PartialFailureError markers even when authoritative result is empty. Both fresh permanent acceptance regressions pass; stale aggregate GoDoc synchronized. |
| T12.C04 | fulfilled | Typed-nil required ports and optional QueryEncoder rejected; optional resolver defaults and ResolverProvider capability tested. Threshold regression distinguishes chain prefilter and terminal output. Winner evidence regression differs/equal payload, strict RRF and GroupBy retained. Artifact value/Diagnostics snapshot tested; unused Embedder removed. |
| T12.C05 | **unfulfilled** | Budget scopes, no call refunds, unknown reservations and cooperative bounds are documented and targeted tests pass. Independent revision2 eight-package race PASS; lint-r2 zero issues. Fresh examples FAIL: TestRescueFallbackAggregate_AllowWebTrue_VectorOutage still expects web rescue from an empty aggregate partial; main.go also promises/resolves the outdated policy. Must synchronize example and rerun full examples/conformance before acceptance. Cache/copy measurement N/A remains justified. |

## Findings

Revision1 blockers (empty partial rescue and stale aggregate GoDoc) are fixed. Fresh revision2 independent targeted race passes permanent direct and wrapped/joined empty-partial regressions. Preflight now rejects builtin invalid topology before callbacks, including typed-nil secondary/planner/case and invalid DegradingMerger ports; value/pointer rebinding respects required ports. Primary success cancellation is gated. Protection wrapper retains joined graph.

**Revision2 blocker:** fresh `/private/tmp/ragy-t12-r2-examples.log` fails `TestRescueFallbackAggregate_AllowWebTrue_VectorOutage` at topology_test.go:52. Correct new engine returns aggregate PartialFailureError with empty payload; old example expects nil error and web payload. `examples/planner/rescue_fallback_aggregate/main.go` also promises vector-outage rescue and panics against corrected policy. Update runnable example/test/docs to the adopted contract and rerun current examples. Conformance revision2 was still executing when reviewed; do not substitute prior candidate PASS.

Independent check `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 ./retrieval ./recipe/... ./adapters/cohere/rerank ./adapters/observability/otel`: exit0, eight packages PASS. Log `docs/task20/acceptance/T12-completeness-r2-check.log`. `git diff --check`: exit0. Former ranking test function inventory equals migrated test inventory (13 tests). No source edits or remote-service claims.

## Assigned source coverage

| Source | Decision | Review | Concrete evidence inspected |
|---|---|---|---|
| D01 | retain | checked | `README.md`, `recipe/README.md` |
| D02 | change | checked | `docs/task20/T12-consumers.md`, `retrieval/orchestrator.go`, `retrieval/ranking_contract_test.go` |
| D03 | change | checked | `retrieval/fusion_failure.go`, `retrieval/fusion_failure_test.go`, `retrieval/orchestrator.go` |
| D04 | change | checked | `retrieval/execution.go`, `retrieval/execution_test.go` |
| D05 | change | checked | `retrieval/partial_failure.go`, `retrieval/access.go`, `retrieval/partial_authority_test.go` |
| D06 | change | checked | `retrieval/composition_config_test.go`, `retrieval/identity.go`, `retrieval/resultset.go`, `recipe/config_ownership_test.go` |
| D07 | change | checked | `retrieval/composition_config_test.go`, `retrieval/score_contract_test.go`, `retrieval/execution.go`, `README.md` |
| D08 | change | checked | `retrieval/resultset.go`, `retrieval/document_source_test.go`, `retrieval/rrf_duplicates_test.go` |
| D09 | change | checked | `recipe/run.go`, `recipe/config_ownership_test.go`, `recipe/README.md`, `retrieval/execution.go` |
| D10 | retain | checked | `recipe/budget/budget.go`, `recipe/budget/README.md`, `recipe/budget/budget_test.go`, `recipe/README.md` |
| D11 | change | checked | `recipe/encoding.go`, `recipe/README.md`, `recipe/negative_test.go` |
| D12 | contract | checked | `README.md`, `recipe/README.md`, `recipe/contracts.go` |
| D13 | retain | checked | `retrieval/cache.go`, `retrieval/cache_test.go`, `retrieval/cache_profile_partition_test.go`, `README.md` |
| D14 | change | checked; r1 gap fixed | `retrieval/execution.go`, `retrieval/orchestrator_test.go`, `README.md` |
| retrieval:01 | retain | checked | `README.md`, `recipe/README.md` |
| retrieval:02 | change | checked | `docs/task20/T12-consumers.md`, `retrieval/orchestrator.go`, `retrieval/ranking_contract_test.go` |
| retrieval:03 | change | checked | `docs/task20/T12-consumers.md`, `retrieval/orchestrator.go`, `retrieval/ranking_contract_test.go` |
| retrieval:04 | change | checked | `retrieval/fusion_failure.go`, `retrieval/fusion_failure_test.go`, `retrieval/orchestrator.go` |
| retrieval:05 | change | checked | `retrieval/execution.go`, `retrieval/execution_test.go` |
| retrieval:06 | change | checked | `retrieval/partial_failure.go`, `retrieval/access.go`, `retrieval/partial_authority_test.go` |
| retrieval:07 | change | checked | `retrieval/composition_config_test.go`, `retrieval/identity.go`, `retrieval/resultset.go`, `recipe/config_ownership_test.go` |
| retrieval:08 | change | checked | `retrieval/composition_config_test.go`, `retrieval/identity.go`, `retrieval/resultset.go`, `recipe/config_ownership_test.go` |
| retrieval:09 | change | checked | `retrieval/composition_config_test.go`, `retrieval/score_contract_test.go`, `retrieval/execution.go`, `README.md` |
| retrieval:10 | change | checked | `retrieval/resultset.go`, `retrieval/document_source_test.go`, `retrieval/rrf_duplicates_test.go` |
| retrieval:11 | change | checked | `recipe/run.go`, `recipe/config_ownership_test.go`, `recipe/README.md`, `retrieval/execution.go` |
| retrieval:12 | retain | checked | `recipe/budget/budget.go`, `recipe/budget/README.md`, `recipe/budget/budget_test.go`, `recipe/README.md` |
| retrieval:13 | retain | checked | `recipe/budget/budget.go`, `recipe/budget/README.md`, `recipe/budget/budget_test.go`, `recipe/README.md` |
| retrieval:14 | change | checked | `recipe/encoding.go`, `recipe/README.md`, `recipe/negative_test.go` |
| retrieval:15 | retain | checked | `retrieval/cache.go`, `retrieval/cache_test.go`, `retrieval/cache_profile_partition_test.go`, `README.md` |
| retrieval:16 | retain | checked | `retrieval/cache.go`, `retrieval/cache_test.go`, `retrieval/cache_profile_partition_test.go`, `README.md` |
| retrieval:17 | contract | checked | `README.md`, `recipe/README.md`, `recipe/contracts.go` |
| retrieval:18 | change | checked; r1 gap fixed | `retrieval/execution.go`, `retrieval/orchestrator_test.go`, `README.md` |
| arch-docs:01 | retain | checked | `README.md`, `recipe/README.md` |
| arch-docs:07 | change | checked | `retrieval/execution.go`, `retrieval/execution_test.go` |
| arch-docs:08 | change | checked; r1 gap fixed | `retrieval/execution.go`, `retrieval/orchestrator_test.go`, `README.md`, `retrieval/fusion_failure.go`, `retrieval/fusion_failure_test.go`, `retrieval/orchestrator.go` |

## Candidate fingerprint

Changed implementation/contract SHA256; acceptance/bookkeeping excluded. Deleted files show baseline digest. Any further change requires reacceptance.

| File | SHA256 |
|---|---|
| `README.md` | `1f8dc24fc4946fa19f2bf0868ab1d8a352e79c0e0c114112ba8c8c8790549389` |
| `adapters/cohere/rerank/client.go` | `202ffeac3441aaba87af2d3b87c1497dc7ba679b352e1ba6618b58ea1725e89b` |
| `adapters/observability/otel/otel.go` | `067a92f82ed3d4e3f6dd74851b40b99cd7c1f0da16dccbfe0e27c769868664d4` |
| `adapters/observability/otel/otel_test.go` | `9f8e96dba40f0db07bb0771b2b927a3e923b89538fa5f6cc6b0736381b1f385b` |
| `docs/contracts/remediation.md` | `258fa48bab2f46a69dfc9de2846cd6b38b9f4e9642fc9cace7dc3b61491f7d8f` |
| `internal/nilvalue/nil.go` | `050119e830092bf4f2fa03781043869e83765b68856a9fa5da9b64ce67fa5594` |
| `ranking/ranking.go (deleted)` | `baseline d9addea8dba746ad0be8fbd56a77a339ae19fb5011c9401e704329136334648c` |
| `ranking/ranking_test.go (deleted)` | `baseline fa0a50dbd65738972aa63946d87fbe69b33f84256ae92609e0a77e4965c94d7b` |
| `recipe/README.md` | `0d52a46d61c820cdfa1562f1c4db6a6659c32a7c46eed3a157fef1925fe84ef6` |
| `recipe/budget/README.md` | `0f645969004cdfb493c9207000aa252442bb02fa1e4d96f7bd5e337d185ef3ff` |
| `recipe/budget/budget.go` | `8f0612a82716eda0670d93411af8c4ea921be952e9b9b101b6039e07312b8749` |
| `recipe/config_ownership_test.go` | `1646ee48d150684951acbc7fa6731e1517918ceeb5fc7371f9284e760bffb1c8` |
| `recipe/encoding.go` | `961e741e39d13e4139756d60b83e3b27406fa398438684d5d04d1e4b41630184` |
| `recipe/run.go` | `76907304a8e70dfbea3e720e957417d6496ba9a4d2aae367a3d578d087c69a04` |
| `retrieval/access.go` | `feab1dd9a9da0e15ff3bc96e4a264ba069ad6b36917ccf3122d4aa67de6cae19` |
| `retrieval/admission.go` | `86187e7c34179e523cfdf02668555a78f53df9b7b864122dcd7a194801a49b09` |
| `retrieval/artifact.go` | `0841bfb1e69cf42616f33e35b60e6851ae123c42e9215f7a9850e61ffc21f6b6` |
| `retrieval/composition_adversarial_test.go` | `7107d2d669e70a25d736fbb3f6d5e0a80999fb937da0a51a0cb5703b3eac2fd0` |
| `retrieval/composition_config_test.go` | `af8542d592ecf3707df90d1a592bfeb7ac6d086f3badc16beeaa4de32da28a30` |
| `retrieval/composition_empty_partial_test.go` | `210bc60fb102363839221f7696ad1918901a01084a71ecd5cab26667b0be53ae` |
| `retrieval/document.go` | `84ec1fc671d9ab202efb7e2f0b2a65fc3d12ba8a31b03a968486fae6f9284554` |
| `retrieval/document_source_test.go` | `9489b4b632fb7d841a0d7194f1abfdeb122345709c434661a2c1ea043f5c3076` |
| `retrieval/execution.go` | `1fafb41e3a68578f8a52110263f7934c516517120fbc7928d296917189d15fe4` |
| `retrieval/execution_test.go` | `706c143c7a520135335176584484862220f106d9cd92a1daea51833d1a8ca8f9` |
| `retrieval/fusion_failure.go` | `28d0893fe26f3c9e86e18b51114f90dc0bc06fb39d893abc370f91e99d6d451a` |
| `retrieval/fusion_failure_test.go` | `c954cc169ee95dcadf4f792eb54b14df0c838b7f6e85ed01db70b8738f8ccdbe` |
| `retrieval/identity.go` | `c391f01bf0188f805810cef5657bf3b1ddfece795d95ac7959690849159627ac` |
| `retrieval/observation_execution_test.go` | `480c2e255f220e91c4df9beffc4e92d1e7c394e773ec27422bb7518ab5a0efd3` |
| `retrieval/observation_legacy_test.go` | `6b3d843c92b0efe8d5879872790855ad22adbbf3a9a870890284cf8af8471dce` |
| `retrieval/orchestrator.go` | `51c5d26bffd0c2445d15c54b51e0eb3e942a00eb26bc13af521ff867963816c9` |
| `retrieval/orchestrator_test.go` | `ce896889741d5596b1d11d1cd6a003d92738f370672fabd5e46d55d24ed9adf5` |
| `retrieval/partial_authority_test.go` | `423144b17ac044b78d3b5d6dd95129f178cadc6434dac71c32234b63ea24e237` |
| `retrieval/partial_failure.go` | `9bb8a2b6bd87421c08a3c39b34fb4f6b2b14356381c1e544c6565cd19dd49b71` |
| `retrieval/partial_failure_test.go` | `74aeeb4a1bb9fff2323bc63a4452e655878383ff00912fdd667c862261577d19` |
| `retrieval/postprocess.go` | `e2da4c47b60831609405046abf9d02feb42e8dad08b67dda7e1287372569507d` |
| `retrieval/postprocessor_chain.go` | `27d4113b50407ffccf33ef75c8b0ced026e04f4a01937412425ea56e1f50d030` |
| `retrieval/query_reranker.go` | `caf4fe947e7516acb198c0d63827de6d349498bf2293d8904398779ec8c165d5` |
| `retrieval/ranking_contract_test.go` | `145fcaa5ed2adfcf8a48ff631160f87fe05e8ce8987ebd2d4fdbe12591c1c08f` |
| `retrieval/resultset.go` | `f9352c31dd83f00590acddc8c760c1dc951f29667011efe678625455ab585330` |
| `retrieval/route_switch.go` | `8967e1b41140eecf7b33b3072e8f514232c8cd3f88ac07e25545682c22bcdc2c` |
| `retrieval/rrf.go` | `8bca99ce4c33e635b68c7966d39ccebe428367d9d9c3d6ecb47ac7fe3982f0c1` |
| `retrieval/score_merger.go` | `93a2f87d9667f5c2f932f891b1f405ef5a4918b4c7fd834adab44684799af8e6` |
| `retrieval/score_policy.go` | `e4602e3b150cb14bbc7b3866b6e9e8862d01eef3a1526e3d767187bf50c105bc` |
