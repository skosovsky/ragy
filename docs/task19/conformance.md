# TASK19 external public composition evidence

Contract: E02, E09 and E11 in [contracts.md](contracts.md). Source baseline is `13c37b67689be736bb05d575b41f2446d2b90c50`; tests are new uncommitted TASK19 changes. Core production code and public dependencies are unchanged. `examples/conformance/final_contract` uses the existing external `example.com/ragyconsumer` module with `GOWORK=off`, not an in-core fixture package.

## Local evidence classes

| Contract | Actual composition and check | Evidence class / limit |
|---|---|---|
| Decorator capabilities, fusion, scope | Reuse `joint_read`: persistent dense + managed lexical/tensor/graph; cache/projected/OTel permutations; nested execution paths | Local storage integration; no DB service claim |
| Embedding identity | Final dense persistent publication rejects mismatched embedding.Space identity with zero payload reads; matching vector retrieves actual stored payload | Local storage integration; no live encoder claim |
| Cache freshness | Actual scoped BM25 + MemoryCache; deterministic Load barrier revokes before delivery clone | Concurrent adversarial contract; no payload returned |
| Packing/provenance/coverage | Actual scoped BM25Snapshot + JSONCodec + RRF + source.Reader hydration from files; whole formatted non-additive measurement; contributor/support association retained | Local actual context composition |
| Evidence schema/privacy | Actual fused BM25 results and original query contributors are projected by the consumer into evidence.Input, then captured in schema v2 with payload-private export policy and roundtrip | Retrieval-evidence schema/privacy contract; artifact delivery export is not exercised by this test |
| Hydration freshness | Revoke host authority during file load; Reader and artifact reject delivery | Local adversarial contract |
| Shared recipe limits | Actual scoped BM25 and three recipe strategies; shared Ledger across attempts; refusal before scripted planner/model call; cancellation | Scripted host model ports; no provider effectiveness or billing claim |
| Pins and maintenance | Actual filestore raw CAS/artifact fences, capacity and batch ownership; managed lexical cleanup and explicit retirement with registered pins | Local durable lifecycle contract |
| Execution observation | Reuse `observation_contract`: actual BM25, exporter failure, unknown usage, bounded concurrent sessions, cancellation | Local diagnostic contract |

Packing and evidence capture are separate assertions in the same actual retrieval composition. `assertContextEvidence` receives consumer-projected fused hits and original query contributions; it does not export the rendered artifact or certify its delivered contributors in the evidence record. Artifact delivery/contributor export remains covered by existing TASK17 recording integration tests in the repository-wide run.

No fake backend is counted as actual scope proof. File payload Catalog/Loader and IAM authority are consumer ports; JSONCodec, source.Reader, storage/retrieval, budgets, artifacts and evidence enforcement are actual library implementations. There is no public `source.Store` type: the source abstraction is Reader over Catalog/Loader. No new public contract helper is necessary; the existing BYOT suites already cover decorator and adapter contracts.

## Adversarial interpretation

Malicious retrieved instructions remain text data and preserve their original source mapping. Tests do not execute an external agent and do not establish agent prompt-injection immunity. Same-document contributor association and forged contributors additionally retain the focused root regressions in `retrieval/document_source_test.go`, `recipe/recording/contributions_test.go` and `evidence/contributions_test.go`; these are checked by the all-module runner rather than copied here.

Publication pins protect lifecycle metadata, not target payloads. Cleanup must receive an explicit retained-reference policy from the host. The actual managed lexical cleanup is allowed to remove old target payloads while the registered pin still protects metadata; tests then release and explicitly retire the old publication. Existing joint read tests prove availability only for their separately retained target profile. Metadata retention is never reported as proof of physical availability.

## Reproduction and raw results

Use the commands in [the external package README](../../examples/conformance/final_contract/README.md). Raw race/lint/version output is saved under `results/conformance-*.txt`; the repository-wide runner supplies the all-module matrix separately. Go 1.27.1, darwin/arm64, golangci-lint 2.14.0. Race execution uses a dedicated cache and `-count=1`; this is not a cold-cache timing experiment. Local elapsed test times are not performance percentiles. Live services, providers and actual PDF engines are not exercised by this package and cannot be inferred from its pass status.

Executed local results: `final_contract` passed race in 1.940s, reused `joint_read` passed race in 37.744s, reused `observation_contract` passed race in 2.065s. Scoped lint for all three packages passed with 0 issues; its upstream deprecated-linter warning is retained in the raw log. All 13 new top-level test cases (including table subcases) ran; no skip or live-service substitution occurred.
