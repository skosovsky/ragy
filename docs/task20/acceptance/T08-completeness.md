# T08 completeness — independent reacceptance

Baseline: `dd81b21abe6e49f1f52b4c03c09c5496a6f25c83`. Reviewer did not implement the change and did not read the correctness report. Verdict: **ACCEPTED — 3/3 criteria = 100%**. No SKIP or unresolved completeness issue. Previous protocol-cause finding independently reproduced as fixed.

| Criterion | Status | Evidence |
|---|---|---|
| T08.C01 | Completed | Pre/post MatchDocument gates in scoring/capture/managed build, temporary snapshot field-index codec gates, existing strengthened capture/output clone gates and final delivery suppress payload and stop next callback on cancellation/revocation. Scoped readfailure.Join retains callback identities/public classifications through errors.Is alongside gate causes without exposing typed private siblings. Independent callback and clone cause/privacy probes pass (details below). Construction codec removed before publishing snapshot; no authorization cache/retry. |
| T08.C02 | Completed | Permanent AAA snapshot capture/index/retrieve cancel/revoke × ordinary error and managed miss/hit variants assert exact callback counts, empty protected result/no later clones; failures additionally retain ErrProtocol for codec cases. Snapshot normal equal-score ordering and managed cold/warm scores/ranks/IDs controls pass. Independent denied-first metadata probe demonstrates protection even on false match. Independent managed clone probe strengthens permanent clone regression with ErrProtocol assertions on both miss/hit and passes. |
| T08.C03 | Completed | Raw Index/Upsert borrowed metadata storage unchanged; readonly snapshot explicit CloneMeta ownership and host stability/concurrency contract retained. Current contract and lexical/managed README explain scoped errors.Is vs private errors.As/text suppression. Independent fresh race four packages PASS, lint 0 issues, git diff --check PASS. |

Assigned source coverage:

| Source | Status | Direct evidence |
|---|---|---|
| F07 | Completed | Context/Binding reaches filterScoredDocs; sorted first candidate cancellation/revocation dispatches one Encode, no later clone/output; ordinary simultaneous failure retains protection plus classification. Snapshot/managed hit/miss normal controls pass. |
| T-F01 | Completed | Original two-candidate callback-boundary defect eliminated in snapshot and raw shared scoring path; capture, metadata-field construction and managed miss also gated. Raw borrowed metadata semantics preserved. No cache authority decisions or retries introduced. |

Assigned coverage **2/2 = 100%**. Raw design remarks remain assigned to later tasks and are not misrepresented as remediated by T08.

Independent verification:

- `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 -overlay=/private/tmp/ragy-t08-completeness/overlay.json ./internal/readfailure ./lexical/... ./retrieval/...` — PASS internal/readfailure, lexical, lexical/managed, retrieval. This full race run included the previously failing cancellation+ErrProtocol probe and denied-match probe.
- `GOCACHE=/private/tmp/ragy-task20-go-cache GOLANGCI_LINT_CACHE=/private/tmp/ragy-task20-lint-cache golangci-lint run --allow-serial-runners ./internal/readfailure ./lexical/... ./retrieval/...` — 0 issues.
- After strengthening independent probes: `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -overlay=/private/tmp/ragy-t08-completeness/overlay.json -run '^(TestT08Independent|TestManagedCloneFailure)' -v ./lexical/...` — PASS all probes.
- `git diff --check` — PASS.

Independent overlay evidence `/private/tmp/ragy-t08-completeness/probe_test.go` / `overlay.json`:

1. Denied first metadata callback cancels: nil snapshot, one Encode, zero clones, protected context.Canceled.
2. Previous failure: first retrieval Encode cancels+returns ErrProtocol: empty result, one Encode, zero output clones, both errors.Is(context.Canceled) and errors.Is(ErrProtocol) true.
3. Private typed callback error wraps ErrProtocol and revokes authority in capture / metadata-field build / retrieval: nil snapshot/empty result, no next callback, errors.Is(private callback identity), ErrProtocol and ErrUnavailable retained; errors.As(private payload) false, errors.As(ProtectionError) true; secret text excluded.
4. Same typed private error in capture/output CloneMeta: nil snapshot/empty result, first failing clone stops next clone, both gate and callback identity/classification retained, private type/text excluded.
5. Managed clone failure+revocation on miss/hit: independent overlay adds required errors.Is(ErrProtocol) assertions to both cases; PASS with empty output, one clone and expected previous Encode counts.

No implementation files modified by reviewer. No backend/parser profile applies; no optimization measurement claimed.

Reviewed SHA256:

- `internal/readfailure/callback.go`: `0ffbb31ad96c96548e5c6840b5eda9e9c974d25ff246bd452781b956a8693913`
- `internal/readfailure/callback_test.go`: `c40f3e98f29a0c70cb59eb80bfabbdebebe73b0c5509a9553976097f11da2297`
- `docs/contracts/remediation.md`: `2c11991ebda8865f3a69c5636d3a54aac637008c0654d28b95bf191d4f9664f0`
- `lexical/bm25.go`: `3f85b325e080a4230d4ad29bbad30f2b8ee849c0ccd5722e7b1059c4f26a600e`
- `lexical/snapshot.go`: `d09c40acee698f4fbf9c09d6493ac84e5b504d69d9ad2141cc083fde1cb7868c`
- `lexical/snapshot_codec.go`: `45331225102232b1f135968d9db9446adeab31f03e632b6fdeba41898328d478`
- `lexical/managed/read.go`: `a717f564615338593d5f8d3efed0c9fb5ec3ea0d3b1a731a3ec4d1e6bcfc39b9`
- `lexical/callback_gate_test.go`: `35c911b0734e01be036c62491ed4b8e8a786214157963d8e128694fd9db5460f`
- `lexical/managed/callback_gate_test.go`: `46bffafcc9552f0109457f213983142df9e055e310f759e08b72fff96de244cf`
- `lexical/bm25_test.go`: `73d0b8c399a170f22050fe1ee77877751d5d742be3f1e1d5f98011be5986e5fd`
- `lexical/managed/managed_test.go`: `abb2a3acffd23ff4a9efc201c6b02b8aea04cf3224b35472c46732b32c92891b`
- `lexical/README.md`: `8fa6f7567dd7329c14c9cc33e8a9b99f10f7b8d0c11886722469ffed874dd9e2`
- `lexical/managed/README.md`: `b15d62b93f255a7d1f5cca61a8bc4f092a50b43179682ba622251f118b39142b`
