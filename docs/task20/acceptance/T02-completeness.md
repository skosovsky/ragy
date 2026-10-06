# T02 independent completeness acceptance

Revision 2: reaccepted after scoped standalone renderer taxonomy preservation fix.

Verdict: **PASS — 100% (4/4 criteria fulfilled)**. Baseline HEAD: `4776e89c12b83d1db41385426d982f25f5a51145` (verified with git rev-parse HEAD). Reviewer did not implement changes, did not modify substantive files, and did not read the correctness report before this verdict.

## Criteria coverage

| Criterion | Status | Evidence |
|---|---|---|
| T02.C01 | выполнено | `recipe/run.go`: callback provenance wrapper, all-leaf bounded classification, joined settlement/callback/gate causes, independent parent/read check; `TestCallbackFailureSurvivesLocalDeadline` verifies failed-journal policy and zero protected side payload across Run/RunOwn/RunObserved/RunOwnObserved. Artifact callback provenance and post-callback joins retain explicit protected failures. |
| T02.C02 | выполнено | 96 cases: planner/backend/assessor/encoder × ordinary/protocol/protection/joined-protection+deadline/joined-ordinary+deadline/joined-ordinary+budget × four entrypoints. Assertions retain original callback and deadline causes, exact model counts, occupied retrieval/model leases and zero outstanding leases. `TestDeadlineUsageAccounting` covers known overrun and conservative unknown reservation through shared settlement; malformed output tests cover planner/assessor/encoder. Parent cancel + protocol and revocation + protocol suppress journals. Existing pure timer/injected expiry tests pass. |
| T02.C03 | выполнено | Three graph regression suites retain callback, deadline and (model paths) usage-overrun causes, zero output, single model/traversal dispatch, settled leases and protection classification. Code joins call/settlement/gate together; summary only permits direct pre-dispatch budget admission bounded stop. |
| T02.C04 | выполнено | Updated Run GoDoc and four package READMEs state precedence, zero graph output and no retry; independent full targeted race rerun passes all eight packages including access. `git diff --check` passes. |

## Original source and contract coverage

| Source | Status | Evidence |
|---|---|---|
| Master F01 / retrieval R-01 | выполнено | Full public-API boundary/cause matrix above covers prior original evidence and callback expiry; pure deadline continues bounded partial; known/unknown accounting is preserved. Original raw acceptance reviewed in `docs/task20/reviews/retrieval.md:24–31`. |
| Master D32 / graph:20 | выполнено | extraction, graphexpand and graphsummary joined error tests and implementations; raw graph item 20 reviewed. F04 earlier shared extraction deadline remains explicitly assigned to T05 and is not claimed completed. |
| remediation error-precedence contract | выполнено | Independent callback/settlement/gate causes remain discoverable; protection suppresses journals; ordinary failure retains only observed failed journal; local timer-only bounded result survives while parent/read authority valid; no retry/refund. |
| Malformed success + expiry | выполнено | Pure output-shape validation occurs before settle/gate for planner/assessor/encoder; regression asserts ErrProtocol plus DeadlineExceeded and zero selected evidence. |
| Parent/read authority | выполнено | Parent cancellation + protocol test and modified revocation + protocol test preserve independent protocol classification while suppressing payload-bearing journal. |
| Renderer regression | выполнено | Explicit host ProtectionError+deadline and protocol+deadline are not rescued; existing actual child timer measurement deadline still yields bounded incomplete delivery. Revision 2 additionally verifies standalone cancellation+protocol and joined protection+protocol preserves public sentinel classes while private typed callback payloads remain inaccessible (both valid-parent and cancelled-parent paths). Global access.Protect is unchanged. |

The matrix is not an exhaustive cross-product of accounting, parent and malformed-output variants: known/unknown usage are independently tested at the shared settlement implementation, parent/read failures at the shared stop/delivery implementation, and malformed model success at all three model output validators. This matches original acceptance scope without claiming all possible callback combinations were executed.

## Independent verification

Executed on the reviewed tree:

`GOCACHE=/private/tmp/ragy-task20-completeness-cache go test -race -count=1 ./recipe/... ./graphingest/extraction/... ./retrieval/... ./access/...`

Exit 0, PASS (revision 2 independent rerun):

- recipe 1.708s
- recipe/budget 1.381s
- recipe/graphexpand 1.784s
- recipe/graphsummary 2.787s
- recipe/recording 1.354s
- graphingest/extraction 1.324s
- retrieval 1.343s
- access 1.219s

No SKIP, live service requirement or unresolved completeness blocker in T02 scope. Independently executed `git diff --check`: exit 0. No release/push performed.

## Substantive reviewed files (SHA256)

The acceptance binds the following implementation/test/documentation bytes. Administrative backlog/status/acceptance logs are excluded so acceptance recording can occur without invalidating the code review.

| File | SHA256 |
|---|---|
| `graphingest/extraction/README.md` | `69759c10eeda0229109a4060a12428d8b8481de78e31707a27bb5fc4f5fde944` |
| `graphingest/extraction/extraction.go` | `e014639b558c54ac84ec54b228a2ebd1b66181025dd90c238ed5dc50c337fb80` |
| `graphingest/extraction/deadline_precedence_test.go` | `6befc9e88df667c32c2fa92e8961342ad6eedea6761eb61aa633fc77e8f314f6` |
| `recipe/README.md` | `9ffc28f846629b31f71f3a8d24573688b16dce1d6286c7119af65c504ff2a906` |
| `recipe/run.go` | `1134df7599a2388f1ffcc84e02e819605fc4f307b95a8c1659f1e42581d7cb4e` |
| `recipe/evidence.go` | `b1010c69636e974292bd480eddce45b077dfa16665bc32e648e8b089b82b1082` |
| `recipe/artifact_scope.go` | `20d853ca65aa929536622a6e7a29d86283848fb6fa3a5fe470f3c9db0a9c85cf` |
| `recipe/failed_journal_test.go` | `c92fd5dc9cc7765b71e4f0ea9419895ca52444567097f2ae2f37798d76ce336f` |
| `recipe/deadline_precedence_test.go` | `466c794a3460dc7d2661a75dfd6d03621c5550928b823af18f015ab153df05a5` |
| `recipe/graphexpand/README.md` | `9e79ce72873cb24636cadc7beb783149024fdf95558d39a0a8ff886dfdb48d78` |
| `recipe/graphexpand/expand.go` | `57f2327a18a098a8c31cd2511622e36cf30426141a6b8053276d8496b11deb9b` |
| `recipe/graphexpand/deadline_precedence_test.go` | `62f25c36a850bc4c97636ff58375421b060d1af6bf8864adf332beaa6a14bc06` |
| `recipe/graphsummary/README.md` | `1456f0d739bd04a551974c9ef801bc45aeaa0a0d809bdea25f70a6dd6e3c19f0` |
| `recipe/graphsummary/run.go` | `4a3933600c2e90716583f993338b66119dc016b548220f01b3a732e1257a7a0f` |
| `recipe/graphsummary/deadline_precedence_test.go` | `e91b18369851b09d8ed44eb2d50281140dfdcd88dc82fa90ffb14fb46b9cf8ac` |
| `retrieval/artifact.go` | `1336026ac46357f34c00a6c9ace0c6ecc2264f8dd0fc75cdae831a1c60f286d3` |
| `retrieval/artifact_deadline_precedence_test.go` | `72995b4aeaaa7d859a1a8b69c85bc6dc4d30757944d9105ebadef2c0b083f45a` |
