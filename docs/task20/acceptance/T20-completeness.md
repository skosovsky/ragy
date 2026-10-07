# T20 independent completeness acceptance

Baseline: `1672c0e029f81480003f797dac174e16f5e141e2`. Reviewer `/root/t20_completeness`; read-only implementation review, independent of implementation and correctness acceptance. No correctness report was read. Report generation is the only repository write by this reviewer.

Verdict: **PASS — criteria 5/5 = 100%; assigned source coverage 7/7 = 100%**. No unmet criteria or current-scope blockers discovered. This verdict is requirement completeness, not production readiness or completion of T21/T22.

## Criteria

| Criterion | Status | Evidence |
|---|---|---|
| T20.C01 | выполнено | README reduced from 534 lines to 158, uniformly English current prose; installation/module and Go1.26.1 requirement, complete typed metadata/finalized schema/local BM25, explicit unrestricted Read and protection-first display. README Go fence equals `examples/local-bm25/main.go` exactly; gofmt clean; independent run produces documented tenant-filtered output. All seven trace dispositions have rationale/evidence. |
| T20.C02 | выполнено | Public `docs/integration.md`, ownership, limits, errors/recovery, architecture and capabilities guides indexed by README. Detailed lifecycle/source/layout/chunking/graph package READMEs remain public authority. Scores/fusion in integration. Local links outside code fences independently resolve. `.cursor` integration is a redirect; original bytes retained as historical snapshot. |
| T20.C03 | выполнено | DOC1–8 individually mapped below. API naming and boundaries checked against current source and package guides, including embedding defaults, BM25 zero parameters, artifact resource caps, managed snapshot count and release implementation/manifest. No inferred RSS/CPU/live-service guarantees. Exact release scope/recovery/clean-consumer runbook describes actual script rather than invoking production publication. |
| T20.C04 | выполнено | `docs/project-policies.md` records absent root license/changelog/external-contribution/private-security decisions and current version behavior, contributor fresh checks and platform scope. No invented license/contact. Publication/license selection excluded by approved goal; absent templates explicitly not runtime blocker. Future owner decisions must precede affected public maintenance/publication. |
| T20.C05 | выполнено | Independent baseline comparison of454tracked task13–19files: no content changes, four CSVs only existing CRLF-vs-LF normalization; gitdiff empty. Seven separate REMEDIATION references plus history index. Failed historical quality promotion not overwritten. Current capabilities separates wire/local/actual PG/PDF/process-crash/quality and expressly rejects checklist100%=readiness; BYOT suite limitations and Codex CLI advisory port stated. |

## Assigned source coverage

Original master D61 and original arch-docs report lines50–61 read directly. Source coverage is separate from criterion percentage.

| Source | Status / disposition | Evidence and rationale |
|---|---|---|
| D61 | выполнено / change | Public guides and runnable onboarding delivered; reviewed release manifest and semantic-version guard documented. Trace explicitly leaves blacklist replacement, standalone/fresh CI/module/fuzz implementation for approved T21, with final verification T22. No premature claim that entire tooling work is complete. |
| arch-docs:09 (A-D09) | выполнено / change | Executable short local onboarding, setup/schema/meta/Read/error policy and Go requirement; advanced topology moved into public integration. |
| arch-docs:10 (A-D10) | выполнено / change | README primary API links point to stable public topic/package guides; historical integration retained with date/hash and redirected entry. Task results remain evidence, not primary contract. |
| arch-docs:11 (A-D11) | выполнено / contract | Owner policy inventory preserves absent decisions, actual release versioning/platform and concrete contributor commands, without selecting legal terms or fabricating private channel. |
| arch-docs:13 (A-D13) | выполнено / retain | `contracttest/read_scope.go` exposes host-owned generic fixture and directly invokes leaf adapter with instrumented payload I/O. Matrix documents older fixed fixtures and supplied-profile-only certification. Retaining concrete historical fixtures is justified. |
| arch-docs:19 (A-D19) | выполнено / retain | BYOT business payload remains `TMeta`; only codec/DSL boundaries use maps/RawAttributes. README, architecture and ownership distinguish shallow copies from explicit metadata cloning. |
| arch-docs:20 (A-D20) | выполнено / contract | Historical acceptance/quality evidence unchanged, separate remediation refs. Matrix separates completeness from readiness and allows external Codex CLI advisory review without demanding new paid live reference-budget run. |

## Documentation requirements

| Requirement | Status | Current authority |
|---|---|---|
| DOC1 runnable local BYOT | выполнено | README + exact executable `examples/local-bm25/main.go`. |
| DOC2 public contracts/history separation | выполнено | README index, current topic/package guides, dated snapshot/history and separate task13–19 REMEDIATION pointers. |
| DOC3 resolution stale ending | выполнено | Existing accepted T14 `graphingest/resolution/README.md` explicitly links extraction/materialization/history/summary/composition, describes current boundaries; retained unchanged. |
| DOC4 units/defaults/admission bounds | выполнено | `docs/limits.md` and current detailed package guides. Calls/tokens/rows/bytes/candidates distinct; wire vs allocation, candidate-exactness vs recall, snapshot count vs memory, source bytes vs full envelope expressly separated. |
| DOC5 error/recovery | выполнено | `docs/errors-and-recovery.md`: protection suppressive, parent vs pure local deadline vs independent callback fault, authoritative partial return, dispatch/cleanup unknown vs read-only uncertainty, raw administration vs scoped retrieval. |
| DOC6 ownership/callback/source | выполнено | `docs/ownership.md`: host types shallow references, explicit clones, stable/pure/concurrent callbacks and cooperative cancellation/error methods; retained source authority and citations vs derivation dependencies. |
| DOC7 capability/evaluation profiles | выполнено | `docs/capabilities.md`: local/wire/real-service/process-crash/quality separate; paid-live SKIP not PASS; conformance helper profile limits, advisory runner and historical completeness scope. |
| DOC8 exact release/runbook | выполнено | `docs/release/runbook.md` synchronized with release.py/state/manifests: full sourceSHA,11publishablemodules versus3nested development examples, allowlisted go.mod edits, lightweight exact refs/atomic push, platform, none/partial/unknown/collision recovery, isolated clean-consumer gate, v2guard. T21/T22 verification remains explicit. |
| DOC9 owner policies | выполнено | `docs/project-policies.md`; choices inventoried, owner authority preserved, no runtime blocker invented from missing templates. |

## Independent checks and scope

Executed fresh commands with Go1.26.5 from `/opt/homebrew/Cellar/go/1.26.5/bin/go`, `GOWORK=off GOTOOLCHAIN=local GOCACHE=/tmp/ragy-t20-completeness-cache`:

- `go run ./examples/local-bm25`: exit0, output `reset [acme]: Reset your password from account settings.`
- `go test -count=1 -race ./examples/local-bm25 ./lexical ./retrieval`: exit0; example compiles (no test files), lexical/retrieval fresh race PASS.
- `gofmt -l examples/local-bm25/main.go`: exit0, empty output; README fence exact equality independently confirmed.
- Python relative-link check on current public/index/remediation/redirect files after excluding fenced code: zero unresolved local links.
- Independent `git show BASELINE:path` comparison of454historicaltrackedfiles: no changed content; `docs/task18/results/{process-costs,round1-process-costs,round1-scaling,scaling}.csv` have existing CRLF worktree normalization. `git diff BASELINE -- docs/task13 … docs/task19`: empty.
- Historical original-integration payload SHA256 independently reconstructed: `fe825c0cac40b90172af8b70ae9c95a8dd95e3e6a06c7f4b88d9964d3ddff4ba`, matches snapshot label.
- `git diff --check`: exit0. Author scoped lint/race/docs-check logs inspected; lint0issues. No correctness-acceptor report inspected.

Known all-root broad wording blacklist remains assigned T21.C01; no all-root PASS is claimed. No production release/push, new paid provider quality profile or all-module final consumer certification was executed by this reviewer. Documentation-only task makes no speedup/optimization claim. Unrelated `docs/task18/correctness 2.md` excluded. Existing historical relative links inside the deliberately preserved old integration payload are not current public API authority.

## Frozen substantive candidate SHA256 manifest

Includes every substantive changed/new T20 documentation/example/contract artifact, including ignored `.cursor/docs/INTEGRATION.md`. Mutable backlog/plan/traceability and acceptance reports/results are intentionally excluded; their final dispositions were inspected separately. Any subsequent substantive edit requires renewed acceptance.

| File | SHA256 |
|---|---|
| `README.md` | `8c3354d2635a39e95de250864a39c04d352b1fb167184f265c77307dd6531970` |
| `docs/architecture.md` | `11018e0c4a76ba0eba62bb5a2611710f463a70bc051182d1657e9aab7b790294` |
| `docs/capabilities.md` | `cbf31666ec6ce47496877980b35b7abf001efce99eb9f5e7eaedfa3a30ed9f4a` |
| `docs/errors-and-recovery.md` | `0771252c5054b8c7ec01b70cf2c9d8fb99d2639b93334ef6365b29be1a2d9fba` |
| `docs/integration.md` | `43e2990b3a53aa29392ec8e729c283c0d555cd1ef4120b6dde274d9057b5b8f6` |
| `docs/limits.md` | `07777ab132ccd0a1c8a9b4968c81fdb82f7e691f23d624f65202d667cf9ccf22` |
| `docs/ownership.md` | `db4b71bd6b5337d0771e4e1113aa5aff24513336878c63db5ff392d0bd9087cf` |
| `docs/project-policies.md` | `e190312101bf2e84e34c2b7471cafd32e953d56bc815c78afff1eab00d23b36b` |
| `docs/release/runbook.md` | `da06b64015487ee56e466e9aa81095d92bb4560b8394f5a22e2b1af9ad6b526c` |
| `docs/task20/T20.md` | `67160d44118673ea7acecdc310b90703c675a8e7314549bc82a1d6ededa9048a` |
| `docs/task20/history.md` | `333afc6152dbfefa00c0ff2c4683d79f539b7025b95a2ab1bc33a897fcea5058` |
| `docs/task20/historical-integration.md` | `b341dd0e2aca2446a24ae3d4463d44ee46010b6e22a78c35f4ae4839067caa20` |
| `.cursor/docs/INTEGRATION.md` | `499db394d5995b7e1baf6d8c0b2da9be270121ee865d2fb6b2b7a846291300fa` |
| `examples/local-bm25/main.go` | `f8cecee210fa0c501f4283059126c528b3e4c9063a274895e6c1402639450775` |
| `docs/task13/REMEDIATION.md` | `c1cbc42550329ad4c25b2f7550d31b8103a26750c0a8a823ce4f1e8ed662f423` |
| `docs/task14/REMEDIATION.md` | `fd7c8aaa747d25422b03b9397d311c9ecf97f5483f3b9f7540fb304c537a78c6` |
| `docs/task15/REMEDIATION.md` | `513abe9e7854641a5550f9140ea89f81d75cf76931afa3fc0d973a8477d498b1` |
| `docs/task16/REMEDIATION.md` | `e48e0c3ac7a08e30f31f4ec453a8f2ce4a22194d5904a05ba436e61d9dca7d1a` |
| `docs/task17/REMEDIATION.md` | `2293daa7058247eaf89bc01e6d9e334f2f1f30ad3846101e5da6f75bad2528a5` |
| `docs/task18/REMEDIATION.md` | `a9700aa820cbc1022956652e905dc0d20579fd4f7597a340d7428f8c527957c6` |
| `docs/task19/REMEDIATION.md` | `b6b4fb8a21ff1559bf17c445a0de372a4c1301edd09eedf8c5ac53060e16e96d` |
