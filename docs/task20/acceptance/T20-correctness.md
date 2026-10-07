# T20 independent correctness acceptance

Verdict: **PASS — no unresolved errors found** in the substantive T20 candidate. Baseline HEAD: `1672c0e029f81480003f797dac174e16f5e141e2`. Read-only implementation review; this acceptor neither implemented nor edited substantive candidate files and did not read the other acceptance report.

## Reviewed scope

Read T20 contract, five backlog criteria, master DOC1–DOC9, assigned original arch-docs findings A-D09/A-D10/A-D11/A-D13/A-D19/A-D20 and D61. Compared all current public guides with current implementation and relevant package contracts. The 21-file frozen manifest below includes ignored `.cursor/docs/INTEGRATION.md`; excludes mutable backlog/plan/traceability, acceptance reports/results and unrelated `docs/task18/correctness 2.md`.

- Complete local quickstart uses actual finalized typed schema, BM25 constructor, explicit unrestricted binding and execution pipeline. Tenant filter is explicitly search selection; no IAM or managed-snapshot claim. Protection precedes diagnostic/ordinary partial handling. Only returned documents are delivered; errors never resurrect diagnostic payload.
- Integration agrees with `retrieval/execution.go`, `query.go`, `orchestrator.go`, `partial_failure.go`, `document.go`, `resultset.go` and artifact APIs: zero Executed authoritative, explicit conditional configuration, fallback/rescue rules, terminal returned-result authority, comparable native scales, rank fusion and no implicit score fallback. Artifact `Resource` and exact support/contributor semantics agree with current contracts.
- Ownership/limits/recovery agree with current borrowed BYOT and core-owned copies, managed snapshot entry bounds, embedding limits, structured/full-envelope accounting, PDF bounds, ledger settlement, callback cooperation and observer attempt/pair units. No bounds are promoted into peak memory, arbitrary callback preemption or exhaustive candidate recall. Source citation and derivation-dependency roles remain separate.
- Capability matrix qualifies local/wire/live/process-crash/quality evidence and retained historical profiles. It makes no paid live PASS, power-loss, universal IAM, driver or semantic-quality certification claim. Scoped helper limits and legitimate external advisory runner are explicit.
- Release runbook checked directly against `scripts/release.py`, `release_state.py`, `release-modules.txt` and Makefile: exact reviewed SHA; eleven publishable modules and three excluded nested development modules; v2 guard; canonical manifests; isolated lightweight refs; atomic missing-ref push; persisted unknown/collision/recovery; clean-consumer gate. Disposable actual Git/Go fixture checks passed below. No production release/push performed.
- Project policy inventory does not select a license or invent a security address/SLA. Missing templates are not runtime blockers; publication/maintenance owner decisions and T21/T22 verification dependencies remain explicit.
- All 454 tracked docs/task13–19 historical files have unchanged tracked content relative to the baseline; only separate REMEDIATION pointers are new. Four existing task18 CSV worktree files (`process-costs.csv`, `round1-process-costs.csv`, `round1-scaling.csv`, `scaling.csv`) use CRLF while Git baseline blobs use LF; independent byte comparison confirms only that line-ending normalization, and the tracked diff is empty. Archived ignored integration bytes match the declared pre-T20 SHA256. The old ignored original is not available in Git HEAD, so this verifies archive integrity, not an independently recovered original.

## Independent commands and results

Environment: Darwin arm64, Go 1.26.5 (`/opt/homebrew/Cellar/go/1.26.5/bin/go`), Git 2.55.0, Python 3.14.6. Go commands use `GOWORK=off GOTOOLCHAIN=local GOCACHE=/tmp/ragy-t20-correctness-go` (acceptor-owned fresh cache).

| Check | Result |
|---|---|
| `go run ./examples/local-bm25` | PASS; exact `reset [acme]: Reset your password from account settings.` |
| `go test -count=1 -race ./retrieval ./lexical ./filter ./access ./examples/local-bm25` | PASS, exit 0. Four actual package suites; example has no repository test files and compiles. |
| Independent copied-example public probe `go test -count=1 -race -v ./...` | PASS, exit 0. Five display cases plus full-run and canceled-parent checks; no leaked SECRET/error text. |
| Exact README fenced Go/source comparison; gofmt check; current relative Markdown links | PASS. Code fences excluded from link parsing; historical obsolete links are preserved as history, not current-link assertions. |
| Tracked historical task13–19 diff check, archive SHA256, substantive final freeze check | PASS. |
| `PYTHONDONTWRITEBYTECODE=1 python3 scripts/release_test.py -v` | PASS; 9 tests, actual Git/Go and disposable local remotes. |
| `PYTHONDONTWRITEBYTECODE=1 python3 scripts/release_recovery_test.py -v` | PASS; 12 tests, actual failure/recovery/collision/unknown/atomic profiles. |
| `bash -n scripts/release.sh`; `git diff --check` | PASS. |

The independent display probe is retained in [correctness-display-probe.go.txt](T20-results/correctness-display-probe.go.txt). It copied the canonical example into a temporary module with an explicit root checkout replacement for changed-scope API verification; this is not the clean-consumer gate. Probes verify joined protection suppresses nonempty admitted/diagnostic payload; diagnostic-only partial cannot resurrect documents; ordinary empty failure reports failure; ordinary partial and success display only admitted documents; full local output; canceled parent delivers no output. Probe structure is Arrange–Act–Assert. Release fixture PATH selected the intact Go executable and all tests completed; no SKIP interpreted as PASS.

Root wording-blacklist replacement and all-module/fuzz/CI tooling belong to T21. Final all-module/live applicable profiles and clean consumer belong to T22. This PASS makes no all-root suite, all-module linter, public-proxy availability or final-goal completion claim. No runtime optimization was changed or speedup claimed, so before/after measurement is inapplicable.

## Frozen substantive SHA256 manifest

```json
{
  ".cursor/docs/INTEGRATION.md": "499db394d5995b7e1baf6d8c0b2da9be270121ee865d2fb6b2b7a846291300fa",
  "README.md": "8c3354d2635a39e95de250864a39c04d352b1fb167184f265c77307dd6531970",
  "docs/architecture.md": "11018e0c4a76ba0eba62bb5a2611710f463a70bc051182d1657e9aab7b790294",
  "docs/capabilities.md": "cbf31666ec6ce47496877980b35b7abf001efce99eb9f5e7eaedfa3a30ed9f4a",
  "docs/errors-and-recovery.md": "0771252c5054b8c7ec01b70cf2c9d8fb99d2639b93334ef6365b29be1a2d9fba",
  "docs/integration.md": "43e2990b3a53aa29392ec8e729c283c0d555cd1ef4120b6dde274d9057b5b8f6",
  "docs/limits.md": "07777ab132ccd0a1c8a9b4968c81fdb82f7e691f23d624f65202d667cf9ccf22",
  "docs/ownership.md": "db4b71bd6b5337d0771e4e1113aa5aff24513336878c63db5ff392d0bd9087cf",
  "docs/project-policies.md": "e190312101bf2e84e34c2b7471cafd32e953d56bc815c78afff1eab00d23b36b",
  "docs/release/runbook.md": "da06b64015487ee56e466e9aa81095d92bb4560b8394f5a22e2b1af9ad6b526c",
  "docs/task13/REMEDIATION.md": "c1cbc42550329ad4c25b2f7550d31b8103a26750c0a8a823ce4f1e8ed662f423",
  "docs/task14/REMEDIATION.md": "fd7c8aaa747d25422b03b9397d311c9ecf97f5483f3b9f7540fb304c537a78c6",
  "docs/task15/REMEDIATION.md": "513abe9e7854641a5550f9140ea89f81d75cf76931afa3fc0d973a8477d498b1",
  "docs/task16/REMEDIATION.md": "e48e0c3ac7a08e30f31f4ec453a8f2ce4a22194d5904a05ba436e61d9dca7d1a",
  "docs/task17/REMEDIATION.md": "2293daa7058247eaf89bc01e6d9e334f2f1f30ad3846101e5da6f75bad2528a5",
  "docs/task18/REMEDIATION.md": "a9700aa820cbc1022956652e905dc0d20579fd4f7597a340d7428f8c527957c6",
  "docs/task19/REMEDIATION.md": "b6b4fb8a21ff1559bf17c445a0de372a4c1301edd09eedf8c5ac53060e16e96d",
  "docs/task20/T20.md": "67160d44118673ea7acecdc310b90703c675a8e7314549bc82a1d6ededa9048a",
  "docs/task20/historical-integration.md": "b341dd0e2aca2446a24ae3d4463d44ee46010b6e22a78c35f4ae4839067caa20",
  "docs/task20/history.md": "333afc6152dbfefa00c0ff2c4683d79f539b7025b95a2ab1bc33a897fcea5058",
  "examples/local-bm25/main.go": "f8cecee210fa0c501f4283059126c528b3e4c9063a274895e6c1402639450775"
}
```
