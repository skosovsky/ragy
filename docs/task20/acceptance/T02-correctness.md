# T02 independent correctness acceptance

Baseline HEAD: `4776e89c12b83d1db41385426d982f25f5a51145`.
Reviewer did not implement this task or read the completeness report. Reviewed current tracked diff, new Go files, T02 description and remediation contract. F04 shared extraction deadline is excluded and remains T05.

## Verdict: PASS — no outstanding correctness findings

Reviewed the latest implementation after the renderer fix. Static adversarial review covered local versus parent timers, stage-origin wrapping, settlement exactly once, ordinary failed journals, early planner/assessor/encoder output-shape validation, artifact callback provenance, graph simultaneous causes and summary budget-stop admission. No additional actionable finding established.

Initial review identified one P2: standalone Render lost ErrProtocol when a measurement callback canceled its parent. `access.NonSkippable` intentionally collapses joined siblings through Protect. The new renderer-local `protectArtifactFailure` recovers lost public sentinel classification while preserving that baseline privacy behavior. Global Protect remains unchanged. The independent repro now passes, including joined protection + protocol + private typed side payload, with and without cancellation: deadline/protocol/protection survive, artifact remains empty and errors.As cannot recover the private sibling. Recipe-local provenance continues to prevent host protection from becoming bounded local expiry. Arbitrary private side-error identity is intentionally not promised by the existing protection sanitization contract.

Independent temporary repro files: `/private/tmp/ragy-t02-adversarial/artifact_adversarial_test.go`, overlay `/private/tmp/ragy-t02-adversarial/overlay.json`. These tests were prepared independently before the implementation repair and retained for repeat verification. Reviewer did not modify implementation or regression sources and did not read the completeness report.

## Commands and results on latest diff

- `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 ./access/... ./recipe/... ./graphingest/extraction/... ./retrieval/...`: exit 0; eight packages PASS.
- After the private test error-type rename required by errname, `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 ./retrieval ./access`: exit 0, both packages PASS.
- `GOCACHE=/private/tmp/ragy-task20-go-cache GOLANGCI_LINT_CACHE=/private/tmp/ragy-task20-lint-review-cache golangci-lint run --allow-serial-runners ./access/... ./recipe/... ./graphingest/extraction/... ./retrieval/...`: final exit 0, 0 issues; installed exhaustruct deprecation warning only. An earlier run observed errname before the concurrent rename; final run uses the renamed type.
- `git diff --check`: exit 0.
- `GOCACHE=/private/tmp/ragy-task20-go-cache go test -overlay=/private/tmp/ragy-t02-adversarial/overlay.json -race -count=1 -run TestT02StandaloneRenderer ./retrieval`: exit 0; independent cancellation/cause/privacy repros PASS. Before the repair this command reproduced the lost protocol class.

Acceptance covers T02 only. F04 shared extraction deadline, live services, release and global completion are not claimed here.

## Reviewed implementation SHA256

```
1134df7599a2388f1ffcc84e02e819605fc4f307b95a8c1659f1e42581d7cb4e  recipe/run.go
b1010c69636e974292bd480eddce45b077dfa16665bc32e648e8b089b82b1082  recipe/evidence.go
20d853ca65aa929536622a6e7a29d86283848fb6fa3a5fe470f3c9db0a9c85cf  recipe/artifact_scope.go
1336026ac46357f34c00a6c9ace0c6ecc2264f8dd0fc75cdae831a1c60f286d3  retrieval/artifact.go
57f2327a18a098a8c31cd2511622e36cf30426141a6b8053276d8496b11deb9b  recipe/graphexpand/expand.go
4a3933600c2e90716583f993338b66119dc016b548220f01b3a732e1257a7a0f  recipe/graphsummary/run.go
e014639b558c54ac84ec54b228a2ebd1b66181025dd90c238ed5dc50c337fb80  graphingest/extraction/extraction.go
72995b4aeaaa7d859a1a8b69c85bc6dc4d30757944d9105ebadef2c0b083f45a  retrieval/artifact_deadline_precedence_test.go
25b35941dd95ecf662c79dbef8c92da1da56e6d01a579945859a72f6e88695ab  docs/task20/T02.md
```
