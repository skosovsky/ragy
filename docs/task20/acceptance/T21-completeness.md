# T21 independent completeness acceptance — R2

Baseline: `b92813c472105ebf2f654c15e912048fafb37a79`. Repeated acceptance on repaired final diff; R1 report remains historical and does not establish final acceptance. Reviewer did not implement T21 and did not read the correctness report. Reviewed current substantive diff/new artifacts, normative `docs/task20/T21.md`, original master D61 and original arch-docs A-D12/A-D14–18 with corresponding traceability dispositions.

Verdict: **PASS. Criteria 5/5 = 100%. Assigned source coverage 7/7 = 100% (six arch-docs findings plus one D61 tooling dependency).** No unfulfilled or blocked T21 requirement found. T22 final acceptance/live profiles remain their separately assigned scope.

| Criterion | Status | Evidence |
|---|---|---|
| T21.C01 | выполнено | Global prose blacklist removed. `docs_test.go` checks exact removed retrieval selectors, current relative link existence, parsed onboarding fence and byte equality to runnable canonical source. New public lifecycle schema exists at its documented path, byte-identical to accepted historical schema. Cached developer checks and fresh acceptance are explicit distinct modes. |
| T21.C02 | выполнено | Deterministic validated inventory has exactly fourteen actual modules; drift/duplicates reject. Runner and normal CI force GOWORK=off. Actual acceptance log has exactly fourteen unique lint commands and fourteen unique `test -count=1 -race` commands, four owning-module example builds and no failing exits. Shared toolchain JSON pins actual validated Go1.26.5/lint2.14.0; CI derives both pins. |
| T21.C03 | выполнено | Actual Go package/function listing, anchored individual fuzz selector, positive1–300s budget, shared hard subprocess timeout and process-group cancellation, including TERM-ignoring descendants after leader exit; failed listing propagates. Multiple-name/Unicode/bare-prefix and actual timeout fixtures pass. Exact Go-listed U+037A identifiers are no longer filtered by Python identifier rules; actual Go fuzz fixture executed successfully. Both command wrappers share process_runner and actual descendant-heartbeat regression. CI and acceptance invoke this regression. Actual all-module1s campaign dispatches `FuzzRecursiveCoordinates` independently, timeout61s, with no failed exits. Empty bench-hotpath removed; examples alias implements build. |
| T21.C04 | выполнено | Existing tracked-source eleven-publishable-module manifest excludes examples; portable Python editor and v2+ guard retained. Runbook documents exact source/manifests/version/tag/recovery/platform boundaries. Actual source b92813c → manifest-only candidate0e6cba09e539845606f33f9cea530c05ebf9d079, versionv0.0.1, all11modules/48publicpackages: external consumer race/build PASS, no ragy replacements, ZIP hashes checked, exact tags and ancestry checked, caller/candidate state preserved. |
| T21.C05 | выполнено | Independently executed ten runner and three consumer tooling fixtures plus one both-wrapper descendant regression PASS; fresh race current documentation checks PASS; `git diff --check` PASS. Recorded release isolation9/recovery12 and public schema8positive/42negative cases PASS. Source inspection limits publication to disposable repositories/local bare remote; no production release or GitHub run claimed. |

| Assigned source | Status | Disposition/evidence |
|---|---|---|
| D61 tooling dependency | выполнено | Targeted docs checks, standalone/fresh fourteen-module CI, separate publishable manifest and repeatable clean consumer fulfilled. Stable public guide portion accepted T20; final repeat explicitly T22. |
| arch-docs:A-D12 | выполнено | Word blacklist replaced with targeted symbols/links/executable source checks; historical and legitimate Deprecated prose no longer rejected. Public schema repair/8+42 validation also closes an actual target link. |
| arch-docs:A-D14 | выполнено | GOWORK=off normal matrix and repeatable actual exact-source clean consumer; external BYOT module included. |
| arch-docs:A-D15 | выполнено | Cached test versus fresh acceptance, once-per-module test scope, explicit example compilation and no redundant example suite. |
| arch-docs:A-D16 | выполнено | Enumerated fuzz functions, exact selectors/budget/cancellation and multiple-name fixture; empty PHONY targets removed/implemented. |
| arch-docs:A-D17 | выполнено | Retain justified: portable editor/scope/semantic import guard already accepted T03/T04; release fixtures and actual all-module consumer revalidate those boundaries. Current v0 not misrepresented as v2 bug. |
| arch-docs:A-D18 | выполнено | Actual compiler/linter pins and mismatch rejection; capability/verification documentation preserves default versus actual service/parser/quality distinctions without new paid benchmark demand. |

Independent executed commands (all exit0; task-owned caches reused only for compilation, tests fresh):

- `GO=/opt/homebrew/Cellar/go/1.26.5/bin/go GOTOOLCHAIN=local GOCACHE=/tmp/ragy-t21-completeness-gocache GOPATH=/tmp/ragy-t21-completeness-gopath PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_test.py -v`: 10 tests PASS; actual U+037A Go campaign PASS, 125103 executions.
- Same environment, `python3 scripts/check_release_consumer_test.py -v`: 3 tests PASS, including actual Go package selection and actual Git/proxy archive fixture.
- `GOWORK=off GOTOOLCHAIN=local GOCACHE=/tmp/ragy-t21-completeness-gocache GOPATH=/tmp/ragy-t21-completeness-gopath /opt/homebrew/Cellar/go/1.26.5/bin/go test -count=1 -race -run '^(TestQuickstartMatchesExecutableSource|TestCurrentDocumentationLinks|TestCurrentDocsDoNotReferenceRemovedRetrievalSymbols)$' .`: PASS1.227s.
- `python3 scripts/verify.py modules --json`: fourteen exact modules, inventory validation PASS.
- Same Python/Go environment, `python3 scripts/process_runner_test.py -v`: one test with both verification and consumer subprofiles PASS2.644s; child readiness and stopped heartbeat establish no surviving descendant work.
- `git diff --check`: PASS.

Inspected recorded evidence: `T21-results/acceptance.log`, `final-root-acceptance-R2.log` (fresh root lint/race/build and repaired runner/consumer/process fixture completion), `versions.log`, `all-module-fuzz.log`, `fuzz.log`, `clean-consumer-R2.log`, `release-isolation.log`, `release-recovery.log`, `public-schema.log`, and failed development attempts documented in `attempts.md`. Failed development logs are retained as failures and never counted as PASS. No cached result establishes fresh acceptance. Actual GitHub/Linux execution is not claimed; local macOS executed checks and source/CI configuration validation establish this T21 scope. T22 repeats the final accepted source/candidate and required actual PostgreSQL/parser profiles. No runtime algorithm optimization or speedup claim; before/after N/A.

## Substantive SHA256 manifest

Seventeen substantive changed/new files. Mutable backlog/plan/traceability, acceptance reports/results, generated Python bytecode and unrelated `docs/task18/correctness 2.md` excluded. `.gitignore` now ignores verification bytecode.

| File | SHA256 |
|---|---|
| `.github/workflows/ci.yml` | `be754b5642a5cee931021895abc2189cceffda8862316d8fcf83281ce1fa5c33` |
| `.gitignore` | `bfea6c33ad0a67806c7d960b44abf9098cf639a4bc2c4d13078e293ca995e431` |
| `Makefile` | `5e0a44e9f7aca3c8a674da074ffe997c750fe138a9e838b9d9e74120634bdaa0` |
| `docs/project-policies.md` | `638b23d7636597052818f09e7aee7170b3b1937f1cee46bc5feb29a1ad022716` |
| `docs/release/runbook.md` | `2fe96959ba7df5aee5b977f38a5ad23272ad7ea654233b95c603f6dbb512697c` |
| `docs/task20/T21.md` | `adc1131573a84d778357b9bb8fef3a2b3f5e47c8c35c6b1e3be6fc3bfac4f9db` |
| `docs/verification.md` | `d2140e61029d5ef9bd075c143e7ef9a29324bd9738dfcb996caf2d35973dba5e` |
| `docs_test.go` | `44e05d445597def9a7dc358c4c0db2e970d3da0fe35e9f8844f66d0407aff13f` |
| `schemas/lifecycle.schema.json` | `c4978dda3d7a7918f49d5031f49e71c7729913600d2f19abf53bec3e4dd5ce05` |
| `scripts/check-modules.txt` | `3c2951334fb22ba61b8dad44354a3b3c5a9c4cfdf2e3b3a84881758ab8a8bc19` |
| `scripts/check_release_consumer.py` | `e3cf10f8763ac9ac7034c825a405a32f34d77196fdd3c4b9344dda741229317e` |
| `scripts/check_release_consumer_test.py` | `84f525869c1b931c7575d0d0fb5ce6d912eabcc723950ccf06b2ea788149396c` |
| `scripts/process_runner.py` | `ef7f403e15a15f45204d388bf7e16fd249481df313455808d6c8c1dc8fb861fd` |
| `scripts/process_runner_test.py` | `bd045088bdf3aba9e4360fe90e14a492a35d860593cc91a00b116be09cda250c` |
| `scripts/toolchain.json` | `8488cb7969df141d357ce8966f114a48b4fc28acce62e34332f510de11692a82` |
| `scripts/verify.py` | `fe5aa110cb450f209362458923453e938ec6e245eb388a75a05b946726c7ddc5` |
| `scripts/verify_test.py` | `22b20bb3de7cb113b36a6054048688a5788184020c0bd7e938d0fc6f6898b25a` |
