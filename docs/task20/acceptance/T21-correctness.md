# T21 independent correctness acceptance — R2 PASS

Baseline `b92813c472105ebf2f654c15e912048fafb37a79`. Read-only independent acceptance; no implementation participation. Completeness reports were not read. Normative T21 contract, original D61/DOC8 and arch-docs A-D12/A-D14–18 inspected. Current substantive diff is covered by the SHA256 manifest below. **PASS: no remaining correctness findings.** Initial failures are preserved in `T21-correctness-R1.md`; that report is historical failed acceptance.

## Repairs independently verified

| Previous finding | Final verification |
|---|---|
| P2 verify timeout leaves TERM-ignoring descendant after leader exits | Both wrappers use owned sessions in shared process_runner.py. TERM followed by unconditional group KILL in finally stops descendants even when leader communicate returns. Actual process regression, verification profile, PASS. |
| P2 consumer timeout leaves descendant | Same shared runner with bounded consumer timeout300, test configurable shorter timeout. Actual consumer profile with closed pipes and TERM-ignoring heartbeat descendant PASS. |
| P2 valid Go Fuzz U+037A silently omitted | Authoritative Go listing names retained, exact escaped anchored selectors. Independent actual two-function ASCII/U+037A package listed and fuzzed both functions for1s; PASS. |

An additional actual KeyboardInterrupt probe interrupted the outer verification runner while its leader owned a TERM-ignoring descendant. Nonzero interruption propagated and descendant heartbeat stopped after return: PASS. These checks execute real subprocesses; mocked dispatch alone was not accepted as cancellation evidence. Group cancellation is a Unix/macOS/Linux profile, not a promise to sandbox deliberately escaped sessions.

## Evidence

All independent commands exited0; logs under `T21-results/`:

- `correctness-R2-verify.log`: Python verification suite10 tests, including real Go Unicode listing/execution, inventory drift/duplicates, fresh/cached dispatch, bounded budgets, exact tool pins and nonzero failure propagation.
- `correctness-R2-consumer-tools.log`:3 consumer tests, actual Go package selection, machine stdout separate from diagnostics, exact tracked ZIP excluding nested module/untracked payload.
- `correctness-R2-process-groups.log`: actual timeout test covers both wrappers and descendant heartbeat cessation.
- `correctness-R2-adversarial.log`: independent actual multiple fuzz functions with Go-only Unicode letter, each anchored selector executed separately; actual SIGINT group cleanup PASS.
- `correctness-R2-doc-tests.log`: fresh Go1.26.5 `GOWORK=off go test -count=1 -race` root targeted quickstart equality/parsing, current public links and removed selectors PASS.
- `correctness-R2-clean-consumer.log`: independently reran full exact-source disposable release/consumer verification. Source `b92813c472105ebf2f654c15e912048fafb37a79`, candidate `d49e4ef344565195fcf32bd7a46fa68355f34355`, version `v0.0.1`;11 publishable modules,48 public packages,11 exact tags, no examples tags, no ragy replacements, local candidate ZIP SHA identity checked. External consumer canonical onboarding fresh race test and build PASS. Publication was exclusively into disposable local bare remote; caller Git state unmodified.

Actual compiler `/opt/homebrew/Cellar/go/1.26.5/bin/go`, GOTOOLCHAIN=local and independent task-owned `/tmp/t21-correctness-gocache`. Public external dependency checksum validation remains enabled; local candidate uses scoped GONOSUMDB only for unpublished ragy identities. No production release/push occurred.

Existing actual all-module acceptance.log inspected by parsing dispatched command records: exactly14 separate lint commands,14 fresh `-count=1 -race ./...` commands and4 explicit owning-module/root example builds, with GOWORK=off. T21 root checks refreshed after tooling repairs. All-module fuzz log records actual bounded campaign. Release isolation/recovery fixture9+12 evidence retains prior contract. CI inspected: same deterministic matrix/pins, standalone GOWORK=off, fresh race once per module, owning-module example compile, root tooling tests including shared process runner, full-history exact-source clean consumer. Neither duplicate examples test loop nor empty PHONY target remains. CI YAML behavior itself has not been remotely executed by this local review.

Targeted documentation tests replace word blacklist; historical/Deprecated prose remains legitimate. New public lifecycle schema is byte-identical to accepted task18 schema (independently checked); recorded schema revalidation covers8 positive/42 negative cases. No runtime algorithm/optimization changes or speedup claim; before/after N/A. Candidate source intentionally precedes tooling-only uncommitted T21 changes per normative contract; T22 must repeat final-source acceptance. Optional live/paid service profiles, parser/DB runtime profiles, semantic quality and power-loss are distinct; this tooling PASS does not promote SKIP into live certification or declare whole goal complete.

## Substantive SHA256 manifest

17 files. Mutable bookkeeping (`backlog.json`, `plan.md`, `traceability.json`), reports/results and unrelated `docs/task18/correctness 2.md` excluded.

| Path | SHA256 |
|---|---|
| `.github/workflows/ci.yml` | `be754b5642a5cee931021895abc2189cceffda8862316d8fcf83281ce1fa5c33` |
| `.gitignore` | `bfea6c33ad0a67806c7d960b44abf9098cf639a4bc2c4d13078e293ca995e431` |
| `Makefile` | `5e0a44e9f7aca3c8a674da074ffe997c750fe138a9e838b9d9e74120634bdaa0` |
| `docs/project-policies.md` | `638b23d7636597052818f09e7aee7170b3b1937f1cee46bc5feb29a1ad022716` |
| `docs/release/runbook.md` | `2fe96959ba7df5aee5b977f38a5ad23272ad7ea654233b95c603f6dbb512697c` |
| `docs/verification.md` | `d2140e61029d5ef9bd075c143e7ef9a29324bd9738dfcb996caf2d35973dba5e` |
| `docs/task20/T21.md` | `adc1131573a84d778357b9bb8fef3a2b3f5e47c8c35c6b1e3be6fc3bfac4f9db` |
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
