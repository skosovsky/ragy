# T21 independent correctness acceptance — initial FAIL

Baseline `b92813c472105ebf2f654c15e912048fafb37a79`. Read-only source acceptance; no implementation participation. Completeness report was not read. Normative T21 contract, D61/DOC8 and arch-docs A-D12/A-D14–18 inspected.

## Blocking findings

1. **P2 — timeout cleanup leaves subprocess descendants alive.** `scripts/verify.py:run` sends SIGTERM to the process group, but successful `communicate()` after leader exit skips SIGKILL. Actual Python parent starts a child ignoring SIGTERM, then sleeps; timeout=0.5 kills leader but child remains alive (`os.kill(pid, 0)` succeeds). Probe child was manually killed afterward. This violates bounded process-group cancellation. Reproduce with child `signal.signal(signal.SIGTERM, signal.SIG_IGN); write_pid(); time.sleep(30)` and parent `subprocess.Popen(child); time.sleep(30)`, capture=False. Ensure cleanup of the group even if leader has already exited; add actual adversarial descendant fixture.
2. **P2 — clean consumer timeout also leaves descendants alive.** `scripts/check_release_consumer.py:command` uses `subprocess.run(timeout=300)` without an isolated group. Independent actual child probe, shortening timeout to 0.5 using a wrapper, leaves descendant alive after TimeoutExpired. A timed-out release/Go process can continue mutating its disposable repository after cleanup begins. Add shared bounded process-group termination/cancellation and regression.
3. **P2 — actual valid Go fuzz function silently omitted.** `scripts/verify.py:fuzz_names` checks Python `str.isidentifier`, which is narrower than Go Unicode identifier admission. Actual intact Go1.26.5 fixture `func Fuzzͺ(f *testing.F)` (suffix U+037A GREEK YPOGEGRAMMENI, Unicode letter) passes `go test -list=^Fuzz` and lists `Fuzzͺ`; parser returns `[]`. This contradicts enumeration of each actual fuzz name. Parse authoritative Go listing without Python-specific identifier constraints; add actual Go fixture with multiple fuzz functions including this name.

## Independent checks and reviewed evidence

Independent actual verify_test.py 9 tests and check_release_consumer_test.py 3 tests PASS under Go1.26.5, GOWORK=off and task-owned GOCACHE. They do not cover the three adversarial failures above. Current module inventory contains fourteen distinct actual modules; eleven-entry publishable manifest excludes examples. CI explicit GOWORK=off, same module/pin matrix, fresh race once per module and example compile inspected. Targeted public docs checks and canonical source equality replace blacklist. Actual clean-consumer implementation inspected for exact committed source, ancestry/manifest-only changes, disposable remote refs, all publishable imports, local ZIP identity and no ragy replacements. Public lifecycle schema inspected. No runtime algorithm or optimization changes; before/after metrics N/A. Existing actual recorded results remain scoped evidence, not cancellation proof. This FAIL must be replaced by a repeated independent acceptance on the repaired final diff.

## Substantive SHA256 manifest

Mutable bookkeeping, reports/results and unrelated duplicate excluded.

| Path | SHA256 |
|---|---|
| `.github/workflows/ci.yml` | `8067628f5ce369adac34dc6a3f5d46a2c627a08034d3b9b5b9acd4b1532ae37e` |
| `Makefile` | `5e0a44e9f7aca3c8a674da074ffe997c750fe138a9e838b9d9e74120634bdaa0` |
| `docs/project-policies.md` | `638b23d7636597052818f09e7aee7170b3b1937f1cee46bc5feb29a1ad022716` |
| `docs/release/runbook.md` | `2fe96959ba7df5aee5b977f38a5ad23272ad7ea654233b95c603f6dbb512697c` |
| `docs/verification.md` | `d2140e61029d5ef9bd075c143e7ef9a29324bd9738dfcb996caf2d35973dba5e` |
| `docs/task20/T21.md` | `adc1131573a84d778357b9bb8fef3a2b3f5e47c8c35c6b1e3be6fc3bfac4f9db` |
| `docs_test.go` | `44e05d445597def9a7dc358c4c0db2e970d3da0fe35e9f8844f66d0407aff13f` |
| `schemas/lifecycle.schema.json` | `c4978dda3d7a7918f49d5031f49e71c7729913600d2f19abf53bec3e4dd5ce05` |
| `scripts/check-modules.txt` | `3c2951334fb22ba61b8dad44354a3b3c5a9c4cfdf2e3b3a84881758ab8a8bc19` |
| `scripts/check_release_consumer.py` | `9bb520b30fdc439319703b392b99f0d60bb4dc3e337656bb205f0067b93271b5` |
| `scripts/check_release_consumer_test.py` | `84f525869c1b931c7575d0d0fb5ce6d912eabcc723950ccf06b2ea788149396c` |
| `scripts/toolchain.json` | `8488cb7969df141d357ce8966f114a48b4fc28acce62e34332f510de11692a82` |
| `scripts/verify.py` | `be12b01281341294dddb458e8c8a231099c707d00c014ab602787e9322c79424` |
| `scripts/verify_test.py` | `2ef7e82cad878b89de897e80d2eefb187f83f3bc08db1bf75d74d04ac8b95ee0` |
