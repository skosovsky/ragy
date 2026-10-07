# Repeatable verification

The command runner uses Python 3.9+ and Unix process-group cancellation (macOS/Linux); CI exercises Linux and local remediation records the actual macOS toolchain.

`scripts/check-modules.txt` is the validated fourteen-module development inventory. New/missing modules fail inventory validation. It includes the three nested example modules. `scripts/release-modules.txt` separately lists eleven publishable modules; development examples do not receive release tags. Root local BM25 is a package inside the core module.

| Command | Scope |
|---|---|
| `make test` | Cached developer race tests, once per module, standalone GOWORK=off. |
| `make acceptance` | Recorded exact validated Go/lint versions, lint and fresh `-count=1 -race` tests once per module, then explicit owning-module example builds. |
| `make lint` | All-module lint with standalone GOWORK=off. |
| `make examples` / `make test-examples` | Compile nested examples and root local onboarding; no second test loop. |
| `make fuzz FUZZ_SECONDS=2` | List actual packages/functions; independently fuzz each exact name with a 1–300 second per-function budget plus bounded process timeout. |
| `make versions` | Print actual Go/lint and recorded validated pins. |
| `python3 scripts/verify.py test --fresh --module adapters/openai` | Fresh changed-scope test selection; duplicate/unknown modules reject. |
| `python3 scripts/check_release_consumer.py <reviewed-full-sha>` | Exact source-derived release candidate and external consumer through local module proxy/disposable bare remote; all publishable modules, no ragy replacements. |

`GO`, `GOLANGCI_LINT` and `PYTHON` can select local executable paths. Use writable task-owned Go/GOPATH/module/lint caches in a restricted environment; keep public dependency checksum validation enabled. Fresh acceptance rejects a compiler/linter version mismatch against `scripts/toolchain.json`. The validated pin describes executed tooling, not a promise that it is the newest release or proof of every supported platform.

CI sets GOWORK=off globally, derives the matrix and Go/lint pins from the same files, performs each module's fresh race suite once and compiles its examples. Its separate full-history release-consumer job publishes only into a disposable local bare remote. The root documentation tests target current links, removed selectors and equality/parsing of the canonical executable onboarding; they do not ban natural-language migration words or rewrite historical acceptance.

These default commands verify local/wire/core contracts. Paid provider checks are opt-in. Actual PostgreSQL and configured native PDF-engine profiles have their own prerequisites, commands and captured runtime/corpus; use [capability verification scope](capabilities.md) and package guides. SKIP never proves an applicable required profile passed. Whole-module PASS does not certify remote service behavior, semantic quality or hardware power loss.
