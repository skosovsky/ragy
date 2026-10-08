# Repository verification

Make invokes standard Go commands and golangci-lint directly. Modules are discovered
from go.mod files, excluding hidden directories and vendor. All commands use
GOWORK=off. There is no aggregate check target or tool version validation target.

| Command | Scope |
|---|---|
| `make modules` | List all discovered development modules. |
| `make test` | Fresh ordinary race tests across all modules. |
| `make test-integration` | Files with integration build tag; execute TestIntegration… functions only. |
| `make test-e2e` | Files with e2e build tag; execute TestE2E… functions only. |
| `make test-live` | Files with live build tag; execute TestLive… functions only, including paid calls. |
| `make lint` | Formatting diff and lint without rewriting files. |
| `make fix` | Go fix, formatting and lint fixes; modifies files. |
| `make fuzz` | Every discovered fuzz function separately, 30 seconds per function. |
| `make bench` / `make cover` | Benchmarks / per-module coverage. |

Build tags alone do not exclude ordinary test files. Profile targets combine the tag
with a matching test-name prefix so a module without such tests executes none.
Use the same convention for new tests; each profile runs directly through Go too:

```sh
GOWORK=off go test -race -tags=integration -run '^TestIntegration' ./...
GOWORK=off go test -race -tags=e2e -run '^TestE2E' ./...
```

Recipes use tools from PATH and explicitly propagate command failures.
Tool versions are pinned in CI, not enforced by Make. CI and source release gates
run lint, fresh unit tests, integration and e2e sequentially.

## Project content

`examples/` contains runnable usage/consumer examples. `tooling/` is a separate
Go test module for Make/release contracts, historical schema/receipt guarantees and
consumer composition; its dependencies do not enter the core module. All discovered
modules are tested and published, including examples and tooling.
There is no separate publication inventory or directory exclusion.

Example packages compile as part of ordinary tests. Integration and e2e dispatch
is common to every module and needs no project hooks.
The complete linter configuration remains a regular .golangci.yml, with local
import grouping and fixture exceptions marked in place.

PostgreSQL tests use the integration tag. Each test starts a uniquely named container
from a pinned image, waits for TCP readiness and cleans it up with t.Cleanup. Missing
Docker or a failed prerequisite fails the selected profile, never skips it.

PDF parser checks use integration; projection, durable publication and retrieval
pipelines use e2e. The retained backend requires RAGY_PDF_PYTHON pointing to an
interpreter with pdfplumber/pypdf. Missing configuration fails selected profiles.
Ordinary PDF tests need no Python. The evaluated pure-Go replacements did not meet
the required geometry/layout and cancellation contracts.

Performance measurements, fuzz campaigns and paid provider calls remain separate.
Historical reports retain the commands used when their results were recorded.
