# Repository verification

Run `make check` before release. Make invokes Go and the pinned linter directly;
there is no Python verification runner. Go/tooling versions are in
`Makefile`. All modules run with GOWORK=off. Development modules are discovered from go.mod files, excluding hidden directories and vendor.
Publication modules are also discovered automatically: project.mk selects root/adapter modules and excludes development modules. `make release-modules` shows the result; Go tests check it against tracked manifests.

| Command | Scope |
|---|---|
| `make modules` | List automatically discovered development modules. |
| `make test` | Race tests in every development module, including tooling tests. |
| `make lint` | Configuration, formatting diff and all-module lint; no rewriting. |
| `make check` | Prerequisites, lint, fresh race tests, example builds, PostgreSQL and consumer/release integrations. |
| `make test-integration` | Isolated PostgreSQL, artifact/peer/published consumers and actual PDF parser. |
| `make examples` | Build all examples in their owning modules. |
| `make test-live` | Tests named TestLive with the live build tag across all modules; missing credentials/configuration fail. |
| `make fuzz FUZZ_SECONDS=2` | Every Go-listed fuzz function separately; a positive number of seconds each, default 30. |
| `make bench` / `make cover` | Benchmarks / per-module coverage. |
| `make versions` | Show and verify recorded Go/linter versions. |

Tooling unit/release-fixture tests and integration consumer tests use separate build profiles; they run once each in `check`.

Use `V=1` to show shell commands. `GO` and `GOLANGCI_LINT` override executable paths.
For a targeted test use ordinary Go: `cd adapters/openai && GOWORK=off go test -race ./...`.
The full check requires Git, Make, Bash, the pinned Go/linter, network access for
pinned dependencies, and a running Docker daemon. PostgreSQL gets an isolated
container with a fixed image digest, readiness deadline and cleanup on exit.
No caller/sibling working tree or production database is used by integration tests.

**Pending PDF migration:** the pure-Go candidates failed the existing PDF contract
(see [evidence](pdf-go-feasibility.md)). The optional adapter is retained. Ordinary
Go tests and tooling need no Python; the full actual-PDF profile currently requires
`RAGY_PDF_PYTHON` pointing to the existing interpreter with pdfplumber/pypdf. Missing
configuration fails `make check` explicitly. This exception must be removed only
after a compatible backend or contract change is approved.

Live provider calls, performance measurements and fuzz campaigns are separate from
`check`. Core/fixture success does not attest live provider quality or hardware
power-loss behavior. Historical task reports retain the commands used at that time.

## Shared infrastructure and project checks

`Makefile` contains the reusable Go/lint commands and ordered check sequence.
`project.mk` supplies ragy's prerequisites, example builds and integrations through
`prerequisites-project`, `examples-project` and `check-project`. Without a project
file these three extension points are empty. Release-specific project targets are
mandatory and documented in the [release contract](release/runbook.md).

The full `.golangci.yml` remains directly consumable by the linter and editors.
Project import grouping and fixture exceptions are marked in place; there is no
configuration generation or merge step. Toolchain versions remain pinned.
