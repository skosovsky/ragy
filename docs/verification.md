# Repository verification

Run `make check` before release. Make invokes Go and the pinned linter directly;
there is no Python verification runner. Go/tooling versions are in
`scripts/toolchain.mk`. All modules run with GOWORK=off. Development and publishable
inventories are explicit and independently checked by Go tests.

| Command | Scope |
|---|---|
| `make test` | Race tests in every development module, including tooling tests. |
| `make lint` | Configuration, formatting diff and all-module lint; no rewriting. |
| `make check` | Prerequisites, lint, fresh race tests, example builds, PostgreSQL and consumer/release integrations. |
| `make test-integration` | Isolated PostgreSQL, artifact/peer/published consumers and actual PDF parser. |
| `make examples` | Build all examples in their owning modules. |
| `make test-live` | Explicit paid-provider profile; missing credentials/configuration fail. |
| `make fuzz FUZZ_SECONDS=2` | Every Go-listed fuzz function separately; 1–300 seconds each. |
| `make bench` / `make cover` | Benchmarks / per-module coverage. |
| `make versions` | Show and verify recorded Go/linter versions. |

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
