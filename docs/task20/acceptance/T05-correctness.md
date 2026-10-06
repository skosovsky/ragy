# T05 — independent correctness acceptance

Verdict: **PASS** on the current substantive diff. Previous P2 is resolved; no unresolved findings. Baseline `f282696c8fdf15e13bcbba70fca8e6bb146e6b14`. No implementation edits by reviewer; completeness report not consulted.

## Resolved P2 — Prepare completion deadline cause

`Adapter.admit` invokes `read.Prepare` after its initial composed gate. Prepare calls the host Authority.ValidateRead. If that callback advances the shared ledger clock to equality and returns an error, admit returns that error immediately without a shared/local completion gate. The result is empty and protection/callback error remains, but simultaneous `context.DeadlineExceeded` is lost. This contradicts F04 callback completion gates and the T01/T02 simultaneous-cause contract.

Independent reproduction: `/private/tmp/ragy-t05-probes/overlay.json`, test `TestT05IndependentPrepareFailureRetainsSharedDeadline`. The authority first call succeeds at entry; second call during Prepare advances independent shared clock by its one-second remainder and returns a stable callback cause. Assert zero output/model calls plus errors.Is(cause) and errors.Is(DeadlineExceeded). Initial implementation failed the deadline assertion: `lost Prepare deadline 2 read protection failed`. On the current implementation this independent test passes with callback cause, protection and deadline all retained.

The current implementation checks both independent clocks before and after freshness callbacks, adds a clock-only check after Prepare and at admit/call/project completions, and preserves protection for real context cancellation. The clock-only path avoids a new authority dispatch after expiry. Permanent AAA Prepare regressions cover both shared and local clock leaps plus an authority failure; final delivery authority regressions cover both scopes. These were inspected and executed in the independently rerun race scope.

## Independent checks

- `GOCACHE=/private/tmp/ragy-t05-correctness-cache go test -race -count=1 ./recipe/... ./graphingest/extraction/...`: PASS, six packages.
- `GOCACHE=/private/tmp/ragy-task20-go-cache GOLANGCI_LINT_CACHE=/private/tmp/ragy-t05-correctness-lint golangci-lint run --allow-serial-runners ./recipe/... ./graphingest/extraction/...`: PASS, zero issues; installed exhaustruct deprecation warning.
- Independent overlay probes: real shared timer after backward clock leap suppresses late known model output and settles once PASS; quote/counter/validation/clone simultaneous callback failure and shared expiry retain both causes PASS; local leap during initial Authority callback suppresses next host callback PASS on latest gate ordering.
- `git diff --check`: PASS.
- Recheck after final implementation changes: independent six-package race PASS, lint zero issues, full independent overlay (`-run T05Independent -v`) PASS including formerly failing Prepare path.

Checks above are targeted and do not certify unrelated tasks or paid provider behavior. Known/unknown settlement, parent/shared/local independent epochs, exact equality, source-byte vs complete host envelope counter agreement and no hidden retry were inspected.

## Reviewed substantive SHA256

- `graphingest/extraction/extraction.go`: `6c97cc7d7a0eafe3d74484461c6f56640363a15647f3b7ff450a5e6f2996d677`
- `graphingest/extraction/contracts.go`: `e35ed0289f1a4861aa6fa83881a3c7df2824e5d1905f5f7bf5eea39a3561daab`
- `graphingest/extraction/README.md`: `39e66b41f36dda0a2923571e378c6dbbaf2b22ff316900b404c2471e6d019e9f`
- `graphingest/extraction/shared_deadline_test.go`: `ea9ff486c5c2c1c1c099345a07c3d1fcf9958e37a8653cdcc6a4844fcad65ffa`
- `recipe/budget/budget.go`: `ab6d50234ca54405868f40e2e446cfc73bb1c1839f8dbaa975c0c8327299c2e3`
- `recipe/budget/deadline_check_test.go`: `bf6d79230fe64190553f34d02fda39ac4854c62520eb7190ac95a531e7de8eb2`
