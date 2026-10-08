# Go verification and shell release migration

## Contract

Make is the developer interface. `test` runs race tests in every development module;
`lint` checks formatting and the pinned linter; `check` adds fresh tests, example
builds, real PostgreSQL, exact-source consumers and isolated release tests.
`test-integration` runs the latter profiles explicitly. Missing prerequisites fail.
`bench`, `fuzz`, `cover` and paid `test-live` are separate commands.
All commands use GOWORK=off. Tooling dependencies belong to the unpublished tooling
module. CI invokes the same Make targets. No production publication is authorized
by this migration.

Release remains Bash plus Git and standard Go commands, with `patch`/`break`, exact
source/candidate validation, isolated checkout, explicit atomic tag pushes and
inspect/resume/finish recovery. Caller contents and unrelated refs are preserved.

PDF replacement requires a pure-Go backend passing the existing layout, source,
limits and cancellation contracts. If neither evaluated backend qualifies, retain
the existing adapter explicitly as a blocker; never silently reduce its contract.

## Python disposition inventory

| Existing files | Replacement / disposition |
|---|---|
| scripts/verify.py | Direct Make targets; module inventory tests in tooling |
| scripts/verify_test.py | Inventory, failure propagation and fuzz-selection Go tests; discard dispatcher-specific mocks |
| scripts/process_runner.py, scripts/process_runner_test.py | Standard Go test timeouts, context-aware subprocess helpers and cleanup; no shared runtime runner |
| scripts/release.py, scripts/release_state.py | Bash release and plain-text recovery records |
| scripts/release_test.py, scripts/release_recovery_test.py | Go tests using disposable real Git repositories |
| scripts/check_release_consumer.py, scripts/check_release_consumer_test.py | Go integration tests for module artifacts and clean consumers |
| scripts/check_context_bridge.py | Go integration tests with pinned peer checkout and published dependency modes |
| scripts/task19_verify.py, scripts/task19_runner_test.py | Remove duplicate historical runner; preserve schema/corpus/evaluation checks in Go |
| scripts/task19_wire.py | Existing provider behavior tests plus explicit fixture validation; remove source-marker auditing |
| docs/task12/verify_coverage_schema.py | Go JSON Schema positive/negative tests |
| docs/task12/verify_evidence_schema.py | Go JSON Schema positive/negative tests |
| docs/task12/verify_lifecycle_schema.py | Go JSON Schema positive/negative tests |
| docs/task12/verify_locator_schema.py | Go JSON Schema positive/negative tests |
| docs/task12/verify_tensor_run_schema.py | Go schema and relational checks |
| docs/task12/build_evidence_schema.py | Retain versioned schema; remove historical generator after schema tests cover it |
| docs/task12/audits/check_cli_capture_consistency.py | Go receipt consistency tests, no live model calls |
| docs/task17/verify_evidence_schema.py | Go evidence corpus/schema and relational checks |
| docs/task18/verify_lifecycle_schema.py | Go lifecycle v2 schema positive/negative tests |
| docs/task18/summarize-benchmarks.py | Historical analysis: preserve existing reports; use standard Go benchmarks going forward |
| examples/conformance/datasets/task19/verify.py | Go corpus/split/schema/digest tests |
| adapters/pdf/engine.py | Conditional pure-Go replacement after feasibility gate |
| adapters/pdf/testdata/create_fixture.py | Conditional Go fixture generation after feasibility gate |
| adapters/pdf/testdata/verify_fixture.py | Conditional Go fixture assertions after feasibility gate |
| adapters/pdf/testdata/verify_engine_errors.py | Conditional Go backend error tests after feasibility gate |

Historical reports retain their original claims and commands as dated evidence.
Active documentation must describe only the current implementation. Migration is
complete only after a clean-source check, release fixtures and the PDF gate pass.

## Implementation status

The tooling migration is implemented: Make invokes tools directly, the unpublished
tooling module contains schema/corpus/receipt and consumer tests, PostgreSQL is an
owned Docker fixture, and release is Bash with plain-text recovery state. Live
provider calls use the explicit `live` build tag. The Python dispatchers and their
duplicate historical entrypoints have been removed: 25 of the 29 Python files.

The remaining four files belong to the PDF adapter and its fixtures. Both pinned
pure-Go candidates failed existing geometry/layout and cancellation requirements;
see [the feasibility report](pdf-go-feasibility.md). No public PDF API or limits
were weakened. Until the owner resolves that blocker, `make check` requires the
retained actual parser and fails explicitly when its prerequisite is absent. This
is a partial migration, not a Python-free acceptance claim.
