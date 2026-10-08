# Tooling migration acceptance — 2026-10-08

Runtime source checked: `4af708e` (clean, independent clones of local main).

| Environment | Command | Result |
|---|---|---|
| macOS / darwin arm64, Go 1.27.1, golangci-lint 2.14.0 | `RAGY_PDF_PYTHON=<retained interpreter> make check` | PASS, exit 0 |
| Linux / amd64, Docker Desktop on arm64, same pinned toolchain | `make check` with the retained PDF interpreter configured | PASS, exit 0 |

Linux source was mounted read-only. Both source checkouts remained clean after
the checks. All 16 development modules and 11 publishable modules were accounted
for. Checks included fresh race tests, formatting/lint, example builds, the fixed
PostgreSQL image, artifact and published consumers, pinned context peers, release
isolation/recovery and the actual PDF fixture profile. No production refs were
written; release tests used disposable bare remotes.

Negative Go fixtures cover tool failures, formatting diffs with both formatter
exit conventions, dirty/staged source, selected-source gate failure, unrelated
caller files/tags, annotated tag collisions, atomic-push rejection, interrupted
preparation, changed candidate/destination/records, unknown observations and a
transport error after successful publication. Retries preserve the candidate.
Breaking v0 releases advance minor; an unprepared v1-to-v2 migration is rejected.
Missing retained PDF prerequisites returned a nonzero Make result before tests.
The live-tag suites compiled separately without paid provider calls.

Earlier attempts correctly failed on Docker disk exhaustion, external container
stops and a transient dependency download error. A fresh PostgreSQL start also
exposed a readiness race; the final script waits for TCP, retains startup logs and
removes its own container and anonymous volume. Final checks passed afterward.

**PDF exception remains:** 25 Python files were removed; four PDF engine/fixture
files remain. Both evaluated pure-Go backends failed the existing contract. These
results certify tooling with the retained PDF prerequisite, not Python-free
acceptance. See [the feasibility report](pdf-go-feasibility.md).

## Shared library infrastructure follow-up

Runtime source checked: `653ae36`, using fresh independent local clones. This
supersedes the earlier command/release behavior; the PDF exception is unchanged.

| Environment | Command | Result |
|---|---|---|
| macOS / darwin arm64, Go 1.27.1, golangci-lint 2.14.0 | `make check` with the retained PDF interpreter | PASS, exit 0 |
| Linux / amd64, Docker Desktop on arm64, same pinned toolchain | `make check` with the retained PDF interpreter | PASS, exit 0 |

Both clones remained clean; Linux source was mounted read-only. The complete gate
covered all 16 development modules, lint, fresh race tests, example builds, real
PostgreSQL, artifact/installability/published/context consumers and actual PDF
fixtures. PostgreSQL containers and their anonymous volumes were removed by their
own cleanup traps. The temporary Linux validation image was removed after the run.

Additional Go fixtures verify the common Makefile with single and nested-module
foreign libraries, optional/failing project hooks and sequential stages under
`make -j8`. Release fixtures use disposable bare repositories and modules under
`packages/`, independent of ragy's adapter layout. They verify source publication
to main, candidate-only release manifests, older-source selection without moving
the caller, branch/ancestry rejection, required project targets, legacy state
rejection, remote advancement, interrupted/unknown publication and immutable
candidate recovery. Recursive Make target preflight and recovery after an initial
missing remote main are covered. No production refs or sibling repositories were
modified, and no paid-provider tests were executed.

### Removal of redundant configuration and checksum preparation

The subsequent cleanup removes the publication TXT, toolchain include and custom
checksum preparation script. Make discovers publication modules using the project
rule; compiler/linter pins live directly in Makefile and CI reads them there.

Targeted validation after this cleanup passed: all tooling tests with fresh race
execution, the complete tooling integration profile (candidate artifacts, exact
published consumers and pinned context peers), lint for both build profiles, Make
version/module outputs and shell syntax. Release fixtures assert that preparing a
release does not manufacture go.sum changes. Real artifact consumers install and
compile without precomputing future root checksums in adapter manifests.
The full macOS/Linux gate above applies to runtime commit 653ae36; it was not
repeated for this follow-up.
