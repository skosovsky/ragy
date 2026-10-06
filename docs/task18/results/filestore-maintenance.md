# TASK18 filestore maintenance handoff

Implemented in `lifecycle/filestore/store_unix.go`:

- CAS and Maintain share one exclusive nonblocking flock, generation comparison, JSON byte-budget check, fsync/rename/directory-fsync persistence, and owned result decoding.
- Both replacement paths invoke lifecycle.ValidateReplacement while holding the lock. Maintain loads and compacts that locked current generation and increments exactly once.
- Reads enforce maxSnapshotBytes against actual opened-descriptor bytes through a limit+1 read. Oversized reads/writes return lifecycle.ErrCapacity.
- Decoded incompatible schemas return joined ragy.ErrUnsupported and ragy.ErrProtocol without overwriting durable data.
- Existing budget expectations updated to ErrCapacity. Benchmark files and shared lifecycle APIs were untouched.

Validation: `GOCACHE=/tmp/ragy-task18-filestore-cache go test -race ./lifecycle/filestore` passed (3.332s); scoped `git diff --check` passed.

New real-filesystem coverage:

- Maintenance survives restart, owns return data, and preserves retired reservations; stale generation and active-publication retirement are rejected.
- CAS cannot remove a released publication pin, including after retirement.
- Independent CAS and Maintain calls start from a channel barrier and compete for a single generation.
- Actual Maintain executes in a child process. Its context pauses at persist's pre-rename checkpoint after temporary fsync/close, signaling through a pipe. Parent observes the held flock, SIGKILLs the child, reloads unchanged committed bytes, then successfully retires history through a fresh store. This proves process-death lock release and ignoring orphan temporary files.
- Pre-canceled Maintain preserves the full snapshot.
- persist cancellation on both sides of rename deterministically verifies pre-commit preservation and post-commit uncertain cancellation semantics.
- A destination directory produces an actual rename failure; temporary data is removed and unrelated committed state is preserved.
- v1 Load, Maintain and CAS reject without byte changes; an undersized reopened store rejects existing state on Maintain/CAS without replacement.

The broader `go test ./lifecycle/...` invocation observed an independently owned lifecycle test fixture failure (`TestRetirementRejectsProtectedStatesAndMalformedSelection/active_plan_ancestor`, maintenance_test.go:62); shared test owner was notified. Initial default GOCACHE access was denied by the workspace sandbox, so validation uses the explicit /tmp cache above. No commits created.

Lint follow-up: zero diagnostics in the owned store/test files after `/opt/homebrew/bin/golangci-lint fmt` on the exact three owned files. Compatible `/opt/homebrew/bin/golangci-lint run ./lifecycle/filestore/...` reports only two diagnostics in independently owned task18_benchmark_test.go (usetesting / misplaced nolint); benchmark owner notified through parent. Default user-bin golangci-lint is stale (built Go 1.26) and panics for the repository's Go 1.27 dependency. Explicit caches also needed for lint (`GOLANGCI_LINT_CACHE=/tmp/ragy-task18-filestore-lint-cache`).

The initial wider invocation ultimately reported integration success (97.454s) and failed because of the earlier shared fixture failure; parent fixed that fixture and owns wider revalidation. Latest scoped race pass before the final formatter run: 2.788s.
Final validation after the exact formatter run: scoped race tests passed (2.876s), scoped git diff --check passed. No owned lint diagnostics remain.

Contract strengthening follow-up: the oversized CAS test now appends a separately owned valid manifest with distinct ID/key/source and oversized Payload, preserving all prior manifests unchanged. Focused `go test ./lifecycle/filestore -run '^TestSnapshotBudgetRejectsOversizedWriteWithoutReplacingPublication$' -count=1` passed (0.479s). Production code unchanged.
