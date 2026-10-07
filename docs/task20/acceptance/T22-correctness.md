# T22 independent correctness acceptance

Verdict: **PASS**. No actionable correctness findings detected on this final verification diff. Reviewer `/root/t22_correctness` did not implement the task and did not inspect the other final acceptor's report.

Reviewed accepted runtime/tooling source: `fc2040c45a1ae50d147f801ff4b8f42c50044db6`. Working diff contains final traceability/evidence/bookkeeping only; runtime, package contracts, manifests and CI remain that accepted source. Original master and all eight role reviews, their registry dispositions, original DOC1–9/DoD1–8 requirements, task contracts/reports and concrete final artifacts were inspected. This is final-task acceptance before its signed commit, not a claim that the goal is already complete.

## Requirement and claim audit

- Independently verified all nine source SHA256 fingerprints, exact F01–F11/D01–D61 ID sets, all161 role finding original source line locations, all12 role defect aliases, and nonpending rationale/disposition/evidence paths for all245 master/role records. New four explicit alias dispositions agree with the accepted master result. Final DOC/DoD mappings include their owning acceptance and final evidence without claiming a self-referential final commit SHA.
- Independently read all22 distinct T00–T21 commit objects: actual SSH signature headers exist, status is accepted, completeness100% and correctnessPASS, and both report bytes equal the blobs at each task's own commit. Signature presence is verified; cryptographic signer trust is not newly attested. Independently compared every one of454 original tracked task13–19 blobs with original baseline: identical Git object identity. This does not claim raw CRLF worktree byte identity.
- Independently parsed raw all-module log commands, rather than relying on generated summary: exact14 unique inventoried modules each have one count=1/race and one lint command, all commands force GOWORKoff and have exit0. Four owning-module example builds are recorded. Actual compiler/linter pins agree with tracked JSON. Linux CI remains configuration evidence, not an executed remote job.
- Exact clean-consumer log and tracked helper inspected: release executes reviewed script only in disposable local Git checkout/bare remote, candidate derives directly from exact source with manifest-only allowlist changes; all11 tags identify candidate88fc3c581c700850c7cb8b5cb49442d5a5a53a5a. External consumer imports48 real public packages, requires all11 versioned modules with no replacements, and downloaded ZIP digests must equal authored candidate archives before fresh race/build. No real publication performed.
- Required actual PG profile is real SQL execution/query/delete over the complete portable scalar/nested predicate corpus, canonical custom codec/quoted table, adjacent >2^53 scoped tenant pairs/omission/conflicting optional filter and pinned-before-I/O rejection. Source enforces owned-container isolation label and fails rather than skips. Actual PDF profile exercises configured real interpreter including retained text/cell/image resolution, geometry, limits, projection/scoped resolution and durable publication/reopen. Parent raw logs show PG3 parents/56 subtests and PDF8 parents, no SKIP; independent reruns below passed.
- D40 source range validation is actual performance change; retained before/after benchmark source and logs contain512/2048-word multibyte mappings, measured26.96ms→9.37ms and429.79ms→13.08ms with roughly unchanged allocation populations. Current scaling doc limits these short100ms observations; boundary differential regression remains. D13/D20/D27/D46/D47/D50 cache/clone/union/storage algorithms and D44 metric algorithms are retained with no speedup claim. ES tokenize-once fixes callback consistency rather than claiming CPU reduction. Cancellation/shape caps do not imply universal CPU/RSS bounds.
- Current public ownership/protection/publication/unknown outcome, authorization versus retention, byte versus count, citations versus derivation and actual-service versus wire/quality/power-loss boundaries agree with original master. License selection and real release/push remain explicitly outside goal; absent owner policy templates are acknowledged per original DOC9 rather than fabricated. No paid benchmark is required for these code-review fixes. No wire fake or optional SKIP is promoted to real-service PASS.

## Independent executed checks

Common Go environment: Go `/opt/homebrew/Cellar/go/1.26.5/bin/go`, GOTOOLCHAINlocal, GOWORKoff, own `/tmp/ragy-t22-correctness-cache`, existing task-owned modcache/GOPATH.

| Command/profile | Result/evidence |
|---|---|
| PG module `go test -race -count=1 -tags=integration_pg -run '^TestRealPostgres' ./...`, owned `RAGY_PG_TEST_CONTAINER=ragy-task20-t22-pg`, Docker on PATH | exit0,103.448s; [log](T22-correctness-results/pg.log). Source inspection confirms these are all3 actual profiles, no skip path. |
| PDF module `go test -race -count=1 -run '^TestActual' ./...`, actual bundled Python3.12.14 via RAGY_PDF_PYTHON | exit0,5.705s; [log](T22-correctness-results/pdf.log). Actual8 parent tests selected. |
| Root `go test -race -count=1 ./recipe/... ./graphingest/... ./layout/... ./lexical/... ./source/...` | exit0; [log](T22-correctness-results/core.log), substantive boundary/accounting/provenance/freshness/ownership tests. |
| Root `go test -race -count=1 ./` | exit0,1.263s; [log](T22-correctness-results/docs-full.log), exact README executable snippet, current links and removed API selectors. |
| `PYTHONPATH=scripts python3 -m unittest scripts/verify_test.py scripts/check_release_consumer_test.py scripts/process_runner_test.py` |14 tests PASS,15.875s; [log](T22-correctness-results/tools-fixed.log). Actual valid Go Unicode fuzz118834 executions; timeout cleanup tests exercise both wrappers with TERM-ignoring descendants. |
| Actual Python `adapters/pdf/testdata/verify_fixture.py` and `verify_engine_errors.py` | both exit0: synthetic text/table/image/rotation and11 sanitized imported engine exception classes PASS. |
| All new final-audit Markdown evidence links and `git diff --check` | PASS; no implementation edits performed. |

An initial reviewer unittest invocation omitted `PYTHONPATH=scripts` and failed import before executing any tests; [failed setup log](T22-correctness-results/tools.log) is preserved and never counted as PASS. Corrected explicit script import environment passed all14 tests.

## Substantive freeze

Mutable backlog/plan, reports and result logs are excluded. Acceptance is for these exact three artifacts plus unchanged accepted runtime source. Any substantive edit requires this review to be repeated.

| File | SHA256 |
|---|---|
| `docs/task20/T22.md` | `71eeb2122456a1d82ef9e0ff5f62d315d073c333eafd8f74d17858cf379cddeb` |
| `docs/task20/final-audit.md` | `92f2acab1be76ef533f6e66571bd09b37b2afd8194dc91f7c9f2c185a7f885b1` |
| `docs/task20/traceability.json` | `273618f1a1cd7b385949457edc6c54b52d5587fd1be92dc6e3cc7f53ec249a00` |
