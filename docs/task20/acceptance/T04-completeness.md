# T04 — independent completeness acceptance

Verdict on latest substantive diff: **ACCEPTED — 4/4 criteria fulfilled, 100% completeness.** No incomplete or blocked criterion remains for T04. Canonical-content recheck and corrected arrangement both independently verified. Baseline: `b32a7e35106b8442f5a306559bf5589ff8e391ad`. Reviewer did not implement the change and did not read the correctness report before this verdict.

Authoritative requirements reviewed: backlog T04.C01–C04; original task20 F03; architecture report A-F02; `docs/contracts/remediation.md` release scope and recovery. T03 caller-isolation guarantees remain applicable. T21/T22 final consumer/profile checks are separate pending tasks and are not claimed here.

| Criterion | Status | Evidence |
|---|---|---|
| T04.C01 — failures before tags, after tags and during push preserve caller and unrelated refs | Fulfilled | `prepare` uses independent persistent checkout, never caller checkout/ref mutation. Recovery tests cover real Go failure before tags, failure at second actual tag operation, actual rejecting bare pre-receive hook and unsupported atomic server. `test_read_checks_preserve_caller_index_bytes` additionally proves exact binary index retention with unchanged tracked content and altered mtime; `GIT_OPTIONAL_LOCKS=0` disables optional Git read refreshes. Caller snapshot covers HEAD, symbolic branch, status, tracked index entries and every payload-file byte; dedicated isolation tests also assert unrelated caller/remote refs. All passed independently. |
| T04.C02 — manifest identity and fixed-candidate retry | Fulfilled | Record precedes fetch/edit/tag and binds kind/version/source/candidate/modules/files/destination/expected object IDs/owned refs/observations/status. Preparation persists each tag and supports incomplete preparation. `intended_manifests` independently reconstructs exact authorized contents from source; working manifests must be original/canonical, stored/recovered commit blobs canonical exactly, so filename allowlisting cannot authorize unreviewed content. Reject/retry, before-tags retry, between-tags retry and archived-next-version tests verify exact source/version/candidate retention. New source/kind cannot bypass active candidate. Failed local tags are isolated and version calculation reads published remote root refs. |
| T04.C03 — atomic publication and explicit none/partial/unknown reconciliation | Fulfilled | Only missing exact refs use `push --atomic`; no non-atomic/force/deletion path. Remote exact object inspection determines none/partial/complete/collision, failed observation unknown. Unknown is persisted before dispatch and refuses resume until inspect. Partial fixture externally publishes one ref then asserts hook receives only the missing module ref. Unsupported server leaves zero published refs; transport failure after actual successful push reconciles complete. Exact annotated-tag object collision preserves the conflicting remote object. |
| T04.C04 — AAA fixtures and matching runbook | Fulfilled | Independently ran 12 recovery methods and 9 isolation methods: all PASS. Tests use real synthetic Git repositories and local bare remotes, with transport wrappers only injecting exit/observation failure. Runbook matches actual inspect/resume/finish, fixed version, archived records, atomic-only behavior, states and host resolution. Four additional independent integration cases below passed. Final suites and supplementary cases were rerun after both the optional-index-refresh and canonical-manifest fixes; latest review is the identities below. The four tamper subcases reject changed module identity/external dependencies in incomplete working files and recovered committed files. An additional independent valid canonical recovered commit succeeds and preserves the exact preexisting candidate commit/version. |

Original-source coverage: F03 and A-F02 both fulfilled. Explicit matrix: before-tag failure; between-tag failure; full rejecting push; same-candidate retry; existing caller/remote module collisions; remote partial subset; unknown after actual publication; unknown before dispatch; successful remote complete despite failed transport exit; annotated object collision; atomic capability rejection; complete idempotence; active candidate prevents new choice; finish archives before next version. No blind ref deletion or production publication was used.

## Independent commands and outcomes

- `PYTHONDONTWRITEBYTECODE=1 python3 scripts/release_recovery_test.py -v` — exit 0, 12 tests, 43.641s, OK.
- `PYTHONDONTWRITEBYTECODE=1 python3 scripts/release_test.py -v` — exit 0, 9 tests, 13.244s, OK.
- `PYTHONDONTWRITEBYTECODE=1 python3 /private/tmp/ragy-t04-completeness.py` — exit 0, four independently arranged actual bare fixtures passed:
  1. Changed origin push destination after rejecting publication: resume rejected, record unchanged, new destination received zero refs, caller unchanged.
  2. Held common-directory flock with active candidate: concurrent resume rejected before changing record or caller.
  3. Lost second remote observation before dispatch: zero refs, unknown recorded, resume refused; explicit inspect established none, then resumed exact candidate/version, caller unchanged.
  4. Completed candidate plus fetched matching caller tags: resume reported Already published; caller refs/payload and remote refs unchanged, candidate unchanged.
- `bash -n scripts/release.sh` — exit 0.
- AST parse of four Python release/test modules — PASS.
- `git diff --check` — exit 0.

Independent integration script is retained at `/private/tmp/ragy-t04-completeness.py` for this session; the permanent recovery suite already covers the required F03 matrix. These fixtures did not run the release command against ragy's production repository or remote. Hardware power-loss certification is not inferred.

## Reviewed substantive file identities

Bookkeeping/backlog/traceability/acceptance files are excluded. Unchanged normative contract and release entrypoint/manifest are included to identify the acceptance inputs.

| File | SHA256 |
|---|---|
| `scripts/release.py` | `3173e65c294d8bbe59dd6c8a57538eb4d10d1935d9dfac1ea07fbc90b5356daf` |
| `scripts/release_state.py` | `6f4ccee78a4ab00ec3860b2536f226bf122d0745fb3cfc469f2c43e6c37da429` |
| `scripts/release_test.py` | `79fe4b4469f5895b6fa11b6408b68c506278db5bf09cc535ae05c9a5282d5466` |
| `scripts/release_recovery_test.py` | `92d8ee6adf627ffcfbc4b898d362d05c2c802775608eb4db93a98fe715c68103` |
| `docs/release/runbook.md` | `61934d84d5597097920c12ff6214d8bf3cf07db6319abc2775077272ba2eac3e` |
| `docs/contracts/remediation.md` | `65d31729d0f80ef8b85ca548fce419ff1019400bbd640592a013bb9a837b45df` |
| `scripts/release.sh` | `d255795829845d1d3d0512827940e126fdd25f90403de65b7cae7037fefed459` |
| `scripts/release-modules.txt` | `e154f42f1322b562aa160a19354fa34800b419257da01726039599412a5aae8d` |

## Canonical-manifest recheck history and resolution

`PYTHONDONTWRITEBYTECODE=1 python3 scripts/release_recovery_test.py -v`: exit1, 12 methods in 40.058s, FAILED (errors=4). All four new `test_incomplete_and_recovered_commit_reject_unreviewed_manifest_content` subcases fail during Arrange at line89 reading absent isolated `checkout/go.mod`; canonical reconstruction invokes failing Go before the source checkout is materialized. No rejection assertion in these subcases executes. The author then materialized the reviewed-source checkout before canonical reconstruction. Final recheck: all12 methods PASS in43.641s, including all four formerly failing subcases. Content rejection assertions now execute and verify preserved source/version/zero remote refs/caller unchanged. Isolation9 PASS in14.312s and supplemental4 PASS on this diff. File identities above have been refreshed to the finally accepted diff. No implementation edits by this reviewer.

Independent valid recovered-commit check (inline `PYTHONDONTWRITEBYTECODE=1 python3`, exit0): arrange real failed Go preparation; use real Go allowed manifest rewrite in isolated checkout and commit canonical manifests with source as sole parent while persistent candidate remains null; resume reconstructs the same committed SHA and version and publishes complete without caller changes. PASS. This complements the four rejecting cases by proving interrupted canonical commit recovery remains functional.
