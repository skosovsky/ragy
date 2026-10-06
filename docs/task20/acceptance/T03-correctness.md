# T03 independent correctness acceptance

Verdict: **PASS — no unresolved errors found in the final T03 scope.**

Reviewed baseline: `6d18bca7403386643bc4b7a0e6be3039785a86e5`; latest substantive working-tree diff, including hook, repository-selector and relative-destination hardening. This reviewer did not implement the change and did not consult the completeness report before the verdict.

## Scope and findings

Checked T03.C01–C04 against F02 in `review-baseline.md`, A-F01 in `reviews/arch-docs.md`, and the release isolation section of `docs/contracts/remediation.md`. The reviewed source is an exact full commit SHA; the source-version-controlled manifest and regular module manifests determine the module/file/tag allowlists. A later clean caller commit, private untracked data and unrelated local tags do not enter the release. The release commit derives directly from the reviewed commit and changes only permitted module manifests. Push uses exact refs, without force or broad tag publication, as one atomic transaction.

Dirty tracked/index state, staged unrelated files, invalid scope and intended local/remote module-tag collisions stop before preparation. Temporary repositories do not register a caller worktree or alter caller refs. Inherited repository selectors are rejected before Git commands; global checkout hooks are disabled, and staged and committed diffs are checked independently. Independent inherited pre-commit hook override experimentation also failed closed at the committed-diff guard rather than publishing its out-of-scope mutation.

No severity findings remain. Persistent candidate records, same-candidate retries and none/partial/unknown reconciliation remain T04. Clean-consumer and broad tooling acceptance remain T21/T22. This PASS does not certify those deferred requirements or production release readiness.

## Independent checks

All commands ran from the repository root; production release/push was never invoked.

- `PYTHONDONTWRITEBYTECODE=1 python3 scripts/release_test.py -v`: exit 0, **9/9 methods PASS**, including parameterized staged/dirty/tag collision and inherited-selector cases. This was rerun on the final selector-hardened implementation; actual disposable bare remotes, Git manifest commits and Go manifest editing were exercised. Final run: 16.460 seconds; relative-origin method includes plain and colon-containing local path subcases.
- `bash -n scripts/release.sh`: exit 0.
- Python `ast.parse` of implementation and test source: PASS.
- `git diff --check`: exit 0.
- Additional independent disposable fixtures in `/private/tmp/ragy-t03-correctness/adversarial.py`: abort confirmation, multiple push destinations, symbolic source instead of reviewed SHA, symlink module manifest, and inherited pre-commit hook attempting to modify/stage `reviewed.txt`. Every case rejected, caller snapshot remained identical and remote had no published tags. The last case explicitly reported `Candidate commit changed files outside manifest allowlist`.
- `/private/tmp/ragy-t03-correctness/global-worktree.py`: inherited global `core.worktree` setting plus a later caller commit; actual publication succeeded with identical caller snapshot and later caller source text preserved. No production destination was used.

The fixture caller snapshots include HEAD, branch, status, index entries and all non-Git file bytes; exact local tags and published remote tree/refs are checked separately. The successful fixture additionally verifies private-file exclusion, reviewed ancestry, permitted changed files and unchanged example manifests.

## Reviewed substantive file SHA256

| File | SHA256 |
|---|---|
| `Makefile` | `ca52736f8172f8a549301ca3009e8056f002aa0105b951ebe73d6d20edb81886` |
| `scripts/release.sh` | `d255795829845d1d3d0512827940e126fdd25f90403de65b7cae7037fefed459` |
| `scripts/release.py` | `3c7ec874d36a0782fa0b2f665e2346515c63424ab5ee67d343847088bc7c67da` |
| `scripts/release_test.py` | `b4774b83d43964102bcc1a613ffc627b51af119c0b2e74302532dc3d43a51e0e` |
| `scripts/release-modules.txt` | `e154f42f1322b562aa160a19354fa34800b419257da01726039599412a5aae8d` |
| `docs/release/runbook.md` | `bcd36767a731b93075a32a946258c52d727ca98da7ba8ccea218fdd3d1bb8977` |

Execution journal and traceability bookkeeping can be updated after acceptance without changing this substantive verdict. Any substantive change requires re-review.

## Reacceptance finding and resolution

**[P2, resolved] Bind all Git local relative paths before isolation.** After the ordinary `../remote.git` fix, `destination()` identifies any string containing `:` as a non-filesystem URL. Git also accepts local paths with a slash before the colon (for example `../remote:fixture.git`). Such a path is successfully inspected in the caller checkout, but subsequently resolves against the temporary checkout and the atomic push fails with exit 128. This contradicts the runbook promise that relative filesystem paths are bound to the caller repository.

Independent reproduction: `/private/tmp/ragy-t03-correctness/relative-colon.py`, an actual local bare repository named `remote:fixture.git`, configured as `../remote:fixture.git`. Release reaches confirmation/preparation, push fails; no tags published and caller snapshot unchanged. The final implementation now recognizes a local path when no colon exists or a slash precedes the colon, then resolves it against the caller repository before isolation. The permanent relative-origin fixture covers both ordinary and colon-containing paths. Independent reproduction was repeated on the final implementation: exit 0, only two intended tags published, caller snapshot identical. All nine standard methods and the five additional adversarial rejection cases were repeated on the final revision; all PASS. The SHA table above is refreshed to the final reviewed files.
