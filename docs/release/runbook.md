# Reviewed release scope

The release tool requires Bash, Python 3.9+, Git with atomic push support, and Go. It works on macOS and Linux without BSD-specific `sed`. Git repository/index/config selection environment overrides are rejected so isolated Git writes cannot be redirected to the caller. Run from a clean tracked checkout; untracked and ignored files are excluded and remain untouched. Do not pass example-module lists or derive release scope from `find`.

Choose and review a full source commit SHA that contains `scripts/release-modules.txt`. The manifest lists root and publishable adapters; examples are not independently published. Every entry must name a tracked regular `go.mod` with the expected root/module path. The tool reads this manifest and the source files from the reviewed commit, rather than from a later caller checkout.

```sh
make release-patch RELEASE_SOURCE=<reviewed-full-commit-sha>
# For a reviewed v0 incompatible change:
make release-break RELEASE_SOURCE=<reviewed-full-commit-sha>
# Direct equivalent, after required acceptance checks:
./scripts/release.sh patch <reviewed-full-commit-sha>
```

These commands publish to the single configured `origin` push destination. Relative filesystem remote paths are bound to the caller repository before entering the isolated checkout. They are operational release commands, not part of task20 verification. Task20 runs the actual tool only in disposable repositories with local bare remotes.

Before new candidate preparation, the tool rejects dirty tracked files, any staged changes, invalid module scope and any intended local/remote tag collision. Existing recorded matching refs are idempotent evidence; differing objects remain collisions. It previews the reviewed SHA, destination, candidate version, modules, permitted manifest files and exact refs for confirmation. Version calculation uses published remote root tags, not arbitrary unpublished caller tags. v0 patches increment patch; v0 breaking changes increment minor; v1 breaking changes that require v2 are rejected pending a reviewed semantic import-version migration.

Preparation uses a persistent independent Git repository fetched at the exact reviewed commit, stored under the caller Git common directory in `ragy-releases/active/checkout`. Only publishable `go.mod` files can be staged. Go's manifest editor updates dependencies within the root module namespace to the candidate version and removes their local replacements; external dependencies/replacements and example manifests remain unchanged. If these edits change manifests, a release commit derives directly from the reviewed SHA; otherwise tags point to that SHA. Commit identity/signing settings are copied for the isolated release commit. Caller/global hooks are disabled in the isolated repository; staged and committed diffs are independently checked against the same file allowlist. The caller's HEAD, branch, index, files and refs are never modified.

Only the listed lightweight root/module tags are created in the isolated repository. Push supplies exact `refs/tags/...:refs/tags/...` refspecs with `--atomic`, without force, broad `--tags` or non-atomic fallback. Unrelated local refs and files cannot enter the release. A concurrent tag collision fails rather than overwriting the remote ref. Preparation and its exact owned tags are retained on success and failure for inspection/retry. Caller tags are never created, deleted or overwritten.

## Inspection and recovery

Before any preparation or publication, `ragy-releases/active/state.json` records the reviewed source, chosen version/kind, modules, allowed files and destination. Preparation adds the exact candidate SHA, expected lightweight tag object IDs and individually created isolated refs. Atomic record replacement and directory/file synchronization retain the record; a process lock in the Git common directory serializes release commands across caller worktrees. Ordinary checkout files and index remain unchanged; optional Git locks/refresh writes are disabled for caller read checks. This is not a hardware power-loss certification.

```sh
./scripts/release.sh inspect  # observe exact remote refs; print the candidate record
./scripts/release.sh resume   # continue this candidate, never calculate another version
./scripts/release.sh finish   # inspect complete publication and archive the record/checkout
```

Repeating `patch|break` with the original source/kind also resumes its existing candidate. A different source or kind is rejected while an active record exists, including a completed record; explicitly `finish` it before selecting the next version. Records and their repositories are archived under `ragy-releases/history/<version>-<candidate-sha>` rather than deleted. `inspect` may run with dirty caller files because it changes only recovery evidence; new publication, resume and finish require a clean tracked/index state.

| Status | Evidence and next step |
|---|---|
| `none` | Successful observation proves all intended remote refs absent. Fix the failure and resume the same candidate. |
| `partial` | A proper subset of exact expected objects is present, other expected refs absent. Resume checks collisions and pushes only missing refs atomically. |
| `complete` | Every intended ref has the exact expected object. Resume is idempotent; finish archives evidence. |
| `unknown` | Remote observation failed or publication is in flight/interrupted. Resume refuses any push until explicit inspect establishes a known state. |
| `collision` | An intended ref differs from the candidate object, including an annotated tag with the same peeled commit. Resolve explicitly as the host, then inspect; the tool never force-pushes or deletes refs. |

The tool writes `unknown` before dispatching push, then observes exact remote objects regardless of push exit. A failed transport exit can still be proven complete; a successful exit with unavailable observation remains unknown. There is no non-atomic fallback when the server lacks atomic support. Failure before tags retains the source/version with an incomplete preparation record; failure between tags resumes the same commit and fills only missing isolated refs. Completed candidates retain their identity independently of later version calculations and caller local tags.

Keep the configured origin push identity equal to the persisted destination during recovery. If it changes, restore/resolve that host configuration explicitly; the tool refuses to silently publish the candidate elsewhere. Canonical manifest bytes are reconstructed from the exact source using only the allowed Go edits. Incomplete preparation accepts only original or canonical manifest content; a recovered/already-recorded commit must contain exactly canonical manifests. Unreviewed module identities, external dependencies or other manifest edits fail closed. Malformed/missing records, changed candidate HEAD, dirty isolated checkout or differing owned refs also fail closed and preserve evidence for investigation. Do not remove recovery state or tags blindly. The final clean-consumer/profile checks remain T21/T22; they are not inferred from release command exits.

## Local acceptance

```sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/release_test.py -v
PYTHONDONTWRITEBYTECODE=1 python3 scripts/release_recovery_test.py -v
bash -n scripts/release.sh
```

Fixtures use actual Git and Go, synthetic source/files and disposable bare remotes. They check exact source ancestry, manifest-only candidate diff, excluded private files/tags/examples, unchanged caller checkout, dirty/index rejection, tag collisions and rejected atomic push. Recovery fixtures additionally exercise failures before/between tags, same-candidate retry, partial external publication, unavailable observation, transport exit ambiguity, exact annotated-object collisions, unsupported atomic capability and explicit completed-record archival. They do not perform production publication.
