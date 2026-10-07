# Reviewed release scope

The release tool requires Bash, Python 3.9+, Git with atomic push support, and the Go/toolchain pins in `scripts/toolchain.json`. Full release checks also require the registry prerequisites, including PostgreSQL integration and the Linux profile (native Linux or Docker). Missing required infrastructure blocks publication. It works on macOS and Linux without BSD-specific `sed`. Git repository/index/config selection environment overrides are rejected so isolated Git writes cannot be redirected to the caller. Run from a clean tracked checkout; untracked and ignored files are excluded and remain untouched. Do not pass example-module lists or derive release scope from `find`.

Choose and review a full source commit SHA that contains `scripts/release-modules.txt`. The manifest lists root and publishable adapters; examples are not independently published. Every entry must name a tracked regular `go.mod` with the expected root/module path. The tool reads this manifest and the source files from the reviewed commit, rather than from a later caller checkout.

```sh
make release-patch RELEASE_SOURCE=<reviewed-full-commit-sha>
# For a reviewed v0 incompatible change:
make release-break RELEASE_SOURCE=<reviewed-full-commit-sha>
# Direct equivalent, with the same internal full gates:
./scripts/release.sh patch <reviewed-full-commit-sha>
```

These commands publish to the single configured `origin` push destination. Relative filesystem remote paths are bound to the caller repository before entering the isolated checkout. They are operational release commands, not part of task20 verification. Task20 runs the actual tool only in disposable repositories with local bare remotes.

Before new candidate preparation, the tool rejects dirty tracked files, any staged changes, invalid module scope and any intended local/remote tag collision. Existing recorded matching refs are idempotent evidence; differing objects remain collisions. It previews the reviewed SHA, destination, candidate version, modules, permitted manifest files and exact refs for confirmation. Version calculation uses published remote root tags, not arbitrary unpublished caller tags. v0 patches increment patch; v0 breaking changes increment minor; v1 breaking changes that require v2 are rejected pending a reviewed semantic import-version migration.

Preparation uses a persistent independent Git repository fetched at the exact reviewed commit, stored under the caller Git common directory in `ragy-releases/active/checkout`. Only publishable `go.mod` and their associated `go.sum` files can be staged. Child-module sum paths are derived from the committed module inventory, including an initially absent sum file; unrelated new files remain outside the allowlist. Go's manifest editor updates dependencies within the root module namespace to the candidate version and removes their local replacements; external dependencies/replacements and example manifests remain unchanged. For adapters, checksum preparation creates an isolated root artifact proxy from the exact source and intended root manifest, downloads that candidate version with Go, and adds only its ZIP/go.mod checksums to adapter sums, creating the allowlisted file when the source has none. Root own-module dependencies are rejected to avoid checksum cycles. These checksum bytes are part of the reconstructed manifest allowlist. If these edits change manifests, a release commit derives directly from the reviewed SHA; otherwise tags point to that SHA. Commit identity/signing settings are copied for the isolated release commit. Caller/global hooks are disabled in the isolated repository; staged and committed diffs are independently checked against the same file allowlist. The caller's HEAD, branch, index, files and refs are never modified.

Before creating any release tags, the tool runs `scripts/verify.py check` on an isolated checkout of the selected source and again on the immutable manifest/sum-rewritten candidate. Each uses the same registry as local and CI checks. The candidate uses a frozen artifact proxy for its unpublished module version. The runner does not invoke the release entrypoint. Every planned required lane must have fresh PASS evidence; duplicate/missing lanes, required SKIP/BLOCKED, wrong report identity/version or nonzero exit reject the gate. Tracked fingerprints, checkout cleanliness and exact candidate ancestry are checked after execution. Gate output is outside the source trees and is preserved in recovery state. Changes to source/candidate invalidate PASS. Recovery reruns both gates rather than trusting a previous result.

Only after these gates are the listed lightweight root/module tags created in the isolated repository. Push supplies exact `refs/tags/...:refs/tags/...` refspecs with `--atomic`, without force, broad `--tags` or non-atomic fallback. Unrelated local refs and files cannot enter the release. A concurrent tag collision fails rather than overwriting the remote ref. Preparation and its exact owned tags are retained on success and failure for inspection/retry. Caller tags are never created, deleted or overwritten.

## Inspection and recovery

Before any preparation or publication, `ragy-releases/active/state.json` records the reviewed source, chosen version/kind, modules, allowed files and destination. Preparation adds the exact candidate SHA, expected lightweight tag object IDs and individually created isolated refs. Atomic record replacement and directory/file synchronization retain the record; a process lock in the Git common directory serializes release commands across caller worktrees. Ordinary checkout files and index remain unchanged; optional Git locks/refresh writes are disabled for caller read checks. This is not a hardware power-loss certification.

```sh
./scripts/release.sh inspect  # observe exact remote refs; print the candidate record
./scripts/release.sh resume   # continue this candidate, never calculate another version
./scripts/release.sh finish   # archive only complete refs plus verified public artifacts/consumer
```

Repeating `patch|break` with the original source/kind also resumes its existing candidate. A different source or kind is rejected while an active record exists, including a completed record; explicitly `finish` it before selecting the next version. Records and their repositories are archived under `ragy-releases/history/<version>-<candidate-sha>` rather than deleted. `inspect` may run with dirty caller files because it changes only recovery evidence; new publication, resume and finish require a clean tracked/index state.

| Status | Evidence and next step |
|---|---|
| `none` | Successful observation proves all intended remote refs absent. Fix the failure and resume the same candidate. |
| `partial` | A proper subset of exact expected objects is present, other expected refs absent. Resume checks collisions and pushes only missing refs atomically. |
| `complete` | Every intended ref has the exact expected object. This alone is not release success: public artifact/checksum/consumer verification must also pass. Resume retries verification without rewriting tags; finish requires `publication_verified: true`. |
| `unknown` | Remote observation failed or publication is in flight/interrupted. Resume refuses any push until explicit inspect establishes a known state. |
| `collision` | An intended ref differs from the candidate object, including an annotated tag with the same peeled commit. Resolve explicitly as the host, then inspect; the tool never force-pushes or deletes refs. |

The tool writes `unknown` before dispatching push, then observes exact remote objects regardless of push exit. A failed transport exit can still be proven complete; a successful exit with unavailable observation remains unknown. There is no non-atomic fallback when the server lacks atomic support. Failure before tags retains the source/version with an incomplete preparation record; failure between tags resumes the same commit and fills only missing isolated refs. Completed candidates retain their identity independently of later version calculations and caller local tags.

Keep the configured origin push identity equal to the persisted destination during recovery. If it changes, restore/resolve that host configuration explicitly; the tool refuses to silently publish the candidate elsewhere. Canonical manifest bytes are reconstructed from the exact source using only the allowed Go edits. Incomplete preparation accepts only original or canonical manifest content; a recovered/already-recorded commit must contain exactly canonical manifests. Unreviewed module identities, external dependencies or other manifest edits fail closed. Malformed/missing records, changed candidate HEAD, dirty isolated checkout or differing owned refs also fail closed and preserve evidence for investigation. Do not remove recovery state or tags blindly. Public verification uses the exact candidate/version, a fresh module cache, `GOWORK=off`, public Go proxy and `sum.golang.org`, with no replacements/private checksum exemptions. It compares each downloaded module ZIP's file names/bytes and separate `.mod` with the candidate, including Go's inherited root LICENSE behavior. It verifies all module checksums, records the resolved module graph, compiles every public package, executes canonical onboarding with fresh race tests, and runs the context bridge semantic tests/demo against the exact new ragy version and centrally pinned published peers. The bridge evidence must report that exact version. Up to six attempts with ten-second waits allow proxy propagation; exhausted attempts fail the release and preserve complete tags plus `publication_verified: false`. `resume` repeats gates and public verification for the same candidate; it never rewrites already published tags. A complete remote-ref observation does not replace this proof.

The root `v*` tag triggers the shared GitHub verification workflow once; submodule tags do not fan out duplicate full workflows. Completion of those exact commit/tag Actions runs must be checked separately before issue closeout; the local release tool does not label old green runs as evidence.

## Local acceptance

```sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/release_test.py -v
PYTHONDONTWRITEBYTECODE=1 python3 scripts/release_recovery_test.py -v
bash -n scripts/release.sh
```

Fixtures use actual Git and Go, synthetic source/files and disposable bare remotes. They check exact source ancestry, manifest-only candidate diff, excluded private files/tags/examples, unchanged caller checkout, dirty/index rejection, tag collisions and rejected atomic push. Negative fixtures inject a required source-lane failure while a later caller HEAD is green, and a lane that reports PASS while mutating the candidate: both leave zero caller/isolated/public tags. A tracked adapter checksum fixture checks exact future-version sum preparation and rejects tampered recovery bytes. A failing postpublication verifier leaves tags intact, refuses finish and succeeds only after verified resume. Recovery fixtures additionally exercise failures before/between tags, same-candidate retry, partial external publication, unavailable observation, transport exit ambiguity, exact annotated-object collisions, unsupported atomic capability and explicit completed-record archival. They do not perform production publication.

## Clean consumer verification

For each reviewed candidate, build outside this checkout with `GOWORK=off` and a fresh module/cache/proxy profile, without source-tree replacements. Verify the root and every publishable adapter from the exact candidate source; building examples through local replacements is insufficient. The source manifest has eleven publishable modules: root and ten adapter modules. The three nested planner/resilience/conformance example modules are development consumers and receive no independent release tags.

After an owner-authorized publication, a clean consumer can select the exact published version as follows (replace the placeholders; these are installation commands, not release dispatch):

```sh
mkdir ragy-consumer
cd ragy-consumer
GOWORK=off go mod init example.com/ragy-consumer
GOWORK=off go get github.com/skosovsky/ragy@<candidate-version>
GOWORK=off go get github.com/skosovsky/ragy/adapters/openai@<candidate-version>
# Add the reviewed complete onboarding or adapter consumer source.
GOWORK=off go test -count=1 -race ./...
GOWORK=off go build ./...
```

Pre-publication acceptance uses an isolated local module proxy built from the exact candidate; it must cover all manifest modules, correct source ancestry/version selection and absence of local ragy replacements. A local candidate install does not prove public proxy availability. Record platform, Go/Python/Git versions, exact candidate SHA/version, commands, exit status and executed backend/parser profiles. Never run production release commands merely to validate the consumer.

The repeatable local candidate gate is:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_release_consumer.py <source-sha> --version <candidate-version>
# Final manifest/sum-rewritten bytes must already match the version:
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_release_consumer.py <candidate-sha> --version <candidate-version> --exact-candidate
# After publication, verify public bytes and checksum-verified installation:
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_release_consumer.py <candidate-sha> --version <published-version> --published
```

The consumer prepares artifacts directly in a disposable repository. It never calls `release.sh`, creates release tags or pushes any refs. It creates a local module proxy and compiles all public packages from every publishable module in an external consumer without ragy replacements, then executes canonical onboarding with race enabled. `--exact-candidate` rejects any manifest/sum rewrite; `--published` additionally uses the public proxy and checksum database and compares published bytes. Local artifact success does not prove public availability.

Use `make check` for the full prerelease registry, `make test` for all mandatory test lanes, `make lint` for the shared lint baseline, and `make test-fast` only for the explicitly shortened cycle. `python3 scripts/verify.py check --list` prints the machine-readable plan; output directories must be outside the repository to preserve checked bytes. Context bridge supported checkout, published baseline and unsupported immutable-peer negative cases are separate required lanes. Peer refs and published baseline versions come from `scripts/check-registry.json`, with no sibling working-tree dependency.
