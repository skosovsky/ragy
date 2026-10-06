# Reviewed release scope

The release tool requires Bash, Python 3, Git with atomic push support, and Go. It works on macOS and Linux without BSD-specific `sed`. Git repository/index/config selection environment overrides are rejected so isolated Git writes cannot be redirected to the caller. Run from a clean tracked checkout; untracked and ignored files are excluded and remain untouched. Do not pass example-module lists or derive release scope from `find`.

Choose and review a full source commit SHA that contains `scripts/release-modules.txt`. The manifest lists root and publishable adapters; examples are not independently published. Every entry must name a tracked regular `go.mod` with the expected root/module path. The tool reads this manifest and the source files from the reviewed commit, rather than from a later caller checkout.

```sh
make release-patch RELEASE_SOURCE=<reviewed-full-commit-sha>
# For a reviewed v0 incompatible change:
make release-break RELEASE_SOURCE=<reviewed-full-commit-sha>
# Direct equivalent, after required acceptance checks:
./scripts/release.sh patch <reviewed-full-commit-sha>
```

These commands publish to the single configured `origin` push destination. Relative filesystem remote paths are bound to the caller repository before entering the isolated checkout. They are operational release commands, not part of task20 verification. Task20 runs the actual tool only in disposable repositories with local bare remotes.

Before preparation, the tool rejects dirty tracked files, any staged changes, invalid module scope and any intended local/remote tag collision. It previews the reviewed SHA, destination, candidate version, modules, permitted manifest files and exact refs for confirmation. Version calculation uses published remote root tags, not arbitrary unpublished caller tags. v0 patches increment patch; v0 breaking changes increment minor; v1 breaking changes that require v2 are rejected pending a reviewed semantic import-version migration.

Preparation uses a temporary independent Git repository fetched at the exact reviewed commit. Only publishable `go.mod` files can be staged. Go's manifest editor updates dependencies within the root module namespace to the candidate version and removes their local replacements; external dependencies/replacements and example manifests remain unchanged. If these edits change manifests, a release commit derives directly from the reviewed SHA; otherwise tags point to that SHA. Commit identity/signing settings are copied for the isolated release commit. Caller/global hooks are disabled in the isolated repository; staged and committed diffs are independently checked against the same file allowlist. The caller's HEAD, branch, index, files and refs are never modified.

Only the listed lightweight root/module tags are created in the isolated repository. Push supplies exact `refs/tags/...:refs/tags/...` refspecs with `--atomic`, without force, broad `--tags` or non-atomic fallback. Unrelated local refs and files cannot enter the release. A concurrent tag collision fails rather than overwriting the remote ref. Temporary preparation is removed on success or failure.

## Recovery implementation status

T03 establishes release isolation. Persistent candidate records, remote none/partial/unknown reconciliation and safe retry of the same candidate are required by [the target recovery contract](../contracts/remediation.md#release-scope-and-recovery-f02-f03-d61) and remain T04. Until that implementation is accepted, a failed publication is not a completed release and this tool is not claimed to provide the final recovery workflow. No candidate recovery or consumer-install proof is inferred from an exit code. The final clean-consumer/profile checks are tracked separately in T21/T22.

## Local acceptance

```sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/release_test.py -v
bash -n scripts/release.sh
```

Fixtures use actual Git and Go, synthetic source/files and disposable bare remotes. They check exact source ancestry, manifest-only candidate diff, excluded private files/tags/examples, unchanged caller checkout, dirty/index rejection, tag collisions and rejected atomic push. They do not perform production publication.
