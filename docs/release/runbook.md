# Shell release

The release implementation uses Bash, Git, Make and the standard Go CLI. Its Go
integration tests validate module artifacts and consumer behavior. There is no
Python release runner or additional executable to install.

```sh
make check
make release-patch
# Incompatible v0 API change:
make release-break
# Optional exact source selection:
make release-patch RELEASE_SOURCE=<full-commit-sha>
```

Source defaults to HEAD. Patch increments patch; break increments minor before v1.
A breaking transition from v1 to v2 remains blocked pending import-path migration.
Versions come from remote root tags. The publishable inventory excludes examples
and tooling. All adapters share the root version and receive directory-prefixed tags.

A clean tracked/index state is required. Untracked files are preserved and excluded.
The script fetches only the selected source into an independent checkout under the
Git common directory. It runs that source's `make check`, prepares only allowlisted
module manifests/checksums, then tests candidate artifacts with an isolated proxy
and fresh external consumers. Development replacements never enter published
manifests. The user confirmation shows source, candidate, destination and exact refs.
The caller's HEAD, branch, index and tags are unchanged.

Only explicit root/adapter refs are pushed, atomically and without force. No broad
`--tags` or non-atomic fallback is allowed. Publication is confirmed by exact remote
object IDs and an exact-version public consumer smoke. Proxy delays get bounded
retries; an incomplete smoke is reported as a failure even if tags exist.

## Recovery

```sh
./scripts/release.sh inspect
./scripts/release.sh resume
./scripts/release.sh finish
```

The candidate and plain-text records live in `ragy-releases/shell-active` in the
Git common directory. Resume retains the same version and candidate. Unknown
publication requires inspect before retry. A differing tag or changed origin is
an error, never permission to overwrite a public ref. Finish archives only a
complete, smoke-verified release. An interrupted process can leave a shell lock;
verify no release process remains before removing that lock manually.

Historical JSON archives remain readable evidence. A legacy `active` record blocks
new shell publication: finish/recover it using its original tooling revision first.
Records are not sourced as shell code. Local recovery directories are private
operational state; no hardware power-loss durability certification is claimed.

The PDF backend migration is currently blocked. Full check/release requires the
retained actual-parser prerequisite described in [verification](../verification.md).

## Tests

`make test` exercises isolation, rejecting hooks, dirty source, candidate identity
and recovery in disposable local repositories. `make test-integration` additionally
checks all real module artifacts and consumers. These commands never publish to
the configured production origin. CI invokes `make check` on Linux.
