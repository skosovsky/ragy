# Shell release

The common release implementation uses Bash, Git, Make and the standard Go CLI.
Repository-specific artifact and consumer checks are defined in `project.mk`.
There is no Python release runner or additional Go executable.

```sh
make check
make release-patch
make release-break
# Optional older source on main:
make release-patch RELEASE_SOURCE=<full-commit-sha>
make release-inspect
make release-resume
make release-finish
```

The caller must be on local `main` with a clean tracked/index state. Untracked
files are preserved and excluded. Source defaults to HEAD; an explicit full SHA
must be an ancestor of local main. Origin must have exactly one push destination
and an existing main that can fast-forward to the selected source. An older
source cannot rewind remote main. The caller's branch, HEAD, files and tags remain
unchanged throughout the transaction.

Versions come from remote root tags. Patch increments patch; break increments
minor before v1. A breaking transition from v1 to v2 is blocked until an import-path
migration is implemented. The explicit publication inventory excludes development
modules. Root tags use `vX.Y.Z`; nested module tags use their directory prefix.

The release runs the selected source's `make check` in an independent checkout.
It updates internal requirements to the new version and removes internal development
replacements only there. The project prepares checksums and validates artifacts and
consumers; only publishable go.mod/go.sum files may change in the release commit.
A failed prerequisite, source gate, preparation or candidate check prevents push.

After displaying the source, candidate, destination and exact refs for confirmation,
one atomic push publishes:

- source SHA to `refs/heads/main`, keeping its development replacements;
- candidate SHA to the exact root/module release tags, with prepared manifests.

There is no force, broad `--tags`, or non-atomic fallback. Remote main is rechecked
before push; a concurrent incompatible update rejects publication. Remote tags must
match exact candidate identities. Remote main must equal source or contain it in
its history. Finally, the exact published version is checked through public module
resolution, with bounded retries for proxy delay. Published refs alone do not mean
that the release has passed its consumer check.

## Project contract

The common Makefile provides standard developer commands and optional
`prerequisites-project`, `examples-project`, `check-project` targets. Its release
script requires these additional targets in the selected source:

| Target | Contract |
|---|---|
| `release-prepare-project` | Prepare candidate checksums/artifacts; may change only the allowed manifests in the isolated checkout. |
| `release-check-project` | Validate the final candidate without changing it. |
| `release-published-project` | Verify exact-version public consumers without publishing refs. |

All three receive `RELEASE_SOURCE` (source SHA), `RELEASE_CANDIDATE_DIR` (absolute
checkout directory), `RELEASE_VERSION` and `RELEASE_ARTIFACT_DIR` (private artifact
scratch directory). Missing targets fail before publication. The current ragy
implementation delegates validation to Go integration tests and root dependency
checksum preparation to a small project script. These tests are not a release CLI.

## Recovery

Format-2 plain-text records live under `library-releases/active` in the Git common
directory. They retain destination, branch, observed remote main, source, candidate,
version, phase and publication status. Records are never evaluated as shell code.
A local lock prevents concurrent releases; after process termination, verify the
process is gone before manually removing a stale lock.

`inspect` refreshes remote observations. `resume` reuses the same version and
candidate; unknown results require inspection first. A lost response after push is
resolved by reading refs. Later forward movement of remote main is acceptable if
source ancestry is verified; a different tag is always a collision. Recovery never
rewinds main or overwrites/deletes a tag. `finish` archives only a verified release.

Historical records remain untouched. An active record under an older release state
directory, or an unsupported record format, blocks new publication: finish/recover
it using its original tooling revision before using the new script.

## Verification

Go tooling tests exercise the common protocol against disposable bare repositories,
including libraries with nested modules outside `adapters/`. Real module artifacts,
installability and consumer composition are checked by project integration tests.
These checks never publish to the production remote. CI runs the same `make check`.

The PDF backend migration remains blocked. ragy's full check/release still requires
the retained PDF runtime documented in [verification](../verification.md); this
prerequisite belongs to the project layer, not the common release implementation.
