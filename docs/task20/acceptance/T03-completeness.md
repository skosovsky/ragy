# T03 independent completeness acceptance

Verdict: **ACCEPTED — 4/4 criteria fulfilled, 100%**. Source coverage: **2/2 assigned defect records, 100%** (master F02 and architecture A-F01). Baseline: `6d18bca7403386643bc4b7a0e6be3039785a86e5`; reviewed working-tree changes against that HEAD. Reviewer did not implement changes and did not read the correctness report before this verdict.

| Criterion | Status | Evidence |
|---|---|---|
| T03.C01 | Fulfilled | Source must be full exact commit SHA. Scope and tracked regular manifests read through `ls-tree/show` at that commit. Independent candidate fetches exact source into temporary repository; stage uses explicit allowlisted go.mod paths; isolated hooks are disabled and staged/committed diffs are checked separately; tags pushed with explicit full refs and atomic transaction. Suite asserts candidate parent equals older reviewed source, changes only adapter manifest, later tracked files absent, root/module tags exact. |
| T03.C02 | Fulfilled | Actual disposable bare-remote suite proves top-level/nested untracked payload and scratch tag absent remotely, caller HEAD/branch/index/status/files preserved. Additional independent fixture confirms ignored private payload absent, existing unrelated remote tag unchanged and no branch publication. |
| T03.C03 | Fulfilled | Inherited repository-selection environment variables reject before any Git command. Tracked dirty/staged modifications and new staged private file reject before preparation; local and remote intended adapter-tag collisions reject without changing caller or remote. Manifest exactly covers 11 root/adapter modules and excludes examples. Invalid examples entry, symlink manifest and abbreviated SHA rejection verified. |
| T03.C04 | Fulfilled | Runbook and implementation agree on exact source, manifests, ref scope, platform requirements, dirty/collision policy, isolated preparation, atomic transaction and caller preservation. Nine suite methods pass, additional three independent fixture cases pass, shell syntax/AST/manifest/diff checks pass. Caller-relative filesystem destinations are bound before isolated preparation; relative push URL with spaces and colon is independently verified. Recovery and clean-consumer acceptance explicitly deferred. |

F02/A-F01 coverage includes original synthetic leak reproduction, untracked exclusion, exact staged file scope, unrelated local tag exclusion, exact reviewed ancestry, explicit preview, dirty/index rejection, tag collision handling and caller preservation. F03 persistent candidate/retry/none-partial-unknown recovery belongs to T04 and is **not** accepted here. Consumer/module tooling remains T21/T22.

## Independent commands and outcomes

- `PYTHONDONTWRITEBYTECODE=1 python3 scripts/release_test.py -v`: exit 0; all nine methods PASS, including relative origin binding (plain and colon-containing paths), three inherited repository-selector rejections and synthetic global post-checkout staging hook exclusion and parameterized local/remote collisions and staged/unstaged dirty variants. Uses actual Git/Go and synthetic local bare remotes only.
- `PYTHONDONTWRITEBYTECODE=1 python3 /private/tmp/ragy-t03-completeness.py`: exit 0; independent disposable fixtures PASS for ignored private payload + unrelated remote tag preservation + no branch publication, abbreviated source rejection, tracked symlink manifest rejection; caller snapshots identical in every case.
- Independent inline disposable fixture for relative push URL with spaces and colon and distinct fetch destination: PASS; exact release tags published only at configured push destination, fetch remote untouched, caller snapshot preserved.
- `bash -n scripts/release.sh`: exit 0.
- Python AST parse of `scripts/release.py` and `scripts/release_test.py`: PASS.
- Enumerate repository go.mod files under root/adapters against release manifest: exact equality, 11/11; example modules excluded.
- `git diff --check`: exit 0.

No production release/push. No SKIP credited as PASS. No completeness findings remain.

## Reviewed substantive file SHA256

Execution bookkeeping and this acceptance report are excluded. T03 evidence note hash records the reviewed implementation-pending note; subsequent acceptance-status-only bookkeeping updates do not expand runtime scope.

| File | SHA256 |
|---|---|
| `Makefile` | `ca52736f8172f8a549301ca3009e8056f002aa0105b951ebe73d6d20edb81886` |
| `scripts/release.sh` | `d255795829845d1d3d0512827940e126fdd25f90403de65b7cae7037fefed459` |
| `scripts/release.py` | `3c7ec874d36a0782fa0b2f665e2346515c63424ab5ee67d343847088bc7c67da` |
| `scripts/release-modules.txt` | `e154f42f1322b562aa160a19354fa34800b419257da01726039599412a5aae8d` |
| `scripts/release_test.py` | `b4774b83d43964102bcc1a613ffc627b51af119c0b2e74302532dc3d43a51e0e` |
| `docs/release/runbook.md` | `bcd36767a731b93075a32a946258c52d727ca98da7ba8ccea218fdd3d1bb8977` |
| `docs/task20/T03.md` | `485aadf0f030ec9516a62abc61fdbbe5c3ef72a3372083f4654ad9574073eb7e` |
