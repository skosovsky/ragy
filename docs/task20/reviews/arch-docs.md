# ragy: architecture, public documentation, observation, conformance, CI/release review

Baseline: `b63d5e19`; read-only source review. Actual release script exercised only in disposable repositories with local bare remotes. No GitHub publication, no production source edits. No additional reference-budget or paid/live provider benchmark requested: TASK19's separated mechanical, scripted and external advisory profiles remain legitimate.

## Confirmed defects

### A-F01 [P1] Release includes unrelated untracked files and publishes unrelated local tags

References: `scripts/release.sh:31-35` only `diff-index` checks tracked changes; `:93` uses `git add .`; `:114` pushes `--tags`.

Trigger: clean tracked tree, a local untracked fixture/document, and unrelated local `scratch-local` tag. Run current script with patch confirmation. Observed actual exit 0, remote `v0.0.1` commit includes `private-untracked.txt`; remote includes both `scratch-local` and `v0.0.1`. Thus normal release can disclose files never reviewed for publication; confirmation only names version, not that payload/ref expansion.

Repro: `/tmp/ragy-review-release.py` (copied exact ragy script into fixture), stdout below. Fixture local config disables commit/tag signing; no global configuration change.

Fix: explicit tracked-source release manifest / isolated worktree; reject untracked files or ignore them without staging; stage only intentionally rewritten module manifests; enumerate intended release refs and push only those refs. Preview exact source commit + module/tag list, not blanket push.

AAA acceptance: Arrange unrelated untracked file and private local tag; Act release into local bare remote; Assert neither file nor unrelated tag becomes reachable remotely, only intended module tags refer to reviewed release source. Include pre-existing staged files and tracked dirty state rejection.

### A-F02 [P2] Failed release leaves caller detached with locally reserved version

References: `scripts/release.sh:78-80,103-114,118`; `set -e` exits before checkout, no failure trap/reconciliation.

Trigger: remote hook rejects push. Actual exit 1, `git branch --show-current` empty, local `v0.0.1` remains and remote no release tags. Next invocation infers a higher version from local failed tag; interrupted partial multi-tag push is similarly not reconciled.

Fix: isolated release checkout preferred; deterministic restore on every exit if mutating caller checkout; explicit intended release manifest/status and safe retry of same candidate. Never blindly delete user tags or remote published refs. Use atomic push if supported or expose partial publication and reconciliation.

AAA: Arrange local rejecting hook; Act release; Assert original branch/worktree preserved, candidate state identified and retry does not silently increment version. Add pre-existing module-tag collision and partial remote acceptance fixture; recovery must preserve pre-existing tags.

Actual fixture results:
```
case=success exit=0 branch=main
local_tags=[scratch-local,v0.0.1] remote_tags=[scratch-local,v0.0.1]
published_files=[go.mod,private-untracked.txt,release.sh]
case=failure exit=1 branch="" local_tags=[v0.0.1] remote_tags=[]
```
Logs: `/var/folders/46/5ywmz5gj26n7mnky51gd60g00000gn/T/ragy-release-review-g21w6up0/success.log`, sibling `failure.log`.

## Separate design / oddities register (not additional confirmed runtime defects)

| ID | Observation / reference | Decision and acceptance |
|---|---|---|
| A-D01 | `retrieval` includes planners, branches, fallback/rescue, execution metadata, recipes. | Retain only bounded retrieval composition. This is not a reason to absorb agent loop, durable workflow runtime, UI or Ask/Plan/Action. Flowy owns general workflow; harness owns agent state. Add concise boundary/examples map. |
| A-D02 | `access`, `lifecycle`, managed indexes repeat authority checks and retain fences. | These are necessary retrieval pre-admission and exact publication guarantees, not duplicate IAM implementation. Host policy/authority remain ports. Do not simplify by moving checks after retrieval. |
| A-D03 | Core observation and OTel module plus metry elsewhere. | Keep payload-free operation facts and optional adapter. Do not add collector/export queue/retry/dashboard/billing service to core. Host may bridge observer to metry. |
| A-D04 | `observation/README.md`; synchronous serialized callbacks may delay/cannot recursively re-enable session. | Already explicit cooperative contract. Preserve zero-worker/no-retry design. Add example offload at host if needed; do not pretend ctx timeout forcibly interrupts callback. |
| A-D05 | Fixed enum observation, counts and usage can have different units and unknown state. | Keep known/unknown and finite capacity; no string labels/IDs/query/error payload in core. Document dropped counter unit (pairs) versus Events (callbacks), event count includes attempts even observer fails. |
| A-D06 | `observation.Classify` errors.Is invokes custom Is/Unwrap. | Do not claim arbitrary hostile error objects are sandboxed; documented never Error() is narrower and correct. Callbacks and custom errors remain cooperative host contracts. No speculative panic swallowing around every boundary. |
| A-D07 | `retrieval/execution.go:993-998`, README explicitly zero ExecMeta means omitted. | Zero-value sentinel with reflect.DeepEqual prevents intentional reset to legitimate zero and conflates missing/present. Contract-first clean break to explicit presence/update result if reset use case required; preserve generic BYOT. Not undocumented implementation bug. |
| A-D08 | README: nil conditional predicate runs child, explicit warning; aggregate merger fallback changes rank semantics. | Replace implicit enable and rank degradation with explicit construction/policy where possible. Test documented policy. Do not silently add fallback beyond intended retrieval topology. Other reviewer handles execution defects. |
| A-D09 | root README now ~500+ lines and mixes Russian wire table with English API prose; Quick start is a parameterized helper, no schema/provider setup. | Short executable onboarding first: installation, supported Go, pure local BM25 sample, domain meta/schema, explicit Read, errors. Move advanced topology prose into public docs indexed by capability; gofmt snippets. |
| A-D10 | Public README links `.cursor/docs/INTEGRATION.md` and `docs/task18/19` as core contract locations. | Promote stable integration/limits/ownership/score/lifecycle docs to durable public topic paths. Keep task acceptance/history as dated evidence. Avoid forcing consumer to navigate task chronology. Links currently valid, not missing-link bug. |
| A-D11 | No root LICENSE/CHANGELOG/CONTRIBUTING/SECURITY tracked in inspected checkout. | Owner chooses license before public redistribution; document versioning and supported targets, contribution/test commands and private vulnerability reporting route if repository maintained publicly. Do not fabricate license/legal decisions. |
| A-D12 | `docs_test.go:22-29` bans words legacy/deprecated/instead of across all public source and README. | Replace broad word blacklist with targeted stale-symbol/contract/link/snippet checks. Current rule blocks legitimate Go `Deprecated:` docs and harmless prose while not validating behavior; no current runtime failure claimed. |
| A-D13 | `contracttest` public scoped suite accepts BYOT and instruments payload I/O; older structmeta/index suites use fixed fixtures. | Keep adapter test helper package, advertise what each suite certifies and limitations. New general suites BYOT; do not rewrite all reference fixture types just to remove useful concrete examples. Successful fixture is not universal IAM certificate. |
| A-D14 | `examples/conformance` real external module name and local replace, GOWORK=off documented; CI executes under ambient workspace. | Add repeatable standalone GOWORK=off module matrix to normal CI, plus release-candidate consumer smoke test after rewriting manifests. TASK19 recorded off-workspace proof exists but shouldn't remain one-shot history. |
| A-D15 | Makefile:20-37 tests planner/resilience twice (module loop then test-examples), no -count=1, ambient go.work. | Separate cached developer quick target from fresh acceptance target; enumerate modules once and run explicit examples compile where tests don't cover. Keep race target; no need re-run unchanged suite redundantly. |
| A-D16 | Makefile:45-54 `-fuzz=.` assumes one fuzz function per package; presently only one in chunking. | Future-proof enumerate fuzz names; explicit bounded budget/cancellation. This is design robustness, not present failure. Empty .PHONY `examples`, `bench-hotpath` should be implemented or removed. |
| A-D17 | release.sh BSD sed -i '', version policy increments major for >=1 but no /vN import migration; tags generated for examples too. | Declare release platform or use portable manifest editor; classify publishable modules separate example modules; before v2 require semantic import-version planning. Current v0 path not v2 bug. |
| A-D18 | CI pinned golangci-lint2.11.3, local reviewer may newer; TASK19 actual parser/provider/DB profile distinctions clear. | Record tool versions and maintain validated CI toolchain; preserve opt-in actual service profiles. No claim local bridge fake proves live DB; don't demand new live benchmark solely because review occurs. |
| A-D19 | README exposes typed documents and filter schema but private wire maps allowed. | BYOT boundary sound; retain wire maps only adapter codec/DSL layer. Do not impose universal business metadata, chat history, account model or service config. Metadata deep-copy ownership should remain explicit host policy. |
| A-D20 | TASK19 accepted with 100% task checklist and quality profiles failing promotion; public matrix qualifies dated/profile limits. | Preserve historical acceptance and raw evidence, add current review remediation link rather than rewriting old acceptance as if never passed. Never equate completeness percentages with production readiness. Codex CLI external runner remains advisory outside core. |

## Positive conclusions / reviewed limits

Core root go.mod has no runtime external deps; providers/storage/OTel/PDF each separate modules. No new library split justified merely by package count. Public matrix is unusually explicit about live-only external stores vs pinned local capabilities, original-source retention, candidate-universe exactness and host authority. Observation strips arbitrary text and serializes callbacks; cooperative/reentrancy constraints are documented rather than disguised as background guarantees. External conformance validates before materialization with negative fixtures, not only final allowed IDs. README/integration current score semantics correctly distinguish native vs normalized and do not claim all scores [0,1].

No extra core runtime bug established within this scope. Parent and other agents own deep retrieval/access/persistence/provider/quality review and full tests. Do not turn the design register into 20 mandatory architectural rewrites: each item needs explicit retain/change and proportional acceptance criteria.
