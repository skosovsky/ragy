# T04 correctness — PASS

Independent reviewer; baseline `b32a7e35106b8442f5a306559bf5589ff8e391ad`. No implementation edits; completeness report not read. Reviewed F03, architecture A-F02, T04.C01–C04 and normative release scope/recovery contract. All publication checks used actual synthetic disposable local bare repositories; no production publication/refs/global configuration changes. No unresolved correctness findings in current diff.

## Initial P2 and resolved regression

Initial review was FAIL: pending preparation accepted arbitrary changed contents within filename-allowlisted manifests. Actual reproduction `/private/tmp/ragy-t04-correctness/preparation_tamper.py` failed Go before editing, changed retained root module to `example.invalid/unreviewed`, then observed successful actual publication of that unreviewed root manifest. Commit-before-record recovery had the same authorization gap.

Repair reviewed: `intended_manifests` independently reconstructs exact source manifest bytes through the allowed Go edit operations and persisted version. Pending manifests must equal original or canonical content; recovered/stored committed manifests must equal canonical content. Recovered HEAD is validated through a temporary record before persistent candidate identity changes. Source ancestry, file allowlist, isolated clean state, owned ref identity and destination checks remain. The test initially needed its checkout/Go-failure arrangement corrected; final full suite passes.

Independent `/private/tmp/ragy-t04-correctness/canonical_final.py` rerun against latest code: module-path alteration and external-dependency insertion both rejected in incomplete working preparation and independently committed recovery windows, four cases PASS. Each case preserved source/version, absent candidate identity, original altered evidence, caller bytes/index and zero remote publication. Permanent parameterized regression covers the same two mutation classes and two preparation states. This resolves the original P2; module-name-only validation was not accepted.

## Independent final checks

| Command / scenario | Outcome | Evidence |
|---|---|---|
| `PYTHONDONTWRITEBYTECODE=1 python3 scripts/release_test.py -v` | PASS, 9 methods, 13.009s | `/private/tmp/ragy-t04-correctness/isolation-latest.log` |
| `PYTHONDONTWRITEBYTECODE=1 python3 scripts/release_recovery_test.py -v` | PASS, 12 methods including parameterized cases, 43.959s | `/private/tmp/ragy-t04-correctness/recovery-latest.log` |
| `PYTHONDONTWRITEBYTECODE=1 python3 /private/tmp/ragy-t04-correctness/adversarial.py` | PASS, 5 scenarios | `/private/tmp/ragy-t04-correctness/adversarial-latest.log` |
| `PYTHONDONTWRITEBYTECODE=1 python3 /private/tmp/ragy-t04-correctness/canonical_final.py` | PASS, 4 mutation/window cases | `/private/tmp/ragy-t04-correctness/canonical-latest.log` |
| `bash -n scripts/release.sh`, AST parse four Python modules, `git diff --check` | PASS | Independent command exits 0 |

Supplemental adversarial tests issue actual SIGKILL to the release process immediately after successful real isolated commit, first tag and real push. Each recovers the exact same candidate/version; post-push record is unknown and refuses resume until explicit inspect. Changed origin rejects without changing stored record or caller; changed owned isolated tag rejects and leaves remote empty. These supplement ordinary subprocess-return failure tests, rather than assuming interrupt behavior from exit codes.

The full suites independently cover actual rejecting bare hook, caller branch/index/files/foreign-tag preservation, caller index bytes under changed tracked-file stat metadata, retries without version increments, partial external exact ref publication and missing-only atomic push, inspection loss after actual push, failed transport exit despite actual complete publication, preexisting different objects and annotated-tag-object collisions, unsupported atomic capability without fallback, exact completed-record archive before a new candidate/source, and private/untracked/example exclusion. None/partial/complete/collision derives from exact observed object IDs; unknown cannot dispatch until explicit inspect. Caller refs are never cleanup targets. Public runbook agrees with the inspected state machine and canonical authorization.

This PASS is scoped to T04. It makes no production release, hardware power-loss, clean-consumer, all-module CI or T21/T22 claim.

## Reviewed substantive SHA256

- `scripts/release.py`: `3173e65c294d8bbe59dd6c8a57538eb4d10d1935d9dfac1ea07fbc90b5356daf`
- `scripts/release_state.py`: `6f4ccee78a4ab00ec3860b2536f226bf122d0745fb3cfc469f2c43e6c37da429`
- `scripts/release_test.py`: `79fe4b4469f5895b6fa11b6408b68c506278db5bf09cc535ae05c9a5282d5466`
- `scripts/release_recovery_test.py`: `92d8ee6adf627ffcfbc4b898d362d05c2c802775608eb4db93a98fe715c68103`
- `docs/release/runbook.md`: `61934d84d5597097920c12ff6214d8bf3cf07db6319abc2775077272ba2eac3e`
- `docs/task20/T04.md`: `85f576b67a5a8f8854cce57bf9704cfd5c428422488ca839efd83e85f08c8f92`
