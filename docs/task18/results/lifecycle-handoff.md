# TASK18 lifecycle implementation handoff

Sources frozen after the final exact-reference raw-CAS reservation guard.
No commits created. Existing task12 wire fixtures/schemas remain historical.

Implemented public contracts:

- Single namespace schema `ragy.lifecycle/v2`; old schema explicitly unsupported.
- `MaintenanceStore`, exact `RetirementRequest`, `CompactHistory` and mandatory
  `ValidateReplacement`; `ErrCapacity`, `ErrProtected` and `ErrRetired`.
- Durable `AcquirePublicationPin` / `ReleasePublicationPin`, canonical requested
  target profile, historical live-ID replay, permanent released-ID reservation.
- Retired manifest skeletons retain ID/key/identity/plan/ancestry/target checkpoints
  and exact artifact identity SHA-256 fences while clearing artifact/support arrays.
- Unknown/active work, unfinished cleanup reference closure, current publication
  and registered live pin tuples prevent metadata retirement.
- Direct CAS cannot erase/rebind any previous operation plan, retired skeleton,
  artifact fence, cleanup owner/item/confirmed completion, bootstrap watermark
  receipt or publication pin. New rows cannot overlap prior exact artifact identities.
- Operational replay rejects retired handles; new cleanup follows retained ancestry
  while omitting already retired inventories. Existing completed receipts remain.

Validation of the final frozen sources:

- `lifecycle-frozen-race.txt`: lifecycle 4.698s; filestore 3.315s;
  graphingest/materialization 2.080s; all PASS with `-race -count=1`.
- `lifecycle-frozen-final-lint.txt`: compatible `/opt/homebrew/bin/golangci-lint`,
  lifecycle/... and materialization/..., 0 issues.
- `lifecycle-schema.txt`: independently executed Draft 2020-12 schema verifier,
  8 positive and 42 negative examples passed.
- `git diff --check`: PASS.
- `lifecycle-final-race.txt`: full lifecycle integration `-race` PASS, 88.179s.
  This run preceded helper-only lint refactors and the last additional raw-CAS
  fence guard; root's frozen full checks provide final all-package confirmation.

Actual filesystem tests include child-process SIGKILL inside Maintain before
rename, orphan temporary-file handling, lock release, fresh-process restart,
independent concurrent CAS/maintenance writers, deterministic cancellation before
and after rename, actual rename failure, byte-budget refusal and incompatible
schema preservation. Pin tests use actual filestore restart, historical publication
advance/cleanup, explicit release, retirement interleaving and acknowledgement loss.

Earlier failure history is intentionally separate from final success:

- Initial `active_plan_ancestor` fixture used an already superseded expectation
  that necessarily belonged to the completed cleanup inventory; it was corrected
  to a future plan whose expectation is the current owner, retaining protection
  through its ancestry. The initial filestore pin fixture lacked the newly required
  RequestedTargets profile; its author corrected the fixture.
- Earlier lint logs retain temporary complexity/format/profile diagnostics.
  The final frozen lint log supersedes them. Default user-bin lint was compiled
  with Go 1.26 and unsuitable for the Go 1.27 module; compatible Homebrew lint ran.
- Existing cleanup crash fixture previously erased committed job state via raw
  CAS. It now constructs an independent pre-Begin initial snapshot, respecting
  the strengthened append-only identity/fence contract.

The new `BenchmarkTask18Retirement` measures identical confirmed-cleaned history
pairs before/after explicit compaction in the current implementation, distinct
from the original active-publication baseline. Setup/maintenance are outside timed
ordinary CAS; history sizes 10/100/1000, 32 artifacts per old manifest. Root captures
these measurements during the shared quiet benchmark window.

Limits: full snapshot CAS and history-dependent validation remain; reserved
identities, digest fences, jobs, receipts and pins eventually consume the explicit
byte budget. Metadata retention does not restore source/target bytes or authorize
read access. No background GC, scheduler, distributed protocol or automatic upgrade.
