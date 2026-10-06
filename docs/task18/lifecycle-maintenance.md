# Local lifecycle maintenance

The current namespace schema is `ragy.lifecycle/v2`. Maintenance is an optional
`lifecycle.MaintenanceStore` capability. Host retention decisions stay outside the
library. The local implementation operates on the complete namespace snapshot:
nonblocking flock, generation CAS, temporary file fsync, atomic rename and directory
fsync. It requires a local filesystem providing those operations.

## Retention and reference model

`AcquirePublicationPin(ctx, store, namespace, pinID, targets)` captures current
publication metadata and registers exact target revision tuples in the same CAS.
Register before advertising retained lifecycle metadata to readers. The returned
`access.Publication` owns its target inventory; the durable registration survives
restart. `ReleasePublicationPin` ends protection explicitly. Released IDs remain
reserved and cannot be acquired again. There is no expiry, scheduler or background
worker. A failed commit may already have succeeded: load the snapshot to reconcile
`ErrOutcomeUnknown`; do not assume rollback or blindly repeat another operation.

`CapturePublication`, `CapturePartialPublication`, and host-created `access` pins
remain observations. Copies of these values are not discoverable leases. Hosts
using them must register the corresponding active target profile before promising
metadata retention, coordinate publication changes and synchronize release with
readers. Metadata protection does not prevent target cleanup, restore a volatile
index or preserve source bytes; target/source retention and IAM remain separate
host contracts. Acquisition rejects unavailable/unknown active target inventories.
An existing live pin ID replays its original inventory after publication advances.

`Maintain(ctx, expectedGeneration, RetirementRequest{Namespace, Manifests})`
selects exact manifest IDs, with no age heuristic or implicit deletion. Only
noncurrent inventories with confirmed per-target cleanup are eligible. Unknown
manifest/target outcomes, every unfinished operation's reference chain, all
unfinished cleanup owners/items/ancestry, and registered live pin tuples protect
history. Empty or duplicate selections are invalid; unknown IDs are unavailable;
protected selections return `ErrProtected`. The selection is atomic: one unsafe
candidate leaves the whole snapshot unchanged. A stale generation returns
`ErrConflict`; hosts own retry/reconciliation policy.

Retirement clears artifact/support arrays while retaining manifest identities,
idempotency keys, payload fingerprints, publication ancestry, target names and
checkpoints. SHA-256 digests reserve exact `(target, source.Reference)` identities.
Completed cleanup receipts stay intact. Later cleanup jobs omit already retired
inventories while following their retained ancestry. Prepare, Stage, Reconcile,
Publish and bootstrap replay of retired handles fail `ErrRetired`. IDs, keys and
artifact identities never alias different data. Cleanup and maintenance are
separate operations; maintenance never invokes a target port or deletes source
payload. Old value-only reader observations must still pass target admission and
freshness gates; retired skeletons cannot establish payload availability.

Custom stores must apply `lifecycle.ValidateReplacement(current, next)` before
committing. Every prior operation plan and artifact inventory stays reserved; only
confirmed workflow checkpoints or explicit safe compaction may change. New rows
reserve exact target/reference identities against current history and every earlier
new row in the same batch. Independent artifacts may share source supports, and
separate targets have distinct ownership keys. Previous retired skeletons/fences,
cleanup and bootstrap receipts, and every registered pin identity remain present; a released pin cannot become live again. Normal
CompareSwap and maintenance share this invariant and a single namespace generation.

## Capacity and scale

`filestore.New(root, maxSnapshotBytes)` requires an explicit finite byte budget.
The limit includes live inventories, retired skeletons/fences, cleanup receipts,
bootstrap receipts and pin registrations, and applies to both reads and writes.
Exceeding it returns `ErrCapacity` without deleting history or replacing committed
state. A host opening an existing snapshot under a smaller budget must deliberately
choose a sufficient budget before it can inspect or compact that state.

Compaction reduces artifact/support-heavy history. It does not promise constant
storage forever: permanently reserved identities, digests, pins and receipts grow
with history. CAS still reads, validates and serializes the whole snapshot; cleanup
reference validation has a history-dependent CPU cost. Hosts exceeding this local
profile should use a store adapter that preserves the same contracts. Metadata
only workloads can gain little space from compaction.

## Migrating an old namespace

The former `ragy.lifecycle` schema is explicitly unsupported. No automatic upgrade,
overwrite, payload deletion or reindex runs when opening it.

1. Stop namespace writers and readers that depend on the old runtime; back up the
   complete old root and independently retained source/target data.
2. Offline, validate the original snapshot against the retained historical schema
   and semantic inventory contract. Copy it to a new root, preserving every
   manifest, ID/key, publication, cleanup checkpoint and inventory receipt.
3. In the copy only, set `schema` to `ragy.lifecycle/v2`; set `pins` to `null`; add
   `retired:false` and `artifact_fences:null` to every manifest. Do not compact or
   invent retirement during migration. The current executable schema is
   `schemas/lifecycle.schema.json`; opening and `Snapshot.Validate` must also pass.
4. Register the host's still-retained publication profiles before enabling
   maintenance, then switch readers/writers to the new root. Preserve the backup.

Alternatively bootstrap/reindex a fresh root through verified inventories and
source revision contracts. Keep old roots and retained source bytes until the host
explicitly authorizes their retention change. Migration cannot restore lost
volatile lexical/graph corpus or infer source ownership from opaque records.
