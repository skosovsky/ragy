# Lifecycle publication and maintenance

`lifecycle` coordinates revision-bound ingestion, publication and managed cleanup.
Executor and Cleaner own checkpoint/CAS invariants. The host owns scheduling,
retry decisions, retention policy, authorization and source/target availability.
Neither component runs an agent loop, background worker or hidden retry.
All context-taking APIs require a non-nil context; use `context.Background()` or
`context.TODO()` without a parent. Nil is unsupported. Some APIs defensively reject
it, but that is not a package-wide nil-context guarantee.

## Three publication concepts

| Value | Meaning | Guarantee |
|---|---|---|
| `lifecycle.Publication` | Source → active manifest pointer within a namespace snapshot | Atomically selected committed source inventory |
| `access.Publication` | Immutable read inventory captured for target branches | Exact admitted revision tuples at capture, owned value containers |
| `lifecycle.PublicationPin` | Durable registered metadata reservation | Protects referenced metadata until explicit release |

`CapturePublication` loads once and checks that Store returned the requested
namespace and a valid current-schema snapshot. A misrouted or malformed response
returns `ragy.ErrProtocol`, zero publication and no writes; invalid caller
namespace/targets/store returns `ragy.ErrInvalidArgument` before Load.
Strict capture requires each requested target ready for every active non-tombstone
source. `CapturePartialPublication` explicitly excludes an entire unavailable
branch, carries its coverage and never substitutes another revision. Complete-empty
namespace/tombstone inventory is a pinned empty observation; all-excluded partial
inventory is unavailable, not complete-empty.

Capture does not create a discoverable lease. Use
`AcquirePublicationPin(ctx, store, namespace, pinID, targets)` before promising
retained metadata. It registers exact current ready tuples under namespace CAS.
A live ID replays its original inventory after publication advances; a released ID
cannot be acquired again. `ReleasePublicationPin` ends metadata protection, with
no expiry or TTL. Hosts synchronize reader release and registration.
Neither capture nor pin prevents target cleanup, restores a volatile index,
retains source bytes or grants access. Those are separate host/adapter contracts;
readers still pass admission, publication and freshness checks.

## Explicit operations and legal replay

| Operation | Durable effect and replay |
|---|---|
| `Executor.Prepare` | Reserves the unchanged plan/ID/key; no target dispatch. Exact replay returns owned state; changed identity conflicts. |
| `Executor.Stage` | Persists unknown checkpoint before one target Stage. Already-ready target does not restage; unknown requires Reconcile. |
| `Executor.Reconcile` | Calls Inspect once for a previously uncertain target; never repeats Stage. Confirmed target state determines the next legal step. |
| `Executor.Publish` | Expected source publication and namespace generation CAS together; only admitted readiness/explicit partial policy publishes. |
| `Cleaner.Begin` | Captures exact previously published ancestry; no target cleanup. |
| `Cleaner.Attempt` | At most one due destructive Cleanup call, preceded by durable unknown checkpoint. Unknown requires Reconcile. |
| `Cleaner.Reconcile` | At most one InspectCleanup; no cleanup dispatch. |
| `MaintenanceStore.Maintain` | Atomically retires exact eligible metadata handles; no target/source payload deletion. |

Manifest State and Checkpoint distinguish uncertain work from last confirmed
progress. Stage confirms exact planned revision before readiness; publication is
separate from physically staged target records. Cleanup tracks waiting → unknown
→ done per captured item, with receipts retained. `Complete` confirms all captured
cleanup items; it does not certify tenant deletion, privacy erasure or destruction
of all historical/source data. Tombstone publication, target cleanup and source
retention are distinct operations.

`Prepare` replay compares the serialized original plan, including target/artifact
order and wire nil/empty distinctions. Replacement validation also preserves
reserved target/artifact order and support identities. Preserve the exact original plan when retrying; do not
sort/canonicalize reserved plans during an upgrade. Sorting capture target selection
for stable publication identity does not change plan replay semantics.

## Conflict, unknown state and recovery

Generation covers the entire namespace, including pins and unrelated source
mutations. That deliberately conservative CAS can conflict with a single-source
operation. A host reloads and re-evaluates its expected publication/plan before
choosing a retry; the library does not retry automatically.

`CheckReuse` is read-only. Only a positive `ReuseConfirmed` decision rechecks the
namespace generation after exact backend inventory inspection. Earlier Absent,
Changed and Incomplete reasons do not reload. Even Confirmed is an observation
point, not a permission or lease; it can become stale immediately after final
Load. Do not cache `CanSkip` indefinitely or bypass subsequent publication gates.

| Error/operation | What is unknown or rejected | Host recovery |
|---|---|---|
| `ErrConflict` | Generation/expected publication changed, or local writer busy | Reload and re-evaluate; choose explicit retry/reconciliation. |
| `ErrOutcomeUnknown` from Stage/Attempt | Dispatch checkpoint may precede an operation whose response was lost, or dispatch may not have occurred | Load exact persisted handle, then Reconcile using Inspect/InspectCleanup; no blind destructive replay. |
| `ErrOutcomeUnknown` from Inspect/CheckReuse | Actual backend state cannot be confirmed by a read-only observation | Re-observe when host policy permits; this error alone does not prove a destructive call occurred. |
| `ErrOutcomeUnknown` from CAS/publication/pin/maintenance | Commit may already have succeeded despite failed acknowledgement | Load the same namespace and reconcile exact IDs/inventory/generation; do not assume rollback. |
| `ErrCleanupNotDue` / `ErrCleanupOverdue` | Host schedule/deadline disallows this attempt | Use persisted NextAt/backoff; overdue work requires explicit recovery flag. |
| `ErrCapacity` | Full retained snapshot exceeds configured byte profile | Provision a sufficient profile or explicitly move/convert the store; no implicit eviction. |
| `ErrRetired` / `ErrProtected` | Handle permanently retired, or requested retirement intersects protected metadata | Respect reservations/references; choose another valid operation or explicit host retention transition. |
| `ErrIdempotencyConflict` | ID/key reserved for a different plan | Recover original exact plan; never rebind its identity. |
| `ragy.ErrProtocol` | Store/port response violates the declared contract | Reject the response and repair the adapter; no guessed success. |

Cleaner receives a host clock, deadline and positive backoff schedule. Backoff and
NextAt are schedule outputs, not sleeping retries. The recovery bool explicitly
permits overdue work; it does not override unknown-outcome inspection. No scheduler
or generic retry middleware is implied by any error.

## Reservations, ownership and authority

Retired skeletons, artifact identity digests, released pin IDs and bootstrap/cleanup
receipts are permanent ABA/replay reservations. They cannot disappear under TTL or
capacity pressure. Maintenance removes selected artifact/support-heavy history
only after exact reference closure and confirmed cleanup checks; it retains IDs,
keys, fingerprints, ancestry, checkpoints, target names and reservation digests.
Unfinished operations/jobs and live metadata pin tuples protect referenced history.

`CompactHistory` uses a full JSON roundtrip to own all nested wire collections;
manifest/job/pin copies likewise preserve independent ragy containers. These are
ownership guarantees, not unexplained fallback. Current clone and scan algorithms
are retained, with no speedup claim. Optimization requires measured profiles and
differential checks for ownership, nil/empty distinctions, order and reservation
semantics; no second persisted index/service is introduced here.

`Store.CompareSwap` is a trusted low-level checkpoint interface. Custom stores must
apply `ValidateReplacement` before committing, preserving immutable reservations
and monotonic receipts. It is not a command authorization API or a validator for
all arbitrary transitions supplied outside Executor/Cleaner. Host adapter admission,
atomic generation/durability and correct namespace responses remain required.

`FencedInventoryVerifier` acquires target fences in sorted order, observes exact
inventory (including opaque unmanaged keys for complete coverage), and requires
each observer to invoke its continuation synchronously exactly once. Ports are
cooperative; there is no forced goroutine detachment or hidden timeout worker.
Confirmation is an observation point, not a distributed commit or a guarantee
about subsequent source/target mutations. Preserve these fences when integrating
with managed indexes.

## Scale, filesystem profile and schema

Finite port-call count is not constant CPU, bytes, allocation or execution time.
Snapshot validation/cloning, ancestry/reservation scans, cleanup reference closure
and whole-snapshot serialization scale with retained namespace history and may
perform nested scans. Compaction reduces heavy inventories but permanent identities,
digests/pins/receipts still grow. Host callbacks also enforce their own bounded work.

`filestore` supports darwin/linux and trusted local filesystems providing
nonblocking flock, atomic rename, file fsync and directory fsync. A competing writer
returns `ErrConflict` rather than waiting indefinitely. Locks are an OS/filesystem
contract, not a hostile-filesystem sandbox or network-filesystem fallback. Host
owns root permissions, provisioning, backups and source/target retention.

`filestore.New(root, maxSnapshotBytes)` requires a finite full-snapshot byte budget
for reads and writes. The limit includes retained reservations/receipts; it is not
an RSS, peak serialization allocation or CPU cap. New can create directories, but
file/root fsync does not establish power-loss durability of newly created ancestors.
A host requiring that guarantee must provision and durably initialize the root
and ancestors. Process-restart fixtures do not certify hardware power-loss behavior.

Current schema is [`ragy.lifecycle/v2`](../schemas/lifecycle.schema.json). Earlier
schema snapshots are rejected; opening never auto-converts, deletes payloads or
reindexes. Any offline host conversion must preserve all IDs, keys, inventories,
checkpoints, receipts and retained source/target data, validate the resulting
snapshot, and register active metadata profiles before enabling retirement. Keep
historical evidence in [task18 maintenance notes](../docs/task18/lifecycle-maintenance.md);
use this guide for current runtime contracts.
