# Managed BM25 target

`managed.Adapter[TMeta]` implements lexical retrieval, lifecycle `StagePort` and
`CleanupPort`. Register it in `Executor` and `Cleaner`, supplying a finalized
metadata schema, host metadata clone function, durable manifest store and validated
payload fingerprint. Stage the full `[]managed.Record[TMeta]` inventory.

Use `lifecycle.CapturePublication` before planning/fan-out and construct the trusted
binding with that publication. This adapter requires pinned reads and rejects live
admission before I/O. Complete capture requires every requested target ready. Explicit partial capture
freezes excluded targets; this adapter rejects a pin excluding its own target. Tombstoned sources are
excluded; an empty capture is pinned complete-empty, never a live fallback.

Records are keyed by exact namespace/source/revision/transformation/access identity.
Returned document IDs are canonical artifact-reference hashes; original artifact IDs
remain in each record's reference and host metadata. Stage preserves supplied source locators and adds the canonical artifact reference.
Supply original mappings/supports on the staged document; an artifact reference alone
does not establish original quotation precision. Query scope metadata is checked before payload cloning. Readonly BM25 snapshots
retain the original binding fingerprint, gating callbacks and final delivery.

Staging and exact-version cleanup share the adapter lock. New source publications
cannot make another version's records disappear through cleanup of an old manifest.
Old pinned reads can complete from already captured records; a physically removed
snapshot returns unavailable with no latest substitution or partial sibling results.

This reference adapter stores documents **in process memory**. Durable manifest state
does not imply durable lexical records. A fresh adapter without restored records
returns snapshot unavailable; host record persistence/restoration is not supplied by
this adapter. Persistent dense/tensor acceptance remains separate.

`Adapter.InventoryObserver(maxEntries, maxRecords)` provides a bounded observation
port for `lifecycle.NewFencedInventoryVerifier`. It holds the actual mutation mutex
through the nested observation callback. Complete inventory accounts for retained
versions by `manifest:<operation-id>` opaque keys; missing memory is unavailable even
when a durable ledger retains a ready checkpoint. The observer performs no indexing,
source adoption, host metadata callback or deletion.
